// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/pipeline.hpp"

#include <cstdlib>
#include <iomanip>
#include <ostream>

#include "cpplink/id_index.hpp"
#include "cpplink/merge_edges.hpp"

namespace cpplink {

bool LoadSchemaFor(const std::string& schema_path,
                   const std::vector<std::string>& data_paths, Schema* schema,
                   std::string* error) {
    return LoadSchema(schema_path, schema, error) &&
           ResolveColumnTypes(data_paths, schema, error);
}

bool ParseMode(const std::string& text, PairMode* mode, std::string* error) {
    if (text == "dedup" || text == "link-and-dedup") {
        *mode = PairMode::kAll;
        return true;
    }
    if (text == "link") {
        *mode = PairMode::kCrossDataset;
        return true;
    }
    *error = "--mode must be dedup, link or link-and-dedup, not '" + text + "'";
    return false;
}

PairMode DefaultMode(bool given, PairMode mode, size_t inputs) {
    if (given) return mode;
    return inputs > 1 ? PairMode::kCrossDataset : PairMode::kAll;
}

bool SchemaForPlan(const Schema& full, bool all_pairs, bool for_estimation,
                   Schema* schema, std::string* error) {
    *schema = for_estimation ? SchemaForEstimation(full) : SchemaForPrediction(full);
    if (all_pairs) {
        BlockingSpec spec;
        spec.kind = SourceKind::kAllPairs;
        spec.name = "all pairs";
        schema->blocking.assign(1, spec);
    } else if (schema->blocking.empty()) {
        *error =
            "the schema declares no \"blocking\" sources; pass --all-pairs to\n"
            "         enumerate every pair instead";
        return false;
    }
    return true;
}

bool BuildPlan(const Schema& full, const RecordStore& store, PairMode mode,
               bool all_pairs, bool for_estimation, BlockingPlan* plan,
               std::string* error) {
    Schema schema;
    return SchemaForPlan(full, all_pairs, for_estimation, &schema, error) &&
           plan->Build(schema, store, mode, error);
}

bool LoadForBlocking(const std::string& schema_path,
                     const std::vector<std::string>& data_paths, PairMode mode,
                     Schema* schema, std::unique_ptr<RecordStore>* store,
                     BlockingPlan* plan, std::string* error, bool all_pairs,
                     LoadStats* stats, bool for_estimation) {
    Schema full;
    if (!LoadSchemaFor(schema_path, data_paths, &full, error)) return false;
    if (!SchemaForPlan(full, all_pairs, for_estimation, schema, error)) return false;
    *store = std::make_unique<RecordStore>(*schema);
    if (!LoadParquetFiles(data_paths, *schema, store->get(), stats, error)) return false;
    return plan->Build(*schema, **store, mode, error);
}

bool LoadIdsOnly(const std::string& schema_path,
                 const std::vector<std::string>& data_paths,
                 std::unique_ptr<RecordStore>* store, std::string* error) {
    Schema schema;
    if (!LoadSchema(schema_path, &schema, error)) return false;
    schema.columns.clear();
    schema.comparisons.clear();
    schema.blocking.clear();
    *store = std::make_unique<RecordStore>(schema);
    return LoadParquetFiles(data_paths, schema, store->get(), nullptr, error);
}

bool ResolveEdgeOutput(const std::string& command, const std::string& out,
                       bool format_given, std::string* out_dir, std::string* merge_path,
                       std::string* error) {
    MergeFormat format = MergeFormat::kCsv;
    if (!MergedFormatOf(out, &format)) {
        *out_dir = out;
        return true;
    }
    if (format_given) {
        *error = command + ": --format names the shard format, and --out " + out +
                 " asks for a single file; the file's extension picks csv or parquet";
        return false;
    }
    *merge_path = out;
    *out_dir = out + ".shards";
    return true;
}

void BuildBallTables(const ComparisonSet& comparisons, const RecordStore& store,
                     const BallOptions& options, BallTables* balls, std::ostream& out) {
    balls->Build(comparisons, store.NumRecords(), options);
    out << "Neighbourhood masses in " << std::fixed << std::setprecision(1)
        << balls->seconds << " s";
    if (store.NumDatasets() > 1) {
        out << ", over the " << store.NumDatasets()
            << " inputs pooled: a value's neighbourhood is as heavy as it is across all "
               "of them";
    }
    out << "\n";
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        out << "  " << comparisons.at(c).spec->name << ": ";
        if (balls->Has(c)) {
            out << balls->tables[c].Values() << " values, "
                << balls->tables[c].ValuePairs() << " value pairs, "
                << std::setprecision(2) << balls->tables[c].Seconds() << " s\n";
        } else {
            out << balls->reasons[c] << "\n";
        }
    }
    out << "\n";
}

bool ResolveId(const RecordStore& store, const std::string& text, uint64_t* row,
               std::string* error) {
    const IdLookup found = FindRowById(store, text, row);
    if (found == IdLookup::kMissing) {
        *error = "no record with id '" + text + "'";
        return false;
    }
    if (found == IdLookup::kAmbiguous) {
        *error = "more than one record has id '" + text +
                 "'; name its input as <dataset>:<id>, where the datasets are";
        for (size_t d = 0; d < store.NumDatasets(); ++d) {
            *error += (d == 0 ? " " : ", ") + store.DatasetName(d);
        }
        return false;
    }
    return true;
}

namespace {

// Splits "a,b" into its two halves.
bool SplitPair(const std::string& text, std::string* first, std::string* second) {
    const size_t comma = text.find(',');
    if (comma == std::string::npos || comma == 0 || comma + 1 >= text.size()) {
        return false;
    }
    *first = text.substr(0, comma);
    *second = text.substr(comma + 1);
    return true;
}

}  // namespace

bool ResolvePair(const RecordStore& store, const std::string& text, bool by_row,
                 uint64_t* row_a, uint64_t* row_b, std::string* error) {
    std::string first;
    std::string second;
    if (!SplitPair(text, &first, &second)) {
        *error = by_row ? "wants <i>,<j>" : "wants <id_a>,<id_b>";
        return false;
    }
    if (by_row) {
        char* end_a = nullptr;
        char* end_b = nullptr;
        *row_a = std::strtoull(first.c_str(), &end_a, 10);
        *row_b = std::strtoull(second.c_str(), &end_b, 10);
        if (*end_a != '\0' || *end_b != '\0') {
            *error = "wants <i>,<j>";
            return false;
        }
        if (*row_a >= store.NumRecords() || *row_b >= store.NumRecords()) {
            *error = "row out of range; the file has " +
                     std::to_string(store.NumRecords()) + " records";
            return false;
        }
        return true;
    }
    return ResolveId(store, first, row_a, error) &&
           ResolveId(store, second, row_b, error);
}

}  // namespace cpplink
