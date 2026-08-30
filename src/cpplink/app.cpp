// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/app.hpp"

#include <ostream>

#include "cpplink/comparison.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/inspect.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

const char* const kVersion = "0.1.0";

namespace {

void PrintUsage(std::ostream& out) {
    out << "usage: cpplink <command> [options]\n"
        << "\n"
        << "commands:\n"
        << "  inspect     load a parquet file and report cardinality and memory\n"
        << "  explain     show the comparison levels a single pair lands on\n"
        << "  gen-sample  write a sample parquet file with planted duplicates\n"
        << "\n"
        << "options:\n"
        << "  -h, --help       show this message and exit\n"
        << "  -v, --version    show the version and exit\n"
        << "\n"
        << "cpplink inspect --schema <schema.json> <file.parquet>\n"
        << "cpplink explain --schema <schema.json> --pair <id_a>,<id_b> "
           "<file.parquet>\n"
        << "cpplink gen-sample --out <file.parquet> [--rows N] [--seed N]\n"
        << "                   [--duplicate-rate F] [--truth <file.csv>]\n";
}

// Reads "--name value" pairs. Returns false and reports on a missing value.
bool TakeValue(const std::vector<std::string>& args, size_t* index, std::string* value,
               std::ostream& err) {
    if (*index + 1 >= args.size()) {
        err << "cpplink: " << args[*index] << " needs a value\n";
        return false;
    }
    *value = args[++(*index)];
    return true;
}

int RunInspect(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::string data_path;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink inspect: unknown option '" << args[i] << "'\n";
            return 1;
        } else if (data_path.empty()) {
            data_path = args[i];
        } else {
            err << "cpplink inspect: unexpected argument '" << args[i] << "'\n";
            return 1;
        }
    }
    if (schema_path.empty() || data_path.empty()) {
        err << "cpplink inspect: --schema <schema.json> and a parquet file are "
               "required\n";
        return 1;
    }

    Schema schema;
    std::string error;
    if (!LoadSchema(schema_path, &schema, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    RecordStore store(schema);
    LoadStats stats;
    if (!LoadParquet(data_path, schema, &store, &stats, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    out << "File         " << data_path << "\n";
    PrintInspection(store, stats, out);
    return 0;
}

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

int RunExplain(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::string data_path;
    std::string pair;
    std::string rows;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--pair") {
            if (!TakeValue(args, &i, &pair, err)) return 1;
        } else if (args[i] == "--rows") {
            if (!TakeValue(args, &i, &rows, err)) return 1;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink explain: unknown option '" << args[i] << "'\n";
            return 1;
        } else if (data_path.empty()) {
            data_path = args[i];
        } else {
            err << "cpplink explain: unexpected argument '" << args[i] << "'\n";
            return 1;
        }
    }
    if (schema_path.empty() || data_path.empty()) {
        err << "cpplink explain: --schema <schema.json> and a parquet file are "
               "required\n";
        return 1;
    }
    if (pair.empty() == rows.empty()) {
        err << "cpplink explain: give exactly one of --pair <id_a>,<id_b> or "
               "--rows <i>,<j>\n";
        return 1;
    }

    Schema schema;
    std::string error;
    if (!LoadSchema(schema_path, &schema, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    if (schema.comparisons.empty()) {
        err << "cpplink explain: the schema declares no \"comparisons\"\n";
        return 1;
    }

    RecordStore store(schema);
    if (!LoadParquet(data_path, schema, &store, nullptr, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    ComparisonSet comparisons;
    if (!comparisons.Bind(schema, store, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    uint64_t row_a = 0;
    uint64_t row_b = 0;
    std::string first;
    std::string second;
    if (!pair.empty()) {
        if (!SplitPair(pair, &first, &second)) {
            err << "cpplink explain: --pair wants <id_a>,<id_b>\n";
            return 1;
        }
        if (!FindRowById(store, first, &row_a)) {
            err << "cpplink explain: no record with id '" << first << "'\n";
            return 1;
        }
        if (!FindRowById(store, second, &row_b)) {
            err << "cpplink explain: no record with id '" << second << "'\n";
            return 1;
        }
    } else {
        if (!SplitPair(rows, &first, &second)) {
            err << "cpplink explain: --rows wants <i>,<j>\n";
            return 1;
        }
        row_a = std::stoull(first);
        row_b = std::stoull(second);
        if (row_a >= store.NumRecords() || row_b >= store.NumRecords()) {
            err << "cpplink explain: row out of range; the file has "
                << store.NumRecords() << " records\n";
            return 1;
        }
    }

    PrintGammaLayout(comparisons, out);
    out << "\n";
    PrintPairExplanation(store, comparisons, row_a, row_b, out);
    return 0;
}

int RunGenSample(const std::vector<std::string>& args, std::ostream& out,
                 std::ostream& err) {
    SampleOptions options;
    std::string out_path;
    std::string value;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--out") {
            if (!TakeValue(args, &i, &out_path, err)) return 1;
        } else if (args[i] == "--rows") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.rows = std::stoull(value);
        } else if (args[i] == "--seed") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.seed = std::stoull(value);
        } else if (args[i] == "--duplicate-rate") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.duplicate_rate = std::stod(value);
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &options.truth_path, err)) return 1;
        } else {
            err << "cpplink gen-sample: unknown option '" << args[i] << "'\n";
            return 1;
        }
    }
    if (out_path.empty()) {
        err << "cpplink gen-sample: --out <file.parquet> is required\n";
        return 1;
    }

    std::string error;
    if (!WriteSampleParquet(out_path, options, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    out << "Wrote " << options.rows << " rows to " << out_path << "\n";
    if (!options.truth_path.empty()) {
        out << "Planted duplicate pairs listed in " << options.truth_path << "\n";
    }
    return 0;
}

}  // namespace

int Run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (args.empty()) {
        PrintUsage(out);
        return 0;
    }
    const std::string& first = args[0];
    if (first == "-h" || first == "--help") {
        PrintUsage(out);
        return 0;
    }
    if (first == "-v" || first == "--version") {
        out << kVersion << "\n";
        return 0;
    }

    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (first == "inspect") return RunInspect(rest, out, err);
    if (first == "explain") return RunExplain(rest, out, err);
    if (first == "gen-sample") return RunGenSample(rest, out, err);

    err << "cpplink: unknown command '" << first << "'\n";
    PrintUsage(err);
    return 1;
}

}  // namespace cpplink
