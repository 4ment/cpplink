// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/waterfall.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cpplink/arrow_export.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/format.hpp"
#include "cpplink/id_index.hpp"
#include "cpplink/merge_edges.hpp"
#include "cpplink/parquet_io.hpp"

namespace cpplink {
namespace {

// One prediction as the merged file names it: a record is its id, qualified by
// its dataset where the file carries one. `gamma` is what the run stored, kept
// so a schema that has drifted from the file can be noticed.
struct NamedPrediction {
    std::string_view dataset_a;
    std::string_view id_a;
    std::string_view dataset_b;
    std::string_view id_b;
    uint32_t gamma = 0;
};

using PredictionSink = std::function<bool(const NamedPrediction&, std::string*)>;

bool ForEachCsvPrediction(const std::string& path, const PredictionSink& sink,
                          std::string* error) {
    std::ifstream file(path);
    if (!file) {
        *error = "explain: could not open '" + path + "'";
        return false;
    }
    std::string line;
    uint64_t number = 0;
    EdgeCsvLayout layout;
    while (std::getline(file, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (number == 1 && ParseEdgeCsvHeader(line, &layout)) continue;
        EdgeRow edge;
        if (!ParseEdgeCsvLine(line, layout, &edge)) {
            *error = "explain: '" + path + "' line " + std::to_string(number) +
                     " is not a cpplink prediction row";
            return false;
        }
        NamedPrediction prediction;
        prediction.dataset_a = edge.dataset_a;
        prediction.id_a = edge.id_a;
        prediction.dataset_b = edge.dataset_b;
        prediction.id_b = edge.id_b;
        prediction.gamma = edge.gamma;
        if (!sink(prediction, error)) return false;
    }
    return true;
}

// Every row of the merged parquet, through the same reader clustering uses. A
// row with a null id is skipped, as the csv reader skips what it cannot parse.
bool ForEachParquetPrediction(const std::string& path, const PredictionSink& sink,
                              std::string* error) {
    EdgeColumns columns;
    return ReadPredictionFile(
        path, "explain", {"gamma"}, &columns,
        [&](const EdgeRow& edge, std::string* visit_error) {
            if (edge.id_a.empty() || edge.id_b.empty()) return true;
            if (columns.datasets && (edge.dataset_a.empty() || edge.dataset_b.empty())) {
                return true;
            }
            NamedPrediction prediction;
            prediction.dataset_a = edge.dataset_a;
            prediction.id_a = edge.id_a;
            prediction.dataset_b = edge.dataset_b;
            prediction.id_b = edge.id_b;
            prediction.gamma = edge.gamma;
            return sink(prediction, visit_error);
        },
        error);
}

// Column names are the comparison's name with a suffix, and an interaction's is
// the scorer's "left x right" made a legal identifier.
std::string InteractionColumn(const std::string& name) {
    std::string out = name;
    const size_t at = out.find(" x ");
    if (at != std::string::npos) out.replace(at, 3, "_x_");
    return out + "_bits";
}

constexpr size_t kFixedColumns = 11;

class WideSink {
   public:
    virtual ~WideSink() = default;
    virtual bool Write(const PairWaterfall& w, std::string* error) = 0;
    virtual bool Close(std::string* error) = 0;
};

class CsvWideSink : public WideSink {
   public:
    bool Open(const std::string& path, const std::vector<std::string>& columns,
              std::string* error) {
        path_ = path;
        file_.open(path);
        if (!file_) {
            *error = "explain: cannot create " + path;
            return false;
        }
        for (size_t i = 0; i < columns.size(); ++i) {
            if (i > 0) buffer_ += ',';
            buffer_ += columns[i];
        }
        buffer_ += '\n';
        return true;
    }

    bool Write(const PairWaterfall& w, std::string* error) override {
        char numbers[160];
        buffer_ += w.id_a;
        buffer_ += ',';
        buffer_ += w.id_b;
        std::snprintf(numbers, sizeof(numbers), ",%u,%.6f,%.6f,%.9f,%.6f,%.6f,%.6f,",
                      w.gamma, w.prior, w.weight, w.probability, w.bracket_low,
                      w.bracket_high, w.threshold);
        buffer_ += numbers;
        buffer_ += ZoneName(w.zone);
        buffer_ += w.emitted ? ",true" : ",false";
        for (const WaterfallStep& step : w.steps) {
            std::snprintf(numbers, sizeof(numbers), ",%u,%.6f,%.6f,%u",
                          static_cast<unsigned>(step.level), step.bits, step.tf,
                          step.frequency);
            buffer_ += numbers;
        }
        for (const WaterfallInteraction& term : w.interactions) {
            std::snprintf(numbers, sizeof(numbers), ",%.6f", term.bits);
            buffer_ += numbers;
        }
        buffer_ += '\n';
        if (buffer_.size() >= (1u << 20)) return Flush(error);
        return true;
    }

    bool Close(std::string* error) override {
        if (!Flush(error)) return false;
        file_.close();
        return true;
    }

   private:
    bool Flush(std::string* error) {
        file_.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
        buffer_.clear();
        if (!file_) {
            *error = "explain: writing " + path_ + " failed";
            return false;
        }
        return true;
    }

    std::string path_;
    std::ofstream file_;
    std::string buffer_;
};

// The wide row as C Data batches handed to the parquet writer: the same builder
// a data frame is handed in Python, so the two cannot name a column differently.
class ParquetWideSink : public WideSink {
   public:
    ParquetWideSink(size_t batch_rows, size_t comparisons, size_t interactions)
        : batch_rows_(batch_rows == 0 ? 1 : batch_rows),
          comparisons_(comparisons),
          interactions_(interactions) {}

    bool Open(const std::string& path, const std::vector<std::string>& columns,
              std::string* error) {
        path_ = path;
        builder_.AddColumn("id_a", ExportType::kString);
        builder_.AddColumn("id_b", ExportType::kString);
        builder_.AddColumn("gamma", ExportType::kUInt32);
        builder_.AddColumn("prior", ExportType::kDouble);
        builder_.AddColumn("match_weight", ExportType::kDouble);
        builder_.AddColumn("match_probability", ExportType::kDouble);
        builder_.AddColumn("bracket_low", ExportType::kDouble);
        builder_.AddColumn("bracket_high", ExportType::kDouble);
        builder_.AddColumn("threshold", ExportType::kDouble);
        builder_.AddColumn("zone", ExportType::kString);
        builder_.AddColumn("emitted", ExportType::kBoolean);
        size_t at = kFixedColumns;
        for (size_t c = 0; c < comparisons_; ++c) {
            builder_.AddColumn(columns[at++], ExportType::kUInt8);
            builder_.AddColumn(columns[at++], ExportType::kDouble);
            builder_.AddColumn(columns[at++], ExportType::kDouble);
            builder_.AddColumn(columns[at++], ExportType::kUInt32);
        }
        for (size_t i = 0; i < interactions_; ++i) {
            builder_.AddColumn(columns[at++], ExportType::kDouble);
        }
        ArrowSchema schema;
        builder_.ExportSchema(&schema);
        const bool ok = writer_.Open(path, schema, error);
        schema.release(&schema);
        if (!ok) *error = "explain: " + *error;
        return ok;
    }

    bool Write(const PairWaterfall& w, std::string* error) override {
        int at = 0;
        builder_.AppendString(at++, w.id_a);
        builder_.AppendString(at++, w.id_b);
        builder_.AppendUInt32(at++, w.gamma);
        builder_.AppendDouble(at++, w.prior);
        builder_.AppendDouble(at++, w.weight);
        builder_.AppendDouble(at++, w.probability);
        builder_.AppendDouble(at++, w.bracket_low);
        builder_.AppendDouble(at++, w.bracket_high);
        builder_.AppendDouble(at++, w.threshold);
        builder_.AppendString(at++, ZoneName(w.zone));
        builder_.AppendBoolean(at++, w.emitted);
        for (size_t c = 0; c < w.steps.size(); ++c) {
            builder_.AppendUInt8(at++, w.steps[c].level);
            builder_.AppendDouble(at++, w.steps[c].bits);
            builder_.AppendDouble(at++, w.steps[c].tf);
            builder_.AppendUInt32(at++, w.steps[c].frequency);
        }
        for (size_t i = 0; i < w.interactions.size(); ++i) {
            builder_.AppendDouble(at++, w.interactions[i].bits);
        }
        if (builder_.Rows() >= static_cast<int64_t>(batch_rows_)) return Flush(error);
        return true;
    }

    bool Close(std::string* error) override {
        if (!Flush(error)) return false;
        if (!writer_.Close(error)) {
            *error = "explain: " + *error;
            return false;
        }
        return true;
    }

   private:
    bool Flush(std::string* error) {
        if (builder_.Rows() == 0) return true;
        ArrowArray batch;
        if (!builder_.ExportBatch(&batch, error) || !writer_.Write(&batch, error)) {
            *error = "explain: " + *error;
            return false;
        }
        return true;
    }

    size_t batch_rows_;
    size_t comparisons_;
    size_t interactions_;
    BatchBuilder builder_;
    std::string path_;
    ParquetWriter writer_;
};

}  // namespace

std::vector<std::string> WaterfallColumns(const ComparisonSet& comparisons,
                                          const Scorer& scorer) {
    std::vector<std::string> columns = {
        "id_a",        "id_b",         "gamma",
        "prior",       "match_weight", "match_probability",
        "bracket_low", "bracket_high", "threshold",
        "zone",        "emitted"};
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        const std::string& name = comparisons.at(c).spec->name;
        columns.push_back(name + "_level");
        columns.push_back(name + "_bits");
        columns.push_back(name + "_tf");
        columns.push_back(name + "_frequency");
    }
    for (size_t i = 0; i < scorer.InteractionCount(); ++i) {
        columns.push_back(InteractionColumn(scorer.InteractionName(i)));
    }
    return columns;
}

bool WriteWaterfalls(const RecordStore& store, const ComparisonSet& comparisons,
                     const Scorer& scorer, const Model& model,
                     const WaterfallOptions& options, WaterfallReport* report,
                     std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    *report = WaterfallReport{};
    report->out_path = options.out_path;
    if (!MergedFormatOf(options.out_path, &report->format)) {
        *error = "explain: --out wants a .csv or .parquet file";
        return false;
    }
    MergeFormat in_format = MergeFormat::kCsv;
    if (!MergedFormatOf(options.predictions_path, &in_format)) {
        *error = "explain: --predictions wants the .csv or .parquet file predict wrote";
        return false;
    }

    const std::vector<std::string> columns = WaterfallColumns(comparisons, scorer);
    report->columns = columns.size();
    std::unique_ptr<WideSink> sink;
    if (report->format == MergeFormat::kParquet) {
        auto parquet_sink = std::make_unique<ParquetWideSink>(
            options.batch_rows, comparisons.Size(), scorer.InteractionCount());
        if (!parquet_sink->Open(options.out_path, columns, error)) return false;
        sink = std::move(parquet_sink);
    } else {
        auto csv_sink = std::make_unique<CsvWideSink>();
        if (!csv_sink->Open(options.out_path, columns, error)) return false;
        sink = std::move(csv_sink);
    }

    const IdIndex index(store);
    if (!index.Unique(error)) {
        *error = "explain: " + *error;
        return false;
    }
    // With a dataset the lookup is exact; without one it is answered only where
    // a single record carries the id, as `cluster` reads the same file.
    const auto resolve = [&](std::string_view dataset, std::string_view id,
                             uint32_t* row) {
        if (dataset.empty()) return index.Find(id, row) == IdLookup::kFound;
        size_t which = 0;
        return store.DatasetIndex(dataset, &which) &&
               index.Find(which, id, row) == IdLookup::kFound;
    };
    const PredictionSink each = [&](const NamedPrediction& prediction,
                                    std::string* trouble) {
        ++report->read;
        uint32_t a = 0;
        uint32_t b = 0;
        if (!resolve(prediction.dataset_a, prediction.id_a, &a) ||
            !resolve(prediction.dataset_b, prediction.id_b, &b)) {
            ++report->unresolved;
            return true;
        }
        const PairWaterfall w =
            BuildPairWaterfall(store, comparisons, scorer, a, b, &model);
        if (w.gamma != prediction.gamma) ++report->pattern_changed;
        if (!sink->Write(w, trouble)) return false;
        ++report->written;
        return true;
    };
    const bool ok = in_format == MergeFormat::kParquet
                        ? ForEachParquetPrediction(options.predictions_path, each, error)
                        : ForEachCsvPrediction(options.predictions_path, each, error);
    if (!ok) return false;
    if (!sink->Close(error)) return false;
    report->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return true;
}

void PrintWaterfallReport(const WaterfallReport& report, std::ostream& out) {
    out << "Read " << WithThousands(report.read) << " predictions, wrote "
        << WithThousands(report.written) << " waterfalls to " << report.out_path << " ("
        << (report.format == MergeFormat::kParquet ? "parquet" : "csv") << ", "
        << report.columns << " columns) in " << std::fixed << std::setprecision(2)
        << report.seconds << " s\n";
    if (report.unresolved > 0) {
        out << "  " << WithThousands(report.unresolved)
            << " predictions name an id no record holds and were skipped: was this "
               "file written from these inputs?\n";
    }
    if (report.pattern_changed > 0) {
        out << "  " << WithThousands(report.pattern_changed)
            << " predictions carry a gamma the comparisons no longer produce: the "
               "schema has changed since predict ran, and the ledgers explain the "
               "schema as it is now, not the weights in the file\n";
    }
}

}  // namespace cpplink
