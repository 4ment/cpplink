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

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "cpplink/explain.hpp"
#include "cpplink/format.hpp"
#include "cpplink/id_index.hpp"

namespace cpplink {
namespace {

// One prediction as the merged file names it. `gamma` is what the run stored,
// kept so a schema that has drifted from the file can be noticed.
struct NamedPrediction {
    std::string_view id_a;
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
    while (std::getline(file, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (number == 1 && line.rfind("id_a,", 0) == 0) continue;
        NamedPrediction prediction;
        double weight = 0.0;
        if (!ParseEdgeCsvLine(line, &prediction.id_a, &prediction.id_b, &prediction.gamma,
                              &weight)) {
            *error = "explain: '" + path + "' line " + std::to_string(number) +
                     " is not a cpplink prediction row";
            return false;
        }
        if (!sink(prediction, error)) return false;
    }
    return true;
}

bool ForEachParquetPrediction(const std::string& path, const PredictionSink& sink,
                              std::string* error) {
    auto input = arrow::io::ReadableFile::Open(path);
    if (!input.ok()) {
        *error = "explain: cannot open " + path + ": " + input.status().message();
        return false;
    }
    parquet::arrow::FileReaderBuilder builder;
    arrow::Status status = builder.Open(*input);
    if (!status.ok()) {
        *error =
            "explain: cannot read parquet metadata for " + path + ": " + status.message();
        return false;
    }
    auto reader_result = builder.Build();
    if (!reader_result.ok()) {
        *error = "explain: cannot open parquet reader for " + path + ": " +
                 reader_result.status().message();
        return false;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);
    std::shared_ptr<arrow::Schema> schema;
    status = reader->GetSchema(&schema);
    if (!status.ok()) {
        *error = "explain: cannot read the schema of " + path + ": " + status.message();
        return false;
    }
    std::vector<int> indices;
    for (const char* name : {"id_a", "id_b", "gamma"}) {
        const int at = schema->GetFieldIndex(name);
        if (at < 0) {
            *error = "explain: '" + path + "' has no column \"" + name +
                     "\", so it is not a cpplink prediction file";
            return false;
        }
        indices.push_back(at);
    }
    for (int group = 0; group < reader->num_row_groups(); ++group) {
        auto group_result = reader->ReadRowGroup(group, indices);
        if (!group_result.ok()) {
            *error = "explain: cannot read row group " + std::to_string(group) + " of " +
                     path + ": " + group_result.status().message();
            return false;
        }
        const std::shared_ptr<arrow::Table> table = *group_result;
        const auto ids_a = table->GetColumnByName("id_a");
        const auto ids_b = table->GetColumnByName("id_b");
        const auto gammas = table->GetColumnByName("gamma");
        for (int chunk = 0; chunk < ids_a->num_chunks(); ++chunk) {
            const auto a_array =
                std::dynamic_pointer_cast<arrow::StringArray>(ids_a->chunk(chunk));
            const auto b_array =
                std::dynamic_pointer_cast<arrow::StringArray>(ids_b->chunk(chunk));
            const auto gamma_array =
                std::dynamic_pointer_cast<arrow::UInt32Array>(gammas->chunk(chunk));
            if (a_array == nullptr || b_array == nullptr || gamma_array == nullptr) {
                *error = "explain: '" + path +
                         "' holds id_a, id_b or gamma in a type a cpplink prediction "
                         "file does not use";
                return false;
            }
            for (int64_t row = 0; row < a_array->length(); ++row) {
                if (a_array->IsNull(row) || b_array->IsNull(row)) continue;
                NamedPrediction prediction;
                prediction.id_a = a_array->GetView(row);
                prediction.id_b = b_array->GetView(row);
                prediction.gamma = gamma_array->IsNull(row) ? 0 : gamma_array->Value(row);
                if (!sink(prediction, error)) return false;
            }
        }
    }
    return true;
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

class ParquetWideSink : public WideSink {
   public:
    ParquetWideSink(size_t batch_rows, size_t comparisons, size_t interactions)
        : batch_rows_(batch_rows == 0 ? 1 : batch_rows),
          levels_(comparisons),
          bits_(comparisons),
          tf_(comparisons),
          frequency_(comparisons),
          interaction_(interactions) {}

    bool Open(const std::string& path, const std::vector<std::string>& columns,
              std::string* error) {
        path_ = path;
        std::vector<std::shared_ptr<arrow::Field>> fields = {
            arrow::field("id_a", arrow::utf8()),
            arrow::field("id_b", arrow::utf8()),
            arrow::field("gamma", arrow::uint32()),
            arrow::field("prior", arrow::float64()),
            arrow::field("match_weight", arrow::float64()),
            arrow::field("match_probability", arrow::float64()),
            arrow::field("bracket_low", arrow::float64()),
            arrow::field("bracket_high", arrow::float64()),
            arrow::field("threshold", arrow::float64()),
            arrow::field("zone", arrow::utf8()),
            arrow::field("emitted", arrow::boolean()),
        };
        size_t at = kFixedColumns;
        for (size_t c = 0; c < levels_.size(); ++c) {
            fields.push_back(arrow::field(columns[at++], arrow::uint8()));
            fields.push_back(arrow::field(columns[at++], arrow::float64()));
            fields.push_back(arrow::field(columns[at++], arrow::float64()));
            fields.push_back(arrow::field(columns[at++], arrow::uint32()));
        }
        for (size_t i = 0; i < interaction_.size(); ++i) {
            fields.push_back(arrow::field(columns[at++], arrow::float64()));
        }
        schema_ = arrow::schema(fields);
        auto sink = arrow::io::FileOutputStream::Open(path);
        if (!sink.ok()) {
            *error = "explain: cannot create " + path + ": " + sink.status().message();
            return false;
        }
        auto props = parquet::WriterProperties::Builder()
                         .compression(parquet::Compression::SNAPPY)
                         ->build();
        auto writer = parquet::arrow::FileWriter::Open(
            *schema_, arrow::default_memory_pool(), *sink, props);
        if (!writer.ok()) {
            *error = "explain: cannot open parquet writer for " + path + ": " +
                     writer.status().message();
            return false;
        }
        writer_ = std::move(*writer);
        return true;
    }

    bool Write(const PairWaterfall& w, std::string* error) override {
        arrow::Status status = id_a_.Append(w.id_a);
        status &= id_b_.Append(w.id_b);
        status &= gamma_.Append(w.gamma);
        status &= prior_.Append(w.prior);
        status &= weight_.Append(w.weight);
        status &= probability_.Append(w.probability);
        status &= low_.Append(w.bracket_low);
        status &= high_.Append(w.bracket_high);
        status &= threshold_.Append(w.threshold);
        status &= zone_.Append(ZoneName(w.zone));
        status &= emitted_.Append(w.emitted);
        for (size_t c = 0; c < w.steps.size(); ++c) {
            status &= levels_[c].Append(w.steps[c].level);
            status &= bits_[c].Append(w.steps[c].bits);
            status &= tf_[c].Append(w.steps[c].tf);
            status &= frequency_[c].Append(w.steps[c].frequency);
        }
        for (size_t i = 0; i < w.interactions.size(); ++i) {
            status &= interaction_[i].Append(w.interactions[i].bits);
        }
        if (!status.ok()) {
            *error = "explain: building a row: " + status.message();
            return false;
        }
        ++pending_;
        if (pending_ >= batch_rows_) return Flush(error);
        return true;
    }

    bool Close(std::string* error) override {
        if (!Flush(error)) return false;
        const arrow::Status closed = writer_->Close();
        if (!closed.ok()) {
            *error = "explain: closing " + path_ + ": " + closed.message();
            return false;
        }
        return true;
    }

   private:
    bool Flush(std::string* error) {
        if (pending_ == 0) return true;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        arrays.reserve(static_cast<size_t>(schema_->num_fields()));
        arrow::Status status;
        const auto finish = [&](arrow::ArrayBuilder* builder) {
            std::shared_ptr<arrow::Array> array;
            status &= builder->Finish(&array);
            arrays.push_back(std::move(array));
        };
        finish(&id_a_);
        finish(&id_b_);
        finish(&gamma_);
        finish(&prior_);
        finish(&weight_);
        finish(&probability_);
        finish(&low_);
        finish(&high_);
        finish(&threshold_);
        finish(&zone_);
        finish(&emitted_);
        for (size_t c = 0; c < levels_.size(); ++c) {
            finish(&levels_[c]);
            finish(&bits_[c]);
            finish(&tf_[c]);
            finish(&frequency_[c]);
        }
        for (auto& builder : interaction_) finish(&builder);
        if (!status.ok()) {
            *error = "explain: finishing a batch: " + status.message();
            return false;
        }
        const auto table =
            arrow::Table::Make(schema_, arrays, static_cast<int64_t>(pending_));
        status = writer_->WriteTable(*table, static_cast<int64_t>(pending_));
        if (!status.ok()) {
            *error = "explain: writing a row group to " + path_ + ": " + status.message();
            return false;
        }
        pending_ = 0;
        return true;
    }

    size_t batch_rows_;
    std::shared_ptr<arrow::Schema> schema_;
    arrow::StringBuilder id_a_, id_b_, zone_;
    arrow::UInt32Builder gamma_;
    arrow::DoubleBuilder prior_, weight_, probability_, low_, high_, threshold_;
    arrow::BooleanBuilder emitted_;
    std::vector<arrow::UInt8Builder> levels_;
    std::vector<arrow::DoubleBuilder> bits_, tf_;
    std::vector<arrow::UInt32Builder> frequency_;
    std::vector<arrow::DoubleBuilder> interaction_;
    size_t pending_ = 0;
    std::string path_;
    std::unique_ptr<parquet::arrow::FileWriter> writer_;
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
    const PredictionSink each = [&](const NamedPrediction& prediction,
                                    std::string* trouble) {
        ++report->read;
        uint32_t a = 0;
        uint32_t b = 0;
        if (!index.Find(prediction.id_a, &a) || !index.Find(prediction.id_b, &b)) {
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
