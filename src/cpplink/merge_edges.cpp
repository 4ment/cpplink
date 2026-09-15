// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/merge_edges.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include "cpplink/format.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/score.hpp"

namespace cpplink {
namespace {

// A whole number of edge records, so a read never splits one across two buffers.
constexpr size_t kReadEdges = 4096;
constexpr size_t kFlushBytes = 1u << 20;

// Both shard formats are turned into an EdgeRow before anything is written, so
// the two writers see one shape. Whether the rows carry datasets is decided once,
// before the sink opens: from the store for binary shards, from the header of
// the first shard for csv ones.
class EdgeSink {
   public:
    virtual ~EdgeSink() = default;
    virtual bool Write(const EdgeRow& edge, std::string* error) = 0;
    virtual bool Close(std::string* error) = 0;
};

// The shard csv format again, header and all, so a merged file and a shard are
// the same thing to whatever reads them.
class CsvSink : public EdgeSink {
   public:
    explicit CsvSink(bool datasets) : datasets_(datasets) {}

    bool Open(const std::string& path, std::string* error) {
        path_ = path;
        file_.open(path, std::ios::binary | std::ios::trunc);
        if (!file_) {
            *error = "merging predictions: could not write '" + path + "'";
            return false;
        }
        buffer_ = EdgeCsvHeader(datasets_);
        buffer_.push_back('\n');
        return true;
    }

    bool Write(const EdgeRow& edge, std::string* error) override {
        char numbers[64];
        std::snprintf(numbers, sizeof(numbers), ",%u,%.6f,%.9f", edge.gamma, edge.weight,
                      ProbabilityForWeight(edge.weight));
        if (datasets_) {
            buffer_.append(edge.dataset_a);
            buffer_.push_back(',');
        }
        buffer_.append(edge.id_a);
        buffer_.push_back(',');
        if (datasets_) {
            buffer_.append(edge.dataset_b);
            buffer_.push_back(',');
        }
        buffer_.append(edge.id_b);
        buffer_.append(numbers);
        buffer_.push_back('\n');
        if (buffer_.size() >= kFlushBytes) return Flush(error);
        return true;
    }

    bool Close(std::string* error) override {
        if (!Flush(error)) return false;
        file_.close();
        if (file_.fail()) {
            *error = "merging predictions: failed writing '" + path_ + "'";
            return false;
        }
        return true;
    }

   private:
    bool Flush(std::string* error) {
        if (buffer_.empty()) return true;
        file_.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
        buffer_.clear();
        if (!file_) {
            *error = "merging predictions: failed writing '" + path_ + "'";
            return false;
        }
        return true;
    }

    bool datasets_;
    std::string path_;
    std::ofstream file_;
    std::string buffer_;
};

// The csv columns, typed, in the same order and under the same names.
std::shared_ptr<arrow::Schema> MergedArrowSchema(bool datasets) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    if (datasets) fields.push_back(arrow::field("dataset_a", arrow::utf8()));
    fields.push_back(arrow::field("id_a", arrow::utf8()));
    if (datasets) fields.push_back(arrow::field("dataset_b", arrow::utf8()));
    fields.push_back(arrow::field("id_b", arrow::utf8()));
    fields.push_back(arrow::field("gamma", arrow::uint32()));
    fields.push_back(arrow::field("match_weight", arrow::float64()));
    fields.push_back(arrow::field("match_probability", arrow::float64()));
    return arrow::schema(fields);
}

class ParquetSink : public EdgeSink {
   public:
    ParquetSink(size_t batch_rows, bool datasets)
        : batch_rows_(batch_rows == 0 ? 1 : batch_rows),
          datasets_(datasets),
          schema_(MergedArrowSchema(datasets)) {}

    bool Open(const std::string& path, std::string* error) {
        path_ = path;
        auto sink = arrow::io::FileOutputStream::Open(path);
        if (!sink.ok()) {
            *error = "merging predictions: cannot create " + path + ": " +
                     sink.status().message();
            return false;
        }
        auto props = parquet::WriterProperties::Builder()
                         .compression(parquet::Compression::SNAPPY)
                         ->build();
        auto writer = parquet::arrow::FileWriter::Open(
            *schema_, arrow::default_memory_pool(), *sink, props);
        if (!writer.ok()) {
            *error = "merging predictions: cannot open parquet writer for " + path +
                     ": " + writer.status().message();
            return false;
        }
        writer_ = std::move(*writer);
        return true;
    }

    bool Write(const EdgeRow& edge, std::string* error) override {
        arrow::Status status = id_a_.Append(edge.id_a);
        status &= id_b_.Append(edge.id_b);
        if (datasets_) {
            status &= dataset_a_.Append(edge.dataset_a);
            status &= dataset_b_.Append(edge.dataset_b);
        }
        status &= gamma_.Append(edge.gamma);
        status &= weight_.Append(edge.weight);
        status &= probability_.Append(ProbabilityForWeight(edge.weight));
        if (!status.ok()) {
            *error = "merging predictions: building a row: " + status.message();
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
            *error = "merging predictions: closing " + path_ + ": " + closed.message();
            return false;
        }
        return true;
    }

   private:
    bool Flush(std::string* error) {
        if (pending_ == 0) return true;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        arrays.reserve(7);
        arrow::Status status;
        auto take = [&](arrow::ArrayBuilder* builder) {
            std::shared_ptr<arrow::Array> array;
            status &= builder->Finish(&array);
            arrays.push_back(std::move(array));
        };
        if (datasets_) take(&dataset_a_);
        take(&id_a_);
        if (datasets_) take(&dataset_b_);
        take(&id_b_);
        take(&gamma_);
        take(&weight_);
        take(&probability_);
        if (!status.ok()) {
            *error = "merging predictions: finishing a batch: " + status.message();
            return false;
        }
        const auto table =
            arrow::Table::Make(schema_, arrays, static_cast<int64_t>(pending_));
        status = writer_->WriteTable(*table, static_cast<int64_t>(pending_));
        if (!status.ok()) {
            *error = "merging predictions: writing a row group to " + path_ + ": " +
                     status.message();
            return false;
        }
        pending_ = 0;
        return true;
    }

    size_t batch_rows_;
    bool datasets_;
    std::shared_ptr<arrow::Schema> schema_;
    arrow::StringBuilder dataset_a_, dataset_b_, id_a_, id_b_;
    arrow::UInt32Builder gamma_;
    arrow::DoubleBuilder weight_, probability_;
    size_t pending_ = 0;
    std::string path_;
    std::unique_ptr<parquet::arrow::FileWriter> writer_;
};

bool CollectShards(const std::string& dir, MergeSource wanted,
                   std::vector<std::string>* shards, std::vector<std::string>* ignored,
                   MergeSource* source, std::string* error) {
    std::error_code code;
    if (!std::filesystem::is_directory(dir, code)) {
        *error =
            "merging predictions: '" + dir + "' is not a directory of prediction shards";
        return false;
    }
    std::vector<std::string> binary;
    std::vector<std::string> csv;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("shard-", 0) != 0) continue;
        if (entry.path().extension() == ".bin") {
            binary.push_back(entry.path().string());
        } else if (entry.path().extension() == ".csv") {
            csv.push_back(entry.path().string());
        }
    }
    // Sorted, because a directory iteration is in no particular order and the
    // merged file should not depend on the filesystem.
    std::sort(binary.begin(), binary.end());
    std::sort(csv.begin(), csv.end());

    if (wanted == MergeSource::kBinary && binary.empty()) {
        *error = "merging predictions: no shard-*.bin files under '" + dir + "'";
        return false;
    }
    if (wanted == MergeSource::kCsv && csv.empty()) {
        *error = "merging predictions: no shard-*.csv files under '" + dir + "'";
        return false;
    }
    if (binary.empty() && csv.empty()) {
        *error = "merging predictions: no shard-*.bin or shard-*.csv files under '" +
                 dir + "'";
        return false;
    }
    const bool take_binary = wanted == MergeSource::kBinary ||
                             (wanted == MergeSource::kAuto && !binary.empty());
    *source = take_binary ? MergeSource::kBinary : MergeSource::kCsv;
    *shards = take_binary ? std::move(binary) : std::move(csv);
    *ignored = take_binary ? std::move(csv) : std::move(binary);
    return true;
}

bool ReadBinaryShard(const std::string& path, const RecordStore* store,
                     const MergeOptions& options, EdgeSink* sink, MergeReport* report,
                     std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        *error = "merging predictions: could not open '" + path + "'";
        return false;
    }
    char magic[sizeof(kEdgeMagic)];
    file.read(magic, sizeof(magic));
    if (file.gcount() != static_cast<std::streamsize>(sizeof(magic)) ||
        std::memcmp(magic, kEdgeMagic, sizeof(magic)) != 0) {
        *error = "merging predictions: '" + path + "' is not a cpplink prediction shard";
        return false;
    }
    const uint64_t records = store == nullptr ? 0 : store->NumRecords();
    const bool datasets = store != nullptr && store->NumDatasets() > 1;
    std::vector<char> buffer(kReadEdges * kEdgeBytes);
    char row_a[16];
    char row_b[16];
    std::string dataset_a;
    std::string dataset_b;
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const size_t got = static_cast<size_t>(file.gcount());
        if (got % kEdgeBytes != 0) {
            *error = "merging predictions: '" + path + "' is truncated mid-prediction";
            return false;
        }
        for (size_t at = 0; at < got; at += kEdgeBytes) {
            uint32_t a = 0;
            uint32_t b = 0;
            EdgeRow edge;
            std::memcpy(&a, buffer.data() + at, 4);
            std::memcpy(&b, buffer.data() + at + 4, 4);
            std::memcpy(&edge.gamma, buffer.data() + at + 8, 4);
            std::memcpy(&edge.weight, buffer.data() + at + 12, 8);
            ++report->read;
            if (edge.weight < options.threshold) continue;
            if (store == nullptr) {
                const int wrote_a = std::snprintf(row_a, sizeof(row_a), "%u", a);
                const int wrote_b = std::snprintf(row_b, sizeof(row_b), "%u", b);
                edge.id_a = std::string_view(row_a, static_cast<size_t>(wrote_a));
                edge.id_b = std::string_view(row_b, static_cast<size_t>(wrote_b));
            } else {
                if (a >= records || b >= records) {
                    *error = "merging predictions: '" + path + "' names row " +
                             std::to_string(a >= records ? a : b) + " but the data has " +
                             std::to_string(records) + " records";
                    return false;
                }
                edge.id_a = store->ids().Get(a);
                edge.id_b = store->ids().Get(b);
                if (datasets) {
                    dataset_a = store->DatasetName(store->DatasetOf(a));
                    dataset_b = store->DatasetName(store->DatasetOf(b));
                    edge.dataset_a = dataset_a;
                    edge.dataset_b = dataset_b;
                }
            }
            if (!sink->Write(edge, error)) return false;
            ++report->written;
        }
    }
    return true;
}

// The header of a csv shard, which is what says whether its rows carry datasets.
// Read before the sink opens, because the merged file's own header has to agree
// with every shard's.
bool CsvShardLayout(const std::string& path, EdgeCsvLayout* layout, std::string* error) {
    std::ifstream file(path);
    if (!file) {
        *error = "merging predictions: could not open '" + path + "'";
        return false;
    }
    std::string line;
    *layout = EdgeCsvLayout();
    if (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        ParseEdgeCsvHeader(line, layout);
    }
    return true;
}

bool ReadCsvShard(const std::string& path, const MergeOptions& options,
                  const EdgeCsvLayout& expected, EdgeSink* sink, MergeReport* report,
                  std::string* error) {
    std::ifstream file(path);
    if (!file) {
        *error = "merging predictions: could not open '" + path + "'";
        return false;
    }
    std::string line;
    uint64_t number = 0;
    EdgeCsvLayout layout;
    while (std::getline(file, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (number == 1 && ParseEdgeCsvHeader(line, &layout)) {
            if (layout.datasets != expected.datasets) {
                *error = "merging predictions: '" + path + "' " +
                         (layout.datasets ? "carries" : "lacks") +
                         " dataset columns where the first shard " +
                         (expected.datasets ? "carries" : "lacks") +
                         " them; these shards are not one run";
                return false;
            }
            continue;
        }
        EdgeRow edge;
        if (!ParseEdgeCsvLine(line, expected, &edge)) {
            *error = "merging predictions: '" + path + "' line " +
                     std::to_string(number) + " is not a cpplink prediction row";
            return false;
        }
        ++report->read;
        if (edge.weight < options.threshold) continue;
        if (!sink->Write(edge, error)) return false;
        ++report->written;
    }
    return true;
}

}  // namespace

bool MergeEdges(const RecordStore* store, const MergeOptions& options,
                MergeReport* report, std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    *report = MergeReport();
    if (options.out_path.empty()) {
        *error = "merging predictions: --out names the file to write";
        return false;
    }
    MergeSource source = MergeSource::kBinary;
    if (!CollectShards(options.edge_dir, options.source, &report->shards,
                       &report->ignored, &source, error)) {
        return false;
    }
    report->source = source;
    report->format = options.format;
    report->out_path = options.out_path;
    report->row_indices = source == MergeSource::kBinary && store == nullptr;

    // Whether the merged rows name their datasets: a store of several inputs
    // says so for binary shards, and csv shards say so themselves.
    EdgeCsvLayout layout;
    if (source == MergeSource::kBinary) {
        layout.datasets = store != nullptr && store->NumDatasets() > 1;
    } else if (!CsvShardLayout(report->shards.front(), &layout, error)) {
        return false;
    }
    report->datasets = layout.datasets;

    std::unique_ptr<EdgeSink> sink;
    if (options.format == MergeFormat::kParquet) {
        auto parquet_sink =
            std::make_unique<ParquetSink>(options.batch_rows, layout.datasets);
        if (!parquet_sink->Open(options.out_path, error)) return false;
        sink = std::move(parquet_sink);
    } else {
        auto csv_sink = std::make_unique<CsvSink>(layout.datasets);
        if (!csv_sink->Open(options.out_path, error)) return false;
        sink = std::move(csv_sink);
    }

    for (const std::string& path : report->shards) {
        const bool read =
            source == MergeSource::kBinary
                ? ReadBinaryShard(path, store, options, sink.get(), report, error)
                : ReadCsvShard(path, options, layout, sink.get(), report, error);
        if (!read) return false;
    }
    if (!sink->Close(error)) return false;

    report->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return true;
}

bool ParseEdgeCsvHeader(const std::string& line, EdgeCsvLayout* layout) {
    if (line == EdgeCsvHeader(true)) {
        layout->datasets = true;
        return true;
    }
    if (line == EdgeCsvHeader(false)) {
        layout->datasets = false;
        return true;
    }
    return false;
}

bool ParseEdgeCsvLine(const std::string& line, const EdgeCsvLayout& layout,
                      EdgeRow* row) {
    const size_t probability = line.rfind(',');
    if (probability == std::string::npos || probability == 0) return false;
    const size_t at_weight = line.rfind(',', probability - 1);
    if (at_weight == std::string::npos || at_weight == 0) return false;
    const size_t at_gamma = line.rfind(',', at_weight - 1);
    if (at_gamma == std::string::npos || at_gamma == 0) return false;
    // The naming fields, left to right, each ending at the next comma and the
    // last at the gamma's.
    std::string_view* fields[4];
    size_t count = 0;
    if (layout.datasets) fields[count++] = &row->dataset_a;
    fields[count++] = &row->id_a;
    if (layout.datasets) fields[count++] = &row->dataset_b;
    fields[count++] = &row->id_b;
    size_t begin = 0;
    for (size_t f = 0; f < count; ++f) {
        const size_t end = f + 1 == count ? at_gamma : line.find(',', begin);
        if (end == std::string::npos || end > at_gamma) return false;
        *fields[f] = std::string_view(line.data() + begin, end - begin);
        begin = end + 1;
    }
    if (!layout.datasets) {
        row->dataset_a = std::string_view();
        row->dataset_b = std::string_view();
    }
    // strtod and strtoul stop at the delimiter, so no field has to be copied out.
    row->gamma =
        static_cast<uint32_t>(std::strtoul(line.c_str() + at_gamma + 1, nullptr, 10));
    row->weight = std::strtod(line.c_str() + at_weight + 1, nullptr);
    return true;
}

bool MergeStagedShards(const RecordStore& store, const std::string& staging,
                       const std::string& out_path, bool shards_are_binary,
                       double* seconds, std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    MergeOptions options;
    options.edge_dir = staging;
    options.out_path = out_path;
    options.source = shards_are_binary ? MergeSource::kBinary : MergeSource::kCsv;
    if (!MergedFormatOf(out_path, &options.format)) {
        *error = "merging predictions: \"" + out_path +
                 "\" names neither a .csv nor a .parquet file";
        return false;
    }
    MergeReport merged;
    if (!MergeEdges(&store, options, &merged, error)) return false;
    std::error_code ec;
    std::filesystem::remove_all(staging, ec);
    if (ec) {
        *error = "merging predictions: cannot remove the staging directory \"" + staging +
                 "\": " + ec.message();
        return false;
    }
    *seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return true;
}

bool MergedFormatOf(const std::string& path, MergeFormat* format) {
    const std::string suffix = std::filesystem::path(path).extension().string();
    if (suffix == ".parquet" || suffix == ".pq") {
        *format = MergeFormat::kParquet;
        return true;
    }
    if (suffix == ".csv") {
        *format = MergeFormat::kCsv;
        return true;
    }
    return false;
}

void PrintMergeReport(const MergeReport& report, std::ostream& out) {
    out << "Read " << WithThousands(report.read) << " predictions from "
        << report.shards.size() << " "
        << (report.source == MergeSource::kBinary ? "bin" : "csv") << " shard"
        << (report.shards.size() == 1 ? "" : "s") << " in " << report.seconds << " s\n";
    if (!report.ignored.empty()) {
        out << "Ignored " << report.ignored.size() << " "
            << (report.source == MergeSource::kBinary ? "csv" : "bin") << " shard"
            << (report.ignored.size() == 1 ? "" : "s")
            << " holding the same run; --from picks the other side\n";
    }
    if (report.written != report.read) {
        out << "Kept " << WithThousands(report.written) << " above the threshold\n";
    }
    if (report.row_indices) {
        out << "Rows are named by index: binary shards carry no ids, so pass "
               "--schema and\nthe parquet input to write the record ids "
               "instead\n";
    }
    if (report.datasets) {
        out << "Each record is named by its dataset and its id (dataset_a, id_a, "
               "dataset_b, id_b),\nbecause the run had more than one input\n";
    }
    out << "Wrote " << WithThousands(report.written) << " predictions to "
        << report.out_path << " ("
        << (report.format == MergeFormat::kParquet ? "parquet" : "csv") << ")\n";
}

}  // namespace cpplink
