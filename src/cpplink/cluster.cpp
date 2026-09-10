// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/cluster.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include "cpplink/format.hpp"
#include "cpplink/merge_edges.hpp"
#include "cpplink/predict.hpp"

namespace cpplink {
namespace {

// A whole number of edge records, so a read never splits one across two buffers.
constexpr size_t kReadEdges = 4096;

std::string Percent(uint64_t part, uint64_t whole) {
    if (whole == 0) return "-";
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f%%",
                  100.0 * static_cast<double>(part) / static_cast<double>(whole));
    return buffer;
}

bool CollectShards(const std::string& dir, std::vector<std::string>* shards,
                   std::string* error) {
    std::error_code code;
    if (!std::filesystem::is_directory(dir, code)) {
        *error = "cluster: '" + dir + "' is not a directory of prediction shards";
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir, code)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("shard-", 0) == 0 && entry.path().extension() == ".bin") {
            shards->push_back(entry.path().string());
        }
    }
    if (code) {
        *error = "cluster: could not read '" + dir + "'";
        return false;
    }
    // Sorted so a run is reproducible; union-find does not care about order, but
    // the report's shard list should not depend on the filesystem.
    std::sort(shards->begin(), shards->end());
    if (shards->empty()) {
        *error = "cluster: no shard-*.bin files under '" + dir + "'";
        return false;
    }
    return true;
}

// The size buckets the report prints. Clusters are the thing most likely to go
// wrong at scale -- one bad edge chains two components together -- so the tail is
// broken out rather than summarised by a mean.
struct BucketSpec {
    const char* label;
    uint64_t low;
    uint64_t high;
};

constexpr BucketSpec kBuckets[] = {
    {"1 (singleton)", 1, 1},
    {"2", 2, 2},
    {"3", 3, 3},
    {"4", 4, 4},
    {"5", 5, 5},
    {"6-10", 6, 10},
    {"11-100", 11, 100},
    {"101-1,000", 101, 1000},
    {"1,001+", 1001, UINT64_MAX},
};

// What a reader hands each edge to: the union-find, the threshold, and the two
// counters the report prints. Every format goes through this, so the accounting
// cannot differ between them.
struct EdgeTally {
    EdgeTally(UnionFind* uf, uint64_t records, double threshold)
        : uf(uf), records(records), threshold(threshold) {}

    void Use(uint32_t a, uint32_t b, double weight) {
        ++read;
        if (weight < threshold) return;
        if (used == 0) {
            min_weight = weight;
            max_weight = weight;
        } else {
            min_weight = std::min(min_weight, weight);
            max_weight = std::max(max_weight, weight);
        }
        ++used;
        uf->Union(a, b);
    }

    UnionFind* uf;
    uint64_t records = 0;
    double threshold = 0.0;
    uint64_t read = 0;
    uint64_t used = 0;
    uint64_t unresolved = 0;
    double min_weight = 0.0;
    double max_weight = 0.0;
};

// An id-to-row lookup over the store's id column: one `uint32` per record, kept
// in the order of the id it names. A merged edge file carries `unique_id`s
// rather than row indices, so clustering one has to map them back, and a hash
// map over 20M ids costs an order of magnitude more than the union-find it
// feeds. This is 4 bytes a record and a handful of string compares an edge.
class IdIndex {
   public:
    explicit IdIndex(const RecordStore& store) : ids_(store.ids()) {
        order_.resize(static_cast<size_t>(store.NumRecords()));
        for (size_t row = 0; row < order_.size(); ++row) {
            order_[row] = static_cast<uint32_t>(row);
        }
        std::sort(order_.begin(), order_.end(),
                  [this](uint32_t a, uint32_t b) { return ids_.Get(a) < ids_.Get(b); });
    }

    bool Find(std::string_view id, uint32_t* row) const {
        const auto at =
            std::lower_bound(order_.begin(), order_.end(), id,
                             [this](uint32_t candidate, std::string_view key) {
                                 return ids_.Get(candidate) < key;
                             });
        if (at == order_.end() || ids_.Get(*at) != id) return false;
        *row = *at;
        return true;
    }

   private:
    const IdColumn& ids_;
    std::vector<uint32_t> order_;
};

bool ReadBinaryShard(const std::string& path, EdgeTally* tally, std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        *error = "cluster: could not open '" + path + "'";
        return false;
    }
    char magic[sizeof(kEdgeMagic)];
    file.read(magic, sizeof(magic));
    if (file.gcount() != static_cast<std::streamsize>(sizeof(magic)) ||
        std::memcmp(magic, kEdgeMagic, sizeof(magic)) != 0) {
        *error = "cluster: '" + path + "' is not a cpplink prediction shard";
        return false;
    }
    std::vector<char> buffer(kReadEdges * kEdgeBytes);
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const size_t got = static_cast<size_t>(file.gcount());
        if (got % kEdgeBytes != 0) {
            *error = "cluster: '" + path + "' is truncated mid-prediction";
            return false;
        }
        for (size_t at = 0; at < got; at += kEdgeBytes) {
            uint32_t a = 0;
            uint32_t b = 0;
            double weight = 0.0;
            std::memcpy(&a, buffer.data() + at, 4);
            std::memcpy(&b, buffer.data() + at + 4, 4);
            std::memcpy(&weight, buffer.data() + at + 12, 8);
            if (a >= tally->records || b >= tally->records) {
                *error = "cluster: '" + path + "' names row " +
                         std::to_string(a >= tally->records ? a : b) +
                         " but the data has " + std::to_string(tally->records) +
                         " records";
                return false;
            }
            tally->Use(a, b, weight);
        }
    }
    return true;
}

bool ReadCsvEdges(const std::string& path, const IdIndex& index, EdgeTally* tally,
                  std::string* error) {
    std::ifstream file(path);
    if (!file) {
        *error = "cluster: could not open '" + path + "'";
        return false;
    }
    std::string line;
    uint64_t number = 0;
    while (std::getline(file, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (number == 1 && line.rfind("id_a,", 0) == 0) continue;
        std::string_view id_a;
        std::string_view id_b;
        uint32_t gamma = 0;
        double weight = 0.0;
        if (!ParseEdgeCsvLine(line, &id_a, &id_b, &gamma, &weight)) {
            *error = "cluster: '" + path + "' line " + std::to_string(number) +
                     " is not a cpplink prediction row";
            return false;
        }
        uint32_t a = 0;
        uint32_t b = 0;
        if (!index.Find(id_a, &a) || !index.Find(id_b, &b)) {
            ++tally->read;
            ++tally->unresolved;
            continue;
        }
        tally->Use(a, b, weight);
    }
    return true;
}

// One row group at a time, and only the three columns clustering reads: the
// pattern and the posterior are in the file for whoever else opens it.
bool ReadParquetEdges(const std::string& path, const IdIndex& index, EdgeTally* tally,
                      std::string* error) {
    auto input = arrow::io::ReadableFile::Open(path);
    if (!input.ok()) {
        *error = "cluster: cannot open " + path + ": " + input.status().message();
        return false;
    }
    parquet::arrow::FileReaderBuilder builder;
    arrow::Status status = builder.Open(*input);
    if (!status.ok()) {
        *error =
            "cluster: cannot read parquet metadata for " + path + ": " + status.message();
        return false;
    }
    auto reader_result = builder.Build();
    if (!reader_result.ok()) {
        *error = "cluster: cannot open parquet reader for " + path + ": " +
                 reader_result.status().message();
        return false;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);

    std::shared_ptr<arrow::Schema> schema;
    status = reader->GetSchema(&schema);
    if (!status.ok()) {
        *error = "cluster: cannot read the schema of " + path + ": " + status.message();
        return false;
    }
    std::vector<int> indices;
    for (const char* name : {"id_a", "id_b", "match_weight"}) {
        const int at = schema->GetFieldIndex(name);
        if (at < 0) {
            *error = "cluster: '" + path + "' has no column \"" + name +
                     "\", so it is not a cpplink prediction file";
            return false;
        }
        indices.push_back(at);
    }

    for (int group = 0; group < reader->num_row_groups(); ++group) {
        auto group_result = reader->ReadRowGroup(group, indices);
        if (!group_result.ok()) {
            *error = "cluster: cannot read row group " + std::to_string(group) + " of " +
                     path + ": " + group_result.status().message();
            return false;
        }
        const std::shared_ptr<arrow::Table> table = *group_result;
        const auto ids_a = table->GetColumnByName("id_a");
        const auto ids_b = table->GetColumnByName("id_b");
        const auto weights = table->GetColumnByName("match_weight");
        for (int chunk = 0; chunk < ids_a->num_chunks(); ++chunk) {
            const auto a_array =
                std::dynamic_pointer_cast<arrow::StringArray>(ids_a->chunk(chunk));
            const auto b_array =
                std::dynamic_pointer_cast<arrow::StringArray>(ids_b->chunk(chunk));
            const auto weight_array =
                std::dynamic_pointer_cast<arrow::DoubleArray>(weights->chunk(chunk));
            if (a_array == nullptr || b_array == nullptr || weight_array == nullptr) {
                *error = "cluster: '" + path +
                         "' holds id_a, id_b or match_weight in a type a cpplink "
                         "prediction file does not use";
                return false;
            }
            for (int64_t row = 0; row < a_array->length(); ++row) {
                if (a_array->IsNull(row) || b_array->IsNull(row) ||
                    weight_array->IsNull(row)) {
                    ++tally->read;
                    ++tally->unresolved;
                    continue;
                }
                uint32_t a = 0;
                uint32_t b = 0;
                if (!index.Find(a_array->GetView(row), &a) ||
                    !index.Find(b_array->GetView(row), &b)) {
                    ++tally->read;
                    ++tally->unresolved;
                    continue;
                }
                tally->Use(a, b, weight_array->Value(row));
            }
        }
    }
    return true;
}

}  // namespace

UnionFind::UnionFind(uint64_t records)
    : parent_(static_cast<size_t>(records)), rank_(static_cast<size_t>(records), 0) {
    for (size_t i = 0; i < parent_.size(); ++i) parent_[i] = static_cast<uint32_t>(i);
}

uint32_t UnionFind::Find(uint32_t row) {
    // Path halving rather than full compression: one pass, no recursion, and the
    // tree ends flat enough that the difference does not show up in a run.
    while (parent_[row] != row) {
        parent_[row] = parent_[parent_[row]];
        row = parent_[row];
    }
    return row;
}

bool UnionFind::Union(uint32_t a, uint32_t b) {
    uint32_t ra = Find(a);
    uint32_t rb = Find(b);
    if (ra == rb) return false;
    if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
    parent_[rb] = ra;
    if (rank_[ra] == rank_[rb]) ++rank_[ra];
    ++merges_;
    return true;
}

bool Cluster(const RecordStore& store, const ClusterOptions& options,
             ClusterAssignment* assignment, ClusterReport* report, std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    const uint64_t records = store.NumRecords();
    UnionFind uf(records);
    EdgeTally tally(&uf, records, options.threshold);

    std::vector<std::string> shards;
    std::string edge_file;
    double index_seconds = 0.0;
    std::error_code ec;
    if (std::filesystem::is_directory(options.edge_path, ec)) {
        if (!CollectShards(options.edge_path, &shards, error)) return false;
        for (const std::string& path : shards) {
            if (!ReadBinaryShard(path, &tally, error)) return false;
        }
    } else {
        // A merged file names records by unique_id, which is what makes it worth
        // anything outside cpplink and what costs an index to read back.
        MergeFormat format = MergeFormat::kCsv;
        if (!MergedFormatOf(options.edge_path, &format)) {
            *error = "cluster: '" + options.edge_path +
                     "' is neither a directory of shards nor a .csv or .parquet "
                     "prediction file";
            return false;
        }
        const auto index_started = std::chrono::steady_clock::now();
        const IdIndex index(store);
        index_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                      index_started)
                            .count();
        const bool read = format == MergeFormat::kParquet
                              ? ReadParquetEdges(options.edge_path, index, &tally, error)
                              : ReadCsvEdges(options.edge_path, index, &tally, error);
        if (!read) return false;
        edge_file = options.edge_path;
    }

    const uint64_t edges_read = tally.read;
    const uint64_t edges_used = tally.used;
    const double min_weight = tally.min_weight;
    const double max_weight = tally.max_weight;

    assignment->root.resize(static_cast<size_t>(records));
    assignment->size.assign(static_cast<size_t>(records), 0);
    for (uint64_t row = 0; row < records; ++row) {
        const uint32_t root = uf.Find(static_cast<uint32_t>(row));
        assignment->root[static_cast<size_t>(row)] = root;
        ++assignment->size[root];
    }
    assignment->clusters = 0;
    assignment->singletons = 0;
    assignment->clustered = 0;
    assignment->largest = 0;
    assignment->implied_pairs = 0;
    for (uint64_t row = 0; row < records; ++row) {
        const uint64_t size = assignment->size[static_cast<size_t>(row)];
        if (size == 0) continue;
        if (size == 1) {
            ++assignment->singletons;
            continue;
        }
        ++assignment->clusters;
        assignment->clustered += size;
        assignment->largest = std::max(assignment->largest, size);
        assignment->implied_pairs += size * (size - 1) / 2;
    }

    if (report != nullptr) {
        report->records = records;
        report->edges_read = edges_read;
        report->edges_used = edges_used;
        report->merges = uf.Merges();
        report->min_weight = min_weight;
        report->max_weight = max_weight;
        report->unresolved = tally.unresolved;
        report->shards = std::move(shards);
        report->edge_file = std::move(edge_file);
        report->index_seconds = index_seconds;
        report->clusters = assignment->clusters;
        report->singletons = assignment->singletons;
        report->clustered = assignment->clustered;
        report->largest = assignment->largest;
        report->implied_pairs = assignment->implied_pairs;
        // One pass over the roots rather than one per bucket: the bucket count is
        // small but the record count is not.
        constexpr size_t kBucketCount = sizeof(kBuckets) / sizeof(kBuckets[0]);
        uint64_t tally[kBucketCount] = {0};
        uint64_t covered[kBucketCount] = {0};
        for (uint64_t row = 0; row < records; ++row) {
            const uint64_t size = assignment->size[static_cast<size_t>(row)];
            if (size == 0) continue;
            for (size_t b = 0; b < kBucketCount; ++b) {
                if (size >= kBuckets[b].low && size <= kBuckets[b].high) {
                    ++tally[b];
                    covered[b] += size;
                    break;
                }
            }
        }
        report->buckets.clear();
        for (size_t b = 0; b < kBucketCount; ++b) {
            if (tally[b] == 0) continue;
            SizeBucket bucket;
            bucket.label = kBuckets[b].label;
            bucket.clusters = tally[b];
            bucket.records = covered[b];
            report->buckets.push_back(std::move(bucket));
        }
        report->seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
    }
    return true;
}

bool WriteClusters(const ClusterAssignment& assignment, const RecordStore& store,
                   const ClusterOptions& options, uint64_t* written, std::string* error) {
    *written = 0;
    if (options.out_path.empty()) return true;
    std::ofstream file(options.out_path);
    if (!file) {
        *error = "cluster: could not write '" + options.out_path + "'";
        return false;
    }
    std::string buffer = "unique_id,cluster_id,cluster_size\n";
    for (uint64_t row = 0; row < assignment.root.size(); ++row) {
        const uint32_t root = assignment.root[static_cast<size_t>(row)];
        const uint64_t size = assignment.size[root];
        if (size < options.min_size) continue;
        buffer.append(store.ids().Get(row));
        buffer.push_back(',');
        buffer.append(store.ids().Get(root));
        buffer.push_back(',');
        buffer.append(std::to_string(size));
        buffer.push_back('\n');
        ++*written;
        if (buffer.size() >= (1u << 20)) {
            file.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            buffer.clear();
        }
    }
    file.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    if (file.fail()) {
        *error = "cluster: failed writing '" + options.out_path + "'";
        return false;
    }
    return true;
}

ClusterQuality MeasureClusters(const ClusterAssignment& assignment,
                               const TruthPairs& truth) {
    ClusterQuality quality;
    quality.listed_pairs = truth.rows.size();
    quality.implied_pairs = assignment.implied_pairs;

    const uint64_t records = assignment.root.size();
    UnionFind closure(records);
    for (const auto& pair : truth.rows) closure.Union(pair.first, pair.second);

    std::vector<uint32_t> truth_root(static_cast<size_t>(records));
    std::vector<uint32_t> truth_size(static_cast<size_t>(records), 0);
    for (uint64_t row = 0; row < records; ++row) {
        const uint32_t root = closure.Find(static_cast<uint32_t>(row));
        truth_root[static_cast<size_t>(row)] = root;
        ++truth_size[root];
    }
    for (uint64_t row = 0; row < records; ++row) {
        const uint64_t size = truth_size[static_cast<size_t>(row)];
        if (size < 2) continue;
        ++quality.truth_clusters;
        quality.largest_truth_cluster = std::max(quality.largest_truth_cluster, size);
        quality.truth_pairs += size * (size - 1) / 2;
    }

    // A pair is recovered when both rows share a predicted cluster *and* a truth
    // cluster, so counting rows per (predicted, truth) cell and summing C(n, 2)
    // gives the intersection exactly, without enumerating pairs.
    std::unordered_map<uint64_t, uint32_t> cell;
    cell.reserve(static_cast<size_t>(records));
    for (uint64_t row = 0; row < records; ++row) {
        const uint64_t key =
            (static_cast<uint64_t>(assignment.root[static_cast<size_t>(row)]) << 32) |
            truth_root[static_cast<size_t>(row)];
        ++cell[key];
    }
    for (const auto& entry : cell) {
        const uint64_t n = entry.second;
        if (n >= 2) quality.recovered += n * (n - 1) / 2;
    }
    if (quality.implied_pairs > 0) {
        quality.precision = static_cast<double>(quality.recovered) /
                            static_cast<double>(quality.implied_pairs);
    }
    if (quality.truth_pairs > 0) {
        quality.recall = static_cast<double>(quality.recovered) /
                         static_cast<double>(quality.truth_pairs);
    }
    if (quality.precision + quality.recall > 0.0) {
        quality.f1 = 2.0 * quality.precision * quality.recall /
                     (quality.precision + quality.recall);
    }
    return quality;
}

void PrintClusterReport(const ClusterReport& report, std::ostream& out) {
    out << "Read " << WithThousands(report.edges_read) << " predictions from ";
    if (report.edge_file.empty()) {
        out << report.shards.size() << " shard" << (report.shards.size() == 1 ? "" : "s");
    } else {
        out << report.edge_file;
    }
    out << " in " << report.seconds << " s\n";
    if (report.index_seconds > 0.0) {
        out << "Resolved their ids against the record ids in " << report.index_seconds
            << " s\n";
    }
    if (report.unresolved > 0) {
        out << "WARNING: " << WithThousands(report.unresolved)
            << " predictions name a record this data file does not hold ("
            << Percent(report.unresolved, report.edges_read) << "); they were skipped\n";
    }
    if (report.edges_used != report.edges_read) {
        out << "Kept " << WithThousands(report.edges_used) << " above the clustering "
            << "threshold (" << Percent(report.edges_used, report.edges_read) << ")\n";
    }
    if (report.edges_used > 0) {
        out << "Weights " << report.min_weight << " to " << report.max_weight
            << " bits\n";
    }
    // An edge that merges nothing is an edge whose endpoints were already joined,
    // which is how much redundancy the blocked union carried.
    out << WithThousands(report.merges) << " of them merged two components ("
        << Percent(report.merges, report.edges_used) << ")\n\n";

    out << WithThousands(report.clusters) << " clusters of two or more, covering "
        << WithThousands(report.clustered) << " records ("
        << Percent(report.clustered, report.records) << ")\n";
    out << WithThousands(report.singletons) << " records stayed alone\n";
    out << "Largest cluster " << WithThousands(report.largest) << " records; the "
        << "partition asserts " << WithThousands(report.implied_pairs)
        << " duplicate pairs\n";
    if (report.written > 0) {
        out << WithThousands(report.written) << " rows written\n";
    }

    out << "\nSize            Clusters          Records\n";
    out << "-------------------------------------------\n";
    for (const SizeBucket& bucket : report.buckets) {
        char line[128];
        std::snprintf(line, sizeof(line), "%-14s %10s %16s\n", bucket.label.c_str(),
                      WithThousands(bucket.clusters).c_str(),
                      WithThousands(bucket.records).c_str());
        out << line;
    }
    out << "-------------------------------------------\n";
}

void PrintClusterQuality(const ClusterQuality& quality, std::ostream& out) {
    out << "\nAgainst the known duplicates, with both sides closed transitively:\n";
    out << "  listed     " << WithThousands(quality.listed_pairs)
        << " pairs in the truth file\n";
    out << "  true       " << WithThousands(quality.truth_pairs) << " pairs across "
        << WithThousands(quality.truth_clusters) << " clusters (largest "
        << quality.largest_truth_cluster << ")\n";
    out << "  recovered  " << WithThousands(quality.recovered) << "\n";
    out << "  asserted   " << WithThousands(quality.implied_pairs) << "\n";
    char line[128];
    std::snprintf(line, sizeof(line),
                  "  precision  %.4f\n  recall     %.4f\n  f1  %14.4f\n",
                  quality.precision, quality.recall, quality.f1);
    out << line;
}

}  // namespace cpplink
