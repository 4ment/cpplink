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

#include "cpplink/arrow_export.hpp"
#include "cpplink/format.hpp"
#include "cpplink/id_index.hpp"
#include "cpplink/merge_edges.hpp"
#include "cpplink/parquet_io.hpp"
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
    uint64_t ambiguous = 0;
    double min_weight = 0.0;
    double max_weight = 0.0;
};

// Maps one named record of a merged file to its row. With a dataset the lookup
// is exact; without one it is answered only where a single record carries the
// id, and a file that names records held by several inputs without saying which
// is counted as ambiguous rather than guessed at.
bool Resolve(const IdIndex& index, const RecordStore& store, std::string_view dataset,
             std::string_view id, uint32_t* row, EdgeTally* tally) {
    IdLookup found = IdLookup::kMissing;
    if (dataset.empty()) {
        found = index.Find(id, row);
    } else {
        size_t which = 0;
        if (store.DatasetIndex(dataset, &which)) found = index.Find(which, id, row);
    }
    if (found == IdLookup::kFound) return true;
    if (found == IdLookup::kAmbiguous) ++tally->ambiguous;
    return false;
}

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

bool ReadCsvEdges(const std::string& path, const RecordStore& store, const IdIndex& index,
                  EdgeTally* tally, std::string* error) {
    std::ifstream file(path);
    if (!file) {
        *error = "cluster: could not open '" + path + "'";
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
            *error = "cluster: '" + path + "' line " + std::to_string(number) +
                     " is not a cpplink prediction row";
            return false;
        }
        uint32_t a = 0;
        uint32_t b = 0;
        if (!Resolve(index, store, edge.dataset_a, edge.id_a, &a, tally) ||
            !Resolve(index, store, edge.dataset_b, edge.id_b, &b, tally)) {
            ++tally->read;
            ++tally->unresolved;
            continue;
        }
        tally->Use(a, b, edge.weight);
    }
    return true;
}

// Every row of the merged parquet, through the same reader a data frame of
// predictions goes through. A row with a null id or dataset resolves to nothing
// and is counted as unresolved, as a csv row naming an unknown record is.
EdgeVisitor ResolvingVisitor(const RecordStore& store, const IdIndex& index,
                             EdgeTally* tally) {
    return [&store, &index, tally](const EdgeRow& edge, std::string*) {
        uint32_t a = 0;
        uint32_t b = 0;
        if (!Resolve(index, store, edge.dataset_a, edge.id_a, &a, tally) ||
            !Resolve(index, store, edge.dataset_b, edge.id_b, &b, tally)) {
            ++tally->read;
            ++tally->unresolved;
            return true;
        }
        tally->Use(a, b, edge.weight);
        return true;
    };
}

bool ReadParquetEdges(const std::string& path, const RecordStore& store,
                      const IdIndex& index, EdgeTally* tally, std::string* error) {
    EdgeColumns columns;
    return ReadPredictionFile(path, "cluster", {"match_weight"}, &columns,
                              ResolvingVisitor(store, index, tally), error);
}

// A prediction table from wherever it came, read as the merged file is.
bool ReadStreamEdges(ArrowArrayStream* stream, const RecordStore& store,
                     const IdIndex& index, EdgeTally* tally, std::string* error) {
    EdgeColumns columns;
    return ReadPredictionStream(stream, "cluster", "predictions", {"match_weight"},
                                &columns, ResolvingVisitor(store, index, tally), error);
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
    if (options.edges != nullptr) {
        // The run's own rows: nothing to resolve.
        for (uint64_t i = 0; i < options.edges->Size(); ++i) {
            tally.Use(options.edges->a[i], options.edges->b[i], options.edges->weight[i]);
        }
    } else if (options.stream != nullptr) {
        const auto index_started = std::chrono::steady_clock::now();
        const IdIndex index(store);
        index_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                      index_started)
                            .count();
        if (!index.Unique(error)) {
            *error = "cluster: " + *error;
            if (options.stream->release != nullptr)
                options.stream->release(options.stream);
            return false;
        }
        if (!ReadStreamEdges(options.stream, store, index, &tally, error)) return false;
    } else if (std::filesystem::is_directory(options.edge_path, ec)) {
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
        // An id two records of one input share names neither of them, so no row
        // of the file can be trusted to mean what it says.
        if (!index.Unique(error)) {
            *error = "cluster: " + *error;
            return false;
        }
        const bool read =
            format == MergeFormat::kParquet
                ? ReadParquetEdges(options.edge_path, store, index, &tally, error)
                : ReadCsvEdges(options.edge_path, store, index, &tally, error);
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
        report->ambiguous = tally.ambiguous;
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

const char* ClusterCsvHeader(bool datasets) {
    return datasets ? "dataset,unique_id,cluster_id,cluster_size"
                    : "unique_id,cluster_id,cluster_size";
}

namespace {

bool WriteClustersCsv(const ClusterAssignment& assignment, const RecordStore& store,
                      const ClusterOptions& options, uint64_t* written,
                      std::string* error) {
    std::ofstream file(options.out_path);
    if (!file) {
        *error = "cluster: could not write '" + options.out_path + "'";
        return false;
    }
    const bool datasets = store.NumDatasets() > 1;
    std::string buffer = ClusterCsvHeader(datasets);
    buffer.push_back('\n');
    for (uint64_t row = 0; row < assignment.root.size(); ++row) {
        const uint32_t root = assignment.root[static_cast<size_t>(row)];
        const uint64_t size = assignment.size[root];
        if (size < options.min_size) continue;
        if (datasets) {
            buffer.append(store.DatasetName(store.DatasetOf(row)));
            buffer.push_back(',');
        }
        buffer.append(store.ids().Get(row));
        buffer.push_back(',');
        buffer.append(QualifiedId(store, root));
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

// The same columns as the csv, typed: the names as strings and the size as a
// uint64, built as C Data batches of a million rows and handed to the writer.
bool WriteClustersParquet(const ClusterAssignment& assignment, const RecordStore& store,
                          const ClusterOptions& options, uint64_t* written,
                          std::string* error) {
    constexpr int64_t kRowGroup = 1 << 20;
    const bool datasets = store.NumDatasets() > 1;
    BatchBuilder builder;
    const int dataset = datasets ? builder.AddColumn("dataset", ExportType::kString) : -1;
    const int unique_id = builder.AddColumn("unique_id", ExportType::kString);
    const int cluster_id = builder.AddColumn("cluster_id", ExportType::kString);
    const int cluster_size = builder.AddColumn("cluster_size", ExportType::kUInt64);

    ParquetWriter writer;
    ArrowSchema schema;
    builder.ExportSchema(&schema);
    const bool opened = writer.Open(options.out_path, schema, error);
    schema.release(&schema);
    if (!opened) {
        *error = "cluster: " + *error;
        return false;
    }
    auto flush = [&]() -> bool {
        if (builder.Rows() == 0) return true;
        ArrowArray batch;
        if (!builder.ExportBatch(&batch, error) || !writer.Write(&batch, error)) {
            *error = "cluster: " + *error;
            return false;
        }
        return true;
    };
    for (uint64_t row = 0; row < assignment.root.size(); ++row) {
        const uint32_t root = assignment.root[static_cast<size_t>(row)];
        const uint64_t size = assignment.size[root];
        if (size < options.min_size) continue;
        if (datasets)
            builder.AppendString(dataset, store.DatasetName(store.DatasetOf(row)));
        builder.AppendString(unique_id, store.ids().Get(row));
        builder.AppendString(cluster_id, QualifiedId(store, root));
        builder.AppendUInt64(cluster_size, size);
        ++*written;
        if (builder.Rows() >= kRowGroup && !flush()) return false;
    }
    if (!flush()) return false;
    if (!writer.Close(error)) {
        *error = "cluster: " + *error;
        return false;
    }
    return true;
}

}  // namespace

bool WriteClusters(const ClusterAssignment& assignment, const RecordStore& store,
                   const ClusterOptions& options, uint64_t* written, std::string* error) {
    *written = 0;
    if (options.out_path.empty()) return true;
    MergeFormat format;
    if (!MergedFormatOf(options.out_path, &format)) {
        *error = "cluster: --out '" + options.out_path +
                 "' names neither a .csv nor a .parquet file";
        return false;
    }
    if (format == MergeFormat::kParquet) {
        return WriteClustersParquet(assignment, store, options, written, error);
    }
    return WriteClustersCsv(assignment, store, options, written, error);
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
    if (report.ambiguous > 0) {
        out << "WARNING: " << WithThousands(report.ambiguous)
            << " of those ids are held by more than one input and the file does not "
               "say which;\na prediction file written from these inputs carries "
               "dataset_a and dataset_b\n";
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
