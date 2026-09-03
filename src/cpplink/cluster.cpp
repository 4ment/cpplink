// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/cluster.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "cpplink/predict.hpp"

namespace cpplink {
namespace {

// A whole number of edge records, so a read never splits one across two buffers.
constexpr size_t kReadEdges = 4096;

std::string WithThousands(uint64_t value) {
    std::string digits = std::to_string(value);
    std::string out;
    int count = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (count > 0 && count % 3 == 0) out.push_back(',');
        out.push_back(*it);
        ++count;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

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
        *error = "cluster: '" + dir + "' is not a directory of edge shards";
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

bool Cluster(uint64_t records, const ClusterOptions& options,
             ClusterAssignment* assignment, ClusterReport* report, std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::string> shards;
    if (!CollectShards(options.edge_dir, &shards, error)) return false;

    UnionFind uf(records);
    uint64_t edges_read = 0;
    uint64_t edges_used = 0;
    double min_weight = 0.0;
    double max_weight = 0.0;
    std::vector<char> buffer(kReadEdges * kEdgeBytes);

    for (const std::string& path : shards) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            *error = "cluster: could not open '" + path + "'";
            return false;
        }
        char magic[sizeof(kEdgeMagic)];
        file.read(magic, sizeof(magic));
        if (file.gcount() != static_cast<std::streamsize>(sizeof(magic)) ||
            std::memcmp(magic, kEdgeMagic, sizeof(magic)) != 0) {
            *error = "cluster: '" + path + "' is not a cpplink edge shard";
            return false;
        }
        while (file) {
            file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const size_t got = static_cast<size_t>(file.gcount());
            if (got % kEdgeBytes != 0) {
                *error = "cluster: '" + path + "' is truncated mid-edge";
                return false;
            }
            for (size_t at = 0; at < got; at += kEdgeBytes) {
                uint32_t a = 0;
                uint32_t b = 0;
                double weight = 0.0;
                std::memcpy(&a, buffer.data() + at, 4);
                std::memcpy(&b, buffer.data() + at + 4, 4);
                std::memcpy(&weight, buffer.data() + at + 12, 8);
                ++edges_read;
                if (a >= records || b >= records) {
                    *error = "cluster: '" + path + "' names row " +
                             std::to_string(a >= records ? a : b) + " but the data has " +
                             std::to_string(records) + " records";
                    return false;
                }
                if (weight < options.threshold) continue;
                if (edges_used == 0) {
                    min_weight = weight;
                    max_weight = weight;
                } else {
                    min_weight = std::min(min_weight, weight);
                    max_weight = std::max(max_weight, weight);
                }
                ++edges_used;
                uf.Union(a, b);
            }
        }
    }

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
        report->clusters = assignment->clusters;
        report->singletons = assignment->singletons;
        report->clustered = assignment->clustered;
        report->largest = assignment->largest;
        report->implied_pairs = assignment->implied_pairs;
        report->shards = std::move(shards);
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
    quality.truth_pairs = truth.rows.size();
    quality.implied_pairs = assignment.implied_pairs;
    for (const auto& pair : truth.rows) {
        if (assignment.SameCluster(pair.first, pair.second)) ++quality.recovered;
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
    out << "Read " << WithThousands(report.edges_read) << " edges from "
        << report.shards.size() << " shard" << (report.shards.size() == 1 ? "" : "s")
        << " in " << report.seconds << " s\n";
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
    out << "\nAgainst " << WithThousands(quality.truth_pairs) << " known duplicate "
        << "pairs, over the transitive closure:\n";
    out << "  recovered  " << WithThousands(quality.recovered) << "\n";
    out << "  asserted   " << WithThousands(quality.implied_pairs) << "\n";
    char line[128];
    std::snprintf(line, sizeof(line),
                  "  precision  %.4f\n  recall     %.4f\n  f1  %14.4f\n",
                  quality.precision, quality.recall, quality.f1);
    out << line;
}

}  // namespace cpplink
