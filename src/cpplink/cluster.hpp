// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

#include "cpplink/recall.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

// Union-find with union by rank and path halving. Parent is a row index and rank
// fits a byte, so the whole structure is 5 bytes a record: 100 MB at 20M rows,
// which is why the edges can be streamed past it and thrown away.
class UnionFind {
   public:
    explicit UnionFind(uint64_t records);

    uint32_t Find(uint32_t row);
    // True when the two rows were in different components, i.e. this edge did
    // something. Repeated edges are idempotent and cost only two finds.
    bool Union(uint32_t a, uint32_t b);

    uint64_t Records() const { return parent_.size(); }
    // Components merged so far: records minus the number of successful unions.
    uint64_t Components() const { return parent_.size() - merges_; }
    uint64_t Merges() const { return merges_; }

   private:
    std::vector<uint32_t> parent_;
    std::vector<uint8_t> rank_;
    uint64_t merges_ = 0;
};

// The finished partition: `root` is fully compressed, so two rows are in the same
// cluster exactly when their entries are equal.
struct ClusterAssignment {
    std::vector<uint32_t> root;  // by row, its representative row
    std::vector<uint32_t> size;  // by representative row; 0 for non-roots
    uint64_t clusters = 0;       // components of two or more
    uint64_t singletons = 0;
    uint64_t clustered = 0;  // records in a cluster of two or more
    uint64_t largest = 0;
    uint64_t implied_pairs = 0;  // sum of C(size, 2) over clusters

    bool SameCluster(uint32_t a, uint32_t b) const { return root[a] == root[b]; }
};

struct ClusterOptions {
    // Either the directory of `shard-*.bin` a run wrote, or the single `.csv` or
    // `.parquet` file it was merged into. A merged file names records by
    // `unique_id` rather than by row, so clustering one costs an id index.
    std::string edge_path;
    std::string out_path;  // empty writes no file
    // Edges carry their weight, so clustering at a threshold above the one
    // predict wrote at costs a re-read and no re-scoring.
    double threshold = -std::numeric_limits<double>::infinity();
    uint64_t min_size = 2;  // smallest cluster written out
};

struct SizeBucket {
    std::string label;
    uint64_t clusters = 0;
    uint64_t records = 0;
};

struct ClusterReport {
    uint64_t records = 0;
    uint64_t edges_read = 0;
    uint64_t edges_used = 0;  // above the clustering threshold
    uint64_t merges = 0;      // edges that actually joined two components
    double min_weight = 0.0;
    double max_weight = 0.0;
    uint64_t clusters = 0;
    uint64_t singletons = 0;
    uint64_t clustered = 0;
    uint64_t largest = 0;
    uint64_t implied_pairs = 0;
    uint64_t written = 0;
    // Edges naming an id no loaded record has. Wrong data file, wrong schema, or
    // a merged file from another run; either way they are counted, not fatal.
    uint64_t unresolved = 0;
    std::vector<SizeBucket> buckets;
    std::vector<std::string> shards;  // empty when a single file was read
    std::string edge_file;            // empty when the shards were read
    double index_seconds = 0.0;       // building the id index, for a merged file
    double seconds = 0.0;
};

// Pairwise quality of the partition against known duplicates, with **both sides
// closed transitively**.
//
// The predicted side is a partition, so it already asserts every pair inside a
// cluster. The truth side has to be closed to match: a truth file lists the pairs
// something happened to generate, but being-a-duplicate is transitive, so if a and
// b are both copies of the same record then (a, b) is a true pair whether or not
// any file lists it. Scoring a transitive partition against a non-transitive list
// counts real pairs as false positives and understates precision.
struct ClusterQuality {
    uint64_t listed_pairs = 0;   // pairs as written in the truth file
    uint64_t truth_pairs = 0;    // pairs after closing the truth transitively
    uint64_t recovered = 0;      // true pairs whose rows share a predicted cluster
    uint64_t implied_pairs = 0;  // pairs the partition asserts
    uint64_t truth_clusters = 0;
    uint64_t largest_truth_cluster = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

// Streams every edge under `edge_path` into a union-find and closes it into a
// partition. Nothing holds an edge: a record is read, unioned, forgotten.
//
// A shard directory names rows and needs nothing from the store but its size; a
// merged csv or parquet file names `unique_id`s, and those are resolved through
// an index over the store's id column.
bool Cluster(const RecordStore& store, const ClusterOptions& options,
             ClusterAssignment* assignment, ClusterReport* report, std::string* error);

// Writes `unique_id,cluster_id,cluster_size` for every record in a cluster of at
// least `min_size`. The cluster id is the representative's own unique id, so the
// output says which record the others collapse onto.
bool WriteClusters(const ClusterAssignment& assignment, const RecordStore& store,
                   const ClusterOptions& options, uint64_t* written, std::string* error);

ClusterQuality MeasureClusters(const ClusterAssignment& assignment,
                               const TruthPairs& truth);

void PrintClusterReport(const ClusterReport& report, std::ostream& out);
void PrintClusterQuality(const ClusterQuality& quality, std::ostream& out);

}  // namespace cpplink
