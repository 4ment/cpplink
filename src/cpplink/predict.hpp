// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/score.hpp"

namespace cpplink {

// The binary shard format, shared with the reader in cluster.cpp so the writer
// and the reader cannot drift apart.
inline constexpr char kEdgeMagic[8] = {'C', 'P', 'P', 'L', 'N', 'K', 'E', '1'};
inline constexpr size_t kEdgeBytes = 20;  // a, b, gamma as u32, then weight as f64

enum class EdgeFormat {
    kBinary,  // 20 bytes a row: a, b, gamma, weight. What clustering reads back.
    kCsv,     // record ids and the weight, for reading with your eyes.
};

// One thread's edge output buffer, flushed in bulk so no two threads ever contend
// on the writer. Public because `rescore` writes the same shards from a spill.
class EdgeShardWriter {
   public:
    bool Open(const std::string& path, EdgeFormat format);
    void WriteBinary(uint32_t a, uint32_t b, uint32_t gamma, double weight);
    void WriteCsv(std::string_view id_a, std::string_view id_b, uint32_t gamma,
                  double weight);
    bool Close();

   private:
    void Flush();

    EdgeFormat format_ = EdgeFormat::kBinary;
    std::ofstream file_;
    std::string buffer_;
};

std::string EdgeShardName(unsigned thread);

struct PredictOptions {
    std::string out_dir;
    EdgeFormat format = EdgeFormat::kBinary;
    unsigned threads = 0;
    uint64_t max_edges = 0;  // 0 is unlimited
    // Where to spill (a, b, gamma) alongside the edges, so the run can be
    // re-scored under a new model without paying for the comparison again.
    std::string spill_dir;
    // Additionally keep this fraction of every candidate, above threshold or not.
    // Without it a spill holds only what cleared the threshold, which is enough to
    // re-tune upward and not enough to see what a lower threshold would find.
    double spill_sample = 0.0;
    uint64_t spill_seed = 20260904;
};

struct PredictReport {
    uint64_t enumerated = 0;
    // Bounded away before the pattern was ever produced: the pair could not have
    // cleared the threshold whatever the string metrics said, so none ran.
    uint64_t skipped = 0;
    uint64_t dropped = 0;  // pattern below threshold even at its rarest values
    uint64_t checked = 0;  // bracket straddled the threshold
    uint64_t certain = 0;  // pattern above threshold even at its commonest values
    uint64_t edges = 0;
    uint64_t tf_lookups = 0;
    uint64_t patterns_drop = 0;
    uint64_t patterns_check = 0;
    uint64_t patterns_certain = 0;
    uint64_t pattern_space = 0;
    unsigned threads = 0;
    double seconds = 0.0;
    bool truncated = false;
    uint64_t spilled = 0;
    PairMode mode = PairMode::kAll;
    size_t datasets = 1;
    std::vector<std::string> shards;
};

// Scores the deduplicated union of every source and writes the edges above the
// threshold, one shard per thread. Nothing holds a row per candidate pair: a pair
// is scored, written or dropped, and forgotten.
bool Predict(const RecordStore& store, const ComparisonSet& comparisons,
             const BlockingPlan& plan, const Scorer& scorer,
             const PredictOptions& options, PredictReport* report, std::string* error);

void PrintPredictReport(const PredictReport& report, const Scorer& scorer,
                        std::ostream& out);

}  // namespace cpplink
