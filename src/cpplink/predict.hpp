// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/score.hpp"

namespace cpplink {

enum class EdgeFormat {
    kBinary,  // 20 bytes a row: a, b, gamma, weight. What clustering reads back.
    kCsv,     // record ids and the weight, for reading with your eyes.
};

struct PredictOptions {
    std::string out_dir;
    EdgeFormat format = EdgeFormat::kBinary;
    unsigned threads = 0;
    uint64_t max_edges = 0;  // 0 is unlimited
};

struct PredictReport {
    uint64_t enumerated = 0;
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
