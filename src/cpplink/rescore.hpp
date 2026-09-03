// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/score.hpp"
#include "cpplink/spill.hpp"

namespace cpplink {

struct RescoreOptions {
    std::string spill_dir;
    std::string out_dir;
    EdgeFormat format = EdgeFormat::kBinary;
    unsigned threads = 0;
    uint64_t max_edges = 0;
};

struct RescoreReport {
    uint64_t pairs = 0;  // pairs read back from the spill
    uint64_t edges = 0;  // pairs clearing the new threshold
    uint64_t dropped = 0;
    uint64_t tf_lookups = 0;
    unsigned threads = 0;
    double seconds = 0.0;
    bool truncated = false;
    // True when the new threshold sits below the one the spill was written at, so
    // pairs the original run discarded could have qualified and are simply absent.
    bool below_spill_threshold = false;
    SpillManifest manifest;
    std::vector<std::string> shards;
};

// Re-scores a spilled run under a new model. Every string metric was already paid
// for and its answer is in gamma, so this is a linear read: no pair is enumerated,
// no comparison is evaluated, and the term-frequency tables are the only thing the
// store is still needed for.
bool Rescore(const RecordStore& store, const ComparisonSet& comparisons,
             const Scorer& scorer, const RescoreOptions& options, RescoreReport* report,
             std::string* error);

void PrintRescoreReport(const RescoreReport& report, const Scorer& scorer,
                        std::ostream& out);

}  // namespace cpplink
