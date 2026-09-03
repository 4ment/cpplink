// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"

namespace cpplink {

struct PatternCount {
    uint32_t gamma = 0;
    uint64_t count = 0;
};

// Counts of distinct agreement patterns. This is the only thing estimation reads,
// and it is why estimation costs the same at 18M records as at 18k: a pair enters
// the likelihood only through its pattern, so the pair itself is discarded the
// moment its gamma is folded in here.
//
// Below kDenseWidth the container is a flat array indexed by gamma, which is no
// hashing at all; above it, open addressing on uint32 keys.
class PatternHistogram {
   public:
    static constexpr uint8_t kDenseWidth = 22;

    explicit PatternHistogram(uint8_t width);

    void Add(uint32_t gamma, uint64_t count = 1);
    void Merge(const PatternHistogram& other);

    uint64_t TotalPairs() const { return total_; }
    uint64_t DistinctPatterns() const { return distinct_; }
    bool Dense() const { return dense_; }
    uint64_t Bytes() const;

    // Every observed pattern with its count, heaviest first.
    std::vector<PatternCount> Entries() const;

   private:
    void Grow();

    uint8_t width_ = 0;
    bool dense_ = false;
    // The sparse table marks a free slot with all-ones, which is also a legal
    // gamma at the full 32-bit width. That one pattern gets its own slot rather
    // than a comment promising it cannot happen.
    bool has_sentinel_ = false;
    uint64_t sentinel_count_ = 0;
    std::vector<uint64_t> counts_;  // dense: by gamma. sparse: parallel to keys_.
    std::vector<uint32_t> keys_;    // sparse only; kEmpty marks a free slot
    uint64_t distinct_ = 0;
    uint64_t total_ = 0;
};

struct HistogramOptions {
    unsigned threads = 0;   // 0 asks the hardware
    uint64_t pair_cap = 0;  // 0 folds every pair; otherwise Bernoulli-sampled
    uint64_t seed = 1;
};

struct HistogramStats {
    uint64_t enumerated = 0;  // candidate pairs the sources produced
    uint64_t folded = 0;      // pairs actually compared, after any sampling
    double rate = 1.0;        // the Bernoulli acceptance rate used
    uint64_t distinct = 0;
    unsigned threads = 0;
    double seconds = 0.0;
};

// Folds the deduplicated pair stream of `selected` into one histogram. Nothing
// here holds a row per pair: each thread keeps its own histogram over its own
// slice of the groups, and they merge at join.
//
// A pair_cap makes this a Bernoulli sample of the stream. Because m, u and lambda
// are all ratios of pattern counts, a uniformly sampled histogram estimates them
// without any reweighting -- and it buys the thing that actually costs: the
// comparison, not the enumeration.
void BuildHistogram(const BlockingPlan& plan, const ComparisonSet& comparisons,
                    const std::vector<size_t>& selected, const HistogramOptions& options,
                    PatternHistogram* histogram, HistogramStats* stats);

}  // namespace cpplink
