// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/histogram.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "cpplink/pair_stream.hpp"

namespace cpplink {
namespace {

constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();

// Per-thread scratch, one cache line each. These are touched once per pair, so
// packing them adjacently would put every thread's increment on the same line and
// cost more than the work being counted.
struct alignas(64) ThreadState {
    uint64_t seen = 0;
    uint64_t kept = 0;
    uint64_t random = 0;
};

uint64_t Mix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

uint32_t Scatter(uint32_t gamma) { return static_cast<uint32_t>(Mix64(gamma) >> 32); }

}  // namespace

PatternHistogram::PatternHistogram(uint8_t width) : width_(width) {
    dense_ = width <= kDenseWidth;
    if (dense_) {
        counts_.assign(size_t{1} << width, 0);
    } else {
        keys_.assign(1024, kEmpty);
        counts_.assign(1024, 0);
    }
}

void PatternHistogram::Grow() {
    std::vector<uint32_t> old_keys = std::move(keys_);
    std::vector<uint64_t> old_counts = std::move(counts_);
    keys_.assign(old_keys.size() * 2, kEmpty);
    counts_.assign(old_keys.size() * 2, 0);
    const uint32_t mask = static_cast<uint32_t>(keys_.size() - 1);
    for (size_t i = 0; i < old_keys.size(); ++i) {
        if (old_keys[i] == kEmpty) continue;
        uint32_t slot = Scatter(old_keys[i]) & mask;
        while (keys_[slot] != kEmpty) slot = (slot + 1) & mask;
        keys_[slot] = old_keys[i];
        counts_[slot] = old_counts[i];
    }
}

void PatternHistogram::Add(uint32_t gamma, uint64_t count) {
    total_ += count;
    if (dense_) {
        if (counts_[gamma] == 0) ++distinct_;
        counts_[gamma] += count;
        return;
    }
    if (gamma == kEmpty) {
        if (!has_sentinel_) {
            has_sentinel_ = true;
            ++distinct_;
        }
        sentinel_count_ += count;
        return;
    }
    const uint32_t mask = static_cast<uint32_t>(keys_.size() - 1);
    uint32_t slot = Scatter(gamma) & mask;
    while (keys_[slot] != kEmpty && keys_[slot] != gamma) slot = (slot + 1) & mask;
    if (keys_[slot] == kEmpty) {
        keys_[slot] = gamma;
        counts_[slot] = count;
        ++distinct_;
        if (distinct_ * 4 > keys_.size() * 3) Grow();
        return;
    }
    counts_[slot] += count;
}

void PatternHistogram::Merge(const PatternHistogram& other) {
    if (other.dense_) {
        for (size_t gamma = 0; gamma < other.counts_.size(); ++gamma) {
            if (other.counts_[gamma] != 0) {
                Add(static_cast<uint32_t>(gamma), other.counts_[gamma]);
            }
        }
        return;
    }
    for (size_t i = 0; i < other.keys_.size(); ++i) {
        if (other.keys_[i] != kEmpty) Add(other.keys_[i], other.counts_[i]);
    }
    if (other.has_sentinel_) Add(kEmpty, other.sentinel_count_);
}

uint64_t PatternHistogram::Bytes() const {
    return counts_.size() * sizeof(uint64_t) + keys_.size() * sizeof(uint32_t);
}

std::vector<PatternCount> PatternHistogram::Entries() const {
    std::vector<PatternCount> entries;
    entries.reserve(distinct_);
    if (dense_) {
        for (size_t gamma = 0; gamma < counts_.size(); ++gamma) {
            if (counts_[gamma] != 0) {
                entries.push_back({static_cast<uint32_t>(gamma), counts_[gamma]});
            }
        }
    } else {
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i] != kEmpty) entries.push_back({keys_[i], counts_[i]});
        }
        if (has_sentinel_) entries.push_back({kEmpty, sentinel_count_});
    }
    std::sort(entries.begin(), entries.end(),
              [](const PatternCount& left, const PatternCount& right) {
                  if (left.count != right.count) return left.count > right.count;
                  return left.gamma < right.gamma;
              });
    return entries;
}

void BuildHistogram(const BlockingPlan& plan, const ComparisonSet& comparisons,
                    const std::vector<size_t>& selected, const HistogramOptions& options,
                    PatternHistogram* histogram, HistogramStats* stats) {
    const auto started = std::chrono::steady_clock::now();
    const unsigned threads = ResolveThreads(options.threads);
    uint64_t enumerated = 0;
    uint64_t folded = 0;

    // The acceptance rate is fixed before the run rather than adapted during it:
    // an adaptive rate would weight early groups differently from late ones, and
    // the groups are sorted by key, not shuffled.
    double rate = 1.0;
    if (options.pair_cap > 0) {
        uint64_t bound = 0;
        for (const size_t source : selected) bound += plan.CountPairs(source);
        if (bound > options.pair_cap) {
            rate = static_cast<double>(options.pair_cap) / static_cast<double>(bound);
        }
    }
    const bool sampling = rate < 1.0;
    const uint64_t threshold =
        sampling ? static_cast<uint64_t>(rate * 18446744073709549568.0) : 0;
    // Each thread draws from its own stream, so the sample does not depend on how
    // the tasks happened to be handed out.
    std::vector<ThreadState> scratch(threads);
    std::vector<std::unique_ptr<PatternHistogram>> partials;
    partials.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        scratch[t].random = Mix64(options.seed + t * 0x9E3779B97F4A7C15ull);
        partials.push_back(std::make_unique<PatternHistogram>(comparisons.Width()));
    }

    ForEachPairParallel(plan, selected, threads, [&](unsigned t) {
        ThreadState* state = &scratch[t];
        PatternHistogram* local = partials[t].get();
        return [&, state, local](uint32_t a, uint32_t b) {
            ++state->seen;
            if (sampling) {
                state->random = Mix64(state->random);
                if (state->random >= threshold) return;
            }
            ++state->kept;
            local->Add(comparisons.Evaluate(a, b));
        };
    });

    for (unsigned t = 0; t < threads; ++t) {
        enumerated += scratch[t].seen;
        folded += scratch[t].kept;
    }

    for (const auto& partial : partials) histogram->Merge(*partial);

    if (stats != nullptr) {
        stats->enumerated = enumerated;
        stats->folded = folded;
        stats->rate = rate;
        stats->distinct = histogram->DistinctPatterns();
        stats->threads = threads;
        stats->seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
    }
}

}  // namespace cpplink
