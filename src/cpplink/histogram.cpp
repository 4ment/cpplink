// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/histogram.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace cpplink {
namespace {

constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();
// Roughly a million pairs per task: small enough that no thread is left holding
// the tail alone, large enough that the atomic counter is never contended.
constexpr uint64_t kPairsPerTask = 1u << 20;

uint64_t Mix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

uint32_t Scatter(uint32_t gamma) { return static_cast<uint32_t>(Mix64(gamma) >> 32); }

// One slice of one source's pair stream. Whole groups when row_end is zero,
// otherwise rows [row_begin, row_end) of the single group at `begin`.
struct PairTask {
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t row_begin = 0;
    uint64_t row_end = 0;
};

uint64_t PairsIn(uint64_t group) { return group * (group - 1) / 2; }

// Groups are batched until a task is worth taking, and an oversized group is cut
// into row ranges of its own triangle -- otherwise one "Smith" block is a whole
// thread's run while the others idle.
void PlanGroupTasks(const SourceGroups& groups, std::vector<PairTask>* tasks) {
    tasks->clear();
    const uint64_t count = groups.GroupCount();
    uint64_t batch = 0;
    uint64_t pending = 0;
    for (uint64_t g = 0; g < count; ++g) {
        const uint64_t rows = groups.Size(g);
        if (PairsIn(rows) > kPairsPerTask) {
            if (g > batch) tasks->push_back({batch, g, 0, 0});
            uint64_t low = 0;
            uint64_t chunk = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                chunk += rows - 1 - i;
                if (chunk >= kPairsPerTask) {
                    tasks->push_back({g, g + 1, low, i + 1});
                    low = i + 1;
                    chunk = 0;
                }
            }
            if (low < rows) tasks->push_back({g, g + 1, low, rows});
            batch = g + 1;
            pending = 0;
            continue;
        }
        pending += PairsIn(rows);
        if (pending >= kPairsPerTask) {
            tasks->push_back({batch, g + 1, 0, 0});
            batch = g + 1;
            pending = 0;
        }
    }
    if (batch < count) tasks->push_back({batch, count, 0, 0});
}

void PlanWindowTasks(uint64_t starts, uint32_t window, std::vector<PairTask>* tasks) {
    tasks->clear();
    const uint64_t step =
        std::max<uint64_t>(1, kPairsPerTask / std::max<uint32_t>(1, window));
    for (uint64_t i = 0; i < starts; i += step) {
        tasks->push_back({i, std::min(i + step, starts), 0, 0});
    }
}

unsigned ThreadCount(unsigned requested) {
    if (requested > 0) return requested;
    const unsigned available = std::thread::hardware_concurrency();
    return available > 0 ? available : 1;
}

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
    const unsigned threads = ThreadCount(options.threads);
    std::atomic<uint64_t> enumerated{0};
    std::atomic<uint64_t> folded{0};

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

    std::vector<PatternHistogram> partials(threads,
                                           PatternHistogram(comparisons.Width()));
    SourceGroups groups;
    std::vector<PairTask> tasks;

    for (size_t position = 0; position < selected.size(); ++position) {
        const BoundSource& source = plan.at(selected[position]);
        const bool window = source.kind == SourceKind::kSortedNeighbourhood;
        if (window) {
            PlanWindowTasks(source.order.size(), source.window, &tasks);
        } else {
            plan.BuildGroups(selected[position], &groups);
            PlanGroupTasks(groups, &tasks);
        }

        std::atomic<size_t> next{0};
        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (unsigned t = 0; t < threads; ++t) {
            workers.emplace_back([&, t] {
                PatternHistogram& local = partials[t];
                uint64_t seen = 0;
                uint64_t kept = 0;
                uint64_t state = Mix64(options.seed + t * 0x9E3779B97F4A7C15ull);
                auto fold = [&](uint32_t a, uint32_t b) {
                    ++seen;
                    if (sampling) {
                        state = Mix64(state);
                        if (state >= threshold) return;
                    }
                    ++kept;
                    local.Add(comparisons.Evaluate(a, b));
                };
                for (;;) {
                    const size_t index = next.fetch_add(1);
                    if (index >= tasks.size()) break;
                    const PairTask& task = tasks[index];
                    if (window) {
                        plan.EnumerateWindowRange(selected, position, task.begin,
                                                  task.end, fold);
                    } else if (task.row_end != 0) {
                        plan.EnumerateGroupRange(selected, position, groups, task.begin,
                                                 task.row_begin, task.row_end, fold);
                    } else {
                        for (uint64_t g = task.begin; g < task.end; ++g) {
                            plan.EnumerateGroupRange(selected, position, groups, g, 0,
                                                     groups.Size(g), fold);
                        }
                    }
                }
                enumerated.fetch_add(seen);
                folded.fetch_add(kept);
            });
        }
        for (std::thread& worker : workers) worker.join();
    }

    for (const PatternHistogram& partial : partials) histogram->Merge(partial);

    if (stats != nullptr) {
        stats->enumerated = enumerated.load();
        stats->folded = folded.load();
        stats->rate = rate;
        stats->distinct = histogram->DistinctPatterns();
        stats->threads = threads;
        stats->seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
    }
}

}  // namespace cpplink
