// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/neighbourhood.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "cpplink/pair_stream.hpp"
#include "cpplink/string_metrics.hpp"

namespace cpplink {
namespace {

// Rows of the outer loop handed out at a time. Values are sorted by nothing in
// particular, so a fixed stripe keeps the threads even without any planning.
constexpr uint32_t kStripe = 256;

}  // namespace

uint64_t BallMassTable::BytesUsed() const {
    uint64_t bytes = 0;
    for (const std::vector<uint32_t>& level : mass_) {
        bytes += level.size() * sizeof(uint32_t);
    }
    return bytes;
}

bool BallMassTable::Build(const ComparisonSet& comparisons, size_t comparison,
                          uint64_t records, const BallOptions& options,
                          std::string* reason) {
    const auto started = std::chrono::steady_clock::now();
    mass_.clear();
    covered_.clear();
    u_.clear();
    min_mass_.clear();
    max_mass_.clear();

    const BoundComparison& bound = comparisons.at(comparison);
    const ComparisonSpec& spec = *bound.spec;
    if (bound.strings == nullptr || bound.lists != nullptr || spec.columns.size() != 1) {
        *reason = "not a single string column";
        return false;
    }
    if (bound.signatures == nullptr) {
        *reason = "no fuzzy level, so every ball is the value itself";
        return false;
    }
    bool fuzzy = false;
    for (const LevelSpec& level : spec.levels) {
        if (level.type == LevelType::kLevenshtein ||
            level.type == LevelType::kJaroWinkler) {
            fuzzy = true;
        }
    }
    if (!fuzzy) {
        *reason = "no fuzzy level, so every ball is the value itself";
        return false;
    }
    if (records == 0) {
        *reason = "no records";
        return false;
    }

    const std::vector<uint32_t>& tf = bound.strings->tf;
    const uint32_t values = static_cast<uint32_t>(tf.size());
    values_ = values;
    const uint64_t pairs = static_cast<uint64_t>(values) * (values - 1) / 2;
    if (pairs > options.budget) {
        *reason = "dictionary of " + std::to_string(values) + " values is " +
                  std::to_string(pairs / 1000000) +
                  "M value pairs, over the budget; no fuzzy adjustment here";
        return false;
    }
    value_pairs_ = pairs;

    const size_t levels = spec.levels.size();
    covered_.assign(levels, false);
    for (size_t l = 0; l < levels; ++l) {
        const LevelType type = spec.levels[l].type;
        covered_[l] = type == LevelType::kExact || type == LevelType::kLevenshtein ||
                      type == LevelType::kJaroWinkler;
    }

    // Counts, not probabilities: the sum over a ball of how many records carry
    // each value, which fits a u32 because it can never exceed the record count.
    mass_.assign(levels, {});
    for (size_t l = 0; l < levels; ++l) {
        if (covered_[l]) mass_[l].assign(values, 0);
    }

    // The loosest bound any fuzzy level of this comparison could pass. Two
    // popcounts reject a value pair before the level loop is entered at all,
    // which is what keeps a quadratic scan over the dictionary affordable; it is
    // admissible because every level's own check is at least as strict.
    double loosest_jaro = 2.0;
    int longest_edit = -1;
    for (const LevelSpec& level : spec.levels) {
        if (level.type == LevelType::kJaroWinkler) {
            loosest_jaro = std::min(loosest_jaro, level.threshold);
        } else if (level.type == LevelType::kLevenshtein) {
            longest_edit = std::max(longest_edit, static_cast<int>(level.threshold));
        }
    }
    const SignatureTable& signatures = *bound.signatures;

    const unsigned threads = ResolveThreads(options.threads);
    std::vector<std::vector<std::vector<uint32_t>>> local(threads);
    std::vector<std::vector<double>> local_u(threads);
    std::atomic<uint32_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            // Each thread owns a full copy so the inner loop writes both ends of a
            // pair without a lock; they merge at join, like every other fold here.
            std::vector<std::vector<uint32_t>>& mine = local[t];
            mine.assign(levels, {});
            for (size_t l = 0; l < levels; ++l) {
                if (covered_[l]) mine[l].assign(values, 0);
            }
            std::vector<double>& u = local_u[t];
            u.assign(levels, 0.0);
            for (;;) {
                const uint32_t begin = next.fetch_add(kStripe);
                if (begin >= values) break;
                const uint32_t end = std::min<uint32_t>(begin + kStripe, values);
                for (uint32_t i = begin; i < end; ++i) {
                    const double count_i = static_cast<double>(tf[i]);
                    if (count_i == 0.0) continue;
                    const uint64_t mask_i = signatures.Mask(i);
                    const uint32_t length_i = signatures.Length(i);
                    for (uint32_t j = i + 1; j < values; ++j) {
                        if (tf[j] == 0) continue;
                        const uint64_t mask_j = signatures.Mask(j);
                        const uint32_t length_j = signatures.Length(j);
                        bool possible = false;
                        if (longest_edit >= 0 &&
                            LevenshteinLowerBound(mask_i, length_i, mask_j, length_j) <=
                                longest_edit) {
                            possible = true;
                        }
                        if (!possible && loosest_jaro <= 1.0 &&
                            JaroWinklerUpperBound(mask_i, length_i, mask_j, length_j) >=
                                loosest_jaro) {
                            possible = true;
                        }
                        if (!possible) continue;
                        const uint8_t level =
                            comparisons.LevelForValues(comparison, i, j);
                        if (!covered_[level]) continue;
                        mine[level][i] += tf[j];
                        mine[level][j] += tf[i];
                        // Both orders of the draw, on the same denominator the
                        // closed-form levels use.
                        u[level] += 2.0 * count_i * static_cast<double>(tf[j]);
                    }
                }
            }
        });
    }
    for (std::thread& worker : workers) worker.join();

    u_.assign(levels, 0.0);
    for (unsigned t = 0; t < threads; ++t) {
        for (size_t l = 0; l < levels; ++l) {
            if (!covered_[l]) continue;
            u_[l] += local_u[t][l];
            const std::vector<uint32_t>& source = local[t][l];
            std::vector<uint32_t>& target = mass_[l];
            for (uint32_t v = 0; v < values; ++v) target[v] += source[v];
        }
    }

    // A value is always in its own ball, and a draw of a value against itself is
    // an exact agreement. Adding both here rather than in the loop keeps the
    // inner loop to the pairs that need a metric.
    for (size_t l = 0; l < levels; ++l) {
        if (!covered_[l]) continue;
        const bool exact = spec.levels[l].type == LevelType::kExact;
        for (uint32_t v = 0; v < values; ++v) {
            if (tf[v] == 0) continue;
            if (exact) {
                mass_[l][v] += tf[v];
            } else {
                // A fuzzy level never fires on identical ids -- the exact level
                // above it takes those -- so its ball excludes the value itself.
                continue;
            }
        }
        if (exact) {
            for (uint32_t v = 0; v < values; ++v) {
                const double count = static_cast<double>(tf[v]);
                u_[l] += count * count;
            }
        }
    }

    inverse_records_ = 1.0 / static_cast<double>(records);
    const double denominator =
        static_cast<double>(records) * static_cast<double>(records);
    min_mass_.assign(levels, 0.0);
    max_mass_.assign(levels, 0.0);
    for (size_t l = 0; l < levels; ++l) {
        u_[l] /= denominator;
        if (!covered_[l]) continue;
        uint32_t low = 0;
        uint32_t high = 0;
        bool seen = false;
        for (uint32_t v = 0; v < values; ++v) {
            if (tf[v] == 0) continue;
            const uint32_t here = mass_[l][v];
            if (here == 0) continue;  // no pair can land on this level with v
            if (!seen) {
                low = here;
                high = here;
                seen = true;
                continue;
            }
            low = std::min(low, here);
            high = std::max(high, here);
        }
        // A level no value can reach brackets to nothing, and the scorer's
        // Reachable check keeps it out of the tables anyway.
        min_mass_[l] = seen ? static_cast<double>(low) * inverse_records_ : 0.0;
        max_mass_[l] = seen ? static_cast<double>(high) * inverse_records_ : 0.0;
    }

    seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    reason->clear();
    return true;
}

void BallTables::Build(const ComparisonSet& comparisons, uint64_t records,
                       const BallOptions& options) {
    const auto started = std::chrono::steady_clock::now();
    tables.resize(comparisons.Size());
    reasons.assign(comparisons.Size(), std::string());
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        std::string reason;
        if (!tables[c].Build(comparisons, c, records, options, &reason)) {
            reasons[c] = reason;
            tables[c] = BallMassTable();
            continue;
        }
        bytes += tables[c].BytesUsed();
    }
    seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

}  // namespace cpplink
