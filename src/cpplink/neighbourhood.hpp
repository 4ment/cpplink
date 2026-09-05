// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"

namespace cpplink {

// Neighbourhood mass: how much of the file falls inside one value's ball.
//
// Term-frequency adjustment is defined only for exact agreement, here and in
// splink, where an exact-match level is *required* before an adjustment can be
// made at all. So two records sharing the misspelling "Zolnerowitch" against
// "Zolnerowich" get the averaged fuzzy weight, and the rarity that makes the pair
// convincing is discarded at the moment it matters most.
//
// The generalisation is not the value's own frequency but the mass of its
// neighbourhood under the level's own predicate:
//
//     M_l(v) = sum over w in B_l(v) of p_w      B_l(v) = { w : (v,w) reaches l }
//
// which is exactly p_v when the level is exact and B(v) = {v}. Nobody computes it
// because it is a similarity self-join over every distinct value, which a
// row-oriented or SQL-backed pipeline cannot afford. This one can, for a
// structural reason: values are interned, so the join is over the 609k distinct
// surnames rather than the 18M rows, it is one pass per column, and it is
// amortised over every pair the run scores. The signature table -- length plus a
// 64-bit character-presence mask -- prunes it with two loads and two popcounts,
// the same primitive the pair path already uses.
//
// The same scan answers a second question for free. u for a fuzzy level is the
// probability two random records land on it, which is what this join enumerates:
//
//     u_l = sum over ordered value pairs landing on l of p_v * p_w
//
// so the sampled-and-rescaled u the estimator falls back to for fuzzy levels
// becomes exact, with no Monte Carlo anywhere in it.
//
// What it does not do is scale to a near-unique column: 16.6M distinct emails is
// 1.4e14 value pairs, and no signature filter makes that affordable. Such a
// column is refused by budget and keeps today's behaviour, which the report says
// out loud rather than quietly degrading.

struct BallOptions {
    // Value pairs the join may look at for one comparison. A dictionary too large
    // for this is refused rather than approximated.
    uint64_t budget = 40'000'000'000ull;
    unsigned threads = 0;
};

// The neighbourhood masses of one comparison, by level and by value id.
class BallMassTable {
   public:
    // `comparison` must be a single-column string comparison with at least one
    // fuzzy level. Returns false and fills `reason` when it is not, or when the
    // dictionary is larger than the budget allows.
    bool Build(const ComparisonSet& comparisons, size_t comparison, uint64_t records,
               const BallOptions& options, std::string* reason);

    bool Empty() const { return mass_.empty(); }
    // Whether this level got a mass column: every level a pair of present values
    // can land on except the trailing else, whose ball is everything left over.
    bool Covers(size_t level) const { return level < covered_.size() && covered_[level]; }
    // P(a random record's value falls in this value's ball at this level).
    double Mass(size_t level, uint32_t id) const {
        return static_cast<double>(mass_[level][id]) * inverse_records_;
    }
    double MinMass(size_t level) const { return min_mass_[level]; }
    double MaxMass(size_t level) const { return max_mass_[level]; }
    // Exact u for the level, over ordered draws, on the same denominator the
    // closed-form exact and null levels use.
    double LevelU(size_t level) const { return u_[level]; }

    uint64_t ValuePairs() const { return value_pairs_; }
    uint64_t Values() const { return values_; }
    double Seconds() const { return seconds_; }
    uint64_t BytesUsed() const;

   private:
    std::vector<std::vector<uint32_t>> mass_;  // [level][value id], in records
    std::vector<bool> covered_;
    std::vector<double> u_;
    std::vector<double> min_mass_;
    std::vector<double> max_mass_;
    double inverse_records_ = 0.0;
    uint64_t value_pairs_ = 0;
    uint64_t values_ = 0;
    double seconds_ = 0.0;
};

// One table per comparison the schema declares, built where it can be and left
// empty where it cannot. Reasons are collected rather than thrown away: which
// columns got a fuzzy adjustment is part of what a run has to report.
struct BallTables {
    std::vector<BallMassTable> tables;
    std::vector<std::string> reasons;  // by comparison; empty where one was built
    double seconds = 0.0;
    uint64_t bytes = 0;

    void Build(const ComparisonSet& comparisons, uint64_t records,
               const BallOptions& options);
    bool Has(size_t comparison) const {
        return comparison < tables.size() && !tables[comparison].Empty();
    }
};

}  // namespace cpplink
