// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/signature.hpp"

namespace cpplink {

// A comparison bound to a loaded store. Column pointers are resolved once here so
// that evaluating a pair costs no variant lookup and no virtual call: the loop runs
// billions of times and this is where that budget is spent or wasted.
struct BoundComparison {
    const ComparisonSpec* spec = nullptr;
    uint8_t bits = 0;
    uint8_t shift = 0;

    const StringColumn* strings = nullptr;
    const DateColumn* dates = nullptr;
    const StringListColumn* lists = nullptr;
    const DoubleColumn* numbers = nullptr;   // also latitude for a geo comparison
    const DoubleColumn* numbers2 = nullptr;  // longitude
    // Present only when this comparison has a fuzzy string level, which is the
    // only place the signatures pay for themselves.
    const SignatureTable* signatures = nullptr;
};

class ComparisonSet {
   public:
    // Resolves every comparison in the schema against the loaded store.
    //
    // `use_signatures` exists so a test can run the same data with the filter off
    // and compare the patterns: the filter is only admissible if it changes
    // nothing, and that has to be checked rather than argued.
    bool Bind(const Schema& schema, const RecordStore& store, std::string* error,
              bool use_signatures = true);

    // The packed agreement pattern for a pair. This is the hot path.
    uint32_t Evaluate(uint64_t a, uint64_t b) const;

    // The level index one comparison assigns to a pair.
    uint8_t EvaluateOne(size_t comparison, uint64_t a, uint64_t b) const;

    // Reads a level index back out of a packed pattern.
    uint8_t LevelOf(uint32_t gamma, size_t comparison) const;

    // The level two *values* of a single-column string comparison land on,
    // counting only the levels a pair of present values can reach: the null level
    // is not one of them. This is the same predicate the pair path runs, reached
    // through value ids rather than rows, which is what lets the dictionary be
    // self-joined once instead of the pair stream being walked again.
    uint8_t LevelForValues(size_t comparison, uint32_t left, uint32_t right) const;

    // Whether a level could fire for this pair, decided without evaluating one
    // string metric: exact for the cheap level types, and the signature bounds
    // for the fuzzy ones. False means "certainly not"; true means "maybe".
    //
    // It is never false where `LevelFires` would be true, which is what lets a
    // caller bound the whole pair's weight before comparing any of it.
    bool LevelPossible(size_t comparison, size_t level, uint64_t a, uint64_t b) const;

    // Whether the comparison has no value to read on this row. Estimation counts
    // these once over the data, which makes the null level's u exact rather than
    // sampled.
    bool IsNullValue(size_t comparison, uint64_t row) const;

    size_t Size() const { return bound_.size(); }
    const BoundComparison& at(size_t index) const { return bound_[index]; }
    uint8_t Width() const { return width_; }
    // Resident bytes of the per-value signature tables.
    uint64_t SignatureBytes() const;

   private:
    bool IsNull(const BoundComparison& comparison, uint64_t row) const;
    bool LevelFires(const BoundComparison& comparison, const LevelSpec& level, uint64_t a,
                    uint64_t b) const;
    bool LevelMaybe(const BoundComparison& comparison, const LevelSpec& level, uint64_t a,
                    uint64_t b) const;
    // The string levels, over value ids. LevelFires routes its string cases here
    // so that the pair path and the dictionary self-join cannot drift apart.
    bool StringLevelFires(const BoundComparison& comparison, const LevelSpec& level,
                          uint32_t left, uint32_t right) const;

    std::vector<BoundComparison> bound_;
    // unique_ptr so the tables keep their address as the vector grows, and so a
    // ComparisonSet cannot be copied into one whose comparisons point at another's
    // tables.
    std::vector<std::unique_ptr<SignatureTable>> tables_;
    uint8_t width_ = 0;
};

}  // namespace cpplink
