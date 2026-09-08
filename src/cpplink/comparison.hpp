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
    // The same thing over the list column's own dictionary, for the pairwise
    // levels. It is worth more here than anywhere else: a pairwise level runs
    // |a| x |b| metric evaluations where a scalar level runs one, so the pair of
    // popcounts that rejects an element pair is paid back that many times over.
    const SignatureTable* list_signatures = nullptr;
    // Present only when this comparison has one of the membership levels. Indexed
    // by a
    // value id of `strings`, holding the id the same text has in `lists`'s own
    // dictionary, or kNullId where the list column never saw that text. Dense
    // ids are per column, so "is this first name one of those nicknames" is an
    // integer question only once the two dictionaries have been aligned, and
    // aligning them once at bind time is what keeps the pair path integer-only.
    const uint32_t* alias_ids = nullptr;
    uint32_t alias_size = 0;
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
    // The pairwise levels: whether any element of one row's list is within the
    // level's threshold of any element of the other's.
    //
    // `bounds_only` stops at the signature bound instead of running the metric,
    // which is what `LevelMaybe` needs and is admissible for the same reason the
    // scalar filter is. One walk serves both so the bound and the predicate
    // cannot come to disagree about which element pairs they consider.
    bool ClosestPairFires(const BoundComparison& comparison, const LevelSpec& level,
                          uint64_t a, uint64_t b, bool bounds_only) const;
    // The fuzzy membership levels, in both directions like the exact one: whether
    // either row's value is within the level's threshold of an element of the
    // other row's list. `bounds_only` means the same thing it does above.
    bool NearListFires(const BoundComparison& comparison, const LevelSpec& level,
                       uint64_t a, uint64_t b, bool bounds_only) const;
    // One direction of it: `value`, of the scalar column's dictionary, against
    // every element of `row`'s list.
    bool NearCell(const BoundComparison& comparison, const LevelSpec& level,
                  uint32_t value, uint64_t row, bool bounds_only) const;

    std::vector<BoundComparison> bound_;
    // unique_ptr so the tables keep their address as the vector grows, and so a
    // ComparisonSet cannot be copied into one whose comparisons point at another's
    // tables.
    std::vector<std::unique_ptr<SignatureTable>> tables_;
    // One per list_contains comparison, addressed through BoundComparison. Held
    // by pointer for the same reason the signature tables are: the vector may
    // grow, and a BoundComparison holds the data() of one of these.
    std::vector<std::unique_ptr<std::vector<uint32_t>>> alias_maps_;
    uint8_t width_ = 0;
};

}  // namespace cpplink
