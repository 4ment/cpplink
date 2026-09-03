// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

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
};

class ComparisonSet {
   public:
    // Resolves every comparison in the schema against the loaded store.
    bool Bind(const Schema& schema, const RecordStore& store, std::string* error);

    // The packed agreement pattern for a pair. This is the hot path.
    uint32_t Evaluate(uint64_t a, uint64_t b) const;

    // The level index one comparison assigns to a pair.
    uint8_t EvaluateOne(size_t comparison, uint64_t a, uint64_t b) const;

    // Reads a level index back out of a packed pattern.
    uint8_t LevelOf(uint32_t gamma, size_t comparison) const;

    // Whether the comparison has no value to read on this row. Estimation counts
    // these once over the data, which makes the null level's u exact rather than
    // sampled.
    bool IsNullValue(size_t comparison, uint64_t row) const;

    size_t Size() const { return bound_.size(); }
    const BoundComparison& at(size_t index) const { return bound_[index]; }
    uint8_t Width() const { return width_; }

   private:
    bool IsNull(const BoundComparison& comparison, uint64_t row) const;
    bool LevelFires(const BoundComparison& comparison, const LevelSpec& level, uint64_t a,
                    uint64_t b) const;

    std::vector<BoundComparison> bound_;
    uint8_t width_ = 0;
};

}  // namespace cpplink
