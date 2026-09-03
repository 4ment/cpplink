// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

#include "cpplink/comparison.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/score.hpp"

namespace cpplink {

// Prints the packed layout: which comparison occupies which bits of the pattern.
void PrintGammaLayout(const ComparisonSet& comparisons, std::ostream& out);

// Prints, for one pair, the level each comparison assigns and the values behind it,
// then the packed pattern. This is how a configuration is debugged before it is
// pointed at billions of pairs.
void PrintPairExplanation(const RecordStore& store, const ComparisonSet& comparisons,
                          uint64_t a, uint64_t b, std::ostream& out);

// Prints the match weight as a waterfall: the prior, then each comparison's
// contribution in bits and its term-frequency move, with a running total.
//
// A pattern says which levels fired; it does not say why the pair scored what it
// scored. Two exact surname matches carry the same gamma and can differ by ten
// bits, because one of them is "Smith". This is the view that shows that.
void PrintPairWaterfall(const RecordStore& store, const ComparisonSet& comparisons,
                        const Scorer& scorer, uint64_t a, uint64_t b, std::ostream& out);

// Resolves a unique_id to a row by linear scan. There is no id index: ids are
// almost all distinct, so an index would cost as much as the values and is only
// ever needed for one-off lookups like this one.
bool FindRowById(const RecordStore& store, const std::string& id, uint64_t* row);

}  // namespace cpplink
