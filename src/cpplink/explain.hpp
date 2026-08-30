// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

#include "cpplink/comparison.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

// Prints the packed layout: which comparison occupies which bits of the pattern.
void PrintGammaLayout(const ComparisonSet& comparisons, std::ostream& out);

// Prints, for one pair, the level each comparison assigns and the values behind it,
// then the packed pattern. This is how a configuration is debugged before it is
// pointed at billions of pairs.
void PrintPairExplanation(const RecordStore& store, const ComparisonSet& comparisons,
                          uint64_t a, uint64_t b, std::ostream& out);

// Resolves a unique_id to a row by linear scan. There is no id index: ids are
// almost all distinct, so an index would cost as much as the values and is only
// ever needed for one-off lookups like this one.
bool FindRowById(const RecordStore& store, const std::string& id, uint64_t* row);

}  // namespace cpplink
