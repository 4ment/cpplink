// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// Applies a chain of transforms to one value, leaving the result in `out`.
//
// An empty result is a missing value rather than an empty one: normalising "???"
// or taking the Soundex of "123" leaves nothing two rows could agree on, and a
// column of empty strings would block every one of those rows together and score
// them as agreeing. The caller turns an empty result into kNullId, which is the
// same rule the loader applies to a value the file does not hold.
void ApplyTransforms(const std::vector<Transform>& transforms, std::string_view value,
                     std::string* out);

// The same chain starting from a date, which is where the date transforms take it
// into a string and the rest of the chain continues as one.
void ApplyDateTransforms(const std::vector<Transform>& transforms, int32_t days,
                         std::string* out);

// Fills in every derived column of the store's schema. Finalize calls it once
// every input file has been read and before the term frequencies are counted, so
// a derived column is counted, blocked on, compared and profiled like any other
// and no caller has to remember it exists.
//
// The work is one pass over the source *dictionary* rather than over the rows,
// which is the whole economy of the thing: a phonetic key over 18M records is one
// Soundex per distinct surname and a gather. Nothing here can fail -- that a
// source exists and that its type feeds the chain are both settled at parse time,
// before a file is opened.
void BuildDerivedColumns(RecordStore* store);

}  // namespace cpplink
