// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>

#include "cpplink/blocking.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

// Prices every source before a single pair is enumerated. At 20M records a
// careless source is catastrophic, and the group-size distribution says so in
// seconds rather than after an overnight run.
void PrintBlockingReport(const BlockingPlan& plan, const RecordStore& store,
                         bool count_union, std::ostream& out);

// The same numbers as one JSON object, for a tool that draws them rather than
// reads them: per source its kind, column, knob, EM-safety, candidate pairs and
// largest group, then the sum, the pair space, and the union when counted. A
// MinHash spec is one entry per band, as the plan holds it.
void WriteBlockingJson(const BlockingPlan& plan, const RecordStore& store,
                       bool count_union, std::ostream& out);

}  // namespace cpplink
