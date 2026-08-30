// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>

#include "cpplink/blocking.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

// Prices every source before a single pair is enumerated. At 18M records a
// careless source is catastrophic, and the group-size distribution says so in
// seconds rather than after an overnight run.
void PrintBlockingReport(const BlockingPlan& plan, const RecordStore& store,
                         bool count_union, std::ostream& out);

}  // namespace cpplink
