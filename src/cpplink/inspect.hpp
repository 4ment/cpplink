// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>

#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

// Prints what the store actually cost: per-column cardinality and null counts,
// then the memory breakdown by structure. Phase 0 exists to replace the design's
// estimates with these numbers.
void PrintInspection(const RecordStore& store, const LoadStats& stats, std::ostream& out);

}  // namespace cpplink
