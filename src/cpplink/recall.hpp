// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "cpplink/blocking.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

struct TruthPairs {
    std::vector<std::pair<uint32_t, uint32_t>> rows;
    uint64_t lines = 0;
    uint64_t unresolved = 0;  // ids in the file that are not in the data
};

// Reads an "id_a,id_b" CSV and resolves both ids to rows. Resolution is one pass
// over the id column against a set of only the wanted ids, rather than a full
// id index, which is not otherwise needed.
bool LoadTruthPairs(const std::string& path, const RecordStore& store, TruthPairs* truth,
                    std::string* error);

// With hand-written rules you can reason about what they miss. With automatic
// sources you cannot, so recall must be measured or the approach is unfalsifiable.
// Asking `Produces` of each known pair costs O(pairs x sources) and enumerates
// nothing.
void PrintRecallReport(const BlockingPlan& plan, const TruthPairs& truth,
                       std::ostream& out);

}  // namespace cpplink
