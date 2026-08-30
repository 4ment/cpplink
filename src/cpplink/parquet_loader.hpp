// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

struct LoadStats {
    uint64_t rows = 0;
    int row_groups = 0;
    double seconds = 0.0;
};

// Reads a parquet file into the record store one row group at a time, so the
// Arrow buffers for a group are released before the next is read and the two
// representations are never both resident for the whole file.
bool LoadParquet(const std::string& path, const Schema& schema, RecordStore* store,
                 LoadStats* stats, std::string* error);

}  // namespace cpplink
