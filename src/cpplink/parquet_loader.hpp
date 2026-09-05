// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

struct LoadStats {
    uint64_t rows = 0;
    int row_groups = 0;
    double seconds = 0.0;
    std::vector<uint64_t> dataset_rows;  // rows contributed by each input, in order
};

// Reads a parquet file into the record store one row group at a time, so the
// Arrow buffers for a group are released before the next is read and the two
// representations are never both resident for the whole file.
bool LoadParquet(const std::string& path, const Schema& schema, RecordStore* store,
                 LoadStats* stats, std::string* error);

// The same, over several inputs read in order into one store. Each file becomes a
// dataset, and because rows are appended one file at a time a dataset is a
// contiguous row range -- which is what makes a link mode's cross-product a nested
// loop over two ranges rather than a per-row test. Everything downstream sees one
// store; only the pair mode knows there was more than one file.
bool LoadParquetFiles(const std::vector<std::string>& paths, const Schema& schema,
                      RecordStore* store, LoadStats* stats, std::string* error);

}  // namespace cpplink
