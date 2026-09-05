// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace cpplink {

struct SampleOptions {
    uint64_t rows = 1000000;
    uint64_t seed = 1;
    double duplicate_rate = 0.08;  // fraction of rows that are corrupted copies
    int64_t row_group_size = 200000;
    std::string truth_path;  // optional sidecar of planted duplicate pairs
    // A second output. When set, originals go to the first file and every planted
    // duplicate to this one, so each recorded pair crosses the two files and the
    // link path has something to be measured against. `duplicate_rate` then sets
    // how large the second file is relative to the first.
    std::string link_path;
};

// Writes a parquet file with the column mix cpplink is aimed at: high-cardinality
// email and phone, Zipf-distributed names and postcodes, a date, a coordinate pair
// and a list column. A configurable share of rows are corrupted copies of earlier
// rows, which is what makes the file usable for later phases as well as this one.
//
// Records are generated deterministically from their row index, so a duplicate can
// reproduce its original exactly without either being held in memory.
bool WriteSampleParquet(const std::string& path, const SampleOptions& options,
                        std::string* error);

}  // namespace cpplink
