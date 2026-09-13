// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/merge_edges.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/score.hpp"

namespace cpplink {

// `explain --predictions <file> --out <file>`: the waterfall of every prediction
// a run wrote, as one wide row per pair. The file is what a viewer draws from,
// so the arithmetic stays in the scorer and a tool only reads.
//
// The columns are the pair and its totals, then four per comparison and one per
// two-way correction, named after the comparison:
//
//   id_a, id_b, gamma, prior, match_weight, match_probability,
//   bracket_low, bracket_high, threshold, zone, emitted,
//   <comparison>_level, <comparison>_bits, <comparison>_tf, <comparison>_frequency,
//   <left>_x_<right>_bits
//
// A level's label, m and u are the model's rather than the pair's, so they are
// not repeated here: the model file carries them, indexed by the level.
struct WaterfallOptions {
    std::string predictions_path;
    std::string out_path;
    size_t batch_rows = 65536;  // rows per parquet row group
};

struct WaterfallReport {
    MergeFormat format = MergeFormat::kCsv;
    std::string out_path;
    uint64_t read = 0;
    uint64_t written = 0;
    // Predictions naming an id no record holds. A file from another input.
    uint64_t unresolved = 0;
    // Predictions whose stored gamma is not what the comparisons produce now.
    // The schema changed under the file; the row is written with the pattern
    // the comparisons give today, and the count says the file is stale.
    uint64_t pattern_changed = 0;
    size_t columns = 0;
    double seconds = 0.0;
};

bool WriteWaterfalls(const RecordStore& store, const ComparisonSet& comparisons,
                     const Scorer& scorer, const Model& model,
                     const WaterfallOptions& options, WaterfallReport* report,
                     std::string* error);

void PrintWaterfallReport(const WaterfallReport& report, std::ostream& out);

// The column names the file carries, in order, for the tools that read it.
std::vector<std::string> WaterfallColumns(const ComparisonSet& comparisons,
                                          const Scorer& scorer);

}  // namespace cpplink
