// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "cpplink/record_store.hpp"

namespace cpplink {

// What the merged file is written as. Parquet is typed and a tenth the size;
// csv is what a spreadsheet and every other tool will open.
enum class MergeFormat { kCsv, kParquet };

// Which shards to read. The two shard formats hold the same edges, so a
// directory holding both is one run written twice rather than twice the edges,
// and reading both would double every row. `kAuto` prefers the binary shards,
// which are the ones clustering reads back.
enum class MergeSource { kAuto, kBinary, kCsv };

struct MergeOptions {
    std::string edge_dir;
    std::string out_path;
    MergeFormat format = MergeFormat::kCsv;
    MergeSource source = MergeSource::kAuto;
    // Edges carry their weight, so a merge can drop what a higher threshold
    // would not have written, exactly as `cluster --threshold` does.
    double threshold = -std::numeric_limits<double>::infinity();
    // Rows per parquet row group. Ignored by the csv writer, which flushes on
    // bytes rather than on rows.
    size_t batch_rows = 65536;
};

struct MergeReport {
    std::vector<std::string> shards;   // what was read, in name order
    std::vector<std::string> ignored;  // the other format's shards, left alone
    MergeSource source = MergeSource::kBinary;
    MergeFormat format = MergeFormat::kCsv;
    std::string out_path;
    // Binary shards name rows, not records, so without a store the merged file
    // carries row indices and the report says so.
    bool row_indices = false;
    // Whether the merged rows name each record's dataset beside its id, which a
    // run over more than one input does.
    bool datasets = false;
    uint64_t read = 0;
    uint64_t written = 0;
    double seconds = 0.0;
};

// Streams every shard in `options.edge_dir` past a single writer. Nothing is
// held: a row is read, filtered, written and forgotten, so the merge costs what
// the output costs and not what the run did. `store` supplies the ids that
// binary shards do not carry and may be null, in which case rows are named by
// index; csv shards carry their own ids and need no store.
bool MergeEdges(const RecordStore* store, const MergeOptions& options,
                MergeReport* report, std::string* error);

void PrintMergeReport(const MergeReport& report, std::ostream& out);

// One row of the edge csv, the shard format and the merged one alike, or of
// the merged parquet. The dataset halves are empty where the file has none. The
// views point into the line or the table they were read from.
struct EdgeRow {
    std::string_view dataset_a;
    std::string_view id_a;
    std::string_view dataset_b;
    std::string_view id_b;
    uint32_t gamma = 0;
    double weight = 0.0;
};

// Which of the two csv shapes a file is in, read off its header.
struct EdgeCsvLayout {
    bool datasets = false;  // `dataset_a,id_a,dataset_b,id_b,...`
};

// True when `line` is an edge csv header, setting the layout it announces. A
// file with no header is read in the single-input shape.
bool ParseEdgeCsvHeader(const std::string& line, EdgeCsvLayout* layout);

// Splits one row of the edge csv. The fields are unquoted, so a row is found by
// counting commas: the last three are the numbers and the first two or four
// name the records. This is the one parser, so the writer and every reader of
// that format cannot drift apart.
bool ParseEdgeCsvLine(const std::string& line, const EdgeCsvLayout& layout, EdgeRow* row);

// Folds a staging directory of shards into one file and removes the directory.
// This is the end of a `predict --out <file>` or `rescore --out <file>` run:
// threads write a shard each because a single writer would have them contending
// on it, and the merge is one sequential pass afterwards over edges that have
// already been paid for. `seconds` reports what that pass cost.
bool MergeStagedShards(const RecordStore& store, const std::string& staging,
                       const std::string& out_path, bool shards_are_binary,
                       double* seconds, std::string* error);

// True when the path names a single merged file rather than a directory of
// shards, with the format its extension asks for: `.parquet` or `.pq` is
// parquet, `.csv` is csv. One reading of an extension, shared by every command
// that takes such a path, so `predict --out` and `cluster --edges` cannot
// disagree about what a name means.
bool MergedFormatOf(const std::string& path, MergeFormat* format);

}  // namespace cpplink
