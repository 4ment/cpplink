// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include "cpplink/dictionary.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// A missing date. Like kNullId, it never compares equal to anything.
inline constexpr int32_t kNullDate = std::numeric_limits<int32_t>::min();

// The shape of the pair space a run enumerates.
//
// Deduplication is the upper triangle of one input; linking is the cross-product
// of two; link-and-dedup is the upper triangle of both inputs read as one, which
// is `kAll` over a store holding more than one dataset. Two shapes is therefore
// all the axis needs, and it lives here rather than in the schema because it is a
// property of how the store's inputs are read, not of the file that describes
// their columns.
enum class PairMode : uint8_t {
    kAll = 0,       // every pair of distinct rows
    kCrossDataset,  // only pairs whose rows came from different inputs
};

const char* PairModeName(PairMode mode);

struct StringColumn {
    Dictionary dict;
    std::vector<uint32_t> ids;  // one per record, kNullId where missing
    std::vector<uint32_t> tf;   // indexed by value id
};

// Arrays are sorted and deduplicated at load time, so intersection size, Jaccard
// and any-overlap are a linear merge with no hashing in the hot loop.
struct StringListColumn {
    Dictionary dict;
    std::vector<uint64_t> offsets;  // NumRecords() + 1
    std::vector<uint32_t> ids;      // flat
    std::vector<uint32_t> tf;       // by value id, counting rows not occurrences
};

struct DateColumn {
    std::vector<int32_t> values;  // kNullDate where missing
    int32_t tf_origin = 0;
    std::vector<uint32_t> tf;  // dense over [tf_origin, tf_origin + tf.size())
};

struct DoubleColumn {
    std::vector<double> values;  // NaN where missing
};

using Column = std::variant<StringColumn, StringListColumn, DateColumn, DoubleColumn>;

// Record identifiers, kept as an arena without an index: they are almost all
// distinct, so a hash map would cost as much as the values and buy nothing.
struct IdColumn {
    std::vector<char> text;
    std::vector<uint64_t> offsets;  // NumRecords() + 1

    void Append(std::string_view value);
    std::string_view Get(uint64_t row) const;
    uint64_t BytesUsed() const;
};

struct MemoryLine {
    std::string structure;
    uint64_t bytes = 0;
    std::string basis;
};

struct MemoryReport {
    std::vector<MemoryLine> lines;
    uint64_t Total() const;
};

// Immutable once loaded, and shared const across worker threads: nothing in the
// hot path takes a lock, and per-thread histograms merge at join.
class RecordStore {
   public:
    explicit RecordStore(Schema schema);

    const Schema& schema() const { return schema_; }
    uint64_t NumRecords() const { return num_records_; }
    size_t NumColumns() const { return columns_.size(); }
    const Column& column(size_t index) const { return columns_[index]; }
    Column& mutable_column(size_t index) { return columns_[index]; }
    const IdColumn& ids() const { return ids_; }
    IdColumn& mutable_ids() { return ids_; }

    void set_num_records(uint64_t count) { num_records_ = count; }

    // Rows are appended one input at a time, so a dataset is a contiguous row
    // range: which dataset a row belongs to is a boundary lookup and there is no
    // per-row array to pay for. `starts` holds NumDatasets() + 1 offsets, the last
    // being NumRecords(). A store loaded from a single file needs none of this and
    // reports one dataset covering everything.
    void set_datasets(std::vector<uint64_t> starts);
    const std::vector<uint64_t>& dataset_starts() const { return dataset_starts_; }
    size_t NumDatasets() const {
        return dataset_starts_.empty() ? 1 : dataset_starts_.size() - 1;
    }
    uint64_t DatasetStart(size_t dataset) const {
        return dataset_starts_.empty() ? 0 : dataset_starts_[dataset];
    }
    uint64_t DatasetEnd(size_t dataset) const {
        return dataset_starts_.empty() ? num_records_ : dataset_starts_[dataset + 1];
    }
    // Which input the row came from. Two inputs is the case that matters and costs
    // one comparison; more is a short walk over boundaries there are only a
    // handful of.
    uint32_t DatasetOf(uint64_t row) const {
        if (dataset_starts_.size() < 3) return 0;
        uint32_t dataset = 0;
        while (row >= dataset_starts_[dataset + 1]) ++dataset;
        return dataset;
    }
    // Where the row's own dataset ends. Rows at or past this boundary are in a
    // later input, which is what makes a cross-dataset partner range contiguous.
    uint64_t DatasetEndFor(uint64_t row) const {
        if (dataset_starts_.size() < 3) return num_records_;
        return dataset_starts_[DatasetOf(row) + 1];
    }
    // How many pairs the mode admits over the whole store: the denominator lambda
    // is put back on, and the number a candidate count is a fraction of.
    double PairSpace(PairMode mode) const;

    // Counts values once all rows are loaded. Skips kDouble columns, which carry
    // no term frequencies. Also releases the dictionaries' load-time indexes.
    void Finalize();

    uint32_t DistinctValues(size_t index) const;
    uint64_t NullCount(size_t index) const;
    MemoryReport Memory() const;

   private:
    Schema schema_;
    uint64_t num_records_ = 0;
    std::vector<Column> columns_;
    IdColumn ids_;
    std::vector<uint64_t> dataset_starts_;
};

}  // namespace cpplink
