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
};

}  // namespace cpplink
