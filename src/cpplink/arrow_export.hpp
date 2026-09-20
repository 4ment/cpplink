// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "cpplink/arrow_c.hpp"

namespace cpplink {

// A record batch built column by column and handed out as the C Data structs,
// which own the memory until they are released. It is the producer side of the
// seam `batch_loader` is the consumer side of: every table cpplink writes -- the
// merged predictions, the clusters, the waterfalls, the sample -- is built here
// and handed either to the parquet writer or, in Python, to pandas, without a
// copy in between and without the builder knowing which.

enum class ExportType : uint8_t {
    kString,   // utf8, 32-bit offsets
    kBoolean,  // bit-packed
    kUInt8,
    kUInt32,
    kUInt64,
    kDouble,
    kDate32,       // days since the epoch
    kStringList,   // list<utf8>
    kDictionary,   // uint32 indices into a large_utf8 dictionary; borrowed only
    kLargeString,  // large_utf8, 64-bit offsets; borrowed only
};

// A column the caller already holds, exported as it sits: the buffers are
// pointed at, not copied, and the batch keeps whatever `BatchBuilder::KeepAlive`
// was given until the export is released. It is how a run's predictions leave
// as a table without a row being copied, and how the record ids come out as a
// dictionary over the store's own id arena with the rows as indices.
struct BorrowedColumn {
    ExportType type = ExportType::kUInt32;
    int64_t length = 0;
    // Fixed-width values, the dictionary indices, or a large string's offsets.
    const void* values = nullptr;
    const char* text = nullptr;  // a large string's bytes
    // The dictionary, for kDictionary: 64-bit offsets over the text, as an id
    // arena holds them.
    const int64_t* dictionary_offsets = nullptr;
    const char* dictionary_text = nullptr;
    int64_t dictionary_size = 0;
};

class BatchBuilder {
   public:
    BatchBuilder();
    ~BatchBuilder();
    BatchBuilder(BatchBuilder&&) noexcept;
    BatchBuilder& operator=(BatchBuilder&&) noexcept;

    // Declares the next column; the index is what the appenders take. Columns
    // must be declared before any row is appended.
    int AddColumn(std::string name, ExportType type);
    // Declares a column that is already built, and stays where it is. A
    // borrowed column takes no appends, is exported every time rather than moved
    // out, and must be as long as the rows of the batch.
    int AddBorrowed(std::string name, BorrowedColumn column);
    // What every export of this builder keeps alive until it is released: the
    // owner of whatever the borrowed columns point into.
    void KeepAlive(std::shared_ptr<void> owner);
    size_t NumColumns() const;
    std::vector<std::string> ColumnNames() const;

    // One value of one column. Every column of a batch must end at the same
    // length, which `ExportBatch` checks; `AppendNull` is valid on every type.
    void AppendString(int column, std::string_view value);
    void AppendBoolean(int column, bool value);
    void AppendUInt8(int column, uint8_t value);
    void AppendUInt32(int column, uint32_t value);
    void AppendUInt64(int column, uint64_t value);
    void AppendDouble(int column, double value);
    void AppendDate32(int column, int32_t days);
    void AppendStringList(int column, const std::vector<std::string>& values);
    void AppendNull(int column);

    // Rows appended to the first column since the last export, or a borrowed
    // first column's length.
    int64_t Rows() const;

    // The schema of the columns declared, as a struct with one child per column.
    // The caller releases it.
    void ExportSchema(ArrowSchema* out) const;

    // Moves every row appended so far out as one batch, leaving the builder empty
    // and ready for the next. The caller releases the array, which frees the
    // rows. Fails, leaving the rows in place, where the columns differ in length.
    bool ExportBatch(ArrowArray* out, std::string* error);

    struct Column;  // the buffers of one column, in the layout the C structs want

   private:
    std::vector<Column> columns_;
    std::shared_ptr<void> owner_;
};

}  // namespace cpplink
