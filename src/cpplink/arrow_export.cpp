// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/arrow_export.hpp"

#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cpplink {

// The buffers of one column, in the layout the C Data Interface wants them: a
// validity bitmap where any value is null, offsets and bytes for text, packed
// bits for booleans, a child column for the elements of a list.
struct BatchBuilder::Column {
    std::string name;
    ExportType type = ExportType::kString;
    int64_t length = 0;
    int64_t null_count = 0;
    std::vector<uint8_t> validity;  // grown only once a null is appended
    std::vector<int32_t> offsets;   // text and lists; starts at 0
    std::vector<uint8_t> data;      // text bytes, packed bits or fixed-width values
    std::unique_ptr<Column> child;  // the elements of a list
    bool borrowed = false;
    BorrowedColumn borrow;

    void SetValid(bool valid) {
        const size_t byte = static_cast<size_t>(length >> 3);
        if (!valid && validity.empty()) {
            // Every row so far was valid: fill the bitmap in retroactively.
            validity.assign(byte + 1, 0xFF);
        }
        if (!validity.empty()) {
            if (validity.size() <= byte) validity.push_back(0xFF);
            if (valid) {
                validity[byte] |= static_cast<uint8_t>(1u << (length & 7));
            } else {
                validity[byte] &= static_cast<uint8_t>(~(1u << (length & 7)));
                ++null_count;
            }
        }
        ++length;
    }

    template <typename T>
    void PutFixed(T value) {
        const size_t at = data.size();
        data.resize(at + sizeof(T));
        std::memcpy(data.data() + at, &value, sizeof(T));
    }

    void PutBit(bool value) {
        const size_t byte = static_cast<size_t>(length >> 3);
        if (data.size() <= byte) data.push_back(0);
        if (value) data[byte] |= static_cast<uint8_t>(1u << (length & 7));
    }

    void Clear() {
        length = 0;
        null_count = 0;
        validity.clear();
        offsets.clear();
        data.clear();
        if (child) child->Clear();
    }
};

namespace {

const char* FormatOf(ExportType type) {
    switch (type) {
        case ExportType::kString:
            return "u";
        case ExportType::kBoolean:
            return "b";
        case ExportType::kUInt8:
            return "C";
        case ExportType::kUInt32:
            return "I";
        case ExportType::kUInt64:
            return "L";
        case ExportType::kDouble:
            return "g";
        case ExportType::kDate32:
            return "tdD";
        case ExportType::kStringList:
            return "+l";
        case ExportType::kDictionary:
            return "I";  // the indices; the values are the dictionary's own schema
        case ExportType::kLargeString:
            return "U";
    }
    return "";
}

// What a released schema or array frees. Both hold their children inline, so a
// consumer releasing the parent releases the whole tree; the children's own
// release callbacks only mark them released, as the specification asks.
struct OwnedSchema {
    std::deque<std::string> names;  // a deque keeps every c_str() where it was
    std::vector<ArrowSchema> children;
    std::vector<ArrowSchema*> child_pointers;
    std::vector<std::unique_ptr<OwnedSchema>> nested;
};

void ReleaseChildSchema(ArrowSchema* schema) { schema->release = nullptr; }

void ReleaseSchema(ArrowSchema* schema) {
    delete static_cast<OwnedSchema*>(schema->private_data);
    schema->release = nullptr;
}

// Fills `out` for one column, hanging its child (a list's elements) off it.
void FillSchema(const BatchBuilder::Column& column, OwnedSchema* owner,
                ArrowSchema* out) {
    owner->names.push_back(column.name);
    out->format = FormatOf(column.type);
    out->name = owner->names.back().c_str();
    out->metadata = nullptr;
    out->flags = ARROW_FLAG_NULLABLE;
    out->n_children = 0;
    out->children = nullptr;
    out->dictionary = nullptr;
    out->release = ReleaseChildSchema;
    out->private_data = nullptr;
    if (column.child) {
        auto nested = std::make_unique<OwnedSchema>();
        nested->children.resize(1);
        FillSchema(*column.child, nested.get(), &nested->children[0]);
        nested->child_pointers = {&nested->children[0]};
        out->n_children = 1;
        out->children = nested->child_pointers.data();
        owner->nested.push_back(std::move(nested));
    }
    if (column.type == ExportType::kDictionary) {
        auto nested = std::make_unique<OwnedSchema>();
        nested->children.resize(1);
        ArrowSchema* values = &nested->children[0];
        values->format = "U";
        values->name = "";
        values->metadata = nullptr;
        values->flags = 0;
        values->n_children = 0;
        values->children = nullptr;
        values->dictionary = nullptr;
        values->release = ReleaseChildSchema;
        values->private_data = nullptr;
        out->dictionary = values;
        owner->nested.push_back(std::move(nested));
    }
}

struct OwnedArray {
    std::vector<std::unique_ptr<BatchBuilder::Column>> columns;
    std::vector<ArrowArray> children;
    std::vector<ArrowArray*> child_pointers;
    std::vector<std::vector<const void*>> buffers;
    std::vector<std::unique_ptr<OwnedArray>> nested;
    std::shared_ptr<void> keep_alive;  // what the borrowed columns point into
};

void ReleaseChildArray(ArrowArray* array) { array->release = nullptr; }

void ReleaseArray(ArrowArray* array) {
    delete static_cast<OwnedArray*>(array->private_data);
    array->release = nullptr;
}

// Points `out` at a borrowed column's buffers, and at its dictionary where it
// has one.
void FillBorrowed(const BorrowedColumn& column, OwnedArray* owner, ArrowArray* out) {
    out->length = column.length;
    out->null_count = 0;
    out->offset = 0;
    out->n_children = 0;
    out->children = nullptr;
    out->dictionary = nullptr;
    out->release = ReleaseChildArray;
    out->private_data = nullptr;
    if (column.type == ExportType::kLargeString) {
        owner->buffers.push_back({nullptr, column.values, column.text});
    } else {
        owner->buffers.push_back({nullptr, column.values});
    }
    out->n_buffers = static_cast<int64_t>(owner->buffers.back().size());
    out->buffers = owner->buffers.back().data();
    if (column.type == ExportType::kDictionary) {
        auto nested = std::make_unique<OwnedArray>();
        nested->children.resize(1);
        nested->buffers.reserve(1);
        ArrowArray* values = &nested->children[0];
        values->length = column.dictionary_size;
        values->null_count = 0;
        values->offset = 0;
        values->n_children = 0;
        values->children = nullptr;
        values->dictionary = nullptr;
        values->release = ReleaseChildArray;
        values->private_data = nullptr;
        nested->buffers.push_back(
            {nullptr, column.dictionary_offsets, column.dictionary_text});
        values->n_buffers = 3;
        values->buffers = nested->buffers.back().data();
        out->dictionary = values;
        owner->nested.push_back(std::move(nested));
    }
}

// Points `out` at the column's buffers, which `owner` now keeps alive.
void FillArray(BatchBuilder::Column* column, OwnedArray* owner, ArrowArray* out) {
    if (column->borrowed) {
        FillBorrowed(column->borrow, owner, out);
        return;
    }
    out->length = column->length;
    out->null_count = column->null_count;
    out->offset = 0;
    out->n_children = 0;
    out->children = nullptr;
    out->dictionary = nullptr;
    out->release = ReleaseChildArray;
    out->private_data = nullptr;
    std::vector<const void*> buffers;
    buffers.push_back(column->validity.empty() ? nullptr : column->validity.data());
    switch (column->type) {
        case ExportType::kString:
            buffers.push_back(column->offsets.data());
            buffers.push_back(column->data.data());
            break;
        case ExportType::kStringList:
            buffers.push_back(column->offsets.data());
            break;
        default:
            buffers.push_back(column->data.data());
            break;
    }
    owner->buffers.push_back(std::move(buffers));
    out->n_buffers = static_cast<int64_t>(owner->buffers.back().size());
    out->buffers = owner->buffers.back().data();
    if (column->child) {
        auto nested = std::make_unique<OwnedArray>();
        nested->children.resize(1);
        nested->buffers.reserve(1);
        FillArray(column->child.get(), nested.get(), &nested->children[0]);
        nested->child_pointers = {&nested->children[0]};
        out->n_children = 1;
        out->children = nested->child_pointers.data();
        owner->nested.push_back(std::move(nested));
    }
}

}  // namespace

BatchBuilder::BatchBuilder() = default;
BatchBuilder::~BatchBuilder() = default;
BatchBuilder::BatchBuilder(BatchBuilder&&) noexcept = default;
BatchBuilder& BatchBuilder::operator=(BatchBuilder&&) noexcept = default;

int BatchBuilder::AddColumn(std::string name, ExportType type) {
    Column column;
    column.name = std::move(name);
    column.type = type;
    if (type == ExportType::kString || type == ExportType::kStringList) {
        column.offsets.push_back(0);
    }
    if (type == ExportType::kStringList) {
        column.child = std::make_unique<Column>();
        column.child->name = "item";
        column.child->type = ExportType::kString;
        column.child->offsets.push_back(0);
    }
    columns_.push_back(std::move(column));
    return static_cast<int>(columns_.size() - 1);
}

int BatchBuilder::AddBorrowed(std::string name, BorrowedColumn column) {
    Column owned;
    owned.name = std::move(name);
    owned.type = column.type;
    owned.length = column.length;
    owned.borrowed = true;
    owned.borrow = column;
    columns_.push_back(std::move(owned));
    return static_cast<int>(columns_.size() - 1);
}

void BatchBuilder::KeepAlive(std::shared_ptr<void> owner) { owner_ = std::move(owner); }

size_t BatchBuilder::NumColumns() const { return columns_.size(); }

std::vector<std::string> BatchBuilder::ColumnNames() const {
    std::vector<std::string> names;
    for (const Column& column : columns_) names.push_back(column.name);
    return names;
}

void BatchBuilder::AppendString(int at, std::string_view value) {
    Column& column = columns_[at];
    column.data.insert(column.data.end(), value.begin(), value.end());
    column.offsets.push_back(static_cast<int32_t>(column.data.size()));
    column.SetValid(true);
}

void BatchBuilder::AppendBoolean(int at, bool value) {
    columns_[at].PutBit(value);
    columns_[at].SetValid(true);
}

void BatchBuilder::AppendUInt8(int at, uint8_t value) {
    columns_[at].PutFixed(value);
    columns_[at].SetValid(true);
}

void BatchBuilder::AppendUInt32(int at, uint32_t value) {
    columns_[at].PutFixed(value);
    columns_[at].SetValid(true);
}

void BatchBuilder::AppendUInt64(int at, uint64_t value) {
    columns_[at].PutFixed(value);
    columns_[at].SetValid(true);
}

void BatchBuilder::AppendDouble(int at, double value) {
    columns_[at].PutFixed(value);
    columns_[at].SetValid(true);
}

void BatchBuilder::AppendDate32(int at, int32_t days) {
    columns_[at].PutFixed(days);
    columns_[at].SetValid(true);
}

void BatchBuilder::AppendStringList(int at, const std::vector<std::string>& values) {
    Column& column = columns_[at];
    Column& items = *column.child;
    for (const std::string& value : values) {
        items.data.insert(items.data.end(), value.begin(), value.end());
        items.offsets.push_back(static_cast<int32_t>(items.data.size()));
        items.SetValid(true);
    }
    column.offsets.push_back(static_cast<int32_t>(items.length));
    column.SetValid(true);
}

void BatchBuilder::AppendNull(int at) {
    Column& column = columns_[at];
    switch (column.type) {
        case ExportType::kString:
            column.offsets.push_back(static_cast<int32_t>(column.data.size()));
            break;
        case ExportType::kStringList:
            column.offsets.push_back(static_cast<int32_t>(column.child->length));
            break;
        case ExportType::kBoolean:
            column.PutBit(false);
            break;
        case ExportType::kUInt8:
            column.PutFixed(uint8_t{0});
            break;
        case ExportType::kUInt32:
            column.PutFixed(uint32_t{0});
            break;
        case ExportType::kUInt64:
            column.PutFixed(uint64_t{0});
            break;
        case ExportType::kDouble:
            column.PutFixed(0.0);
            break;
        case ExportType::kDate32:
            column.PutFixed(int32_t{0});
            break;
        case ExportType::kDictionary:
        case ExportType::kLargeString:
            break;  // borrowed only: they take no appends
    }
    column.SetValid(false);
}

int64_t BatchBuilder::Rows() const { return columns_.empty() ? 0 : columns_[0].length; }

void BatchBuilder::ExportSchema(ArrowSchema* out) const {
    auto owner = std::make_unique<OwnedSchema>();
    owner->children.resize(columns_.size());
    for (size_t c = 0; c < columns_.size(); ++c) {
        FillSchema(columns_[c], owner.get(), &owner->children[c]);
        owner->child_pointers.push_back(&owner->children[c]);
    }
    out->format = "+s";
    out->name = "";
    out->metadata = nullptr;
    out->flags = 0;
    out->n_children = static_cast<int64_t>(columns_.size());
    out->children = owner->child_pointers.data();
    out->dictionary = nullptr;
    out->release = ReleaseSchema;
    out->private_data = owner.release();
}

bool BatchBuilder::ExportBatch(ArrowArray* out, std::string* error) {
    const int64_t rows = Rows();
    for (const Column& column : columns_) {
        if (column.length != rows) {
            *error = "column \"" + column.name + "\" has " +
                     std::to_string(column.length) + " rows where \"" + columns_[0].name +
                     "\" has " + std::to_string(rows);
            return false;
        }
    }
    auto owner = std::make_unique<OwnedArray>();
    owner->children.resize(columns_.size());
    owner->buffers.reserve(columns_.size() + 1);
    owner->keep_alive = owner_;
    for (size_t c = 0; c < columns_.size(); ++c) {
        if (columns_[c].borrowed) {
            // Borrowed columns stay where they are and are exported every time.
            FillBorrowed(columns_[c].borrow, owner.get(), &owner->children[c]);
            owner->child_pointers.push_back(&owner->children[c]);
            continue;
        }
        // The rows move out; the column keeps its name, type and an empty start.
        auto moved = std::make_unique<Column>();
        moved->name = columns_[c].name;
        moved->type = columns_[c].type;
        moved->length = columns_[c].length;
        moved->null_count = columns_[c].null_count;
        moved->validity = std::move(columns_[c].validity);
        moved->offsets = std::move(columns_[c].offsets);
        moved->data = std::move(columns_[c].data);
        moved->child = std::move(columns_[c].child);
        columns_[c].Clear();
        columns_[c].validity.clear();
        if (columns_[c].type == ExportType::kString ||
            columns_[c].type == ExportType::kStringList) {
            columns_[c].offsets.push_back(0);
        }
        if (columns_[c].type == ExportType::kStringList) {
            columns_[c].child = std::make_unique<Column>();
            columns_[c].child->name = "item";
            columns_[c].child->type = ExportType::kString;
            columns_[c].child->offsets.push_back(0);
        }
        FillArray(moved.get(), owner.get(), &owner->children[c]);
        owner->child_pointers.push_back(&owner->children[c]);
        owner->columns.push_back(std::move(moved));
    }
    out->length = rows;
    out->null_count = 0;
    out->offset = 0;
    out->n_buffers = 1;
    owner->buffers.push_back({nullptr});
    out->buffers = owner->buffers.back().data();
    out->n_children = static_cast<int64_t>(columns_.size());
    out->children = owner->child_pointers.data();
    out->dictionary = nullptr;
    out->release = ReleaseArray;
    out->private_data = owner.release();
    return true;
}

}  // namespace cpplink
