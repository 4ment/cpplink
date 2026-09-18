// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/parquet_loader.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

namespace cpplink {
namespace {

// Calls fn on the array as its concrete integer type, or returns false when the
// column is not an integer of any width or sign.
template <typename Fn>
bool WithIntegerArray(const arrow::Array& array, Fn&& fn) {
    switch (array.type_id()) {
        case arrow::Type::INT8:
            fn(static_cast<const arrow::Int8Array&>(array));
            return true;
        case arrow::Type::INT16:
            fn(static_cast<const arrow::Int16Array&>(array));
            return true;
        case arrow::Type::INT32:
            fn(static_cast<const arrow::Int32Array&>(array));
            return true;
        case arrow::Type::INT64:
            fn(static_cast<const arrow::Int64Array&>(array));
            return true;
        case arrow::Type::UINT8:
            fn(static_cast<const arrow::UInt8Array&>(array));
            return true;
        case arrow::Type::UINT16:
            fn(static_cast<const arrow::UInt16Array&>(array));
            return true;
        case arrow::Type::UINT32:
            fn(static_cast<const arrow::UInt32Array&>(array));
            return true;
        case arrow::Type::UINT64:
            fn(static_cast<const arrow::UInt64Array&>(array));
            return true;
        default:
            return false;
    }
}

// Visits every value of a text-like column in order as `visit(value, is_null)`.
// A file written from Python rarely holds the types the schema names: pandas and
// polars write `large_string` for text, and a postcode, a year or an id that was
// numeric in the frame arrives as an integer. Both are the string the user meant,
// so an integer is visited as its decimal digits and interned like any other
// value, which is what makes "2000" in one file and 2000 in another the same id.
template <typename StringArray, typename Visit>
void VisitStrings(const arrow::Array& array, Visit& visit) {
    const auto& typed = static_cast<const StringArray&>(array);
    for (int64_t i = 0; i < typed.length(); ++i) {
        if (typed.IsNull(i)) {
            visit(std::string_view(), true);
        } else {
            visit(std::string_view(typed.GetView(i)), false);
        }
    }
}

template <typename Visit>
bool ForEachText(const arrow::Array& array, Visit&& visit, std::string* error) {
    if (array.type_id() == arrow::Type::STRING) {
        VisitStrings<arrow::StringArray>(array, visit);
        return true;
    }
    if (array.type_id() == arrow::Type::LARGE_STRING) {
        VisitStrings<arrow::LargeStringArray>(array, visit);
        return true;
    }
    const bool integer = WithIntegerArray(array, [&](const auto& typed) {
        char digits[24];  // -9223372036854775808 is 20 characters
        for (int64_t i = 0; i < typed.length(); ++i) {
            if (typed.IsNull(i)) {
                visit(std::string_view(), true);
                continue;
            }
            const auto result =
                std::to_chars(digits, digits + sizeof(digits), typed.Value(i));
            visit(std::string_view(digits, result.ptr - digits), false);
        }
    });
    if (integer) return true;
    *error = "expected a string or integer column, found " + array.type()->ToString();
    return false;
}

// Days since the Unix epoch, whatever the column's temporal unit.
bool AppendDates(const arrow::Array& array, DateColumn* column, std::string* error) {
    switch (array.type_id()) {
        case arrow::Type::DATE32: {
            const auto& typed = static_cast<const arrow::Date32Array&>(array);
            for (int64_t i = 0; i < typed.length(); ++i) {
                column->values.push_back(typed.IsNull(i) ? kNullDate : typed.Value(i));
            }
            return true;
        }
        case arrow::Type::DATE64: {
            const auto& typed = static_cast<const arrow::Date64Array&>(array);
            constexpr int64_t kMillisPerDay = 86400000;
            for (int64_t i = 0; i < typed.length(); ++i) {
                column->values.push_back(
                    typed.IsNull(i)
                        ? kNullDate
                        : static_cast<int32_t>(typed.Value(i) / kMillisPerDay));
            }
            return true;
        }
        case arrow::Type::TIMESTAMP: {
            const auto& typed = static_cast<const arrow::TimestampArray&>(array);
            const auto& type = static_cast<const arrow::TimestampType&>(*typed.type());
            int64_t per_day = 86400;
            switch (type.unit()) {
                case arrow::TimeUnit::SECOND:
                    per_day = 86400LL;
                    break;
                case arrow::TimeUnit::MILLI:
                    per_day = 86400000LL;
                    break;
                case arrow::TimeUnit::MICRO:
                    per_day = 86400000000LL;
                    break;
                case arrow::TimeUnit::NANO:
                    per_day = 86400000000000LL;
                    break;
            }
            for (int64_t i = 0; i < typed.length(); ++i) {
                column->values.push_back(
                    typed.IsNull(i) ? kNullDate
                                    : static_cast<int32_t>(typed.Value(i) / per_day));
            }
            return true;
        }
        default:
            *error =
                "expected a date or timestamp column, found " + array.type()->ToString();
            return false;
    }
}

bool AppendDoubles(const arrow::Array& array, DoubleColumn* column, std::string* error) {
    const double kMissing = std::numeric_limits<double>::quiet_NaN();
    if (array.type_id() == arrow::Type::DOUBLE) {
        const auto& typed = static_cast<const arrow::DoubleArray&>(array);
        for (int64_t i = 0; i < typed.length(); ++i) {
            column->values.push_back(typed.IsNull(i) ? kMissing : typed.Value(i));
        }
        return true;
    }
    if (array.type_id() == arrow::Type::FLOAT) {
        const auto& typed = static_cast<const arrow::FloatArray&>(array);
        for (int64_t i = 0; i < typed.length(); ++i) {
            column->values.push_back(typed.IsNull(i) ? kMissing : typed.Value(i));
        }
        return true;
    }
    *error = "expected a floating point column, found " + array.type()->ToString();
    return false;
}

bool AppendBooleans(const arrow::Array& array, BooleanColumn* column,
                    std::string* error) {
    if (array.type_id() == arrow::Type::BOOL) {
        const auto& typed = static_cast<const arrow::BooleanArray&>(array);
        for (int64_t i = 0; i < typed.length(); ++i) {
            column->values.push_back(typed.IsNull(i)  ? kNullBoolean
                                     : typed.Value(i) ? int8_t{1}
                                                      : int8_t{0});
        }
        return true;
    }
    *error = "expected a boolean column, found " + array.type()->ToString();
    return false;
}

bool AppendStrings(const arrow::Array& array, StringColumn* column, std::string* error) {
    return ForEachText(
        array,
        [&](std::string_view value, bool is_null) {
            column->ids.push_back(is_null ? kNullId : column->dict.Intern(value));
        },
        error);
}

// Slices the interned elements of a list chunk into one sorted, deduplicated cell
// per row, whether the offsets are 32-bit (`list`) or 64-bit (`large_list`).
template <typename ListArray>
void AppendListRows(const ListArray& lists, const std::vector<uint32_t>& elements,
                    StringListColumn* column) {
    std::vector<uint32_t> row;
    for (int64_t i = 0; i < lists.length(); ++i) {
        row.clear();
        if (!lists.IsNull(i)) {
            const int64_t begin = lists.value_offset(i);
            const int64_t end = lists.value_offset(i + 1);
            for (int64_t j = begin; j < end; ++j) {
                if (elements[j] != kNullId) row.push_back(elements[j]);
            }
        }
        // Sorted and deduplicated here so overlap is a linear merge later.
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        column->ids.insert(column->ids.end(), row.begin(), row.end());
        column->offsets.push_back(column->ids.size());
    }
}

bool AppendStringLists(const arrow::Array& array, StringListColumn* column,
                       std::string* error) {
    const bool large = array.type_id() == arrow::Type::LARGE_LIST;
    if (array.type_id() != arrow::Type::LIST && !large) {
        *error = "expected a list column, found " + array.type()->ToString();
        return false;
    }
    // The elements are interned first, as one flat run over the child array, so a
    // list of integers goes through the same visitor a scalar column does; the rows
    // are then cut out of that run by the list's own offsets.
    const std::shared_ptr<arrow::Array> values =
        large ? static_cast<const arrow::LargeListArray&>(array).values()
              : static_cast<const arrow::ListArray&>(array).values();
    std::vector<uint32_t> elements;
    elements.reserve(static_cast<size_t>(values->length()));
    std::string element_error;
    const bool ok = ForEachText(
        *values,
        [&](std::string_view value, bool is_null) {
            elements.push_back(is_null ? kNullId : column->dict.Intern(value));
        },
        &element_error);
    if (!ok) {
        *error = "expected a list of strings or integers, found a list of " +
                 values->type()->ToString();
        return false;
    }
    if (column->offsets.empty()) column->offsets.push_back(0);
    if (large) {
        AppendListRows(static_cast<const arrow::LargeListArray&>(array), elements,
                       column);
    } else {
        AppendListRows(static_cast<const arrow::ListArray&>(array), elements, column);
    }
    return true;
}

bool AppendIds(const arrow::Array& array, IdColumn* column, std::string* error) {
    std::string text_error;
    const bool ok = ForEachText(
        array, [&](std::string_view value, bool) { column->Append(value); }, &text_error);
    if (ok) return true;
    *error = "expected a string or integer unique_id column, found " +
             array.type()->ToString();
    return false;
}

bool AppendColumn(const arrow::ChunkedArray& chunked, ColumnType type, Column* column,
                  std::string* error) {
    for (const auto& chunk : chunked.chunks()) {
        bool ok = false;
        switch (type) {
            case ColumnType::kString:
                ok = AppendStrings(*chunk, &std::get<StringColumn>(*column), error);
                break;
            case ColumnType::kStringList:
                ok = AppendStringLists(*chunk, &std::get<StringListColumn>(*column),
                                       error);
                break;
            case ColumnType::kDate:
                ok = AppendDates(*chunk, &std::get<DateColumn>(*column), error);
                break;
            case ColumnType::kBoolean:
                ok = AppendBooleans(*chunk, &std::get<BooleanColumn>(*column), error);
                break;
            case ColumnType::kDouble:
                ok = AppendDoubles(*chunk, &std::get<DoubleColumn>(*column), error);
                break;
        }
        if (!ok) return false;
    }
    return true;
}

bool OpenSchema(const std::string& path, std::shared_ptr<arrow::Schema>* file_schema,
                std::string* error) {
    auto input_result = arrow::io::ReadableFile::Open(path);
    if (!input_result.ok()) {
        *error = "cannot open " + path + ": " + input_result.status().message();
        return false;
    }
    parquet::arrow::FileReaderBuilder builder;
    auto status = builder.Open(*input_result);
    if (!status.ok()) {
        *error = "cannot read parquet metadata: " + status.message();
        return false;
    }
    auto reader_result = builder.Build();
    if (!reader_result.ok()) {
        *error = "cannot open parquet reader: " + reader_result.status().message();
        return false;
    }
    status = (*reader_result)->GetSchema(file_schema);
    if (!status.ok()) {
        *error = "cannot read parquet schema: " + status.message();
        return false;
    }
    return true;
}

// The column type an Arrow type reads as, or false where none does. An integer
// is text: a postcode, a year or a phone number that was numeric in the frame is
// still the value it spells, and a writer who wanted a double wrote one.
bool ColumnTypeOf(const arrow::DataType& type, ColumnType* out) {
    switch (type.id()) {
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
            *out = ColumnType::kString;
            return true;
        case arrow::Type::LIST:
        case arrow::Type::LARGE_LIST:
            *out = ColumnType::kStringList;
            return true;
        case arrow::Type::DATE32:
        case arrow::Type::DATE64:
        case arrow::Type::TIMESTAMP:
            *out = ColumnType::kDate;
            return true;
        case arrow::Type::BOOL:
            *out = ColumnType::kBoolean;
            return true;
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            *out = ColumnType::kDouble;
            return true;
        default:
            return false;
    }
}

// Appends one file's rows to the store without finalizing it: several files may
// still be coming, and the dictionaries and term frequencies are only correct once
// every row that will ever be in the store has been read.
bool AppendOneFile(const std::string& path, const Schema& schema, RecordStore* store,
                   uint64_t* rows_read, int* row_groups_read, std::string* error) {
    auto input_result = arrow::io::ReadableFile::Open(path);
    if (!input_result.ok()) {
        *error = "cannot open " + path + ": " + input_result.status().message();
        return false;
    }

    parquet::arrow::FileReaderBuilder builder;
    auto status = builder.Open(*input_result);
    if (!status.ok()) {
        *error = "cannot read parquet metadata: " + status.message();
        return false;
    }
    auto reader_result = builder.Build();
    if (!reader_result.ok()) {
        *error = "cannot open parquet reader: " + reader_result.status().message();
        return false;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);

    std::shared_ptr<arrow::Schema> file_schema;
    status = reader->GetSchema(&file_schema);
    if (!status.ok()) {
        *error = "cannot read parquet schema: " + status.message();
        return false;
    }

    // Resolve every column up front so a typo fails before any work is done. A
    // derived column is not among them: it is computed from another column when
    // the store is finalized, and the file is not expected to hold it.
    constexpr size_t kIdTarget = static_cast<size_t>(-1);
    std::vector<int> indices;
    std::vector<std::string> names;
    std::vector<size_t> targets;  // the store column each name fills
    int id_index = -1;
    if (!schema.unique_id.empty()) {
        id_index = file_schema->GetFieldIndex(schema.unique_id);
        if (id_index < 0) {
            *error = "unique_id column \"" + schema.unique_id + "\" is not in the file";
            return false;
        }
        indices.push_back(id_index);
        names.push_back(schema.unique_id);
        targets.push_back(kIdTarget);
    }
    for (size_t i = 0; i < schema.columns.size(); ++i) {
        const ColumnSpec& spec = schema.columns[i];
        if (spec.IsDerived()) continue;
        const int index = file_schema->GetFieldIndex(spec.name);
        if (index < 0) {
            *error = "column \"" + spec.name + "\" is not in the file";
            return false;
        }
        indices.push_back(index);
        names.push_back(spec.name);
        targets.push_back(i);
    }

    const int row_groups = reader->num_row_groups();
    uint64_t rows = 0;
    for (int group = 0; group < row_groups; ++group) {
        std::shared_ptr<arrow::Table> table;
        status = reader->ReadRowGroup(group, indices, &table);
        if (!status.ok()) {
            *error = "cannot read row group " + std::to_string(group) + ": " +
                     status.message();
            return false;
        }
        for (size_t i = 0; i < names.size(); ++i) {
            const auto chunked = table->GetColumnByName(names[i]);
            if (chunked == nullptr) {
                *error = "column \"" + names[i] + "\" vanished from row group " +
                         std::to_string(group);
                return false;
            }
            if (targets[i] == kIdTarget) {
                for (const auto& chunk : chunked->chunks()) {
                    if (!AppendIds(*chunk, &store->mutable_ids(), error)) {
                        *error = schema.unique_id + ": " + *error;
                        return false;
                    }
                }
                continue;
            }
            const size_t column_index = targets[i];
            if (!AppendColumn(*chunked, schema.columns[column_index].type,
                              &store->mutable_column(column_index), error)) {
                *error = names[i] + ": " + *error;
                return false;
            }
        }
        rows += static_cast<uint64_t>(table->num_rows());
        // The row group's Arrow buffers are released here, before the next read.
    }

    *rows_read = rows;
    *row_groups_read = row_groups;
    return true;
}

}  // namespace

bool ReadFileColumns(const std::string& path, std::vector<FileColumn>* columns,
                     std::string* error) {
    std::shared_ptr<arrow::Schema> file_schema;
    if (!OpenSchema(path, &file_schema, error)) return false;
    columns->clear();
    for (const auto& field : file_schema->fields()) {
        FileColumn column;
        column.name = field->name();
        column.arrow_type = field->type()->ToString();
        column.readable = ColumnTypeOf(*field->type(), &column.type);
        columns->push_back(std::move(column));
    }
    return true;
}

bool ResolveColumnTypes(const std::vector<std::string>& paths, Schema* schema,
                        std::string* error) {
    if (paths.empty()) {
        *error = "no parquet file to read the column types from";
        return false;
    }
    std::shared_ptr<arrow::Schema> file_schema;
    if (!OpenSchema(paths.front(), &file_schema, error)) return false;
    for (ColumnSpec& spec : schema->columns) {
        if (spec.type_declared || spec.IsDerived()) continue;
        const int index = file_schema->GetFieldIndex(spec.name);
        if (index < 0) {
            *error = "column \"" + spec.name + "\" is not in " + paths.front();
            return false;
        }
        const auto& type = *file_schema->field(index)->type();
        if (!ColumnTypeOf(type, &spec.type)) {
            *error = "column \"" + spec.name + "\" is " + type.ToString() + " in " +
                     paths.front() + ", which cpplink cannot read; cast it to a string";
            return false;
        }
    }
    return CheckTypes(schema, error);
}

std::vector<std::string> DatasetNamesFor(const std::vector<std::string>& paths) {
    std::vector<std::string> names;
    if (paths.size() < 2) return names;
    names.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        std::string name = std::filesystem::path(paths[i]).stem().string();
        if (name.empty()) name = std::to_string(i);
        for (char& c : name) {
            if (c == ',' || c == ':') c = '_';
        }
        if (std::find(names.begin(), names.end(), name) != names.end()) {
            name += "#" + std::to_string(i);
        }
        names.push_back(std::move(name));
    }
    return names;
}

bool LoadParquetFiles(const std::vector<std::string>& paths, const Schema& schema,
                      RecordStore* store, LoadStats* stats, std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    if (paths.empty()) {
        *error = "no input file was given";
        return false;
    }

    // Boundaries are recorded as they are crossed, so a dataset is the row range
    // between two of them and nothing per row has to be stored.
    std::vector<uint64_t> starts;
    std::vector<uint64_t> per_file;
    starts.push_back(0);
    uint64_t total = 0;
    int row_groups = 0;
    for (const std::string& path : paths) {
        uint64_t rows = 0;
        int groups = 0;
        if (!AppendOneFile(path, schema, store, &rows, &groups, error)) {
            // With several inputs the message has to say which one failed; a
            // column missing from the second file reads the same as one missing
            // from the first otherwise.
            if (paths.size() > 1) *error = path + ": " + *error;
            return false;
        }
        total += rows;
        row_groups += groups;
        per_file.push_back(rows);
        starts.push_back(total);
    }

    store->set_num_records(total);
    // One dataset is the absence of a boundary, not a boundary at each end: the
    // dedup path then pays no lookup at all.
    if (paths.size() > 1) {
        store->set_datasets(std::move(starts));
        store->set_dataset_names(DatasetNamesFor(paths));
    }
    store->Finalize();

    if (stats != nullptr) {
        stats->rows = total;
        stats->row_groups = row_groups;
        stats->dataset_rows = std::move(per_file);
        stats->seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
    }
    return true;
}

bool LoadParquet(const std::string& path, const Schema& schema, RecordStore* store,
                 LoadStats* stats, std::string* error) {
    return LoadParquetFiles({path}, schema, store, stats, error);
}

}  // namespace cpplink
