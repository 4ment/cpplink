// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/parquet_loader.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

namespace cpplink {
namespace {

std::string_view ViewOf(const arrow::StringArray& array, int64_t row) {
    const std::string_view value = array.GetView(row);
    return value;
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
    if (array.type_id() != arrow::Type::STRING) {
        *error = "expected a string column, found " + array.type()->ToString();
        return false;
    }
    const auto& typed = static_cast<const arrow::StringArray&>(array);
    for (int64_t i = 0; i < typed.length(); ++i) {
        column->ids.push_back(typed.IsNull(i) ? kNullId
                                              : column->dict.Intern(ViewOf(typed, i)));
    }
    return true;
}

bool AppendStringLists(const arrow::Array& array, StringListColumn* column,
                       std::string* error) {
    if (array.type_id() != arrow::Type::LIST) {
        *error = "expected a list column, found " + array.type()->ToString();
        return false;
    }
    const auto& typed = static_cast<const arrow::ListArray&>(array);
    const auto values = std::static_pointer_cast<arrow::StringArray>(typed.values());
    if (values->type_id() != arrow::Type::STRING) {
        *error =
            "expected a list of strings, found a list of " + values->type()->ToString();
        return false;
    }
    if (column->offsets.empty()) column->offsets.push_back(0);

    std::vector<uint32_t> row;
    for (int64_t i = 0; i < typed.length(); ++i) {
        row.clear();
        if (!typed.IsNull(i)) {
            const int64_t begin = typed.value_offset(i);
            const int64_t end = typed.value_offset(i + 1);
            for (int64_t j = begin; j < end; ++j) {
                if (values->IsNull(j)) continue;
                row.push_back(column->dict.Intern(values->GetView(j)));
            }
        }
        // Sorted and deduplicated here so overlap is a linear merge later.
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        column->ids.insert(column->ids.end(), row.begin(), row.end());
        column->offsets.push_back(column->ids.size());
    }
    return true;
}

bool AppendIds(const arrow::Array& array, IdColumn* column, std::string* error) {
    if (array.type_id() != arrow::Type::STRING) {
        *error = "expected a string unique_id column, found " + array.type()->ToString();
        return false;
    }
    const auto& typed = static_cast<const arrow::StringArray&>(array);
    for (int64_t i = 0; i < typed.length(); ++i) {
        column->Append(typed.IsNull(i) ? std::string_view() : typed.GetView(i));
    }
    return true;
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
        if (!AppendOneFile(path, schema, store, &rows, &groups, error)) return false;
        total += rows;
        row_groups += groups;
        per_file.push_back(rows);
        starts.push_back(total);
    }

    store->set_num_records(total);
    // One dataset is the absence of a boundary, not a boundary at each end: the
    // dedup path then pays no lookup at all.
    if (paths.size() > 1) store->set_datasets(std::move(starts));
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
