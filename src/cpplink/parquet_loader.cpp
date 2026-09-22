// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/parquet_loader.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "cpplink/batch_loader.hpp"
#include "cpplink/parquet_io.hpp"

namespace cpplink {
namespace {

// The exported struct releases itself through the callback the producer put in
// it; this guard makes sure it does whichever way a function leaves.
struct SchemaGuard {
    ArrowSchema schema;
    SchemaGuard() { schema.release = nullptr; }
    ~SchemaGuard() {
        if (schema.release != nullptr) schema.release(&schema);
    }
};

// The columns a schema reads from a file: the id and every column that is not
// derived, so a file is read for what the store holds and nothing else.
std::vector<std::string> ColumnsToRead(const Schema& schema) {
    std::vector<std::string> names;
    if (!schema.unique_id.empty()) names.push_back(schema.unique_id);
    for (const ColumnSpec& spec : schema.columns) {
        if (!spec.IsDerived()) names.push_back(spec.name);
    }
    return names;
}

}  // namespace

bool ReadFileColumns(const std::string& path, std::vector<FileColumn>* columns,
                     std::string* error) {
    SchemaGuard file;
    if (!ReadParquetSchema(path, &file.schema, error)) return false;
    columns->clear();
    for (int64_t i = 0; i < file.schema.n_children; ++i) {
        const ArrowSchema& field = *file.schema.children[i];
        FileColumn column;
        column.name = field.name == nullptr ? "" : field.name;
        column.arrow_type = ArrowTypeName(field);
        column.readable = ColumnTypeOf(field, &column.type);
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
    SchemaGuard file;
    if (!ReadParquetSchema(paths.front(), &file.schema, error)) return false;
    return ResolveColumnTypesFrom(file.schema, paths.front(), schema, error);
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
    if (paths.empty()) {
        *error = "no input file was given";
        return false;
    }
    // Every file is opened before any is read, so a missing column in the last
    // input fails before the first is loaded. Opening reads the footer only.
    const std::vector<std::string> columns = ColumnsToRead(schema);
    std::vector<ArrowArrayStream> streams(paths.size());
    std::vector<InputStream> inputs;
    const std::vector<std::string> names = DatasetNamesFor(paths);
    for (size_t i = 0; i < paths.size(); ++i) {
        streams[i].release = nullptr;
        if (!OpenParquetStream(paths[i], columns, &streams[i], error)) {
            for (size_t j = 0; j < i; ++j) streams[j].release(&streams[j]);
            if (paths.size() > 1) *error = paths[i] + ": " + *error;
            return false;
        }
        inputs.push_back({names.empty() ? std::string() : names[i], &streams[i]});
    }
    if (LoadStreams(inputs, schema, store, stats, error)) return true;
    // `LoadStreams` names the input by its dataset name; the file is what the
    // user typed, so say that where there is more than one.
    if (paths.size() > 1) {
        for (size_t i = 0; i < paths.size(); ++i) {
            const std::string prefix = names[i] + ": ";
            if (error->compare(0, prefix.size(), prefix) == 0) {
                *error = paths[i] + ": " + error->substr(prefix.size());
                break;
            }
        }
    }
    return false;
}

bool LoadParquet(const std::string& path, const Schema& schema, RecordStore* store,
                 LoadStats* stats, std::string* error) {
    return LoadParquetFiles({path}, schema, store, stats, error);
}

}  // namespace cpplink
