// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

namespace cpplink {

// How a column is stored, which decides what can be asked of it later.
enum class ColumnType {
    kString,      // interned to a dense uint32 id
    kStringList,  // CSR of interned ids, sorted and deduplicated per row
    kDate,        // days since 1970-01-01
    kDouble,      // stored as-is; neither interned nor counted
};

const char* ColumnTypeName(ColumnType type);
bool ParseColumnType(const std::string& name, ColumnType* type);

// Term frequencies are kept only where exact agreement on a value is a discrete
// event worth counting. Two doubles agreeing to the last bit says nothing useful,
// so kDouble carries no counts and cannot drive rare-value blocking.
bool HasTermFrequencies(ColumnType type);

struct ColumnSpec {
    std::string name;
    ColumnType type = ColumnType::kString;
};

// The data description: what the columns are and how they are stored. How to
// compare them is a separate concern and is not read here.
struct Schema {
    std::string unique_id;  // optional; empty means the row index is the id
    std::vector<ColumnSpec> columns;

    const ColumnSpec* Find(const std::string& name) const;
};

bool ParseSchema(const std::string& json_text, Schema* schema, std::string* error);
bool LoadSchema(const std::string& path, Schema* schema, std::string* error);

}  // namespace cpplink
