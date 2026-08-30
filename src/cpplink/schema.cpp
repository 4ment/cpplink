// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/schema.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace cpplink {
namespace {

struct TypeName {
    const char* name;
    ColumnType type;
};

constexpr TypeName kTypeNames[] = {
    {"string", ColumnType::kString},
    {"string_list", ColumnType::kStringList},
    {"date", ColumnType::kDate},
    {"double", ColumnType::kDouble},
};

}  // namespace

const char* ColumnTypeName(ColumnType type) {
    for (const TypeName& entry : kTypeNames) {
        if (entry.type == type) return entry.name;
    }
    return "unknown";
}

bool ParseColumnType(const std::string& name, ColumnType* type) {
    for (const TypeName& entry : kTypeNames) {
        if (name == entry.name) {
            *type = entry.type;
            return true;
        }
    }
    return false;
}

bool HasTermFrequencies(ColumnType type) { return type != ColumnType::kDouble; }

const ColumnSpec* Schema::Find(const std::string& name) const {
    for (const ColumnSpec& spec : columns) {
        if (spec.name == name) return &spec;
    }
    return nullptr;
}

bool ParseSchema(const std::string& json_text, Schema* schema, std::string* error) {
    nlohmann::json root = nlohmann::json::parse(json_text, nullptr, false);
    if (root.is_discarded()) {
        *error = "schema is not valid JSON";
        return false;
    }
    if (!root.is_object()) {
        *error = "schema must be a JSON object";
        return false;
    }
    if (!root.contains("columns") || !root["columns"].is_array()) {
        *error = "schema needs a \"columns\" array";
        return false;
    }

    Schema parsed;
    std::unordered_set<std::string> seen;
    for (const nlohmann::json& item : root["columns"]) {
        if (!item.is_object() || !item.contains("name") || !item["name"].is_string()) {
            *error = "every column needs a string \"name\"";
            return false;
        }
        ColumnSpec spec;
        spec.name = item["name"].get<std::string>();
        if (!seen.insert(spec.name).second) {
            *error = "column \"" + spec.name + "\" is declared twice";
            return false;
        }
        const std::string type_name = item.contains("type") && item["type"].is_string()
                                          ? item["type"].get<std::string>()
                                          : std::string("string");
        if (!ParseColumnType(type_name, &spec.type)) {
            *error = "column \"" + spec.name + "\" has unknown type \"" + type_name +
                     "\" (expected string, string_list, date or double)";
            return false;
        }
        parsed.columns.push_back(spec);
    }
    if (parsed.columns.empty()) {
        *error = "schema declares no columns";
        return false;
    }

    if (root.contains("unique_id")) {
        if (!root["unique_id"].is_string()) {
            *error = "\"unique_id\" must be a column name";
            return false;
        }
        parsed.unique_id = root["unique_id"].get<std::string>();
        if (parsed.Find(parsed.unique_id) != nullptr) {
            *error = "\"unique_id\" column \"" + parsed.unique_id +
                     "\" must not also be declared in \"columns\"";
            return false;
        }
    }

    *schema = std::move(parsed);
    return true;
}

bool LoadSchema(const std::string& path, Schema* schema, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        *error = "cannot open schema file: " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return ParseSchema(buffer.str(), schema, error);
}

}  // namespace cpplink
