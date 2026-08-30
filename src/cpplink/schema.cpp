// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/schema.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
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

struct LevelName {
    const char* name;
    LevelType type;
};

constexpr LevelName kLevelNames[] = {
    {"null", LevelType::kNull},
    {"exact", LevelType::kExact},
    {"levenshtein", LevelType::kLevenshtein},
    {"jaro_winkler", LevelType::kJaroWinkler},
    {"date_within", LevelType::kDateWithin},
    {"numeric_within", LevelType::kNumericWithin},
    {"geo_within", LevelType::kGeoWithin},
    {"list_overlap", LevelType::kListOverlap},
    {"list_jaccard", LevelType::kListJaccard},
    {"else", LevelType::kElse},
};

// Thresholds are printed at the precision the level is written in, so an explain
// report reads back as the configuration that produced it.
std::string Number(double value, int decimals) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

struct KindName {
    const char* name;
    SourceKind kind;
};

constexpr KindName kKindNames[] = {
    {"exact_value", SourceKind::kExactValue},
    {"rare_value", SourceKind::kRareValue},
    {"minhash", SourceKind::kMinHash},
    {"sorted_neighbourhood", SourceKind::kSortedNeighbourhood},
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

const char* LevelTypeName(LevelType type) {
    for (const LevelName& entry : kLevelNames) {
        if (entry.type == type) return entry.name;
    }
    return "unknown";
}

bool ParseLevelType(const std::string& name, LevelType* type) {
    for (const LevelName& entry : kLevelNames) {
        if (name == entry.name) {
            *type = entry.type;
            return true;
        }
    }
    return false;
}

std::string LevelSpec::Describe() const {
    if (!label.empty()) return label;
    switch (type) {
        case LevelType::kNull:
            return "null";
        case LevelType::kExact:
            return "exact";
        case LevelType::kLevenshtein:
            return "levenshtein <= " + Number(threshold, 0);
        case LevelType::kJaroWinkler:
            return "jaro_winkler >= " + Number(threshold, 2);
        case LevelType::kDateWithin:
            return "within " + Number(threshold, 0) + " days";
        case LevelType::kNumericWithin:
            return "within " + Number(threshold, 3);
        case LevelType::kGeoWithin:
            return "within " + Number(threshold, 2) + " km";
        case LevelType::kListOverlap:
            return "overlap >= " + Number(threshold, 0);
        case LevelType::kListJaccard:
            return "jaccard >= " + Number(threshold, 2);
        case LevelType::kElse:
            return "else";
    }
    return "unknown";
}

const char* SourceKindName(SourceKind kind) {
    for (const KindName& entry : kKindNames) {
        if (entry.kind == kind) return entry.name;
    }
    return "unknown";
}

bool ParseSourceKind(const std::string& name, SourceKind* kind) {
    for (const KindName& entry : kKindNames) {
        if (name == entry.name) {
            *kind = entry.kind;
            return true;
        }
    }
    return false;
}

std::string BlockingSpec::Describe() const {
    switch (kind) {
        case SourceKind::kExactValue:
            return "exact value";
        case SourceKind::kRareValue:
            return "value seen <= " + std::to_string(max_frequency) + " times";
        case SourceKind::kMinHash:
            return "minhash " + std::to_string(bands) + "x" +
                   std::to_string(rows_per_band) + ", " + std::to_string(ngram) +
                   "-grams";
        case SourceKind::kSortedNeighbourhood:
            return "window " + std::to_string(window);
    }
    return "unknown";
}

uint8_t Schema::GammaWidth() const {
    uint8_t width = 0;
    for (const ComparisonSpec& comparison : comparisons) width += comparison.bits;
    return width;
}

const ColumnSpec* Schema::Find(const std::string& name) const {
    for (const ColumnSpec& spec : columns) {
        if (spec.name == name) return &spec;
    }
    return nullptr;
}

namespace {

// Which column types a level can be evaluated against. Checked at parse time so a
// mistyped configuration fails before a file is opened, not on the first pair.
bool LevelAcceptsColumn(LevelType level, ColumnType column) {
    switch (level) {
        case LevelType::kNull:
        case LevelType::kElse:
            return true;
        case LevelType::kExact:
            return column != ColumnType::kDouble;
        case LevelType::kLevenshtein:
        case LevelType::kJaroWinkler:
            return column == ColumnType::kString;
        case LevelType::kDateWithin:
            return column == ColumnType::kDate;
        case LevelType::kNumericWithin:
        case LevelType::kGeoWithin:
            return column == ColumnType::kDouble;
        case LevelType::kListOverlap:
        case LevelType::kListJaccard:
            return column == ColumnType::kStringList;
    }
    return false;
}

// How many columns a level reads. Everything is single-column except a coordinate
// pair, which is one comparison over two. Zero means the level is agnostic: "null"
// checks whichever columns the comparison names, and "else" reads none.
size_t LevelColumnCount(LevelType level) {
    switch (level) {
        case LevelType::kNull:
        case LevelType::kElse:
            return 0;
        case LevelType::kGeoWithin:
            return 2;
        default:
            return 1;
    }
}

bool LevelNeedsThreshold(LevelType level) {
    switch (level) {
        case LevelType::kNull:
        case LevelType::kExact:
        case LevelType::kElse:
            return false;
        default:
            return true;
    }
}

uint8_t BitsFor(size_t level_count) {
    uint8_t bits = 1;
    while ((1u << bits) < level_count) ++bits;
    return bits;
}

bool ParseComparisons(const nlohmann::json& root, Schema* schema, std::string* error) {
    if (!root.contains("comparisons")) return true;
    if (!root["comparisons"].is_array()) {
        *error = "\"comparisons\" must be an array";
        return false;
    }

    for (const nlohmann::json& item : root["comparisons"]) {
        ComparisonSpec comparison;
        if (!item.is_object() || !item.contains("columns") ||
            !item["columns"].is_array() || item["columns"].empty()) {
            *error = "every comparison needs a non-empty \"columns\" array";
            return false;
        }
        for (const nlohmann::json& column : item["columns"]) {
            if (!column.is_string()) {
                *error = "comparison column names must be strings";
                return false;
            }
            comparison.columns.push_back(column.get<std::string>());
        }
        comparison.name = item.contains("name") && item["name"].is_string()
                              ? item["name"].get<std::string>()
                              : comparison.columns.front();
        comparison.term_frequency = item.contains("term_frequency") &&
                                    item["term_frequency"].is_boolean() &&
                                    item["term_frequency"].get<bool>();

        // Every named column must exist, and a comparison spans one type only.
        ColumnType column_type = ColumnType::kString;
        for (size_t i = 0; i < comparison.columns.size(); ++i) {
            const ColumnSpec* spec = schema->Find(comparison.columns[i]);
            if (spec == nullptr) {
                *error = "comparison \"" + comparison.name + "\" names column \"" +
                         comparison.columns[i] + "\", which is not declared";
                return false;
            }
            if (i == 0) {
                column_type = spec->type;
            } else if (spec->type != column_type) {
                *error = "comparison \"" + comparison.name +
                         "\" mixes column types; all its columns must agree";
                return false;
            }
        }

        if (!item.contains("levels") || !item["levels"].is_array() ||
            item["levels"].empty()) {
            *error = "comparison \"" + comparison.name +
                     "\" needs a non-empty \"levels\" array";
            return false;
        }
        for (const nlohmann::json& entry : item["levels"]) {
            if (!entry.is_object() || !entry.contains("type") ||
                !entry["type"].is_string()) {
                *error =
                    "every level of \"" + comparison.name + "\" needs a string \"type\"";
                return false;
            }
            LevelSpec level;
            const std::string type_name = entry["type"].get<std::string>();
            if (!ParseLevelType(type_name, &level.type)) {
                *error = "comparison \"" + comparison.name +
                         "\" has unknown level type \"" + type_name + "\"";
                return false;
            }
            if (!LevelAcceptsColumn(level.type, column_type)) {
                *error = "comparison \"" + comparison.name + "\" applies level \"" +
                         type_name + "\" to a " + ColumnTypeName(column_type) +
                         " column, which it cannot read";
                return false;
            }
            const size_t needs = LevelColumnCount(level.type);
            if (needs != 0 && needs != comparison.columns.size()) {
                *error = "comparison \"" + comparison.name + "\" level \"" + type_name +
                         "\" reads " + std::to_string(needs) +
                         " column(s) but the comparison names " +
                         std::to_string(comparison.columns.size());
                return false;
            }
            if (LevelNeedsThreshold(level.type)) {
                if (!entry.contains("threshold") || !entry["threshold"].is_number()) {
                    *error = "comparison \"" + comparison.name + "\" level \"" +
                             type_name + "\" needs a numeric \"threshold\"";
                    return false;
                }
                level.threshold = entry["threshold"].get<double>();
            }
            if (entry.contains("label") && entry["label"].is_string()) {
                level.label = entry["label"].get<std::string>();
            }
            comparison.levels.push_back(level);
        }

        // A catch-all must exist and must be last, or a pair can match no level.
        const LevelType last = comparison.levels.back().type;
        if (last != LevelType::kElse) {
            *error = "comparison \"" + comparison.name +
                     "\" must end with an \"else\" level, so every pair lands "
                     "somewhere";
            return false;
        }
        for (size_t i = 0; i + 1 < comparison.levels.size(); ++i) {
            if (comparison.levels[i].type == LevelType::kElse) {
                *error = "comparison \"" + comparison.name +
                         "\" has an \"else\" level before the end; levels after it "
                         "can never fire";
                return false;
            }
        }

        comparison.bits = BitsFor(comparison.levels.size());
        schema->comparisons.push_back(comparison);
    }

    // Assign packed positions and check the whole pattern still fits a uint32.
    uint8_t shift = 0;
    for (ComparisonSpec& comparison : schema->comparisons) {
        comparison.shift = shift;
        shift = static_cast<uint8_t>(shift + comparison.bits);
    }
    if (shift > 32) {
        *error = "the comparisons need " + std::to_string(shift) +
                 " bits, but a packed agreement pattern is a uint32; use fewer "
                 "levels or fewer comparisons";
        return false;
    }
    return true;
}

}  // namespace

namespace {

uint32_t ReadUnsigned(const nlohmann::json& item, const char* key, uint32_t fallback) {
    if (!item.contains(key) || !item[key].is_number()) return fallback;
    const double value = item[key].get<double>();
    return value < 0 ? 0 : static_cast<uint32_t>(value);
}

bool ParseBlocking(const nlohmann::json& root, Schema* schema, std::string* error) {
    if (!root.contains("blocking")) return true;
    if (!root["blocking"].is_array()) {
        *error = "\"blocking\" must be an array";
        return false;
    }
    for (const nlohmann::json& item : root["blocking"]) {
        if (!item.is_object() || !item.contains("type") || !item["type"].is_string()) {
            *error = "every blocking source needs a string \"type\"";
            return false;
        }
        BlockingSpec spec;
        const std::string kind_name = item["type"].get<std::string>();
        if (!ParseSourceKind(kind_name, &spec.kind)) {
            *error = "unknown blocking type \"" + kind_name +
                     "\" (expected exact_value, rare_value, minhash or "
                     "sorted_neighbourhood)";
            return false;
        }
        if (!item.contains("column") || !item["column"].is_string()) {
            *error = "blocking source \"" + kind_name + "\" needs a \"column\"";
            return false;
        }
        spec.column = item["column"].get<std::string>();
        const ColumnSpec* column = schema->Find(spec.column);
        if (column == nullptr) {
            *error = "blocking source names column \"" + spec.column +
                     "\", which is not declared";
            return false;
        }
        // Blocking needs discrete agreement, which a double never provides.
        if (!HasTermFrequencies(column->type)) {
            *error = "blocking source names column \"" + spec.column + "\", a " +
                     ColumnTypeName(column->type) +
                     " column; blocking needs discrete agreement on a value";
            return false;
        }
        if (spec.kind == SourceKind::kMinHash && column->type != ColumnType::kString) {
            *error = "minhash blocking needs a string column, but \"" + spec.column +
                     "\" is " + ColumnTypeName(column->type);
            return false;
        }

        spec.max_frequency = ReadUnsigned(item, "max_frequency", spec.max_frequency);
        spec.window = ReadUnsigned(item, "window", spec.window);
        spec.bands = ReadUnsigned(item, "bands", spec.bands);
        spec.rows_per_band = ReadUnsigned(item, "rows_per_band", spec.rows_per_band);
        spec.ngram = ReadUnsigned(item, "ngram", spec.ngram);
        if (item.contains("seed") && item["seed"].is_number()) {
            spec.seed = static_cast<uint64_t>(item["seed"].get<double>());
        }
        if (spec.kind == SourceKind::kMinHash &&
            (spec.bands == 0 || spec.rows_per_band == 0 || spec.ngram == 0)) {
            *error = "minhash blocking on \"" + spec.column +
                     "\" needs non-zero bands, rows_per_band and ngram";
            return false;
        }
        if (spec.kind == SourceKind::kSortedNeighbourhood && spec.window == 0) {
            *error = "sorted_neighbourhood blocking on \"" + spec.column +
                     "\" needs a non-zero window";
            return false;
        }
        spec.name = item.contains("name") && item["name"].is_string()
                        ? item["name"].get<std::string>()
                        : spec.column + " " + SourceKindName(spec.kind);
        schema->blocking.push_back(spec);
    }
    return true;
}

}  // namespace

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

    if (!ParseComparisons(root, &parsed, error)) return false;
    if (!ParseBlocking(root, &parsed, error)) return false;

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
