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
#include <vector>

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
    {"list_contains", LevelType::kListContains},
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

struct TransformName_ {
    const char* name;
    Transform transform;
};

constexpr TransformName_ kTransformNames[] = {
    {"normalize", Transform::kNormalize},  {"sorted_tokens", Transform::kSortedTokens},
    {"soundex", Transform::kSoundex},      {"year", Transform::kYear},
    {"month", Transform::kMonth},          {"day", Transform::kDay},
    {"year_month", Transform::kYearMonth},
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

const char* TransformName(Transform transform) {
    for (const TransformName_& entry : kTransformNames) {
        if (entry.transform == transform) return entry.name;
    }
    return "unknown";
}

bool ParseTransform(const std::string& name, Transform* transform) {
    for (const TransformName_& entry : kTransformNames) {
        if (name == entry.name) {
            *transform = entry.transform;
            return true;
        }
    }
    return false;
}

ColumnType TransformInput(Transform transform) {
    switch (transform) {
        case Transform::kNormalize:
        case Transform::kSortedTokens:
        case Transform::kSoundex:
            return ColumnType::kString;
        case Transform::kYear:
        case Transform::kMonth:
        case Transform::kDay:
        case Transform::kYearMonth:
            return ColumnType::kDate;
    }
    return ColumnType::kString;
}

// Every transform so far produces a string, which is the point of them: an exact
// level on an interned key is an integer equality. The function exists so that a
// transform producing anything else type-checks rather than being assumed.
ColumnType TransformOutput(Transform transform) {
    switch (transform) {
        case Transform::kNormalize:
        case Transform::kSortedTokens:
        case Transform::kSoundex:
        case Transform::kYear:
        case Transform::kMonth:
        case Transform::kDay:
        case Transform::kYearMonth:
            return ColumnType::kString;
    }
    return ColumnType::kString;
}

std::string DeriveSpec::Describe() const {
    if (transforms.empty()) return std::string();
    std::string text = from;
    for (const Transform transform : transforms) {
        text += " -> ";
        text += TransformName(transform);
    }
    return text;
}

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
        case LevelType::kListContains:
            return "value in list";
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

bool Schema::IsDerived(const std::string& name) const {
    const ColumnSpec* spec = Find(name);
    return spec != nullptr && spec->IsDerived();
}

bool SameSource(const Schema& schema, const std::string& a, const std::string& b) {
    if (a == b) return true;
    const ColumnSpec* left = schema.Find(a);
    const ColumnSpec* right = schema.Find(b);
    if (left == nullptr || right == nullptr) return false;
    // A derivation is one level deep by construction, so the whole relation is
    // these three cases and no walk up a chain is needed.
    if (left->IsDerived() && left->derive.from == b) return true;
    if (right->IsDerived() && right->derive.from == a) return true;
    return left->IsDerived() && right->IsDerived() &&
           left->derive.from == right->derive.from;
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
        case LevelType::kListContains:
            // Answered by LevelAcceptsColumns, which sees both columns at once:
            // this level is the one whose two columns have different types, so a
            // type at a time cannot decide it.
            return false;
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
        case LevelType::kListContains:
            return 2;
        default:
            return 1;
    }
}

// A comparison ordinarily spans one column type. This is the one shape that does
// not: a scalar string beside a list of the aliases it may be known by, which is
// what list_contains reads. Order is part of it -- the scalar first -- because
// "is this name one of those nicknames" is not the question with the columns the
// other way round.
bool IsScalarAndList(const std::vector<ColumnType>& types) {
    return types.size() == 2 && types[0] == ColumnType::kString &&
           types[1] == ColumnType::kStringList;
}

// Whether a level can read the columns the comparison names, in order.
//
// Almost every level wants one type and every column of it. The exception is the
// scalar-and-list shape, where a one-column level reads whichever of the two
// columns has the type it understands: exact and the fuzzy string levels the
// scalar one, list_overlap and list_jaccard the list one. That is deliberate and
// is the reason to want the shape at all. Levels are ordered evidence, so the
// alias bridge belongs in the same comparison as the name levels it ranks
// against -- below an exact match and above a fuzzy one -- not in a second
// comparison whose agreements would then be counted as independent of the
// first's, which they are not.
bool LevelAcceptsColumns(LevelType level, const std::vector<ColumnType>& types) {
    if (level == LevelType::kListContains) return IsScalarAndList(types);
    if (IsScalarAndList(types)) {
        return LevelColumnCount(level) <= 1 && (LevelAcceptsColumn(level, types[0]) ||
                                                LevelAcceptsColumn(level, types[1]));
    }
    for (const ColumnType type : types) {
        if (!LevelAcceptsColumn(level, type)) return false;
    }
    return true;
}

// Checked before the levels are, so a comparison over columns no level could ever
// read together is reported as that rather than as whichever level first tripped
// over it.
bool TypesCompatible(const std::vector<ColumnType>& types) {
    if (IsScalarAndList(types)) return true;
    for (const ColumnType type : types) {
        if (type != types.front()) return false;
    }
    return true;
}

bool LevelNeedsThreshold(LevelType level) {
    switch (level) {
        case LevelType::kNull:
        case LevelType::kExact:
        case LevelType::kListContains:
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

        // Every named column must exist, and the types it spans must be a shape
        // some level can read.
        std::vector<ColumnType> column_types;
        for (const std::string& name : comparison.columns) {
            const ColumnSpec* spec = schema->Find(name);
            if (spec == nullptr) {
                *error = "comparison \"" + comparison.name + "\" names column \"" + name +
                         "\", which is not declared";
                return false;
            }
            column_types.push_back(spec->type);
        }
        if (!TypesCompatible(column_types)) {
            *error = "comparison \"" + comparison.name +
                     "\" mixes column types; all its columns must agree, unless it "
                     "is a string against a string_list, which is what "
                     "\"list_contains\" reads";
            return false;
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
            if (!LevelAcceptsColumns(level.type, column_types)) {
                std::string types;
                for (const ColumnType type : column_types) {
                    if (!types.empty()) types += ", ";
                    types += ColumnTypeName(type);
                }
                *error = "comparison \"" + comparison.name + "\" applies level \"" +
                         type_name + "\" to a " + types + " column, which it cannot read";
                return false;
            }
            // A one-column level in the scalar-and-list shape reads one of the
            // two, which is the one place a level's arity may be under the
            // comparison's without that being a mistake.
            const size_t needs = LevelColumnCount(level.type);
            const bool reads_one_of_two = needs == 1 && IsScalarAndList(column_types);
            if (needs != 0 && needs != comparison.columns.size() && !reads_one_of_two) {
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

std::string KnownTransforms() {
    std::string text;
    for (const TransformName_& entry : kTransformNames) {
        if (!text.empty()) text += ", ";
        text += entry.name;
    }
    return text;
}

// "derive": {"from": "surname", "transform": "soundex"}, or an array of transform
// names applied in order. A chain is how "the sorted tokens of a normalised name"
// is written, which is why one column may not derive from another: composition
// belongs in the list, where it type-checks in one place and the load stays a
// single pass over the source dictionary.
bool ParseDerive(const nlohmann::json& item, const std::string& column,
                 DeriveSpec* derive, std::string* error) {
    if (!item.is_object() || !item.contains("from") || !item["from"].is_string()) {
        *error = "column \"" + column + "\" has a \"derive\" without a string \"from\"";
        return false;
    }
    derive->from = item["from"].get<std::string>();
    if (!item.contains("transform")) {
        *error = "column \"" + column + "\" derives from \"" + derive->from +
                 "\" without a \"transform\"";
        return false;
    }
    std::vector<std::string> names;
    if (item["transform"].is_string()) {
        names.push_back(item["transform"].get<std::string>());
    } else if (item["transform"].is_array()) {
        for (const nlohmann::json& entry : item["transform"]) {
            if (!entry.is_string()) {
                *error = "column \"" + column + "\" has a transform that is not a name";
                return false;
            }
            names.push_back(entry.get<std::string>());
        }
    } else {
        *error = "column \"" + column +
                 "\" needs \"transform\" to be a name or an array of names";
        return false;
    }
    if (names.empty()) {
        *error = "column \"" + column + "\" names no transform to derive it";
        return false;
    }
    for (const std::string& name : names) {
        Transform transform = Transform::kNormalize;
        if (!ParseTransform(name, &transform)) {
            *error = "column \"" + column + "\" has unknown transform \"" + name +
                     "\" (expected " + KnownTransforms() + ")";
            return false;
        }
        derive->transforms.push_back(transform);
    }
    return true;
}

// Resolved once every column is parsed, so "from" may name a column declared
// later, and checked here rather than at load: a chain that cannot type-check is a
// configuration error and should fail before a file is opened.
bool ResolveDerived(Schema* schema, const std::vector<bool>& type_declared,
                    std::string* error) {
    for (size_t i = 0; i < schema->columns.size(); ++i) {
        ColumnSpec& spec = schema->columns[i];
        if (!spec.IsDerived()) continue;
        if (spec.derive.from == spec.name) {
            *error = "column \"" + spec.name + "\" derives from itself";
            return false;
        }
        const ColumnSpec* source = schema->Find(spec.derive.from);
        if (source == nullptr) {
            *error = "column \"" + spec.name + "\" derives from \"" + spec.derive.from +
                     "\", which is not declared";
            return false;
        }
        if (source->IsDerived()) {
            *error = "column \"" + spec.name + "\" derives from \"" + spec.derive.from +
                     "\", which is itself derived; derive from a column the file "
                     "holds and chain the transforms instead";
            return false;
        }
        ColumnType current = source->type;
        for (const Transform transform : spec.derive.transforms) {
            if (TransformInput(transform) != current) {
                *error = "column \"" + spec.name + "\" applies transform \"" +
                         TransformName(transform) + "\" to a " + ColumnTypeName(current) +
                         " value, which it cannot read";
                return false;
            }
            current = TransformOutput(transform);
        }
        if (type_declared[i] && spec.type != current) {
            *error = "column \"" + spec.name + "\" is declared " +
                     ColumnTypeName(spec.type) + ", but its transforms produce a " +
                     ColumnTypeName(current) + " value";
            return false;
        }
        spec.type = current;
    }
    return true;
}

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
    std::vector<bool> type_declared;
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
        const bool declared = item.contains("type") && item["type"].is_string();
        const std::string type_name =
            declared ? item["type"].get<std::string>() : std::string("string");
        if (!ParseColumnType(type_name, &spec.type)) {
            *error = "column \"" + spec.name + "\" has unknown type \"" + type_name +
                     "\" (expected string, string_list, date or double)";
            return false;
        }
        if (item.contains("derive") &&
            !ParseDerive(item["derive"], spec.name, &spec.derive, error)) {
            return false;
        }
        type_declared.push_back(declared);
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

    // Before the comparisons, which check the types a derived column only has once
    // its transforms have been resolved.
    if (!ResolveDerived(&parsed, type_declared, error)) return false;
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
