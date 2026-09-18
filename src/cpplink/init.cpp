// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/init.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace cpplink {
namespace {

struct RoleEntry {
    Role role;
    const char* name;
};

constexpr RoleEntry kRoleNames[] = {
    {Role::kId, "id"},
    {Role::kFirstName, "first_name"},
    {Role::kMiddleName, "middle_name"},
    {Role::kSurname, "surname"},
    {Role::kFullName, "full_name"},
    {Role::kGender, "gender"},
    {Role::kBirthDate, "birth_date"},
    {Role::kDate, "date"},
    {Role::kEmail, "email"},
    {Role::kPhone, "phone"},
    {Role::kPostcode, "postcode"},
    {Role::kAddress, "address"},
    {Role::kCity, "city"},
    {Role::kRegion, "region"},
    {Role::kCountry, "country"},
    {Role::kNationalId, "national_id"},
    {Role::kLatitude, "latitude"},
    {Role::kLongitude, "longitude"},
    {Role::kList, "list"},
    {Role::kBoolean, "boolean"},
    {Role::kNumber, "number"},
    {Role::kText, "text"},
    {Role::kUnknown, "unknown"},
};

// Whole-name synonyms, matched after normalisation. A name in this table is
// taken as its role outright; the substring rules below only see the rest.
struct Synonym {
    const char* name;
    Role role;
};

constexpr Synonym kSynonyms[] = {
    {"id", Role::kId},
    {"unique_id", Role::kId},
    {"uid", Role::kId},
    {"record_id", Role::kId},
    {"rec_id", Role::kId},
    {"row_id", Role::kId},
    {"rowid", Role::kId},
    {"recid", Role::kId},
    {"index", Role::kId},
    {"row_number", Role::kId},
    {"rownum", Role::kId},

    {"first_name", Role::kFirstName},
    {"firstname", Role::kFirstName},
    {"first_names", Role::kFirstName},
    {"fname", Role::kFirstName},
    {"first", Role::kFirstName},
    {"given_name", Role::kFirstName},
    {"givenname", Role::kFirstName},
    {"given_names", Role::kFirstName},
    {"given", Role::kFirstName},
    {"forename", Role::kFirstName},
    {"forenames", Role::kFirstName},
    {"name_first", Role::kFirstName},
    {"prenom", Role::kFirstName},

    {"middle_name", Role::kMiddleName},
    {"middlename", Role::kMiddleName},
    {"middle", Role::kMiddleName},
    {"middle_initial", Role::kMiddleName},
    {"mname", Role::kMiddleName},
    {"second_name", Role::kMiddleName},

    {"surname", Role::kSurname},
    {"surnames", Role::kSurname},
    {"last_name", Role::kSurname},
    {"lastname", Role::kSurname},
    {"lname", Role::kSurname},
    {"last", Role::kSurname},
    {"family_name", Role::kSurname},
    {"familyname", Role::kSurname},
    {"name_last", Role::kSurname},
    {"nom", Role::kSurname},

    {"name", Role::kFullName},
    {"full_name", Role::kFullName},
    {"fullname", Role::kFullName},
    {"person_name", Role::kFullName},
    {"display_name", Role::kFullName},
    {"first_and_surname", Role::kFullName},
    {"forename_surname", Role::kFullName},

    {"gender", Role::kGender},
    {"sex", Role::kGender},

    {"dob", Role::kBirthDate},
    {"date_of_birth", Role::kBirthDate},
    {"birth_date", Role::kBirthDate},
    {"birthdate", Role::kBirthDate},
    {"birthday", Role::kBirthDate},
    {"born", Role::kBirthDate},

    {"date", Role::kDate},

    {"email", Role::kEmail},
    {"e_mail", Role::kEmail},
    {"mail", Role::kEmail},
    {"email_address", Role::kEmail},
    {"emailaddress", Role::kEmail},

    {"phone", Role::kPhone},
    {"telephone", Role::kPhone},
    {"phone_number", Role::kPhone},
    {"phonenumber", Role::kPhone},
    {"tel", Role::kPhone},
    {"mobile", Role::kPhone},
    {"mobile_number", Role::kPhone},
    {"cell", Role::kPhone},
    {"cellphone", Role::kPhone},
    {"cell_phone", Role::kPhone},
    {"fax", Role::kPhone},

    {"postcode", Role::kPostcode},
    {"post_code", Role::kPostcode},
    {"postal_code", Role::kPostcode},
    {"postalcode", Role::kPostcode},
    {"zip", Role::kPostcode},
    {"zipcode", Role::kPostcode},
    {"zip_code", Role::kPostcode},
    {"pincode", Role::kPostcode},
    {"pin_code", Role::kPostcode},

    {"address", Role::kAddress},
    {"addr", Role::kAddress},
    {"street", Role::kAddress},
    {"street_address", Role::kAddress},
    {"street_name", Role::kAddress},
    {"address_line", Role::kAddress},
    {"address_1", Role::kAddress},
    {"address1", Role::kAddress},
    {"address_tokens", Role::kAddress},

    {"city", Role::kCity},
    {"town", Role::kCity},
    {"suburb", Role::kCity},
    {"locality", Role::kCity},
    {"municipality", Role::kCity},

    {"state", Role::kRegion},
    {"county", Role::kRegion},
    {"province", Role::kRegion},
    {"region", Role::kRegion},
    {"territory", Role::kRegion},

    {"country", Role::kCountry},
    {"nation", Role::kCountry},
    {"country_code", Role::kCountry},

    {"ssn", Role::kNationalId},
    {"soc_sec_id", Role::kNationalId},
    {"social_security_number", Role::kNationalId},
    {"nino", Role::kNationalId},
    {"ni_number", Role::kNationalId},
    {"national_id", Role::kNationalId},
    {"national_insurance_number", Role::kNationalId},
    {"nhs_number", Role::kNationalId},
    {"passport", Role::kNationalId},
    {"passport_number", Role::kNationalId},
    {"tax_id", Role::kNationalId},
    {"tin", Role::kNationalId},
    {"sin", Role::kNationalId},
    {"driver_licence", Role::kNationalId},
    {"drivers_license", Role::kNationalId},
    {"licence_number", Role::kNationalId},
    {"license_number", Role::kNationalId},

    {"lat", Role::kLatitude},
    {"latitude", Role::kLatitude},
    {"lon", Role::kLongitude},
    {"lng", Role::kLongitude},
    {"long", Role::kLongitude},
    {"longitude", Role::kLongitude},
};

// Substrings unambiguous enough to name a role on their own, tried in order on
// a name the synonym table did not settle. "name" comes last and only after the
// qualified forms, so `first_name_1` is a first name and `company_name` a name.
struct Fragment {
    const char* text;
    Role role;
};

constexpr Fragment kFragments[] = {
    {"email", Role::kEmail},         {"e_mail", Role::kEmail},
    {"phone", Role::kPhone},         {"telephone", Role::kPhone},
    {"mobile", Role::kPhone},        {"postcode", Role::kPostcode},
    {"postal", Role::kPostcode},     {"zip", Role::kPostcode},
    {"address", Role::kAddress},     {"street", Role::kAddress},
    {"birth", Role::kBirthDate},     {"dob", Role::kBirthDate},
    {"gender", Role::kGender},       {"ssn", Role::kNationalId},
    {"passport", Role::kNationalId}, {"nino", Role::kNationalId},
    {"licen", Role::kNationalId},    {"latitude", Role::kLatitude},
    {"longitude", Role::kLongitude}, {"country", Role::kCountry},
    {"city", Role::kCity},           {"county", Role::kRegion},
    {"state", Role::kRegion},        {"province", Role::kRegion},
    {"middle", Role::kMiddleName},   {"first", Role::kFirstName},
    {"given", Role::kFirstName},     {"forename", Role::kFirstName},
    {"surname", Role::kSurname},     {"last", Role::kSurname},
    {"family", Role::kSurname},      {"date", Role::kDate},
    {"name", Role::kFullName},
};

// Lowercased, every run of non-alphanumerics one underscore, none at the ends:
// "Date Of Birth", "date-of-birth" and "DateOfBirth" are three spellings of one
// name, and the third is split at its capitals so it reaches the same string.
std::string NormalizeName(const std::string& name) {
    std::string out;
    bool pending = false;
    for (size_t i = 0; i < name.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        if (std::isalnum(c)) {
            const bool camel = std::isupper(c) && i > 0 &&
                               std::islower(static_cast<unsigned char>(name[i - 1]));
            if ((pending || camel) && !out.empty()) out.push_back('_');
            out.push_back(static_cast<char>(std::tolower(c)));
            pending = false;
        } else {
            pending = true;
        }
    }
    return out;
}

// The name's own reading, before the type has its say.
Role RoleFromName(const std::string& normalized) {
    for (const Synonym& synonym : kSynonyms) {
        if (normalized == synonym.name) return synonym.role;
    }
    for (const Fragment& fragment : kFragments) {
        if (normalized.find(fragment.text) != std::string::npos) return fragment.role;
    }
    return Role::kText;
}

// Where the draft puts a role's comparison: identifiers first, then names, then
// the rest. The order is what a reader expects and what is kept when the packed
// pattern runs out of bits.
int Priority(Role role) {
    switch (role) {
        case Role::kEmail:
            return 0;
        case Role::kNationalId:
            return 1;
        case Role::kPhone:
            return 2;
        case Role::kBirthDate:
            return 3;
        case Role::kSurname:
            return 4;
        case Role::kFirstName:
            return 5;
        case Role::kFullName:
            return 6;
        case Role::kMiddleName:
            return 7;
        case Role::kPostcode:
            return 8;
        case Role::kAddress:
            return 9;
        case Role::kLatitude:
        case Role::kLongitude:
            return 10;
        case Role::kCity:
            return 11;
        case Role::kRegion:
            return 12;
        case Role::kCountry:
            return 13;
        case Role::kGender:
            return 14;
        case Role::kDate:
            return 15;
        case Role::kList:
            return 16;
        case Role::kBoolean:
            return 17;
        case Role::kText:
            return 18;
        default:
            return 99;
    }
}

LevelSpec Level(LevelType type, double threshold = 0.0, const char* label = "") {
    LevelSpec level;
    level.type = type;
    level.threshold = threshold;
    level.label = label;
    return level;
}

LevelSpec LevelOn(LevelType type, uint8_t column, double threshold = 0.0) {
    LevelSpec level = Level(type, threshold);
    level.column = column;
    level.names_column = true;
    return level;
}

// The default ladder for a role over the type the column actually has. A role
// says what the field is; the type says which levels can read it, and where the
// two disagree (a date of birth the file holds as text) the type wins and the
// role only picks among the levels the type allows.
std::vector<LevelSpec> LadderFor(Role role, ColumnType type) {
    using L = LevelType;
    std::vector<LevelSpec> levels;
    levels.push_back(Level(L::kNull));
    switch (type) {
        case ColumnType::kString:
            switch (role) {
                case Role::kFirstName:
                    levels.push_back(Level(L::kExact));
                    levels.push_back(Level(L::kLevenshtein, 1));
                    levels.push_back(Level(L::kJaroWinkler, 0.88));
                    break;
                case Role::kSurname:
                case Role::kFullName:
                    levels.push_back(Level(L::kExact));
                    levels.push_back(Level(L::kJaroWinkler, 0.92));
                    levels.push_back(Level(L::kJaroWinkler, 0.85));
                    break;
                case Role::kMiddleName:
                case Role::kCity:
                case Role::kRegion:
                case Role::kCountry:
                    levels.push_back(Level(L::kExact));
                    levels.push_back(Level(L::kJaroWinkler, 0.90));
                    break;
                case Role::kAddress:
                    levels.push_back(Level(L::kExact));
                    levels.push_back(Level(L::kJaroWinkler, 0.90));
                    levels.push_back(Level(L::kJaroWinkler, 0.80));
                    break;
                case Role::kGender:
                    levels.push_back(Level(L::kExact));
                    break;
                case Role::kPhone:
                    levels.push_back(Level(L::kExact));
                    levels.push_back(Level(L::kLevenshtein, 2));
                    break;
                case Role::kPostcode:
                case Role::kNationalId:
                case Role::kBirthDate:
                case Role::kDate:
                    levels.push_back(Level(L::kExact));
                    levels.push_back(Level(L::kLevenshtein, 1));
                    break;
                default:
                    levels.push_back(Level(L::kExact));
                    levels.push_back(Level(L::kJaroWinkler, 0.88));
                    break;
            }
            break;
        case ColumnType::kDate:
            levels.push_back(Level(L::kExact));
            if (role == Role::kBirthDate) {
                levels.push_back(Level(L::kDateWithin, 2));
                levels.push_back(Level(L::kDateWithin, 370, "within a year"));
            } else {
                levels.push_back(Level(L::kDateWithin, 7));
            }
            break;
        case ColumnType::kStringList:
            levels.push_back(Level(L::kExact));
            levels.push_back(Level(L::kListJaccard, 0.6));
            levels.push_back(Level(L::kListOverlap, 2));
            break;
        case ColumnType::kBoolean:
            levels.push_back(Level(L::kExact));
            break;
        case ColumnType::kDouble:
            break;
    }
    levels.push_back(Level(L::kElse));
    return levels;
}

uint8_t BitsFor(size_t level_count) {
    uint8_t bits = 1;
    while ((1u << bits) < level_count) ++bits;
    return bits;
}

BlockingSpec Source(SourceKind kind, const std::string& column) {
    BlockingSpec spec;
    spec.kind = kind;
    spec.column = column;
    spec.name = column + " " + SourceKindName(kind);
    return spec;
}

std::string LadderText(const ComparisonSpec& comparison) {
    std::string text;
    for (size_t i = 0; i < comparison.levels.size(); ++i) {
        if (i > 0) text += " / ";
        text += comparison.levels[i].Describe();
    }
    return text;
}

}  // namespace

const char* RoleName(Role role) {
    for (const RoleEntry& entry : kRoleNames) {
        if (entry.role == role) return entry.name;
    }
    return "unknown";
}

bool ParseRole(const std::string& name, Role* role) {
    for (const RoleEntry& entry : kRoleNames) {
        if (name == entry.name) {
            *role = entry.role;
            return true;
        }
    }
    return false;
}

std::string KnownRoles() {
    std::string text;
    for (const RoleEntry& entry : kRoleNames) {
        if (!text.empty()) text += ", ";
        text += entry.name;
    }
    return text;
}

Role GuessRole(const std::string& name, ColumnType type, bool readable) {
    if (!readable) return Role::kUnknown;
    const Role named = RoleFromName(NormalizeName(name));
    switch (type) {
        case ColumnType::kBoolean:
            return Role::kBoolean;
        case ColumnType::kDouble:
            return named == Role::kLatitude || named == Role::kLongitude ? named
                                                                         : Role::kNumber;
        case ColumnType::kDate:
            return named == Role::kBirthDate ? named : Role::kDate;
        case ColumnType::kStringList:
            return named == Role::kAddress ? named : Role::kList;
        case ColumnType::kString:
            if (named == Role::kLatitude || named == Role::kLongitude) return Role::kText;
            return named;
    }
    return Role::kText;
}

bool DraftSchema(const std::string& path, const DraftOptions& options,
                 DraftReport* report, std::string* error) {
    return DraftSchema(std::vector<std::string>{path}, options, report, error);
}

bool DraftSchema(const std::vector<std::string>& paths, const DraftOptions& options,
                 DraftReport* report, std::string* error) {
    if (paths.empty()) {
        *error = "no parquet file to draft a schema from";
        return false;
    }
    const std::string& path = paths.front();
    std::vector<FileColumn> file_columns;
    if (!ReadFileColumns(path, &file_columns, error)) return false;
    if (file_columns.empty()) {
        *error = path + " has no columns";
        return false;
    }

    // The draft is from the first file; the others have to be able to run it. A
    // column the first holds must be in each of them at a type that reads the same
    // way, or the schema would fail at load with a message about the second file
    // that init could have given now. Columns only a later file holds are left
    // out, and said so.
    std::vector<std::string> later_only;
    for (size_t i = 1; i < paths.size(); ++i) {
        std::vector<FileColumn> other;
        if (!ReadFileColumns(paths[i], &other, error)) return false;
        for (const FileColumn& column : file_columns) {
            const FileColumn* match = nullptr;
            for (const FileColumn& candidate : other) {
                if (candidate.name == column.name) match = &candidate;
            }
            if (match == nullptr) {
                *error = "column \"" + column.name + "\" is in " + path + " but not in " +
                         paths[i] +
                         "; a link needs the same columns "
                         "in every input";
                return false;
            }
            if (column.readable && (!match->readable || match->type != column.type)) {
                *error = "column \"" + column.name + "\" is " + column.arrow_type +
                         " in " + path + " and " + match->arrow_type + " in " + paths[i] +
                         "; cast one so both read as the same type";
                return false;
            }
        }
        for (const FileColumn& candidate : other) {
            bool in_first = false;
            for (const FileColumn& column : file_columns) {
                in_first = in_first || column.name == candidate.name;
            }
            if (!in_first) later_only.push_back(candidate.name + " (" + paths[i] + ")");
        }
    }

    DraftReport draft;
    draft.path = path;
    if (!later_only.empty()) {
        std::string list;
        for (const std::string& name : later_only) {
            if (!list.empty()) list += ", ";
            list += name;
        }
        draft.notes.push_back("not in the first input and so not drafted: " + list);
    }
    std::unordered_map<std::string, Role> given;
    for (const auto& [column, role_name] : options.roles) {
        Role role = Role::kText;
        if (!ParseRole(role_name, &role)) {
            *error = "unknown role \"" + role_name + "\" for column \"" + column +
                     "\" (expected " + KnownRoles() + ")";
            return false;
        }
        given[column] = role;
    }
    for (const FileColumn& file_column : file_columns) {
        DraftColumn column;
        column.name = file_column.name;
        column.arrow_type = file_column.arrow_type;
        column.type = file_column.type;
        column.readable = file_column.readable;
        column.role = GuessRole(column.name, column.type, column.readable);
        const auto it = given.find(column.name);
        if (it != given.end()) {
            column.role = it->second;
            column.role_given = true;
            given.erase(it);
        }
        if (!options.unique_id.empty()) {
            column.role = column.name == options.unique_id ? Role::kId
                          : column.role == Role::kId       ? Role::kText
                                                           : column.role;
        }
        draft.columns.push_back(column);
    }
    for (const auto& [column, role] : given) {
        *error = "--role names column \"" + column + "\", which is not in " + path;
        return false;
    }
    if (!options.unique_id.empty()) {
        const auto it = std::find_if(
            draft.columns.begin(), draft.columns.end(),
            [&](const DraftColumn& column) { return column.name == options.unique_id; });
        if (it == draft.columns.end()) {
            *error = "--id names column \"" + options.unique_id + "\", which is not in " +
                     path;
            return false;
        }
    }

    // The first id column is the unique_id; a second one is an identifier like any
    // other and is compared as text rather than silently dropped.
    Schema schema;
    for (DraftColumn& column : draft.columns) {
        if (column.role != Role::kId) continue;
        if (schema.unique_id.empty()) {
            schema.unique_id = column.name;
            column.note = "unique_id";
        } else {
            column.role = Role::kText;
            column.note = "a second id column, compared as text";
        }
    }
    if (schema.unique_id.empty()) {
        draft.notes.push_back(
            "no id column was found, so records are named by row index; --id names "
            "one");
    }

    // Columns as the file holds them, then the derived ones a role asks for.
    for (DraftColumn& column : draft.columns) {
        if (column.role == Role::kId) continue;
        if (!column.readable) {
            column.role = Role::kUnknown;
            column.note = "not read: " + column.arrow_type +
                          " is not a type cpplink reads; cast it to a string";
            continue;
        }
        ColumnSpec spec;
        spec.name = column.name;
        spec.type = column.type;
        spec.type_declared = true;
        schema.columns.push_back(spec);
    }
    auto declared = [&](const std::string& name) {
        return schema.Find(name) != nullptr ||
               std::any_of(draft.columns.begin(), draft.columns.end(),
                           [&](const DraftColumn& c) { return c.name == name; });
    };

    // One comparison per column, in role order, with the two halves of a
    // coordinate pair folded into one. What each role gets is LadderFor.
    std::vector<size_t> order;
    for (size_t i = 0; i < draft.columns.size(); ++i) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return Priority(draft.columns[a].role) < Priority(draft.columns[b].role);
    });

    std::string latitude;
    std::string longitude;
    std::vector<ComparisonSpec> comparisons;
    for (const size_t index : order) {
        DraftColumn& column = draft.columns[index];
        if (column.role == Role::kId || column.role == Role::kUnknown) continue;
        if (column.role == Role::kNumber) {
            column.note =
                "not compared: a double with no known role; add a numeric_within "
                "level with a threshold in the column's units";
            continue;
        }
        if (column.role == Role::kLatitude || column.role == Role::kLongitude) {
            std::string& slot = column.role == Role::kLatitude ? latitude : longitude;
            if (!slot.empty()) {
                column.note =
                    "not compared: a second " + std::string(RoleName(column.role));
                continue;
            }
            slot = column.name;
            continue;
        }

        ComparisonSpec comparison;
        comparison.name = column.name;
        comparison.columns.push_back(column.name);
        comparison.term_frequency = HasTermFrequencies(column.type);
        comparison.levels = LadderFor(column.role, column.type);

        // An email is compared as splink's email comparison does: the address,
        // then the username split from it at load, then either fuzzily.
        if (column.role == Role::kEmail && column.type == ColumnType::kString) {
            const std::string username = column.name + "_username";
            if (declared(username)) {
                column.note = "the derived column \"" + username +
                              "\" was not added because the file holds one";
            } else {
                ColumnSpec derived;
                derived.name = username;
                derived.type = ColumnType::kString;
                derived.derive.from = column.name;
                derived.derive.transforms.push_back(Transform::kEmailUsername);
                schema.columns.push_back(derived);
                comparison.columns.push_back(username);
                comparison.levels = {Level(LevelType::kNull),
                                     Level(LevelType::kExact),
                                     LevelOn(LevelType::kExact, 1),
                                     Level(LevelType::kJaroWinkler, 0.93),
                                     LevelOn(LevelType::kJaroWinkler, 1, 0.93),
                                     Level(LevelType::kElse)};
            }
        }
        comparisons.push_back(comparison);
    }
    if (!latitude.empty() && !longitude.empty()) {
        ComparisonSpec comparison;
        comparison.name = "location";
        comparison.columns = {latitude, longitude};
        comparison.levels = {Level(LevelType::kNull), Level(LevelType::kGeoWithin, 1),
                             Level(LevelType::kGeoWithin, 25), Level(LevelType::kElse)};
        comparisons.push_back(comparison);
    } else if (!latitude.empty() || !longitude.empty()) {
        for (DraftColumn& column : draft.columns) {
            if (column.name == latitude || column.name == longitude) {
                column.note =
                    "not compared: a coordinate needs both a latitude and "
                    "a longitude";
            }
        }
    }

    // The packed pattern is a uint32. The comparisons are in priority order, so
    // what does not fit is the weakest evidence, and the report names it.
    uint8_t width = 0;
    for (size_t i = 0; i < comparisons.size(); ++i) {
        const uint8_t bits = BitsFor(comparisons[i].levels.size());
        if (width + bits > 32) {
            draft.notes.push_back("comparison \"" + comparisons[i].name +
                                  "\" was left out: the packed pattern is full at 32 "
                                  "bits; merge levels or drop a comparison to make "
                                  "room");
            for (DraftColumn& column : draft.columns) {
                const auto& columns = comparisons[i].columns;
                if (std::find(columns.begin(), columns.end(), column.name) !=
                    columns.end()) {
                    column.note = "not compared: the packed pattern is full at 32 bits";
                }
            }
            continue;
        }
        width = static_cast<uint8_t>(width + bits);
        schema.comparisons.push_back(comparisons[i]);
    }

    // Blocking. Identifiers block on exact agreement, which is free and strong;
    // the name column with the widest reach carries a rare-value source and a
    // sorted neighbourhood, because a name is what the fuzzy pairs agree on.
    // Nothing here blocks on a name's first or a low-cardinality column alone.
    std::string name_anchor;
    int anchor_priority = 99;
    for (const DraftColumn& column : draft.columns) {
        if (column.type != ColumnType::kString || !schema.Find(column.name)) continue;
        int priority = 99;
        if (column.role == Role::kSurname) priority = 0;
        if (column.role == Role::kFullName) priority = 1;
        if (column.role == Role::kFirstName) priority = 2;
        if (priority < anchor_priority) {
            anchor_priority = priority;
            name_anchor = column.name;
        }
    }
    for (const ComparisonSpec& comparison : schema.comparisons) {
        const DraftColumn* column = nullptr;
        for (const DraftColumn& candidate : draft.columns) {
            if (candidate.name == comparison.columns.front()) column = &candidate;
        }
        if (column == nullptr) continue;
        switch (column->role) {
            case Role::kEmail:
            case Role::kPhone:
            case Role::kNationalId:
            case Role::kBirthDate:
            case Role::kPostcode:
                schema.blocking.push_back(Source(SourceKind::kExactValue, column->name));
                if (comparison.columns.size() > 1) {
                    schema.blocking.push_back(
                        Source(SourceKind::kExactValue, comparison.columns[1]));
                }
                break;
            default:
                break;
        }
    }
    if (!name_anchor.empty()) {
        BlockingSpec rare = Source(SourceKind::kRareValue, name_anchor);
        rare.max_frequency = 100;
        schema.blocking.push_back(rare);
        BlockingSpec window = Source(SourceKind::kSortedNeighbourhood, name_anchor);
        window.window = 20;
        schema.blocking.push_back(window);
    }
    if (schema.blocking.empty()) {
        draft.notes.push_back(
            "no column carries a blocking source; name one with --role, or run the "
            "plan-building commands with --all-pairs");
    }
    if (schema.comparisons.empty()) {
        *error = "no column of " + path + " can be compared; see the roles it was given";
        return false;
    }

    draft.json = SchemaToJson(schema);
    if (!ParseSchema(draft.json, &draft.schema, error)) {
        *error = "the draft does not parse, which is a bug: " + *error;
        return false;
    }
    // The ladders are described from the parsed schema, which is what the file
    // says: the parser labels a level that names a column, and this does not.
    for (const ComparisonSpec& comparison : draft.schema.comparisons) {
        for (DraftColumn& column : draft.columns) {
            if (!column.note.empty()) continue;
            const auto& columns = comparison.columns;
            if (std::find(columns.begin(), columns.end(), column.name) == columns.end()) {
                continue;
            }
            column.note = comparison.name == column.name
                              ? LadderText(comparison)
                              : comparison.name + ": " + LadderText(comparison);
        }
    }
    *report = std::move(draft);
    return true;
}

std::string SchemaToJson(const Schema& schema) {
    nlohmann::ordered_json root = nlohmann::ordered_json::object();
    if (!schema.unique_id.empty()) root["unique_id"] = schema.unique_id;

    root["columns"] = nlohmann::ordered_json::array();
    for (const ColumnSpec& column : schema.columns) {
        nlohmann::ordered_json entry;
        entry["name"] = column.name;
        if (column.IsDerived()) {
            nlohmann::ordered_json derive;
            derive["from"] = column.derive.from;
            if (column.derive.transforms.size() == 1) {
                derive["transform"] = TransformName(column.derive.transforms.front());
            } else {
                derive["transform"] = nlohmann::ordered_json::array();
                for (const Transform transform : column.derive.transforms) {
                    derive["transform"].push_back(TransformName(transform));
                }
            }
            entry["derive"] = derive;
        } else if (column.type_declared) {
            entry["type"] = ColumnTypeName(column.type);
        }
        if (!column.derived_from.empty()) entry["derived_from"] = column.derived_from;
        root["columns"].push_back(entry);
    }

    root["comparisons"] = nlohmann::ordered_json::array();
    for (const ComparisonSpec& comparison : schema.comparisons) {
        nlohmann::ordered_json entry;
        entry["name"] = comparison.name;
        entry["columns"] = comparison.columns;
        if (comparison.term_frequency) entry["term_frequency"] = true;
        entry["levels"] = nlohmann::ordered_json::array();
        for (const LevelSpec& level : comparison.levels) {
            nlohmann::ordered_json item;
            item["type"] = LevelTypeName(level.type);
            if (level.type != LevelType::kNull && level.type != LevelType::kExact &&
                level.type != LevelType::kElse) {
                item["threshold"] = level.threshold;
            }
            if (level.directed) item["direction"] = "forward";
            if (level.names_column) {
                const std::string& column = comparison.columns[level.column];
                item["column"] = column;
                // The parser makes this label itself; writing it back would pin
                // a default the reader can see without it.
                LevelSpec plain = level;
                plain.label.clear();
                if (!level.label.empty() &&
                    level.label != plain.Describe() + " on " + column) {
                    item["label"] = level.label;
                }
            } else if (!level.label.empty()) {
                item["label"] = level.label;
            }
            entry["levels"].push_back(item);
        }
        root["comparisons"].push_back(entry);
    }

    root["blocking"] = nlohmann::ordered_json::array();
    for (const BlockingSpec& source : schema.blocking) {
        nlohmann::ordered_json entry;
        entry["type"] = SourceKindName(source.kind);
        if (source.kind != SourceKind::kAllPairs) entry["column"] = source.column;
        switch (source.kind) {
            case SourceKind::kRareValue:
                entry["max_frequency"] = source.max_frequency;
                break;
            case SourceKind::kSortedNeighbourhood:
                entry["window"] = source.window;
                break;
            case SourceKind::kMinHash:
                entry["bands"] = source.bands;
                entry["rows_per_band"] = source.rows_per_band;
                entry["ngram"] = source.ngram;
                entry["seed"] = source.seed;
                break;
            default:
                break;
        }
        if (source.use != SourceUse::kBoth) entry["use"] = SourceUseName(source.use);
        const std::string default_name =
            source.kind == SourceKind::kAllPairs
                ? std::string("all pairs")
                : source.column + " " + SourceKindName(source.kind);
        if (!source.name.empty() && source.name != default_name) {
            entry["name"] = source.name;
        }
        root["blocking"].push_back(entry);
    }
    return root.dump(2) + "\n";
}

void PrintDraftReport(const DraftReport& report, std::ostream& out) {
    size_t width = 6;
    for (const DraftColumn& column : report.columns) {
        width = std::max(width, column.name.size());
    }
    out << "File         " << report.path << "\n";
    out << "Unique id    "
        << (report.schema.unique_id.empty() ? std::string("(row index)")
                                            : report.schema.unique_id)
        << "\n\n";
    out << std::left << std::setw(static_cast<int>(width) + 2) << "Column"
        << std::setw(13) << "Type" << std::setw(14) << "Role" << "Comparison\n";
    for (const DraftColumn& column : report.columns) {
        std::string role = RoleName(column.role);
        if (column.role_given) role += "*";
        out << std::left << std::setw(static_cast<int>(width) + 2) << column.name
            << std::setw(13) << (column.readable ? ColumnTypeName(column.type) : "-")
            << std::setw(14) << role << column.note << "\n";
    }

    bool any_derived = false;
    for (const ColumnSpec& column : report.schema.columns) {
        if (!column.IsDerived()) continue;
        if (!any_derived) out << "\nDerived\n";
        any_derived = true;
        out << "  " << column.name << "  " << column.derive.Describe() << "\n";
    }

    out << "\nBlocking\n";
    if (report.schema.blocking.empty()) out << "  (none)\n";
    for (const BlockingSpec& source : report.schema.blocking) {
        out << "  " << std::left << std::setw(21) << SourceKindName(source.kind)
            << std::setw(static_cast<int>(width) + 2) << source.column
            << source.Describe() << "\n";
    }

    out << "\nPattern      " << static_cast<int>(report.schema.GammaWidth())
        << " of 32 bits over " << report.schema.comparisons.size() << " comparisons\n";
    for (const std::string& note : report.notes) out << "Note         " << note << "\n";
    if (std::any_of(report.columns.begin(), report.columns.end(),
                    [](const DraftColumn& c) { return c.role_given; })) {
        out << "* role given with --role\n";
    }
    out << "\nThis is a draft from the column names alone. `cpplink profile` prices "
           "what each\ncolumn is worth and `cpplink levels` fits the thresholds; edit "
           "the file to match\nthe data before estimating.\n";
}

}  // namespace cpplink
