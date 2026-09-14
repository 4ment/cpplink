// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "cpplink/parquet_loader.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// What a column is for, read off its name and the type the file gives it. A role
// is a guess about the field, and the field is what decides the comparison: a
// surname wants a Jaro-Winkler ladder with term frequency, a date of birth wants
// exact then a day or two then a year, an email wants its username split out. The
// roles are what splink's comparison templates are indexed by, and the draft is
// the same idea run from the column names rather than by hand.
enum class Role {
    kId,          // the unique_id; the row index if none is found
    kFirstName,   // forename, given name
    kMiddleName,  // middle name or initial
    kSurname,     // last name, family name
    kFullName,    // a whole name in one field
    kGender,      // sex or gender
    kBirthDate,   // date of birth
    kDate,        // any other date
    kEmail,       // email address
    kPhone,       // telephone number
    kPostcode,    // postcode, zip code
    kAddress,     // street address, a string or a list of tokens
    kCity,        // city, town, suburb
    kRegion,      // state, county, province
    kCountry,     // country
    kNationalId,  // social security number, passport, tax id: an identifier
    kLatitude,    // half of a coordinate pair
    kLongitude,   // the other half
    kList,        // a string list with no better guess
    kBoolean,     // a boolean column
    kNumber,      // a double with no known role; compared only when the user says
    kText,        // a string column with no better guess
    kUnknown,     // an Arrow type cpplink cannot read
};

const char* RoleName(Role role);
bool ParseRole(const std::string& name, Role* role);
std::string KnownRoles();

// The guess from the name alone, given the type the file holds. The name is
// normalised (lowercased, punctuation to underscores) and matched whole against a
// synonym table first, then against a few substrings that are unambiguous enough
// to carry a role on their own ("email", "phone", "postcode"). The type is what
// a name cannot override: a `dob` the file holds as a date is a birth date, one
// it holds as a list is a list.
Role GuessRole(const std::string& name, ColumnType type, bool readable);

// One column of the draft: what the file said, what was guessed, and what the
// draft did with it. `note` is empty where the column was compared as its role
// says; otherwise it says why not.
struct DraftColumn {
    std::string name;
    std::string arrow_type;
    ColumnType type = ColumnType::kString;
    bool readable = false;
    Role role = Role::kUnknown;
    bool role_given = false;  // set by --role rather than guessed
    std::string note;
};

struct DraftOptions {
    std::string unique_id;                                   // --id, or guessed
    std::vector<std::pair<std::string, std::string>> roles;  // --role column=role
};

struct DraftReport {
    std::string path;
    std::vector<DraftColumn> columns;
    std::vector<std::string> notes;  // about the schema as a whole
    Schema schema;                   // the draft, parsed back through ParseSchema
    std::string json;                // the draft as written
};

// Reads the footer of one parquet file, guesses a role per column, and writes a
// schema that runs: a comparison per role with the default ladder, derived
// columns where a role wants one, and the blocking sources a role is strong
// enough to carry. The draft is a starting point and says so; `profile` prices
// what it is worth and `levels` fits the thresholds it guessed.
bool DraftSchema(const std::string& path, const DraftOptions& options,
                 DraftReport* report, std::string* error);

// Every draft is re-parsed before it is returned, so what comes out is a file
// the other commands accept, not a fragment.
std::string SchemaToJson(const Schema& schema);

void PrintDraftReport(const DraftReport& report, std::ostream& out);

}  // namespace cpplink
