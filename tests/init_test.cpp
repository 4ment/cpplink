// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/init.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include "cpplink/app.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/schema.hpp"
#include "tests/process_id.hpp"
#include "tests/temp_dir.hpp"

namespace {

using cpplink::ColumnType;
using cpplink::GuessRole;
using cpplink::Role;

TEST(GuessRoleTest, WholeNameSynonyms) {
    EXPECT_EQ(GuessRole("unique_id", ColumnType::kString, true), Role::kId);
    EXPECT_EQ(GuessRole("first_name", ColumnType::kString, true), Role::kFirstName);
    EXPECT_EQ(GuessRole("forename", ColumnType::kString, true), Role::kFirstName);
    EXPECT_EQ(GuessRole("surname", ColumnType::kString, true), Role::kSurname);
    EXPECT_EQ(GuessRole("last_name", ColumnType::kString, true), Role::kSurname);
    EXPECT_EQ(GuessRole("name", ColumnType::kString, true), Role::kFullName);
    EXPECT_EQ(GuessRole("sex", ColumnType::kString, true), Role::kGender);
    EXPECT_EQ(GuessRole("dob", ColumnType::kDate, true), Role::kBirthDate);
    EXPECT_EQ(GuessRole("email", ColumnType::kString, true), Role::kEmail);
    EXPECT_EQ(GuessRole("mobile", ColumnType::kString, true), Role::kPhone);
    EXPECT_EQ(GuessRole("zip", ColumnType::kString, true), Role::kPostcode);
    EXPECT_EQ(GuessRole("city", ColumnType::kString, true), Role::kCity);
    EXPECT_EQ(GuessRole("state", ColumnType::kString, true), Role::kRegion);
    EXPECT_EQ(GuessRole("country", ColumnType::kString, true), Role::kCountry);
    EXPECT_EQ(GuessRole("soc_sec_id", ColumnType::kString, true), Role::kNationalId);
    EXPECT_EQ(GuessRole("lat", ColumnType::kDouble, true), Role::kLatitude);
    EXPECT_EQ(GuessRole("lng", ColumnType::kDouble, true), Role::kLongitude);
}

TEST(GuessRoleTest, SpellingsNormalise) {
    EXPECT_EQ(GuessRole("Date Of Birth", ColumnType::kDate, true), Role::kBirthDate);
    EXPECT_EQ(GuessRole("DateOfBirth", ColumnType::kDate, true), Role::kBirthDate);
    EXPECT_EQ(GuessRole("date-of-birth", ColumnType::kString, true), Role::kBirthDate);
    EXPECT_EQ(GuessRole("FirstName", ColumnType::kString, true), Role::kFirstName);
    EXPECT_EQ(GuessRole("EMAIL", ColumnType::kString, true), Role::kEmail);
}

TEST(GuessRoleTest, FragmentsCarryARole) {
    EXPECT_EQ(GuessRole("email_address_2", ColumnType::kString, true), Role::kEmail);
    EXPECT_EQ(GuessRole("home_phone", ColumnType::kString, true), Role::kPhone);
    EXPECT_EQ(GuessRole("postcode_fake", ColumnType::kString, true), Role::kPostcode);
    EXPECT_EQ(GuessRole("first_name_1", ColumnType::kString, true), Role::kFirstName);
    EXPECT_EQ(GuessRole("company_name", ColumnType::kString, true), Role::kFullName);
    EXPECT_EQ(GuessRole("registration_date", ColumnType::kDate, true), Role::kDate);
    EXPECT_EQ(GuessRole("notes", ColumnType::kString, true), Role::kText);
}

// A name says what the field is; the type says what can be done with it.
TEST(GuessRoleTest, TypeOverridesName) {
    EXPECT_EQ(GuessRole("registration_date", ColumnType::kDate, true), Role::kDate);
    EXPECT_EQ(GuessRole("surname", ColumnType::kDate, true), Role::kDate);
    EXPECT_EQ(GuessRole("surname", ColumnType::kBoolean, true), Role::kBoolean);
    EXPECT_EQ(GuessRole("surname", ColumnType::kDouble, true), Role::kNumber);
    EXPECT_EQ(GuessRole("surname", ColumnType::kStringList, true), Role::kList);
    EXPECT_EQ(GuessRole("address_tokens", ColumnType::kStringList, true), Role::kAddress);
    EXPECT_EQ(GuessRole("latitude", ColumnType::kString, true), Role::kText);
    EXPECT_EQ(GuessRole("email", ColumnType::kString, false), Role::kUnknown);
}

TEST(RoleNameTest, RoundTrips) {
    for (const Role role : {Role::kId, Role::kFirstName, Role::kSurname, Role::kEmail,
                            Role::kLatitude, Role::kText, Role::kUnknown}) {
        Role parsed = Role::kText;
        ASSERT_TRUE(cpplink::ParseRole(cpplink::RoleName(role), &parsed));
        EXPECT_EQ(parsed, role);
    }
    Role parsed = Role::kText;
    EXPECT_FALSE(cpplink::ParseRole("nope", &parsed));
}

class DraftFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_init_" + cpplink_test::ProcessId());
        std::filesystem::create_directories(dir_);
        path_ = (dir_ / "sample.parquet").string();
        cpplink::SampleOptions options;
        options.rows = 2000;
        options.seed = 7;
        std::string error;
        ASSERT_TRUE(cpplink::WriteSampleParquet(path_, options, &error)) << error;
    }
    void TearDown() override { cpplink_test::RemoveAll(dir_); }

    const cpplink::DraftColumn* Column(const cpplink::DraftReport& report,
                                       const std::string& name) {
        for (const cpplink::DraftColumn& column : report.columns) {
            if (column.name == name) return &column;
        }
        return nullptr;
    }

    std::filesystem::path dir_;
    std::string path_;
};

TEST_F(DraftFixture, GuessesTheSampleFile) {
    cpplink::DraftReport report;
    std::string error;
    ASSERT_TRUE(cpplink::DraftSchema(path_, cpplink::DraftOptions(), &report, &error))
        << error;

    EXPECT_EQ(report.schema.unique_id, "id");
    ASSERT_NE(Column(report, "first_name"), nullptr);
    EXPECT_EQ(Column(report, "first_name")->role, Role::kFirstName);
    EXPECT_EQ(Column(report, "last_name")->role, Role::kSurname);
    EXPECT_EQ(Column(report, "gender")->role, Role::kGender);
    EXPECT_EQ(Column(report, "dob")->role, Role::kBirthDate);
    EXPECT_EQ(Column(report, "dob")->type, ColumnType::kDate);
    EXPECT_EQ(Column(report, "email")->role, Role::kEmail);
    EXPECT_EQ(Column(report, "phone")->role, Role::kPhone);
    EXPECT_EQ(Column(report, "postcode")->role, Role::kPostcode);
    EXPECT_EQ(Column(report, "latitude")->role, Role::kLatitude);
    EXPECT_EQ(Column(report, "longitude")->role, Role::kLongitude);
    EXPECT_EQ(Column(report, "address_tokens")->role, Role::kAddress);
    EXPECT_EQ(Column(report, "address_tokens")->type, ColumnType::kStringList);

    // The email's username is split out and compared inside the email comparison.
    const cpplink::ColumnSpec* username = report.schema.Find("email_username");
    ASSERT_NE(username, nullptr);
    EXPECT_TRUE(username->IsDerived());
    EXPECT_EQ(username->derive.from, "email");
    bool email_names_username = false;
    bool location = false;
    for (const cpplink::ComparisonSpec& comparison : report.schema.comparisons) {
        if (comparison.name == "email") {
            ASSERT_EQ(comparison.columns.size(), 2u);
            for (const cpplink::LevelSpec& level : comparison.levels) {
                if (level.names_column && level.column == 1) email_names_username = true;
            }
        }
        if (comparison.name == "location") {
            location = true;
            EXPECT_EQ(comparison.columns.size(), 2u);
            EXPECT_EQ(comparison.levels[1].type, cpplink::LevelType::kGeoWithin);
        }
        EXPECT_EQ(comparison.levels.back().type, cpplink::LevelType::kElse);
    }
    EXPECT_TRUE(email_names_username);
    EXPECT_TRUE(location);
    EXPECT_LE(report.schema.GammaWidth(), 32);

    // Identifiers block on exact agreement; the surname carries the fuzzy reach.
    bool email_exact = false;
    bool surname_rare = false;
    bool surname_window = false;
    for (const cpplink::BlockingSpec& source : report.schema.blocking) {
        if (source.column == "email" && source.kind == cpplink::SourceKind::kExactValue) {
            email_exact = true;
        }
        if (source.column == "last_name") {
            if (source.kind == cpplink::SourceKind::kRareValue) surname_rare = true;
            if (source.kind == cpplink::SourceKind::kSortedNeighbourhood) {
                surname_window = true;
            }
        }
    }
    EXPECT_TRUE(email_exact);
    EXPECT_TRUE(surname_rare);
    EXPECT_TRUE(surname_window);
}

// The draft is only worth anything if the other commands accept it: the JSON
// must parse, resolve against the file and load into a store.
TEST_F(DraftFixture, DraftLoadsTheFileItWasDrawnFrom) {
    cpplink::DraftReport report;
    std::string error;
    ASSERT_TRUE(cpplink::DraftSchema(path_, cpplink::DraftOptions(), &report, &error))
        << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(report.json, &schema, &error)) << error;
    ASSERT_TRUE(cpplink::ResolveColumnTypes({path_}, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    cpplink::LoadStats stats;
    ASSERT_TRUE(cpplink::LoadParquetFiles({path_}, schema, &store, &stats, &error))
        << error;
    EXPECT_EQ(store.NumRecords(), 2000u);
}

TEST_F(DraftFixture, RolesAndIdCanBeGiven) {
    cpplink::DraftOptions options;
    options.roles.emplace_back("postcode", "text");
    options.roles.emplace_back("gender", "national_id");
    cpplink::DraftReport report;
    std::string error;
    ASSERT_TRUE(cpplink::DraftSchema(path_, options, &report, &error)) << error;
    EXPECT_EQ(Column(report, "postcode")->role, Role::kText);
    EXPECT_TRUE(Column(report, "postcode")->role_given);
    EXPECT_EQ(Column(report, "gender")->role, Role::kNationalId);
    bool gender_blocks = false;
    bool postcode_blocks = false;
    for (const cpplink::BlockingSpec& source : report.schema.blocking) {
        if (source.column == "gender") gender_blocks = true;
        if (source.column == "postcode") postcode_blocks = true;
    }
    EXPECT_TRUE(gender_blocks);
    EXPECT_FALSE(postcode_blocks);

    options = cpplink::DraftOptions();
    options.unique_id = "phone";
    ASSERT_TRUE(cpplink::DraftSchema(path_, options, &report, &error)) << error;
    EXPECT_EQ(report.schema.unique_id, "phone");
    EXPECT_EQ(Column(report, "id")->role, Role::kText);
    EXPECT_NE(report.schema.Find("id"), nullptr);
}

TEST_F(DraftFixture, RefusesWhatItCannotPlace) {
    cpplink::DraftOptions options;
    options.roles.emplace_back("postcode", "zipcode");
    cpplink::DraftReport report;
    std::string error;
    EXPECT_FALSE(cpplink::DraftSchema(path_, options, &report, &error));
    EXPECT_NE(error.find("unknown role"), std::string::npos);

    options = cpplink::DraftOptions();
    options.roles.emplace_back("nope", "text");
    EXPECT_FALSE(cpplink::DraftSchema(path_, options, &report, &error));
    EXPECT_NE(error.find("not in"), std::string::npos);

    options = cpplink::DraftOptions();
    options.unique_id = "nope";
    EXPECT_FALSE(cpplink::DraftSchema(path_, options, &report, &error));
    EXPECT_NE(error.find("--id"), std::string::npos);
}

// A link's draft is from the first input, and the others must be able to run it:
// the same columns, at types that read the same way. The check is made here from
// the footers rather than left to the loader, which would fail on the second file
// after the schema had already been written.
TEST_F(DraftFixture, DraftsALinkFromTheFirstInputAndChecksTheOthers) {
    const std::string second = (dir_ / "second.parquet").string();
    cpplink::SampleOptions options;
    options.rows = 500;
    options.seed = 9;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(second, options, &error)) << error;

    cpplink::DraftReport alone;
    cpplink::DraftReport linked;
    ASSERT_TRUE(cpplink::DraftSchema(path_, cpplink::DraftOptions(), &alone, &error))
        << error;
    ASSERT_TRUE(cpplink::DraftSchema(std::vector<std::string>{path_, second},
                                     cpplink::DraftOptions(), &linked, &error))
        << error;
    EXPECT_EQ(linked.json, alone.json);

    // A second input missing a column the first holds, and one holding it at a
    // type that reads differently, are both refused by name.
    const std::string narrow = (dir_ / "narrow.parquet").string();
    const std::string retyped = (dir_ / "retyped.parquet").string();
    {
        arrow::StringBuilder ids;
        arrow::StringBuilder dobs;
        ASSERT_TRUE(ids.Append("x1").ok());
        ASSERT_TRUE(dobs.Append("1980-01-02").ok());
        std::shared_ptr<arrow::Array> id_array;
        std::shared_ptr<arrow::Array> dob_array;
        ASSERT_TRUE(ids.Finish(&id_array).ok());
        ASSERT_TRUE(dobs.Finish(&dob_array).ok());
        auto write = [&](const std::string& path,
                         const std::vector<std::shared_ptr<arrow::Field>>& fields,
                         const std::vector<std::shared_ptr<arrow::Array>>& arrays) {
            auto table = arrow::Table::Make(arrow::schema(fields), arrays);
            auto sink = arrow::io::FileOutputStream::Open(path);
            ASSERT_TRUE(sink.ok());
            ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                                   *sink, 1024)
                            .ok());
        };
        write(narrow, {arrow::field("id", arrow::utf8())}, {id_array});
        std::vector<cpplink::FileColumn> columns;
        ASSERT_TRUE(cpplink::ReadFileColumns(path_, &columns, &error)) << error;
        // Every column of the sample, with dob as text rather than a date.
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        for (const cpplink::FileColumn& column : columns) {
            fields.push_back(arrow::field(column.name, arrow::utf8()));
            arrays.push_back(column.name == "dob" ? dob_array : id_array);
        }
        write(retyped, fields, arrays);
    }
    cpplink::DraftReport report;
    EXPECT_FALSE(cpplink::DraftSchema(std::vector<std::string>{path_, narrow},
                                      cpplink::DraftOptions(), &report, &error));
    EXPECT_NE(error.find("but not in " + narrow), std::string::npos) << error;
    EXPECT_FALSE(cpplink::DraftSchema(std::vector<std::string>{path_, retyped},
                                      cpplink::DraftOptions(), &report, &error));
    EXPECT_NE(error.find("\"dob\""), std::string::npos) << error;
    EXPECT_NE(error.find("cast one"), std::string::npos) << error;
}

TEST_F(DraftFixture, RunWritesTheDraftAndTheReport) {
    const std::string out_path = (dir_ / "draft.json").string();
    std::ostringstream out;
    std::ostringstream err;
    ASSERT_EQ(cpplink::Run({"init", "--out", out_path, path_}, out, err), 0) << err.str();
    EXPECT_NE(out.str().find("Blocking"), std::string::npos);
    EXPECT_NE(out.str().find("Wrote " + out_path), std::string::npos);

    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::LoadSchema(out_path, &schema, &error)) << error;
    EXPECT_EQ(schema.unique_id, "id");

    // Without --out the schema is stdout and the report stderr.
    out.str("");
    err.str("");
    ASSERT_EQ(cpplink::Run({"init", path_}, out, err), 0) << err.str();
    ASSERT_TRUE(cpplink::ParseSchema(out.str(), &schema, &error)) << error;
    EXPECT_NE(err.str().find("Blocking"), std::string::npos);
}

TEST(RunInitTest, NeedsOneFile) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"init"}, out, err), 1);
    EXPECT_NE(err.str().find("parquet file"), std::string::npos);
    EXPECT_EQ(cpplink::Run({"init", "--role", "bad", "x.parquet"}, out, err), 1);
    EXPECT_NE(err.str().find("COLUMN=ROLE"), std::string::npos);
}

// Writing a schema and reading it back must give the same schema, or a draft
// is describing something other than what it wrote.
TEST(SchemaToJsonTest, RoundTripsTheSampleSchema) {
    const std::string path =
        std::string(CPPLINK_SOURCE_DIR) + "/examples/sample_schema.json";
    std::ifstream file(path);
    ASSERT_TRUE(file.is_open()) << "cannot open " << path;
    std::stringstream text;
    text << file.rdbuf();
    cpplink::Schema first;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(text.str(), &first, &error)) << error;

    cpplink::Schema second;
    ASSERT_TRUE(cpplink::ParseSchema(cpplink::SchemaToJson(first), &second, &error))
        << error;
    EXPECT_EQ(second.unique_id, first.unique_id);
    ASSERT_EQ(second.columns.size(), first.columns.size());
    for (size_t i = 0; i < first.columns.size(); ++i) {
        EXPECT_EQ(second.columns[i].name, first.columns[i].name);
        EXPECT_EQ(second.columns[i].type, first.columns[i].type);
        EXPECT_EQ(second.columns[i].derive.Describe(),
                  first.columns[i].derive.Describe());
    }
    ASSERT_EQ(second.comparisons.size(), first.comparisons.size());
    for (size_t i = 0; i < first.comparisons.size(); ++i) {
        const cpplink::ComparisonSpec& a = first.comparisons[i];
        const cpplink::ComparisonSpec& b = second.comparisons[i];
        EXPECT_EQ(b.name, a.name);
        EXPECT_EQ(b.columns, a.columns);
        EXPECT_EQ(b.term_frequency, a.term_frequency);
        EXPECT_EQ(b.bits, a.bits);
        EXPECT_EQ(b.shift, a.shift);
        ASSERT_EQ(b.levels.size(), a.levels.size());
        for (size_t j = 0; j < a.levels.size(); ++j) {
            EXPECT_EQ(b.levels[j].type, a.levels[j].type);
            EXPECT_EQ(b.levels[j].threshold, a.levels[j].threshold);
            EXPECT_EQ(b.levels[j].column, a.levels[j].column);
            EXPECT_EQ(b.levels[j].Describe(), a.levels[j].Describe());
        }
    }
    ASSERT_EQ(second.blocking.size(), first.blocking.size());
    for (size_t i = 0; i < first.blocking.size(); ++i) {
        EXPECT_EQ(second.blocking[i].kind, first.blocking[i].kind);
        EXPECT_EQ(second.blocking[i].column, first.blocking[i].column);
        EXPECT_EQ(second.blocking[i].name, first.blocking[i].name);
        EXPECT_EQ(second.blocking[i].Describe(), first.blocking[i].Describe());
    }
}

}  // namespace
