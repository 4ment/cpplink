// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// A parquet file written from Python rarely holds the Arrow types the schema
// names. pandas and polars write text as `large_string` and lists as `large_list`,
// and a column that was numeric in the frame -- a postcode, a year, an integer
// id -- arrives as int64 rather than as the string the user meant. These tests
// write those shapes with Arrow builders and check that each loads as the value
// it would have been as text, so a file from a script and the same file with
// every column cast to string produce the same store.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "tests/process_id.hpp"
#include "tests/temp_dir.hpp"

namespace {

class LoaderTypes : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_loader_types_" + cpplink_test::ProcessId());
        cpplink_test::RemoveAll(dir_);
        std::filesystem::create_directories(dir_);
        path_ = (dir_ / "typed.parquet").string();
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    // Two row groups, so the loader appends chunk by chunk as it does at scale.
    void Write(const std::vector<std::shared_ptr<arrow::Field>>& fields,
               const std::vector<std::shared_ptr<arrow::Array>>& arrays) {
        auto table = arrow::Table::Make(arrow::schema(fields), arrays);
        auto sink = arrow::io::FileOutputStream::Open(path_);
        ASSERT_TRUE(sink.ok());
        ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                               *sink,
                                               /*chunk_size=*/2)
                        .ok());
    }

    template <typename Builder, typename T>
    static std::shared_ptr<arrow::Array> Column(const std::vector<T>& values,
                                                const std::vector<bool>& valid = {}) {
        Builder builder;
        for (size_t i = 0; i < values.size(); ++i) {
            if (!valid.empty() && !valid[i]) {
                EXPECT_TRUE(builder.AppendNull().ok());
            } else {
                EXPECT_TRUE(builder.Append(values[i]).ok());
            }
        }
        std::shared_ptr<arrow::Array> array;
        EXPECT_TRUE(builder.Finish(&array).ok());
        return array;
    }

    static std::string Text(const cpplink::StringColumn& column, uint64_t row) {
        return std::string(column.dict.Value(column.ids[row]));
    }

    std::filesystem::path dir_;
    std::string path_;
};

TEST_F(LoaderTypes, IntegerColumnsLoadAsTheirDecimalText) {
    Write({arrow::field("id", arrow::int64()), arrow::field("postcode", arrow::int32()),
           arrow::field("year", arrow::uint16()), arrow::field("delta", arrow::int8())},
          {Column<arrow::Int64Builder>(std::vector<int64_t>{10, 11, 12, 13}),
           Column<arrow::Int32Builder>(std::vector<int32_t>{2000, 2000, 90210, 0},
                                       {true, true, false, true}),
           Column<arrow::UInt16Builder>(std::vector<uint16_t>{1999, 65535, 1999, 7}),
           Column<arrow::Int8Builder>(std::vector<int8_t>{-128, 0, 127, -1})});

    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[
        {"name":"postcode","type":"string"},
        {"name":"year","type":"string"},
        {"name":"delta","type":"string"}]})",
                                     &schema, &error))
        << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error)) << error;

    ASSERT_EQ(store.NumRecords(), 4u);
    EXPECT_EQ(store.ids().Get(0), "10");
    EXPECT_EQ(store.ids().Get(3), "13");

    const auto& postcode = std::get<cpplink::StringColumn>(store.column(0));
    EXPECT_EQ(Text(postcode, 0), "2000");
    EXPECT_EQ(postcode.ids[0], postcode.ids[1]) << "the same integer is one id";
    EXPECT_EQ(postcode.ids[2], cpplink::kNullId);
    EXPECT_EQ(Text(postcode, 3), "0");
    EXPECT_EQ(postcode.tf[postcode.ids[0]], 2u);

    const auto& year = std::get<cpplink::StringColumn>(store.column(1));
    EXPECT_EQ(Text(year, 1), "65535");
    EXPECT_EQ(year.ids[0], year.ids[2]);

    const auto& delta = std::get<cpplink::StringColumn>(store.column(2));
    EXPECT_EQ(Text(delta, 0), "-128");
    EXPECT_EQ(Text(delta, 2), "127");
    EXPECT_EQ(Text(delta, 3), "-1");
}

// A large_string column holds the same bytes behind 64-bit offsets; a script that
// used pyarrow-backed strings or polars wrote one without choosing to.
TEST_F(LoaderTypes, LargeStringLoadsAsString) {
    Write({arrow::field("id", arrow::large_utf8()),
           arrow::field("name", arrow::large_utf8())},
          {Column<arrow::LargeStringBuilder>(std::vector<std::string>{"a", "b", "c"}),
           Column<arrow::LargeStringBuilder>(std::vector<std::string>{"ann", "", "ann"},
                                             {true, false, true})});

    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(
        R"({"unique_id":"id","columns":[{"name":"name","type":"string"}]})", &schema,
        &error))
        << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error)) << error;

    EXPECT_EQ(store.ids().Get(1), "b");
    const auto& name = std::get<cpplink::StringColumn>(store.column(0));
    EXPECT_EQ(Text(name, 0), "ann");
    EXPECT_EQ(name.ids[1], cpplink::kNullId);
    EXPECT_EQ(name.ids[0], name.ids[2]);
}

// A list of integers is a list of their digits, sorted and deduplicated per row
// like any other list, and large_list is the same list with wider offsets.
TEST_F(LoaderTypes, ListsOfIntegersAndLargeListsLoadAsStringLists) {
    arrow::ListBuilder codes(arrow::default_memory_pool(),
                             std::make_shared<arrow::Int64Builder>());
    auto* code = static_cast<arrow::Int64Builder*>(codes.value_builder());
    ASSERT_TRUE(codes.Append().ok());
    ASSERT_TRUE(code->AppendValues({30, 10, 30}).ok());
    ASSERT_TRUE(codes.AppendNull().ok());
    ASSERT_TRUE(codes.Append().ok());
    ASSERT_TRUE(code->Append(10).ok());
    ASSERT_TRUE(code->AppendNull().ok());
    std::shared_ptr<arrow::Array> codes_array;
    ASSERT_TRUE(codes.Finish(&codes_array).ok());

    arrow::LargeListBuilder tags(arrow::default_memory_pool(),
                                 std::make_shared<arrow::LargeStringBuilder>());
    auto* tag = static_cast<arrow::LargeStringBuilder*>(tags.value_builder());
    ASSERT_TRUE(tags.Append().ok());
    ASSERT_TRUE(tag->AppendValues({"x", "y"}).ok());
    ASSERT_TRUE(tags.Append().ok());
    ASSERT_TRUE(tags.Append().ok());
    ASSERT_TRUE(tag->AppendValues({"y"}).ok());
    std::shared_ptr<arrow::Array> tags_array;
    ASSERT_TRUE(tags.Finish(&tags_array).ok());

    Write({arrow::field("id", arrow::utf8()),
           arrow::field("codes", arrow::list(arrow::int64())),
           arrow::field("tags", arrow::large_list(arrow::large_utf8()))},
          {Column<arrow::StringBuilder>(std::vector<std::string>{"a", "b", "c"}),
           codes_array, tags_array});

    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[
        {"name":"codes","type":"string_list"},
        {"name":"tags","type":"string_list"}]})",
                                     &schema, &error))
        << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error)) << error;

    const auto& codes_column = std::get<cpplink::StringListColumn>(store.column(0));
    const std::vector<uint64_t> code_offsets = {0, 2, 2, 3};
    EXPECT_EQ(codes_column.offsets, code_offsets);
    // Cells are sorted by id, and 30 was interned before 10.
    EXPECT_EQ(codes_column.dict.Value(codes_column.ids[0]), "30");
    EXPECT_EQ(codes_column.dict.Value(codes_column.ids[1]), "10");
    EXPECT_EQ(codes_column.ids[2], codes_column.ids[1]) << "10 again on the third row";

    const auto& tags_column = std::get<cpplink::StringListColumn>(store.column(1));
    const std::vector<uint64_t> tag_offsets = {0, 2, 2, 3};
    EXPECT_EQ(tags_column.offsets, tag_offsets);
    EXPECT_EQ(tags_column.dict.Value(tags_column.ids[2]), "y");
}

// An integer is text, not a double: a writer who wanted a double wrote one, so a
// schema declaring `double` over an int64 column is a contradiction with the file.
TEST_F(LoaderTypes, IntegerColumnIsNotADouble) {
    Write({arrow::field("id", arrow::utf8()), arrow::field("height", arrow::int64())},
          {Column<arrow::StringBuilder>(std::vector<std::string>{"a", "b", "c"}),
           Column<arrow::Int64Builder>(std::vector<int64_t>{170, 0, 182},
                                       {true, false, true})});

    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(
        R"({"unique_id":"id","columns":[{"name":"height","type":"double"}]})", &schema,
        &error))
        << error;
    cpplink::RecordStore store(schema);
    EXPECT_FALSE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error));
    EXPECT_NE(error.find("expected a floating point column"), std::string::npos) << error;
}

// What is still refused: a floating point column declared as a string, because
// "2000.0" is a pandas artefact and not a value anyone typed, and a list whose
// elements are neither text nor integers.
TEST_F(LoaderTypes, FloatingPointIsNotReadAsText) {
    Write({arrow::field("id", arrow::utf8()), arrow::field("postcode", arrow::float64())},
          {Column<arrow::StringBuilder>(std::vector<std::string>{"a", "b"}),
           Column<arrow::DoubleBuilder>(std::vector<double>{2000.0, 2001.0})});

    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(
        R"({"unique_id":"id","columns":[{"name":"postcode","type":"string"}]})", &schema,
        &error))
        << error;
    cpplink::RecordStore store(schema);
    EXPECT_FALSE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error));
    EXPECT_NE(error.find("expected a string or integer column"), std::string::npos)
        << error;
    EXPECT_NE(error.find("double"), std::string::npos) << error;
}

// -- Types from the file -----------------------------------------------------------
//
// A parquet file already knows what its columns are, so a schema need not say
// again. A column with no "type" takes the file's: text and integers are strings,
// lists are string lists, dates and timestamps are dates, floats are doubles.

class TypesFromFile : public LoaderTypes {
   protected:
    void WriteMixed() {
        arrow::ListBuilder tags(arrow::default_memory_pool(),
                                std::make_shared<arrow::StringBuilder>());
        auto* tag = static_cast<arrow::StringBuilder*>(tags.value_builder());
        ASSERT_TRUE(tags.Append().ok());
        ASSERT_TRUE(tag->AppendValues({"x", "y"}).ok());
        ASSERT_TRUE(tags.Append().ok());
        ASSERT_TRUE(tags.Append().ok());
        ASSERT_TRUE(tag->Append("y").ok());
        std::shared_ptr<arrow::Array> tags_array;
        ASSERT_TRUE(tags.Finish(&tags_array).ok());

        Write(
            {arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8()),
             arrow::field("postcode", arrow::int32()),
             arrow::field("dob", arrow::date32()), arrow::field("lat", arrow::float64()),
             arrow::field("alive", arrow::boolean()),
             arrow::field("tags", arrow::list(arrow::utf8()))},
            {Column<arrow::Int64Builder>(std::vector<int64_t>{1, 2, 3}),
             Column<arrow::StringBuilder>(std::vector<std::string>{"ann", "bob", "cy"}),
             Column<arrow::Int32Builder>(std::vector<int32_t>{2000, 2001, 2000}),
             Column<arrow::Date32Builder>(std::vector<int32_t>{10, 20, 30}),
             Column<arrow::DoubleBuilder>(std::vector<double>{1.5, 2.5, 3.5}),
             Column<arrow::BooleanBuilder>(std::vector<bool>{true, false, true}),
             tags_array});
    }
};

TEST_F(TypesFromFile, UndeclaredColumnsTakeTheFilesTypes) {
    WriteMixed();
    cpplink::Schema schema;
    std::string error;
    // Not one "type" anywhere, and every level that needs a type to be legal.
    ASSERT_TRUE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[
        {"name":"name"},{"name":"postcode"},{"name":"dob"},{"name":"lat"},
        {"name":"alive"},{"name":"tags"},
        {"name":"dob_year","derive":{"from":"dob","transform":"year"}}],
      "comparisons":[
        {"columns":["name"],"levels":[{"type":"jaro_winkler","threshold":0.9},
                                      {"type":"else"}]},
        {"columns":["dob"],"levels":[{"type":"date_within","threshold":3},
                                     {"type":"else"}]},
        {"columns":["lat","lat"],"levels":[{"type":"geo_within","threshold":5},
                                           {"type":"else"}]},
        {"columns":["tags"],"levels":[{"type":"list_overlap","threshold":1},
                                      {"type":"else"}]},
        {"columns":["alive"],"levels":[{"type":"exact"},{"type":"else"}]}],
      "blocking":[{"type":"exact_value","column":"postcode"}]})",
                                     &schema, &error))
        << error;
    for (const cpplink::ColumnSpec& spec : schema.columns) {
        EXPECT_FALSE(spec.type_declared) << spec.name;
    }

    ASSERT_TRUE(cpplink::ResolveColumnTypes({path_}, &schema, &error)) << error;
    EXPECT_EQ(schema.Find("name")->type, cpplink::ColumnType::kString);
    EXPECT_EQ(schema.Find("postcode")->type, cpplink::ColumnType::kString);
    EXPECT_EQ(schema.Find("dob")->type, cpplink::ColumnType::kDate);
    EXPECT_EQ(schema.Find("lat")->type, cpplink::ColumnType::kDouble);
    EXPECT_EQ(schema.Find("alive")->type, cpplink::ColumnType::kBoolean);
    EXPECT_EQ(schema.Find("tags")->type, cpplink::ColumnType::kStringList);
    EXPECT_EQ(schema.Find("dob_year")->type, cpplink::ColumnType::kString);

    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error)) << error;
    EXPECT_EQ(store.ids().Get(2), "3");
    EXPECT_EQ(Text(std::get<cpplink::StringColumn>(store.column(1)), 0), "2000");
    EXPECT_EQ(std::get<cpplink::DateColumn>(store.column(2)).values[1], 20);
    EXPECT_DOUBLE_EQ(std::get<cpplink::DoubleColumn>(store.column(3)).values[2], 3.5);
    EXPECT_EQ(Text(std::get<cpplink::StringColumn>(store.column(6)), 0), "1970");
}

// The type checks a declared schema fails at parse time are the same checks,
// only run once the file has said what the columns are.
TEST_F(TypesFromFile, LevelChecksAreDeferredNotDropped) {
    WriteMixed();
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[{"name":"dob"}],
      "comparisons":[{"columns":["dob"],
        "levels":[{"type":"jaro_winkler","threshold":0.9},{"type":"else"}]}]})",
                                     &schema, &error))
        << error;
    EXPECT_FALSE(cpplink::ResolveColumnTypes({path_}, &schema, &error));
    EXPECT_NE(error.find("cannot read"), std::string::npos) << error;

    ASSERT_TRUE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[{"name":"lat"}],
      "blocking":[{"type":"exact_value","column":"lat"}]})",
                                     &schema, &error))
        << error;
    EXPECT_FALSE(cpplink::ResolveColumnTypes({path_}, &schema, &error));
    EXPECT_NE(error.find("discrete agreement"), std::string::npos) << error;

    ASSERT_TRUE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[{"name":"name"},
        {"name":"name_year","derive":{"from":"name","transform":"year"}}]})",
                                     &schema, &error))
        << error;
    EXPECT_FALSE(cpplink::ResolveColumnTypes({path_}, &schema, &error));
    EXPECT_NE(error.find("cannot read"), std::string::npos) << error;
}

// What the schema does say is checked against the file, not used instead of it.
TEST_F(TypesFromFile, ADeclaredTypeIsStillCheckedAgainstTheFile) {
    WriteMixed();
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(
        R"({"unique_id":"id","columns":[{"name":"dob","type":"string"}]})", &schema,
        &error))
        << error;
    ASSERT_TRUE(cpplink::ResolveColumnTypes({path_}, &schema, &error)) << error;
    EXPECT_EQ(schema.Find("dob")->type, cpplink::ColumnType::kString);
    cpplink::RecordStore store(schema);
    EXPECT_FALSE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error));
    EXPECT_NE(error.find("expected a string or integer column"), std::string::npos)
        << error;
}

TEST_F(TypesFromFile, RefusesAColumnTheFileLacksOrCannotType) {
    Write({arrow::field("id", arrow::utf8()), arrow::field("blob", arrow::binary())},
          {Column<arrow::StringBuilder>(std::vector<std::string>{"a"}),
           Column<arrow::BinaryBuilder>(std::vector<std::string>{"\x00"})});
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[{"name":"gone"}]})",
                                     &schema, &error))
        << error;
    EXPECT_FALSE(cpplink::ResolveColumnTypes({path_}, &schema, &error));
    EXPECT_NE(error.find("\"gone\" is not in"), std::string::npos) << error;

    ASSERT_TRUE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[{"name":"blob"}]})",
                                     &schema, &error))
        << error;
    EXPECT_FALSE(cpplink::ResolveColumnTypes({path_}, &schema, &error));
    EXPECT_NE(error.find("cannot read"), std::string::npos) << error;
}

}  // namespace
