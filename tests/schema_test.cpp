// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/schema.hpp"

#include <string>

#include <gtest/gtest.h>

namespace {

constexpr const char* kValid = R"({
  "unique_id": "id",
  "columns": [
    {"name": "first_name", "type": "string"},
    {"name": "dob", "type": "date"},
    {"name": "latitude", "type": "double"},
    {"name": "address_tokens", "type": "string_list"},
    {"name": "defaults_to_string"}
  ]
})";

TEST(SchemaTest, ParsesEveryColumnType) {
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(kValid, &schema, &error)) << error;
    EXPECT_EQ(schema.unique_id, "id");
    ASSERT_EQ(schema.columns.size(), 5u);
    EXPECT_EQ(schema.columns[0].type, cpplink::ColumnType::kString);
    EXPECT_EQ(schema.columns[1].type, cpplink::ColumnType::kDate);
    EXPECT_EQ(schema.columns[2].type, cpplink::ColumnType::kDouble);
    EXPECT_EQ(schema.columns[3].type, cpplink::ColumnType::kStringList);
    EXPECT_EQ(schema.columns[4].type, cpplink::ColumnType::kString);
}

TEST(SchemaTest, FindsColumnsByName) {
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(kValid, &schema, &error)) << error;
    ASSERT_NE(schema.Find("dob"), nullptr);
    EXPECT_EQ(schema.Find("dob")->type, cpplink::ColumnType::kDate);
    EXPECT_EQ(schema.Find("absent"), nullptr);
}

TEST(SchemaTest, RejectsMalformedJson) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_FALSE(cpplink::ParseSchema("{not json", &schema, &error));
    EXPECT_NE(error.find("valid JSON"), std::string::npos);
}

TEST(SchemaTest, RejectsUnknownType) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_FALSE(cpplink::ParseSchema(R"({"columns":[{"name":"x","type":"blob"}]})",
                                      &schema, &error));
    EXPECT_NE(error.find("unknown type"), std::string::npos);
}

TEST(SchemaTest, RejectsDuplicateColumns) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_FALSE(cpplink::ParseSchema(R"({"columns":[{"name":"x"},{"name":"x"}]})",
                                      &schema, &error));
    EXPECT_NE(error.find("twice"), std::string::npos);
}

TEST(SchemaTest, RejectsUniqueIdAlsoDeclaredAsAColumn) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_FALSE(cpplink::ParseSchema(R"({"unique_id":"id","columns":[{"name":"id"}]})",
                                      &schema, &error));
    EXPECT_NE(error.find("must not also be declared"), std::string::npos);
}

// A comparison configuration that is wrong should fail at parse time, before a
// file is opened -- not on the first pair, an hour into a run.
class ComparisonConfig : public ::testing::Test {
   protected:
    bool Parse(const std::string& comparisons) {
        const std::string json = R"({"columns":[
            {"name":"surname","type":"string"},
            {"name":"dob","type":"date"},
            {"name":"lat","type":"double"},
            {"name":"lon","type":"double"}],"comparisons":)" +
                                 comparisons + "}";
        return cpplink::ParseSchema(json, &schema_, &error_);
    }
    cpplink::Schema schema_;
    std::string error_;
};

TEST_F(ComparisonConfig, AssignsBitsAndDisjointShifts) {
    ASSERT_TRUE(Parse(R"([
        {"columns":["surname"],"levels":[{"type":"null"},{"type":"exact"},
            {"type":"jaro_winkler","threshold":0.9},{"type":"else"}]},
        {"columns":["dob"],"levels":[{"type":"exact"},{"type":"else"}]}
    ])"))
        << error_;
    ASSERT_EQ(schema_.comparisons.size(), 2u);
    EXPECT_EQ(schema_.comparisons[0].bits, 2);  // four levels
    EXPECT_EQ(schema_.comparisons[0].shift, 0);
    EXPECT_EQ(schema_.comparisons[1].bits, 1);  // two levels
    EXPECT_EQ(schema_.comparisons[1].shift, 2);
    EXPECT_EQ(schema_.GammaWidth(), 3);
    // The name defaults to the first column.
    EXPECT_EQ(schema_.comparisons[0].name, "surname");
}

TEST_F(ComparisonConfig, RequiresATrailingElseSoEveryPairLandsSomewhere) {
    EXPECT_FALSE(Parse(R"([{"columns":["surname"],
        "levels":[{"type":"null"},{"type":"exact"}]}])"));
    EXPECT_NE(error_.find("must end with an \"else\""), std::string::npos);
}

TEST_F(ComparisonConfig, RejectsLevelsAfterElseBecauseTheyCanNeverFire) {
    EXPECT_FALSE(Parse(R"([{"columns":["surname"],
        "levels":[{"type":"else"},{"type":"exact"},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("can never fire"), std::string::npos);
}

TEST_F(ComparisonConfig, RejectsALevelTheColumnTypeCannotSupport) {
    EXPECT_FALSE(Parse(R"([{"columns":["dob"],
        "levels":[{"type":"jaro_winkler","threshold":0.9},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("cannot read"), std::string::npos);
    // A double never agrees exactly in a useful way.
    EXPECT_FALSE(Parse(R"([{"columns":["lat"],
        "levels":[{"type":"exact"},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("cannot read"), std::string::npos);
}

TEST_F(ComparisonConfig, GeoNeedsExactlyTwoColumns) {
    EXPECT_FALSE(Parse(R"([{"columns":["lat"],
        "levels":[{"type":"geo_within","threshold":5},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("reads 2 column(s)"), std::string::npos);
    EXPECT_TRUE(Parse(R"([{"name":"where","columns":["lat","lon"],
        "levels":[{"type":"null"},{"type":"geo_within","threshold":5},
                  {"type":"else"}]}])"))
        << error_;
}

TEST_F(ComparisonConfig, RejectsMixedColumnTypesInOneComparison) {
    EXPECT_FALSE(Parse(R"([{"columns":["lat","dob"],
        "levels":[{"type":"geo_within","threshold":5},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("mixes column types"), std::string::npos);
}

TEST_F(ComparisonConfig, RejectsAnUndeclaredColumn) {
    EXPECT_FALSE(Parse(R"([{"columns":["nickname"],
        "levels":[{"type":"exact"},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("not declared"), std::string::npos);
}

TEST_F(ComparisonConfig, RequiresAThresholdWhereTheLevelNeedsOne) {
    EXPECT_FALSE(Parse(R"([{"columns":["surname"],
        "levels":[{"type":"levenshtein"},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("numeric \"threshold\""), std::string::npos);
}

// The packed pattern is a uint32; a configuration that overflows it has to be
// refused rather than silently truncated.
TEST_F(ComparisonConfig, RefusesAPatternWiderThanAUint32) {
    std::string one = R"({"columns":["surname"],"levels":[{"type":"null"},
        {"type":"exact"},{"type":"levenshtein","threshold":1},
        {"type":"jaro_winkler","threshold":0.9},{"type":"else"}]})";
    std::string many = "[";
    for (int i = 0; i < 11; ++i) {  // 11 comparisons x 3 bits = 33
        if (i > 0) many += ",";
        many += one;
    }
    many += "]";
    EXPECT_FALSE(Parse(many));
    EXPECT_NE(error_.find("33 bits"), std::string::npos);
}

TEST(SchemaTest, DoublesCarryNoTermFrequencies) {
    EXPECT_FALSE(cpplink::HasTermFrequencies(cpplink::ColumnType::kDouble));
    EXPECT_TRUE(cpplink::HasTermFrequencies(cpplink::ColumnType::kString));
    EXPECT_TRUE(cpplink::HasTermFrequencies(cpplink::ColumnType::kDate));
    EXPECT_TRUE(cpplink::HasTermFrequencies(cpplink::ColumnType::kStringList));
}

}  // namespace
