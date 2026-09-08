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
            {"name":"lon","type":"double"},
            {"name":"aliases","type":"string_list"}],"comparisons":)" +
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

// The one level whose two columns have different types, so the check has to see
// the whole signature rather than one type at a time.
TEST_F(ComparisonConfig, ListContainsReadsAScalarAgainstAList) {
    EXPECT_TRUE(Parse(R"([{"name":"nickname","columns":["surname","aliases"],
        "levels":[{"type":"null"},{"type":"list_contains"},{"type":"else"}]}])"))
        << error_;
    // The scalar column comes first: reversed, the level would be asking whether
    // a list is an element of a string.
    EXPECT_FALSE(Parse(R"([{"columns":["aliases","surname"],
        "levels":[{"type":"list_contains"},{"type":"else"}]}])"));
    // Two strings is a different question -- equality -- and two lists is
    // list_overlap. Neither is this level.
    EXPECT_FALSE(Parse(R"([{"columns":["surname","surname"],
        "levels":[{"type":"list_contains"},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("cannot read"), std::string::npos);
    EXPECT_FALSE(Parse(R"([{"columns":["aliases"],
        "levels":[{"type":"list_contains"},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("cannot read"), std::string::npos);
}

// The pairwise levels read one list column, like the set-valued ones beside them:
// what is different is the relation, not the shape of the columns.
TEST_F(ComparisonConfig, PairwiseLevelsReadAListColumn) {
    ASSERT_TRUE(Parse(R"([{"columns":["aliases"],
        "levels":[{"type":"null"},{"type":"list_levenshtein","threshold":1},
                  {"type":"list_jaro_winkler","threshold":0.85},
                  {"type":"else"}]}])"))
        << error_;
    // A description reads back as the configuration that produced it, which is
    // also what keeps it inside the label column every report prints.
    EXPECT_EQ(schema_.comparisons[0].levels[1].Describe(), "list_levenshtein <= 1");
    EXPECT_EQ(schema_.comparisons[0].levels[2].Describe(), "list_jaro_winkler >= 0.85");
    // A scalar column has no cross product to take the closest pair of, and the
    // level that compares two strings is levenshtein itself.
    EXPECT_FALSE(Parse(R"([{"columns":["surname"],
        "levels":[{"type":"list_levenshtein","threshold":1},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("cannot read"), std::string::npos);
}

// The fuzzy membership levels read the same two columns list_contains does, so
// they are checked the same way -- and unlike it, they need a threshold.
TEST_F(ComparisonConfig, FuzzyMembershipReadsTheSameShapeAsListContains) {
    ASSERT_TRUE(Parse(R"([{"name":"nickname","columns":["surname","aliases"],
        "levels":[{"type":"null"},{"type":"list_contains"},
                  {"type":"contains_levenshtein","threshold":1},
                  {"type":"contains_jaro_winkler","threshold":0.9},
                  {"type":"else"}]}])"))
        << error_;
    EXPECT_EQ(schema_.comparisons[0].levels[2].Describe(), "contains_levenshtein <= 1");
    EXPECT_EQ(schema_.comparisons[0].levels[3].Describe(),
              "contains_jaro_winkler >= 0.90");
    // One column is not the shape: there is no list for the value to be near.
    EXPECT_FALSE(Parse(R"([{"columns":["surname"],
        "levels":[{"type":"contains_levenshtein","threshold":1},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("cannot read"), std::string::npos);
    // Reversed, the level would be asking whether a list is near a string.
    EXPECT_FALSE(Parse(R"([{"columns":["aliases","surname"],
        "levels":[{"type":"contains_jaro_winkler","threshold":0.9},
                  {"type":"else"}]}])"));
    // And a metric level with no threshold is a level with no meaning.
    EXPECT_FALSE(Parse(R"([{"columns":["surname","aliases"],
        "levels":[{"type":"contains_levenshtein"},{"type":"else"}]}])"));
    EXPECT_NE(error_.find("threshold"), std::string::npos);
}

// Membership is a yes or no: there is no threshold to give it, and offering one
// would suggest it could be tuned.
TEST_F(ComparisonConfig, ListContainsTakesNoThreshold) {
    ASSERT_TRUE(Parse(R"([{"columns":["surname","aliases"],
        "levels":[{"type":"list_contains","threshold":2},{"type":"else"}]}])"))
        << error_;
    EXPECT_EQ(schema_.comparisons[0].levels[0].threshold, 0.0);
    EXPECT_EQ(schema_.comparisons[0].levels[0].Describe(), "value in list");
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
