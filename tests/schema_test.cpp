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

TEST(SchemaTest, DoublesCarryNoTermFrequencies) {
    EXPECT_FALSE(cpplink::HasTermFrequencies(cpplink::ColumnType::kDouble));
    EXPECT_TRUE(cpplink::HasTermFrequencies(cpplink::ColumnType::kString));
    EXPECT_TRUE(cpplink::HasTermFrequencies(cpplink::ColumnType::kDate));
    EXPECT_TRUE(cpplink::HasTermFrequencies(cpplink::ColumnType::kStringList));
}

}  // namespace
