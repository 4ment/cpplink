// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "cpplink/inspect.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/schema.hpp"

namespace {

constexpr const char* kSampleSchema = R"({
  "unique_id": "id",
  "columns": [
    {"name": "first_name", "type": "string"},
    {"name": "last_name", "type": "string"},
    {"name": "dob", "type": "date"},
    {"name": "email", "type": "string"},
    {"name": "phone", "type": "string"},
    {"name": "postcode", "type": "string"},
    {"name": "latitude", "type": "double"},
    {"name": "longitude", "type": "double"},
    {"name": "address_tokens", "type": "string_list"}
  ]
})";

class RoundTrip : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / "cpplink_roundtrip";
        std::filesystem::create_directories(dir_);
        data_ = (dir_ / "sample.parquet").string();
        truth_ = (dir_ / "sample.truth.csv").string();
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::filesystem::path dir_;
    std::string data_;
    std::string truth_;
};

TEST_F(RoundTrip, GeneratedFileLoadsBackWithTheDeclaredShape) {
    cpplink::SampleOptions options;
    options.rows = 5000;
    options.row_group_size = 1000;
    options.duplicate_rate = 0.1;
    options.truth_path = truth_;

    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kSampleSchema, &schema, &error)) << error;

    cpplink::RecordStore store(schema);
    cpplink::LoadStats stats;
    ASSERT_TRUE(cpplink::LoadParquet(data_, schema, &store, &stats, &error)) << error;

    EXPECT_EQ(store.NumRecords(), 5000u);
    EXPECT_EQ(stats.row_groups, 5);
    EXPECT_EQ(store.NumColumns(), 9u);
    EXPECT_EQ(store.ids().Get(0), "r0");

    // Email is near-unique by construction; first names come from a small Zipf
    // vocabulary. If these ever converge, the sample stopped being representative.
    const uint32_t emails = store.DistinctValues(3);
    const uint32_t first_names = store.DistinctValues(0);
    EXPECT_GT(emails, 3000u);
    EXPECT_LT(first_names, emails);

    // Corruption drops fields, so the nullable columns must actually contain nulls.
    EXPECT_GT(store.NullCount(3), 0u);  // email
    EXPECT_EQ(store.NullCount(6), 0u);  // latitude is never dropped
    EXPECT_GT(store.Memory().Total(), 0u);
}

TEST_F(RoundTrip, ListValuesAreSortedAndDeduplicatedPerRow) {
    cpplink::SampleOptions options;
    options.rows = 500;
    options.row_group_size = 500;
    options.truth_path.clear();

    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kSampleSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(data_, schema, &store, nullptr, &error)) << error;

    const auto& tokens = std::get<cpplink::StringListColumn>(store.column(8));
    ASSERT_EQ(tokens.offsets.size(), 501u);
    for (size_t row = 0; row + 1 < tokens.offsets.size(); ++row) {
        for (uint64_t i = tokens.offsets[row] + 1; i < tokens.offsets[row + 1]; ++i) {
            EXPECT_LT(tokens.ids[i - 1], tokens.ids[i])
                << "row " << row << " is not strictly increasing";
        }
    }
}

TEST_F(RoundTrip, PlantedDuplicatesAreRecordedAsGroundTruth) {
    cpplink::SampleOptions options;
    options.rows = 4000;
    options.row_group_size = 4000;
    options.duplicate_rate = 0.2;
    options.truth_path = truth_;

    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    std::ifstream in(truth_);
    ASSERT_TRUE(in.good());
    std::string line;
    ASSERT_TRUE(std::getline(in, line));
    EXPECT_EQ(line, "id_a,id_b");
    size_t pairs = 0;
    while (std::getline(in, line)) {
        if (!line.empty()) ++pairs;
    }
    // Planting is random, so assert the order of magnitude rather than an exact count.
    EXPECT_GT(pairs, 600u);
    EXPECT_LT(pairs, 1200u);
}

TEST_F(RoundTrip, MissingColumnIsReportedByName) {
    cpplink::SampleOptions options;
    options.rows = 100;
    options.row_group_size = 100;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(
        R"({"columns":[{"name":"not_there","type":"string"}]})", &schema, &error));
    cpplink::RecordStore store(schema);
    EXPECT_FALSE(cpplink::LoadParquet(data_, schema, &store, nullptr, &error));
    EXPECT_NE(error.find("not_there"), std::string::npos);
}

TEST_F(RoundTrip, WrongDeclaredTypeIsReportedByColumn) {
    cpplink::SampleOptions options;
    options.rows = 100;
    options.row_group_size = 100;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(
        R"({"columns":[{"name":"latitude","type":"string"}]})", &schema, &error));
    cpplink::RecordStore store(schema);
    EXPECT_FALSE(cpplink::LoadParquet(data_, schema, &store, nullptr, &error));
    EXPECT_NE(error.find("latitude"), std::string::npos);
    EXPECT_NE(error.find("expected a string column"), std::string::npos);
}

}  // namespace
