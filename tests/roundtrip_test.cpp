// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/inspect.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/recall.hpp"
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
  ],
  "comparisons": [
    {"name": "last_name", "columns": ["last_name"], "levels": [
      {"type": "null"}, {"type": "exact"},
      {"type": "jaro_winkler", "threshold": 0.85}, {"type": "else"}]},
    {"name": "first_name", "columns": ["first_name"], "levels": [
      {"type": "null"}, {"type": "exact"},
      {"type": "jaro_winkler", "threshold": 0.85}, {"type": "else"}]},
    {"name": "dob", "columns": ["dob"], "levels": [
      {"type": "null"}, {"type": "exact"},
      {"type": "date_within", "threshold": 2}, {"type": "else"}]},
    {"name": "postcode", "columns": ["postcode"], "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "location", "columns": ["latitude", "longitude"], "levels": [
      {"type": "null"}, {"type": "geo_within", "threshold": 1.0},
      {"type": "else"}]},
    {"name": "address", "columns": ["address_tokens"], "levels": [
      {"type": "null"}, {"type": "exact"},
      {"type": "list_jaccard", "threshold": 0.6}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "email"},
    {"type": "exact_value", "column": "dob"}
  ]
})";

class RoundTrip : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / "cpplink_roundtrip";
        std::filesystem::create_directories(dir_);
        data_ = (dir_ / "sample.parquet").string();
        link_ = (dir_ / "sample.b.parquet").string();
        truth_ = (dir_ / "sample.truth.csv").string();
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::filesystem::path dir_;
    std::string data_;
    std::string link_;
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

// A planted duplicate is a corruption of another record, so the two rows must
// still resemble each other. They did not: a duplicate could pick an original that
// was itself a duplicate, and corrupting *that row's index* regenerated a record
// the file never held, so the pair shared nothing and the truth file claimed it
// anyway. That put roughly 8% unfindable pairs into the ground truth and made
// blocking recall read almost eight points worse than it was.
TEST_F(RoundTrip, EveryPlantedPairActuallyResemblesItself) {
    cpplink::SampleOptions options;
    options.rows = 4000;
    options.duplicate_rate = 0.25;  // high, so chains of copies are common
    options.truth_path = truth_;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kSampleSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(data_, schema, &store, nullptr, &error)) << error;
    cpplink::ComparisonSet comparisons;
    ASSERT_TRUE(comparisons.Bind(schema, store, &error)) << error;
    cpplink::TruthPairs truth;
    ASSERT_TRUE(cpplink::LoadTruthPairs(truth_, store, &truth, &error)) << error;
    ASSERT_GT(truth.rows.size(), 100u);

    uint64_t share_nothing = 0;
    for (const auto& pair : truth.rows) {
        bool anything = false;
        for (size_t c = 0; c < comparisons.Size() && !anything; ++c) {
            const uint8_t level = comparisons.EvaluateOne(c, pair.first, pair.second);
            const cpplink::LevelType type = comparisons.at(c).spec->levels[level].type;
            anything =
                type != cpplink::LevelType::kNull && type != cpplink::LevelType::kElse;
        }
        if (!anything) ++share_nothing;
    }
    EXPECT_EQ(share_nothing, 0u)
        << share_nothing << " of " << truth.rows.size()
        << " planted pairs share no column agreement at all, so they are not "
           "duplicates of one another";
}

TEST_F(RoundTrip, MissDiagnosisAccountsForEveryMissedPair) {
    cpplink::SampleOptions options;
    options.rows = 3000;
    options.duplicate_rate = 0.15;
    options.truth_path = truth_;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kSampleSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(data_, schema, &store, nullptr, &error)) << error;
    cpplink::BlockingPlan plan;
    ASSERT_TRUE(plan.Build(schema, store, &error)) << error;
    cpplink::ComparisonSet comparisons;
    ASSERT_TRUE(comparisons.Bind(schema, store, &error)) << error;
    cpplink::TruthPairs truth;
    ASSERT_TRUE(cpplink::LoadTruthPairs(truth_, store, &truth, &error)) << error;

    cpplink::MissReport report;
    cpplink::DiagnoseMisses(plan, comparisons, truth, &report);

    // The three fixes are exclusive and cover every missed pair.
    EXPECT_EQ(report.fixable_by_cap + report.fixable_by_window + report.unreachable,
              report.missed);
    // And every source has a reason recorded for every missed pair.
    for (size_t index = 0; index < plan.Size(); ++index) {
        uint64_t total = 0;
        for (const uint64_t count : report.reasons[index]) total += count;
        EXPECT_EQ(total, report.missed) << plan.at(index).name;
        // A source that produced the pair would not have been a miss.
        EXPECT_EQ(report.reasons[index][0], 0u) << plan.at(index).name;
    }

    uint64_t missed = 0;
    for (const auto& pair : truth.rows) {
        if (!plan.ProducedByAny(pair.first, pair.second)) ++missed;
    }
    EXPECT_EQ(report.missed, missed);
    EXPECT_EQ(report.truth_pairs, truth.rows.size());
}

// The link fixture: originals in the first file, every planted duplicate in the
// second. Without this there is nothing to measure the cross-dataset path
// against, and a link run that silently produced no pairs would look like a run
// that produced the right ones.
TEST_F(RoundTrip, SplitSampleRecordsOnlyPairsThatCrossTheTwoFiles) {
    cpplink::SampleOptions options;
    options.rows = 4000;
    options.duplicate_rate = 0.25;
    options.truth_path = truth_;
    options.link_path = link_;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kSampleSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    cpplink::LoadStats stats;
    ASSERT_TRUE(cpplink::LoadParquetFiles({data_, link_}, schema, &store, &stats, &error))
        << error;

    ASSERT_EQ(store.NumRecords(), 4000u);
    ASSERT_EQ(store.NumDatasets(), 2u);
    ASSERT_EQ(stats.dataset_rows.size(), 2u);
    EXPECT_EQ(stats.dataset_rows[0] + stats.dataset_rows[1], 4000u);
    // Roughly the duplicate rate, and in any case both files hold rows.
    EXPECT_GT(stats.dataset_rows[1], 500u);

    cpplink::TruthPairs truth;
    ASSERT_TRUE(cpplink::LoadTruthPairs(truth_, store, &truth, &error)) << error;
    ASSERT_GT(truth.rows.size(), 100u);
    EXPECT_EQ(truth.unresolved, 0u);
    for (const auto& pair : truth.rows) {
        EXPECT_NE(store.DatasetOf(pair.first), store.DatasetOf(pair.second))
            << store.ids().Get(pair.first) << "," << store.ids().Get(pair.second);
    }

    // And the pairs are still genuine duplicates: the split changes which row a
    // pair is recorded against, not what the two rows hold.
    cpplink::ComparisonSet comparisons;
    ASSERT_TRUE(comparisons.Bind(schema, store, &error)) << error;
    uint64_t share_nothing = 0;
    for (const auto& pair : truth.rows) {
        bool anything = false;
        for (size_t c = 0; c < comparisons.Size() && !anything; ++c) {
            const uint8_t level = comparisons.EvaluateOne(c, pair.first, pair.second);
            const cpplink::LevelType type = comparisons.at(c).spec->levels[level].type;
            anything =
                type != cpplink::LevelType::kNull && type != cpplink::LevelType::kElse;
        }
        if (!anything) ++share_nothing;
    }
    EXPECT_EQ(share_nothing, 0u);
}

// Blocking has to reach the same pairs across two files that it reaches inside
// one, or the link path is a different pipeline wearing the same name.
TEST_F(RoundTrip, LinkModeReachesThePlantedPairsAcrossTheTwoFiles) {
    cpplink::SampleOptions options;
    options.rows = 4000;
    options.duplicate_rate = 0.25;
    options.truth_path = truth_;
    options.link_path = link_;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kSampleSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(
        cpplink::LoadParquetFiles({data_, link_}, schema, &store, nullptr, &error))
        << error;
    cpplink::BlockingPlan plan;
    ASSERT_TRUE(plan.Build(schema, store, cpplink::PairMode::kCrossDataset, &error))
        << error;
    cpplink::TruthPairs truth;
    ASSERT_TRUE(cpplink::LoadTruthPairs(truth_, store, &truth, &error)) << error;

    // The claim is not an absolute recall number -- that is a property of the
    // schema's sources, and it is the recall harness's job to report it. The claim
    // is that link mode loses nothing to the split: every planted pair crosses the
    // two files, so cross-dataset blocking must reach exactly the pairs the same
    // sources would have reached with both files read as one.
    cpplink::BlockingPlan both;
    ASSERT_TRUE(both.Build(schema, store, cpplink::PairMode::kAll, &error)) << error;
    uint64_t reached = 0;
    for (const auto& pair : truth.rows) {
        const bool linked = plan.ProducedByAny(pair.first, pair.second);
        EXPECT_EQ(linked, both.ProducedByAny(pair.first, pair.second))
            << store.ids().Get(pair.first) << "," << store.ids().Get(pair.second);
        if (linked) ++reached;
    }
    EXPECT_GT(reached * 10, truth.rows.size() * 9) << "blocking has stopped working";

    // And it is cheaper, because the two inputs' own triangles are never walked.
    EXPECT_LT(plan.CountUnion(), both.CountUnion());
}

}  // namespace
