// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

#include <gtest/gtest.h>
#include <unistd.h>

#include "cpplink/app.hpp"
#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/inspect.hpp"
#include "cpplink/model.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/recall.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/string_metrics.hpp"

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
        // One directory per process, because the test binary is run once per case:
        // a fixed name has concurrent cases of this fixture writing the same files
        // and deleting the directory under one another.
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_roundtrip_" + std::to_string(::getpid()));
        std::filesystem::remove_all(dir_);
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

// A schema whose one comparison bridges a scalar column and a list one. The two
// dictionaries are interned independently by the loader, so the same text holds a
// different id in each -- which is the thing the alias map exists to absorb and
// the thing only a real load exercises.
constexpr const char* kAliasSchema = R"({
  "unique_id": "id",
  "columns": [
    {"name": "first_name", "type": "string"},
    {"name": "address_tokens", "type": "string_list"}
  ],
  "comparisons": [
    {"name": "alias", "columns": ["first_name", "address_tokens"], "levels": [
      {"type": "null"}, {"type": "list_contains"}, {"type": "else"}]}
  ]
})";

TEST_F(RoundTrip, ListContainsAlignsTwoDictionariesOverLoadedData) {
    cpplink::SampleOptions options;
    options.rows = 2000;
    options.row_group_size = 500;
    options.truth_path.clear();

    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kAliasSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(data_, schema, &store, nullptr, &error)) << error;

    cpplink::ComparisonSet comparisons;
    ASSERT_TRUE(comparisons.Bind(schema, store, &error)) << error;

    const auto& names = std::get<cpplink::StringColumn>(store.column(0));
    const auto& tokens = std::get<cpplink::StringListColumn>(store.column(1));

    // The same question answered from the text, which is what the interned form
    // is supposed to be a faster spelling of.
    const auto held = [&](uint64_t row, uint64_t other) {
        const uint32_t id = names.ids[other];
        if (id == cpplink::kNullId) return false;
        const std::string_view name = names.dict.Value(id);
        for (uint64_t i = tokens.offsets[row]; i < tokens.offsets[row + 1]; ++i) {
            if (tokens.dict.Value(tokens.ids[i]) == name) return true;
        }
        return false;
    };

    uint64_t fired = 0;
    for (uint64_t a = 0; a < 400; ++a) {
        for (uint64_t b = a + 1; b < 400; ++b) {
            const bool contains = held(a, b) || held(b, a);
            const bool null = names.ids[a] == cpplink::kNullId &&
                              tokens.offsets[a + 1] == tokens.offsets[a];
            const uint8_t expected = null ? 0
                                     : names.ids[b] == cpplink::kNullId &&
                                             tokens.offsets[b + 1] == tokens.offsets[b]
                                         ? 0
                                     : contains ? 1
                                                : 2;
            ASSERT_EQ(comparisons.EvaluateOne(0, a, b), expected) << a << "," << b;
            if (expected == 1) ++fired;
        }
    }
    // The vocabularies are drawn from one syllable generator, so names and street
    // words collide often enough for this to be testing something.
    EXPECT_GT(fired, 0u);
}

// The pairwise levels over a list column that a real load interned, checked
// against the same question asked of the text. What only a load exercises is the
// signature table built over the *list* column's dictionary: the unit fixtures
// hold a handful of values, and the bound is only interesting over a vocabulary
// wide enough for the masks to differ in every way they can.
constexpr const char* kPairwiseSchema = R"({
  "unique_id": "id",
  "columns": [
    {"name": "address_tokens", "type": "string_list"}
  ],
  "comparisons": [
    {"name": "tokens", "columns": ["address_tokens"], "levels": [
      {"type": "null"},
      {"type": "list_levenshtein", "threshold": 1},
      {"type": "list_jaro_winkler", "threshold": 0.9},
      {"type": "else"}]}
  ]
})";

TEST_F(RoundTrip, PairwiseLevelsAgreeWithTheTextOverLoadedData) {
    cpplink::SampleOptions options;
    options.rows = 2000;
    options.row_group_size = 500;
    options.truth_path.clear();

    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kPairwiseSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(data_, schema, &store, nullptr, &error)) << error;

    cpplink::ComparisonSet comparisons;
    ASSERT_TRUE(comparisons.Bind(schema, store, &error)) << error;

    const auto& tokens = std::get<cpplink::StringListColumn>(store.column(0));
    // The cross product, spelled out over the text and using the metrics
    // directly: no short-circuit, no signature bound, no interning.
    const auto closest = [&](uint64_t a, uint64_t b, int* edits, double* similarity) {
        *edits = -1;
        *similarity = -1.0;
        for (uint64_t i = tokens.offsets[a]; i < tokens.offsets[a + 1]; ++i) {
            for (uint64_t j = tokens.offsets[b]; j < tokens.offsets[b + 1]; ++j) {
                const std::string_view left = tokens.dict.Value(tokens.ids[i]);
                const std::string_view right = tokens.dict.Value(tokens.ids[j]);
                const int distance = cpplink::BoundedLevenshtein(left, right, 1);
                if (*edits < 0 || distance < *edits) *edits = distance;
                *similarity = std::max(*similarity, cpplink::JaroWinkler(left, right));
            }
        }
    };

    uint64_t levenshtein = 0;
    uint64_t jaro = 0;
    for (uint64_t a = 0; a < 250; ++a) {
        for (uint64_t b = a + 1; b < 250; ++b) {
            const bool empty = tokens.offsets[a + 1] == tokens.offsets[a] ||
                               tokens.offsets[b + 1] == tokens.offsets[b];
            int edits = 0;
            double similarity = 0.0;
            if (!empty) closest(a, b, &edits, &similarity);
            const uint8_t expected = empty               ? 0
                                     : edits <= 1        ? 1
                                     : similarity >= 0.9 ? 2
                                                         : 3;
            ASSERT_EQ(comparisons.EvaluateOne(0, a, b), expected) << a << "," << b;
            if (expected == 1) ++levenshtein;
            if (expected == 2) ++jaro;
        }
    }
    // Both levels have to be reached, or the loop asserted nothing about them.
    EXPECT_GT(levenshtein, 0u);
    EXPECT_GT(jaro, 0u);
}

// The fuzzy membership levels over two dictionaries a real load interned
// separately, checked against the same question asked of the text. This is the
// one shape where a level reads a value id of one dictionary against value ids of
// another, so both signature tables and the alias map are in play at once.
constexpr const char* kNearContainsSchema = R"({
  "unique_id": "id",
  "columns": [
    {"name": "first_name", "type": "string"},
    {"name": "address_tokens", "type": "string_list"}
  ],
  "comparisons": [
    {"name": "alias", "columns": ["first_name", "address_tokens"], "levels": [
      {"type": "null"},
      {"type": "list_contains"},
      {"type": "contains_levenshtein", "threshold": 1},
      {"type": "contains_jaro_winkler", "threshold": 0.92},
      {"type": "else"}]}
  ]
})";

TEST_F(RoundTrip, FuzzyMembershipAgreesWithTheTextOverLoadedData) {
    cpplink::SampleOptions options;
    options.rows = 2000;
    options.row_group_size = 500;
    options.truth_path.clear();

    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kNearContainsSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(data_, schema, &store, nullptr, &error)) << error;

    cpplink::ComparisonSet comparisons;
    ASSERT_TRUE(comparisons.Bind(schema, store, &error)) << error;

    const auto& names = std::get<cpplink::StringColumn>(store.column(0));
    const auto& tokens = std::get<cpplink::StringListColumn>(store.column(1));

    // One direction, spelled out over the text: no alias map, no signature bound,
    // no interning. `edits` and `similarity` are the best this direction reaches.
    const auto direction = [&](uint64_t value_row, uint64_t list_row, int* edits,
                               double* similarity) {
        const uint32_t id = names.ids[value_row];
        if (id == cpplink::kNullId) return;
        const std::string_view text = names.dict.Value(id);
        for (uint64_t i = tokens.offsets[list_row]; i < tokens.offsets[list_row + 1];
             ++i) {
            const std::string_view other = tokens.dict.Value(tokens.ids[i]);
            const int distance = cpplink::BoundedLevenshtein(text, other, 1);
            if (*edits < 0 || distance < *edits) *edits = distance;
            *similarity = std::max(*similarity, cpplink::JaroWinkler(text, other));
        }
    };

    uint64_t contains = 0;
    uint64_t levenshtein = 0;
    uint64_t jaro = 0;
    for (uint64_t a = 0; a < 250; ++a) {
        for (uint64_t b = a + 1; b < 250; ++b) {
            const bool null_a = names.ids[a] == cpplink::kNullId &&
                                tokens.offsets[a + 1] == tokens.offsets[a];
            const bool null_b = names.ids[b] == cpplink::kNullId &&
                                tokens.offsets[b + 1] == tokens.offsets[b];
            int edits = -1;
            double similarity = -1.0;
            if (!null_a && !null_b) {
                direction(a, b, &edits, &similarity);
                direction(b, a, &edits, &similarity);
            }
            const uint8_t expected = (null_a || null_b)   ? 0
                                     : edits == 0         ? 1
                                     : edits == 1         ? 2
                                     : similarity >= 0.92 ? 3
                                                          : 4;
            ASSERT_EQ(comparisons.EvaluateOne(0, a, b), expected) << a << "," << b;
            if (expected == 1) ++contains;
            if (expected == 2) ++levenshtein;
            if (expected == 3) ++jaro;
        }
    }
    // Each level has to be reached, or the loop asserted nothing about it.
    EXPECT_GT(contains, 0u);
    EXPECT_GT(levenshtein, 0u);
    EXPECT_GT(jaro, 0u);
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

// The unblocked path, end to end and through the command line, on an input small
// enough that enumerating every pair costs less than the plan that would avoid
// them. The schema carries no "blocking" section at all: --all-pairs is what makes
// that section optional, and a link run is where it pays, because the admissible
// space is the cross product rather than a triangle.
TEST_F(RoundTrip, UnblockedLinkRunGoesEndToEndWithNoBlockingSection) {
    cpplink::SampleOptions options;
    options.rows = 2000;
    options.duplicate_rate = 0.25;
    options.truth_path = truth_;
    options.link_path = link_;
    std::string error;
    ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;

    const std::string schema_path = (dir_ / "unblocked.json").string();
    {
        std::string json(kSampleSchema);
        const size_t blocking = json.find(",\n  \"blocking\"");
        ASSERT_NE(blocking, std::string::npos);
        json = json.substr(0, blocking) + "\n}";
        std::ofstream file(schema_path);
        file << json;
    }

    // Without the flag there is nothing to run and the message says what to pass.
    {
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(
            cpplink::Run({"explain-blocking", "--schema", schema_path, data_}, out, err),
            1);
        EXPECT_NE(err.str().find("--all-pairs"), std::string::npos) << err.str();
    }

    const std::string model_path = (dir_ / "model.json").string();
    {
        std::ostringstream out;
        std::ostringstream err;
        ASSERT_EQ(cpplink::Run({"estimate", "--schema", schema_path, "--all-pairs",
                                "--out", model_path, data_, link_},
                               out, err),
                  0)
            << err.str();
    }
    cpplink::Model model;
    ASSERT_TRUE(cpplink::LoadModel(model_path, &model, &error)) << error;
    // Lambda is a lower bound only because blocking misses matches. This session
    // enumerated every admissible pair, so it missed none and the report must not
    // claim otherwise.
    EXPECT_EQ(model.lambda_basis.find("lower bound"), std::string::npos)
        << model.lambda_basis;
    EXPECT_GT(model.lambda, 0.0);

    const std::string edges = (dir_ / "edges").string();
    {
        std::ostringstream out;
        std::ostringstream err;
        ASSERT_EQ(cpplink::Run(
                      {"predict", "--schema", schema_path, "--model", model_path,
                       "--all-pairs", "--threshold", "10", "--out", edges, data_, link_},
                      out, err),
                  0)
            << err.str();
        EXPECT_NE(out.str().find("link"), std::string::npos) << out.str();
    }

    cpplink::Schema schema;
    ASSERT_TRUE(cpplink::ParseSchema(kSampleSchema, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(
        cpplink::LoadParquetFiles({data_, link_}, schema, &store, nullptr, &error))
        << error;
    cpplink::TruthPairs truth;
    ASSERT_TRUE(cpplink::LoadTruthPairs(truth_, store, &truth, &error)) << error;

    cpplink::Schema unblocked = schema;
    unblocked.blocking.clear();
    cpplink::BlockingSpec spec;
    spec.kind = cpplink::SourceKind::kAllPairs;
    spec.name = "all pairs";
    unblocked.blocking.push_back(spec);
    cpplink::BlockingPlan plan;
    ASSERT_TRUE(plan.Build(unblocked, store, cpplink::PairMode::kCrossDataset, &error))
        << error;
    // Blocking recall is one by construction: there is no pair it can lose.
    for (const auto& pair : truth.rows) {
        EXPECT_TRUE(plan.ProducedByAny(pair.first, pair.second))
            << store.ids().Get(pair.first) << "," << store.ids().Get(pair.second);
    }
    const uint64_t first = store.DatasetEnd(0) - store.DatasetStart(0);
    EXPECT_EQ(plan.CountPairs(0), first * (store.NumRecords() - first));
}

}  // namespace
