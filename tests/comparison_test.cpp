// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/comparison.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include "cpplink/schema.hpp"

namespace {

constexpr const char* kConfig = R"({
  "columns": [
    {"name": "surname", "type": "string"},
    {"name": "dob", "type": "date"},
    {"name": "latitude", "type": "double"},
    {"name": "longitude", "type": "double"},
    {"name": "tokens", "type": "string_list"}
  ],
  "comparisons": [
    {"name": "surname", "columns": ["surname"], "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "levenshtein", "threshold": 1},
      {"type": "jaro_winkler", "threshold": 0.85},
      {"type": "else"}]},
    {"name": "dob", "columns": ["dob"], "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "date_within", "threshold": 3},
      {"type": "else"}]},
    {"name": "location", "columns": ["latitude", "longitude"], "levels": [
      {"type": "null"},
      {"type": "geo_within", "threshold": 5},
      {"type": "else"}]},
    {"name": "tokens", "columns": ["tokens"], "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "list_jaccard", "threshold": 0.5},
      {"type": "list_overlap", "threshold": 1},
      {"type": "else"}]}
  ]
})";

// Rows, in order:
//   0 baseline               3 dob three days out, 3 km away, one shared token
//   1 identical to row 0     4 all columns null / empty
//   2 one-character typo     5 unrelated
class Fixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kConfig, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        const uint32_t smith = surname.dict.Intern("smith");
        const uint32_t smyth = surname.dict.Intern("smyth");
        const uint32_t smithers = surname.dict.Intern("smithe");
        const uint32_t other = surname.dict.Intern("kowalczyk");
        surname.ids = {smith, smith, smyth, smithers, cpplink::kNullId, other};

        auto& dob = std::get<cpplink::DateColumn>(store_->mutable_column(1));
        dob.values = {10000, 10000, 10000, 10003, cpplink::kNullDate, 500};

        const double nan = std::numeric_limits<double>::quiet_NaN();
        auto& lat = std::get<cpplink::DoubleColumn>(store_->mutable_column(2));
        auto& lon = std::get<cpplink::DoubleColumn>(store_->mutable_column(3));
        lat.values = {-33.8688, -33.8688, -33.8688, -33.8957, nan, -37.8136};
        lon.values = {151.2093, 151.2093, 151.2093, 151.2093, nan, 144.9631};

        auto& tokens = std::get<cpplink::StringListColumn>(store_->mutable_column(4));
        const uint32_t king = tokens.dict.Intern("king");
        const uint32_t street = tokens.dict.Intern("street");
        const uint32_t west = tokens.dict.Intern("west");
        const uint32_t rue = tokens.dict.Intern("rue");
        // Sorted per row, as the loader guarantees.
        tokens.ids = {king, street, king, street, king, street, king, king, west, rue};
        tokens.offsets = {0, 2, 4, 6, 7, 7, 10};

        store_->set_num_records(6);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
};

TEST_F(Fixture, IdenticalRecordsHitTheExactLevelEverywhere) {
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 1), 1);  // surname exact
    EXPECT_EQ(comparisons_.EvaluateOne(1, 0, 1), 1);  // dob exact
    EXPECT_EQ(comparisons_.EvaluateOne(2, 0, 1), 1);  // within 5 km
    EXPECT_EQ(comparisons_.EvaluateOne(3, 0, 1), 1);  // tokens exact
}

TEST_F(Fixture, LevelsAreEvaluatedTopDownAndTheFirstHitWins) {
    // "smith" vs "smyth" is one substitution, so it stops at levenshtein and never
    // reaches the jaro_winkler level below it, even though that would also fire.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 2), 2);
    // "smith" vs "smithe" is also one edit.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 3), 2);
    // Nothing matches "kowalczyk".
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 5), 4);
}

TEST_F(Fixture, ThresholdsAreInclusive) {
    EXPECT_EQ(comparisons_.EvaluateOne(1, 0, 3), 2);  // 3 days, threshold 3
    EXPECT_EQ(comparisons_.EvaluateOne(1, 0, 5), 3);  // far apart, else
}

TEST_F(Fixture, GeoLevelReadsBothColumnsAsOneComparison) {
    EXPECT_EQ(comparisons_.EvaluateOne(2, 0, 3), 1);  // ~3 km
    EXPECT_EQ(comparisons_.EvaluateOne(2, 0, 5), 2);  // Sydney to Melbourne
}

// A null is not a value: it must never agree with anything, including another null.
TEST_F(Fixture, NullsAgreeWithNothingIncludingOtherNulls) {
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 4), 0);
    EXPECT_EQ(comparisons_.EvaluateOne(1, 0, 4), 0);
    EXPECT_EQ(comparisons_.EvaluateOne(2, 0, 4), 0);
    EXPECT_EQ(comparisons_.EvaluateOne(3, 0, 4), 0);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 4, 4), 0);  // a row against itself
}

TEST_F(Fixture, ListLevelsUseIntersectionAndJaccard) {
    EXPECT_EQ(comparisons_.EvaluateOne(3, 0, 1), 1);  // {king,street} identical
    // {king,street} against {king}: intersection 1, Jaccard 1/2 -> the 0.5 level.
    EXPECT_EQ(comparisons_.EvaluateOne(3, 0, 3), 2);
    // {king,street} against {king,west,rue}: Jaccard 1/4, so overlap >= 1 instead.
    EXPECT_EQ(comparisons_.EvaluateOne(3, 0, 5), 3);
}

TEST_F(Fixture, PackedPatternRoundTripsThroughEveryComparison) {
    EXPECT_EQ(comparisons_.Width(), 3 + 2 + 2 + 3);
    for (uint64_t a = 0; a < store_->NumRecords(); ++a) {
        for (uint64_t b = 0; b < store_->NumRecords(); ++b) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            for (size_t c = 0; c < comparisons_.Size(); ++c) {
                EXPECT_EQ(comparisons_.LevelOf(gamma, c),
                          comparisons_.EvaluateOne(c, a, b))
                    << "rows " << a << "," << b << " comparison " << c;
            }
        }
    }
}

TEST_F(Fixture, ComparisonsOccupyDisjointBitRanges) {
    uint32_t seen = 0;
    for (size_t c = 0; c < comparisons_.Size(); ++c) {
        const cpplink::BoundComparison& bound = comparisons_.at(c);
        const uint32_t mask = ((1u << bound.bits) - 1u) << bound.shift;
        EXPECT_EQ(seen & mask, 0u) << "comparison " << c << " overlaps an earlier one";
        seen |= mask;
    }
}

}  // namespace
