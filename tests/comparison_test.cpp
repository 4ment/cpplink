// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/comparison.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <variant>
#include <vector>

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

// The signature filter may only skip work the metric would have rejected anyway.
// Asserting that on a handful of hand-picked names proves nothing, so this rebuilds
// the same store over a population wide enough for the masks to differ in every
// interesting way and compares the packed patterns pair for pair.
TEST(SignatureFilterTest, ChangesNoPattern) {
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(kConfig, &schema, &error)) << error;
    cpplink::RecordStore store(schema);

    std::mt19937_64 rng(20260904);
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
    std::uniform_int_distribution<size_t> length(3, 12);
    std::uniform_int_distribution<int> corrupt(0, 3);
    constexpr uint64_t kRows = 300;

    auto& surname = std::get<cpplink::StringColumn>(store.mutable_column(0));
    std::vector<std::string> values;
    for (uint64_t row = 0; row < kRows; ++row) {
        std::string value;
        // Every fourth row is a corruption of an earlier one, so the population
        // carries near misses as well as unrelated pairs -- the filter is only
        // interesting where it has to decide between them.
        if (!values.empty() && corrupt(rng) == 0) {
            value = values[rng() % values.size()];
            if (!value.empty()) value[rng() % value.size()] = alphabet[pick(rng)];
        } else {
            for (size_t i = length(rng); i > 0; --i) value.push_back(alphabet[pick(rng)]);
        }
        values.push_back(value);
        surname.ids.push_back(surname.dict.Intern(value));
    }

    auto& dob = std::get<cpplink::DateColumn>(store.mutable_column(1));
    auto& lat = std::get<cpplink::DoubleColumn>(store.mutable_column(2));
    auto& lon = std::get<cpplink::DoubleColumn>(store.mutable_column(3));
    auto& tokens = std::get<cpplink::StringListColumn>(store.mutable_column(4));
    const uint32_t token = tokens.dict.Intern("king");
    tokens.offsets.push_back(0);
    for (uint64_t row = 0; row < kRows; ++row) {
        dob.values.push_back(static_cast<int32_t>(10000 + row % 7));
        lat.values.push_back(-33.8688);
        lon.values.push_back(151.2093);
        tokens.ids.push_back(token);
        tokens.offsets.push_back(tokens.ids.size());
    }
    store.set_num_records(kRows);
    store.Finalize();

    cpplink::ComparisonSet filtered;
    cpplink::ComparisonSet plain;
    ASSERT_TRUE(filtered.Bind(schema, store, &error, true)) << error;
    ASSERT_TRUE(plain.Bind(schema, store, &error, false)) << error;
    EXPECT_GT(filtered.SignatureBytes(), 0u);
    EXPECT_EQ(plain.SignatureBytes(), 0u);

    for (uint64_t a = 0; a < kRows; ++a) {
        for (uint64_t b = a + 1; b < kRows; ++b) {
            ASSERT_EQ(filtered.Evaluate(a, b), plain.Evaluate(a, b))
                << "'" << values[a] << "' vs '" << values[b] << "'";
        }
    }
}

}  // namespace
