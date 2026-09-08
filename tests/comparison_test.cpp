// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/comparison.hpp"

#include <algorithm>
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

// The pairwise shape: one list column against itself, scored on the *closest*
// pair of elements rather than the elements the two rows share. Levels in order:
// null, exact, list_overlap, list_levenshtein, list_jaro_winkler, else -- the
// pairwise levels below the shared-element ones, because a shared element is the
// closest pair there can be.
constexpr const char* kPairwiseConfig = R"({
  "columns": [
    {"name": "names", "type": "string_list"}
  ],
  "comparisons": [
    {"name": "names", "columns": ["names"], "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "list_overlap", "threshold": 1},
      {"type": "list_levenshtein", "threshold": 1},
      {"type": "list_jaro_winkler", "threshold": 0.85},
      {"type": "else"}]}
  ]
})";

// Rows, in order:
//   0 {smith, jones}       3 {smyth, jonas}     -- a typo in each, none shared
//   1 {smith, jones}       4 {}
//   2 {smith, kowalczyk}   5 {kowalczyk}
//   6 {smithers}
class Pairwise : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kPairwiseConfig, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& names = std::get<cpplink::StringListColumn>(store_->mutable_column(0));
        const uint32_t smith = names.dict.Intern("smith");
        const uint32_t smyth = names.dict.Intern("smyth");
        const uint32_t jones = names.dict.Intern("jones");
        const uint32_t jonas = names.dict.Intern("jonas");
        const uint32_t kowalczyk = names.dict.Intern("kowalczyk");
        const uint32_t smithers = names.dict.Intern("smithers");
        // Sorted by id per row, as the loader guarantees.
        names.ids = {smith,     jones, smith, jones,     smith,
                     kowalczyk, smyth, jonas, kowalczyk, smithers};
        names.offsets = {0, 2, 4, 6, 8, 8, 9, 10};

        store_->set_num_records(7);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
};

// The point of the shape: two rows whose lists intersect in nothing still agree,
// because "smith" and "smyth" are one edit apart and "jones" and "jonas" are too.
// Every set-valued level above reads that pair as no agreement at all.
TEST_F(Pairwise, TheClosestPairDecidesTheLevel) {
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 3), 3);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 3, 0), 3);  // and it is symmetric
    // "smith" against "smithers" is three edits, so the levenshtein level cannot
    // fire and the jaro_winkler one below it does.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 6), 4);
    // Nothing in {kowalczyk} is close to anything in {smith, jones}.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 5), 5);
}

// A shared element is a pair at distance zero, which no other pair can beat, so
// the levels reading the intersection are the ones that fire.
TEST_F(Pairwise, SharedElementsStillWinAbove) {
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 1), 1);  // identical sets
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 2), 2);  // "smith" in both
}

// An empty list holds no element pair, so there is nothing for a closest pair to
// be: it is null, not a disagreement, and not an agreement with another empty.
TEST_F(Pairwise, AnEmptyListIsNull) {
    EXPECT_TRUE(comparisons_.IsNullValue(0, 4));
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 4), 0);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 4, 4), 0);
}

// LevelPossible is what Scorer::Ceiling prices a pair with, so it may never be
// false where the level fires.
TEST_F(Pairwise, LevelPossibleAgreesWithTheLevelItBounds) {
    for (uint64_t a = 0; a < store_->NumRecords(); ++a) {
        for (uint64_t b = 0; b < store_->NumRecords(); ++b) {
            const uint8_t level = comparisons_.EvaluateOne(0, a, b);
            EXPECT_TRUE(comparisons_.LevelPossible(0, level, a, b))
                << "rows " << a << "," << b;
        }
    }
}

// The same admissibility argument the scalar filter is held to, over the cross
// product instead of one pair of values: the signature bound rejects element
// pairs, so it must change no pattern.
TEST(PairwiseSignatureFilterTest, ChangesNoPattern) {
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(kPairwiseConfig, &schema, &error)) << error;
    cpplink::RecordStore store(schema);

    std::mt19937_64 rng(20260909);
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
    std::uniform_int_distribution<size_t> length(3, 12);
    std::uniform_int_distribution<size_t> cell(0, 3);
    std::uniform_int_distribution<int> corrupt(0, 2);
    constexpr uint64_t kRows = 200;

    auto& names = std::get<cpplink::StringListColumn>(store.mutable_column(0));
    std::vector<std::string> written;
    names.offsets.push_back(0);
    for (uint64_t row = 0; row < kRows; ++row) {
        std::vector<uint32_t> ids;
        for (size_t i = cell(rng); i > 0; --i) {
            std::string value;
            // A third of the elements are corruptions of one written earlier, so
            // the cross products carry near misses as well as unrelated pairs --
            // which is the only place the bound has to decide anything.
            if (!written.empty() && corrupt(rng) == 0) {
                value = written[rng() % written.size()];
                if (!value.empty()) value[rng() % value.size()] = alphabet[pick(rng)];
            } else {
                for (size_t k = length(rng); k > 0; --k) {
                    value.push_back(alphabet[pick(rng)]);
                }
            }
            written.push_back(value);
            ids.push_back(names.dict.Intern(value));
        }
        // Sorted and deduplicated, as the loader guarantees.
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (const uint32_t id : ids) names.ids.push_back(id);
        names.offsets.push_back(names.ids.size());
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
                << "rows " << a << "," << b;
        }
    }
}

// The nickname shape: a forename compared against a forename, with the other
// row's alias list as a bridge between them. Levels in order: null, exact,
// list_contains, jaro_winkler, else -- which is the ordering that makes the
// bridge worth having, since it ranks between a name that matches outright and
// one that merely looks similar.
constexpr const char* kNicknameConfig = R"({
  "columns": [
    {"name": "forename", "type": "string"},
    {"name": "aliases", "type": "string_list"}
  ],
  "comparisons": [
    {"name": "forename", "columns": ["forename", "aliases"], "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "list_contains"},
      {"type": "jaro_winkler", "threshold": 0.9},
      {"type": "else"}]}
  ]
})";

// Rows, in order:
//   0 william, aliases {bill, will}    3 robert, aliases {bob, will}
//   1 bill,    aliases {william}       4 no forename, no aliases
//   2 willam,  no aliases              5 bill, no aliases
class Nicknames : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kNicknameConfig, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& forename = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        const uint32_t william = forename.dict.Intern("william");
        const uint32_t bill = forename.dict.Intern("bill");
        const uint32_t willam = forename.dict.Intern("willam");
        const uint32_t robert = forename.dict.Intern("robert");
        forename.ids = {william, bill, willam, robert, cpplink::kNullId, bill};

        auto& aliases = std::get<cpplink::StringListColumn>(store_->mutable_column(1));
        // Interned in a different order from the forename column, and holding a
        // value the forename column never sees: the two dictionaries share no ids,
        // which is exactly what the alias map has to absorb.
        const uint32_t a_bob = aliases.dict.Intern("bob");
        const uint32_t a_bill = aliases.dict.Intern("bill");
        const uint32_t a_will = aliases.dict.Intern("will");
        const uint32_t a_william = aliases.dict.Intern("william");
        // Sorted per row, as the loader guarantees.
        aliases.ids = {a_bill, a_will, a_william, a_bob, a_will};
        aliases.offsets = {0, 2, 3, 3, 5, 5, 5};

        store_->set_num_records(6);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
};

TEST_F(Nicknames, FiresInEitherDirection) {
    // "bill" is one of william's aliases, and "william" is one of bill's. Either
    // direction alone is enough, so the level does not care which row was drawn
    // first -- and the pair evaluates the same both ways round.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 1), 2);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 1, 0), 2);
    // Row 5 is "bill" with no aliases of its own, so only the direction that
    // reads row 0's list can fire. It still does.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 5), 2);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 5, 0), 2);
}

// The reason this is not list_overlap over the two alias lists: rows 0 and 3 are
// "william" and "robert", and both lists hold "will". Intersecting them would
// agree; membership does not, because neither name is in the other's list.
TEST_F(Nicknames, SharingAnAliasIsNotBeingOne) {
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 3), 4);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 1, 3), 4);
}

TEST_F(Nicknames, LevelsAboveAndBelowStillWin) {
    // "bill" against "bill": exact, above the bridge.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 1, 5), 1);
    // "william" against "willam": no alias link, so it falls to jaro_winkler.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 2), 3);
    // "willam" against "robert": nothing at all.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 2, 3), 4);
}

// A row holding neither a name nor a list can produce no match with any partner,
// which is what the null level means here. A row holding a name but no list can,
// so it is not null -- marking it so would let the null level pre-empt a level
// that fires.
TEST_F(Nicknames, NullIsHoldingNeitherPart) {
    EXPECT_TRUE(comparisons_.IsNullValue(0, 4));
    EXPECT_FALSE(comparisons_.IsNullValue(0, 5));
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 4), 0);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 4, 5), 0);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 4, 4), 0);
    // Two present names with no lists between them is a disagreement, not a null.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 2, 5), 4);
}

// LevelPossible is what Scorer::Ceiling prices a pair with, so it may never be
// false where the level fires.
TEST_F(Nicknames, LevelPossibleAgreesWithTheLevelItBounds) {
    for (uint64_t a = 0; a < store_->NumRecords(); ++a) {
        for (uint64_t b = 0; b < store_->NumRecords(); ++b) {
            const uint8_t level = comparisons_.EvaluateOne(0, a, b);
            EXPECT_TRUE(comparisons_.LevelPossible(0, level, a, b))
                << "rows " << a << "," << b;
        }
    }
}

// The fuzzy half of the bridge: the same two columns and the same both-directions
// rule as list_contains, with a metric where that has an id equality. Levels in
// order: null, exact, list_contains, contains_levenshtein, contains_jaro_winkler,
// else -- exact membership above the fuzzy levels it implies.
constexpr const char* kNearNicknameConfig = R"({
  "columns": [
    {"name": "forename", "type": "string"},
    {"name": "aliases", "type": "string_list"}
  ],
  "comparisons": [
    {"name": "forename", "columns": ["forename", "aliases"], "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "list_contains"},
      {"type": "contains_levenshtein", "threshold": 1},
      {"type": "contains_jaro_winkler", "threshold": 0.9},
      {"type": "else"}]}
  ]
})";

// Rows, in order:
//   0 william, aliases {bill, will}   4 no forename, no aliases
//   1 bil,     no aliases             5 wilhelmina, no aliases
//   2 billie,  no aliases             6 bill, no aliases
//   3 robert,  aliases {bob, will}    7 bill, no aliases
class NearNicknames : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kNearNicknameConfig, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& forename = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        const uint32_t william = forename.dict.Intern("william");
        const uint32_t bil = forename.dict.Intern("bil");
        const uint32_t billie = forename.dict.Intern("billie");
        const uint32_t robert = forename.dict.Intern("robert");
        const uint32_t wilhelmina = forename.dict.Intern("wilhelmina");
        const uint32_t bill = forename.dict.Intern("bill");
        forename.ids = {william,          bil,        billie, robert,
                        cpplink::kNullId, wilhelmina, bill,   bill};

        auto& aliases = std::get<cpplink::StringListColumn>(store_->mutable_column(1));
        const uint32_t a_bob = aliases.dict.Intern("bob");
        const uint32_t a_bill = aliases.dict.Intern("bill");
        const uint32_t a_will = aliases.dict.Intern("will");
        // Sorted per row, as the loader guarantees.
        aliases.ids = {a_bill, a_will, a_bob, a_will};
        aliases.offsets = {0, 2, 2, 2, 4, 4, 4, 4, 4};

        store_->set_num_records(8);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
};

// What the level is for: an alias list that holds "bill" reaches a row named
// "bil", which exact membership reads as no agreement at all.
TEST_F(NearNicknames, ReachesAnAliasTheExactLevelMisses) {
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 1), 3);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 1, 0), 3);  // either direction
    // "billie" is two edits from "bill", so the levenshtein level cannot fire and
    // the jaro_winkler one below it does.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 2), 4);
    // "wilhelmina" is neither: 0.86 against a 0.90 threshold.
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 5), 5);
}

// The levels above are untouched by the ones below them, which is what makes the
// fuzzy levels an addition rather than a replacement.
TEST_F(NearNicknames, ExactLevelsStillWinAbove) {
    EXPECT_EQ(comparisons_.EvaluateOne(0, 6, 7), 1);  // two rows named "bill"
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 6), 2);  // "bill" is william's alias
}

// Being weaker evidence than membership, it agrees where membership refuses:
// "william" is 0.91 similar to "will", which is in robert's alias list, so the
// pair william/robert lands on the jaro_winkler level. That is the reason the
// fuzzy levels rank *below* the exact one rather than replacing it, and it is a
// different error from the one the pairwise levels would make on the same two
// columns -- those would compare the alias lists to each other and agree because
// both hold "will", which says nothing about either name.
TEST_F(NearNicknames, IsWeakerEvidenceThanMembership) {
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 3), 4);
}

TEST_F(NearNicknames, NullIsStillHoldingNeitherPart) {
    EXPECT_TRUE(comparisons_.IsNullValue(0, 4));
    // A name with no aliases of its own is reachable through the other row's
    // list, which here is exactly how the level fires.
    EXPECT_FALSE(comparisons_.IsNullValue(0, 1));
    EXPECT_EQ(comparisons_.EvaluateOne(0, 0, 4), 0);
    EXPECT_EQ(comparisons_.EvaluateOne(0, 4, 4), 0);
}

TEST_F(NearNicknames, LevelPossibleAgreesWithTheLevelItBounds) {
    for (uint64_t a = 0; a < store_->NumRecords(); ++a) {
        for (uint64_t b = 0; b < store_->NumRecords(); ++b) {
            const uint8_t level = comparisons_.EvaluateOne(0, a, b);
            EXPECT_TRUE(comparisons_.LevelPossible(0, level, a, b))
                << "rows " << a << "," << b;
        }
    }
}

// The signature filter reads a value of one dictionary against elements of
// another here, so the admissibility check is worth running over this shape too.
TEST(NearNicknameSignatureTest, ChangesNoPattern) {
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(kNearNicknameConfig, &schema, &error)) << error;
    cpplink::RecordStore store(schema);

    std::mt19937_64 rng(20260910);
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
    std::uniform_int_distribution<size_t> length(3, 10);
    std::uniform_int_distribution<size_t> cell(0, 3);
    std::uniform_int_distribution<int> corrupt(0, 2);
    constexpr uint64_t kRows = 300;

    auto& forename = std::get<cpplink::StringColumn>(store.mutable_column(0));
    auto& aliases = std::get<cpplink::StringListColumn>(store.mutable_column(1));
    std::vector<std::string> written;
    const auto value = [&] {
        std::string text;
        // A third of the values are corruptions of one written earlier, so names
        // and alias elements land near each other often enough for the bound to
        // have to decide something.
        if (!written.empty() && corrupt(rng) == 0) {
            text = written[rng() % written.size()];
            if (!text.empty()) text[rng() % text.size()] = alphabet[pick(rng)];
        } else {
            for (size_t k = length(rng); k > 0; --k) text.push_back(alphabet[pick(rng)]);
        }
        written.push_back(text);
        return text;
    };

    aliases.offsets.push_back(0);
    for (uint64_t row = 0; row < kRows; ++row) {
        forename.ids.push_back(forename.dict.Intern(value()));
        std::vector<uint32_t> ids;
        for (size_t i = cell(rng); i > 0; --i)
            ids.push_back(aliases.dict.Intern(value()));
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (const uint32_t id : ids) aliases.ids.push_back(id);
        aliases.offsets.push_back(aliases.ids.size());
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
                << "rows " << a << "," << b;
        }
    }
}

}  // namespace
