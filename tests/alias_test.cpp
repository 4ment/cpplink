// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// Term frequency on a name compared beside its list of aliases. The exact level of
// that shape is the two names being equal, which reads the scalar column alone,
// so it is adjusted by the scalar's own frequency and its u is the scalar's
// collision rate in closed form. The membership and fuzzy levels below it are
// adjusted by nothing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"

namespace {

constexpr const char* kAliasSchema = R"({
  "unique_id": "id",
  "columns": [
    {"name": "forename", "type": "string"},
    {"name": "aliases", "type": "string_list"},
    {"name": "surname", "type": "string"},
    {"name": "town", "type": "string"},
    {"name": "job", "type": "string"}
  ],
  "comparisons": [
    {"name": "forename", "columns": ["forename", "aliases"], "term_frequency": true,
     "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "list_contains"},
      {"type": "jaro_winkler", "threshold": 0.9},
      {"type": "else"}]},
    {"name": "surname", "columns": ["surname"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "town", "columns": ["town"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "job", "columns": ["job"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "surname"}]
})";

// The rows. A name of "-" is missing.
//
//   0  william   {bill, will}   the common name, three times
//   1  william   {}
//   2  william   {bill}
//   3  bill      {william}      an alias of row 0, and row 0 one of its
//   4  zbigniew  {}             the rare name, twice
//   5  zbigniew  {zbysz}
//   6  -         {}             holds neither part, so the comparison is null
//   7  -         {william}      no name, but reachable through its list
//   8  -         {zbysz}        the same
//   9  robert    {bob}
constexpr uint64_t kRows = 10;
constexpr uint8_t kNull = 0;
constexpr uint8_t kExact = 1;
constexpr uint8_t kContains = 2;
constexpr uint8_t kElse = 4;

class AliasFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kAliasSchema, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        const std::vector<std::string> names = {"william",  "william",  "william", "bill",
                                                "zbigniew", "zbigniew", "",        "",
                                                "",         "robert"};
        const std::vector<std::vector<std::string>> lists = {
            {"bill", "will"}, {}, {"bill"},    {"william"}, {},
            {"zbysz"},        {}, {"william"}, {"zbysz"},   {"bob"}};
        auto& forename = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& aliases = std::get<cpplink::StringListColumn>(store_->mutable_column(1));
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        auto& town = std::get<cpplink::StringColumn>(store_->mutable_column(3));
        auto& job = std::get<cpplink::StringColumn>(store_->mutable_column(4));
        aliases.offsets.push_back(0);
        for (size_t row = 0; row < kRows; ++row) {
            forename.ids.push_back(names[row].empty() ? cpplink::kNullId
                                                      : forename.dict.Intern(names[row]));
            std::vector<uint32_t> cell;
            for (const std::string& alias : lists[row]) {
                cell.push_back(aliases.dict.Intern(alias));
            }
            std::sort(cell.begin(), cell.end());  // the loader's guarantee
            aliases.ids.insert(aliases.ids.end(), cell.begin(), cell.end());
            aliases.offsets.push_back(aliases.ids.size());
            surname.ids.push_back(surname.dict.Intern(row < 5 ? "smith" : "jones"));
            town.ids.push_back(town.dict.Intern("t" + std::to_string(row % 3)));
            job.ids.push_back(job.dict.Intern("j" + std::to_string(row % 2)));
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRows);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    uint8_t Level(uint64_t a, uint64_t b) const {
        return comparisons_.LevelOf(comparisons_.Evaluate(a, b), 0);
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
};

TEST_F(AliasFixture, TheFixtureReachesEveryLevelItMeansTo) {
    EXPECT_EQ(Level(0, 1), kExact);
    EXPECT_EQ(Level(4, 5), kExact);
    EXPECT_EQ(Level(0, 3), kContains);
    EXPECT_EQ(Level(0, 7), kContains);  // no name of its own, william in its list
    EXPECT_EQ(Level(5, 8), kElse);
    EXPECT_EQ(Level(0, 6), kNull);
    EXPECT_FALSE(comparisons_.IsNullValue(0, 7));
}

// The exact level's u is the scalar column's collision rate. The rows the null
// level takes hold no name, so they could never have agreed, and the rows holding
// only a list (7 and 8) agree on no name either: counting the pairs one by one
// must give the same number, to the last bit.
TEST_F(AliasFixture, TheExactLevelsUIsClosedFormAndExact) {
    cpplink::BlockingPlan plan;
    std::string error;
    ASSERT_TRUE(plan.Build(schema_, *store_, cpplink::PairMode::kAll, &error)) << error;
    cpplink::EstimateOptions options;
    options.threads = 1;
    options.u_sample = 2000;
    cpplink::Model model;
    cpplink::EstimateReport report;
    ASSERT_TRUE(
        cpplink::Estimate(*store_, comparisons_, plan, options, &model, &report, &error))
        << error;

    uint64_t pairs = 0;
    uint64_t null_pairs = 0;
    uint64_t exact_pairs = 0;
    for (uint64_t a = 0; a < kRows; ++a) {
        for (uint64_t b = a + 1; b < kRows; ++b) {
            const uint8_t level = Level(a, b);
            ++pairs;
            if (level == kNull) ++null_pairs;
            if (level == kExact) ++exact_pairs;
        }
    }
    // william three times and zbigniew twice.
    ASSERT_EQ(exact_pairs, 3u + 1u);
    const double space = static_cast<double>(pairs);
    const cpplink::ModelComparison& learned = model.comparisons[0];
    ASSERT_EQ(learned.name, "forename");
    EXPECT_TRUE(learned.levels[kNull].u_exact);
    EXPECT_TRUE(learned.levels[kExact].u_exact);
    EXPECT_FALSE(learned.levels[kContains].u_exact);
    EXPECT_NEAR(learned.levels[kNull].u, static_cast<double>(null_pairs) / space, 1e-12);
    EXPECT_NEAR(learned.levels[kExact].u, static_cast<double>(exact_pairs) / space,
                1e-12);
}

class AliasScoring : public AliasFixture {
   protected:
    void SetUp() override {
        AliasFixture::SetUp();
        model_.lambda = 0.05;
        model_.records = kRows;
        model_.comparisons = {
            Comparison(
                "forename", true,
                {{1e-9, 1e-9}, {0.6, 0.04}, {0.2, 0.02}, {0.1, 0.05}, {0.1, 0.89}}),
            Comparison("surname", false, {{0.9, 0.3}, {0.1, 0.7}}),
            Comparison("town", false, {{0.9, 0.3}, {0.1, 0.7}}),
            Comparison("job", false, {{0.9, 0.5}, {0.1, 0.5}})};
        cpplink::ScoreOptions options;
        options.threshold = 0.0;
        std::string error;
        ASSERT_TRUE(scorer_.Bind(model_, comparisons_, *store_, options, &error))
            << error;
    }

    static cpplink::ModelComparison Comparison(
        const std::string& name, bool tf,
        const std::vector<std::pair<double, double>>& levels) {
        cpplink::ModelComparison comparison;
        comparison.name = name;
        comparison.term_frequency = tf;
        for (const auto& entry : levels) {
            cpplink::ModelLevel level;
            level.m = entry.first;
            level.u = entry.second;
            level.m_estimated = true;
            comparison.levels.push_back(level);
        }
        return comparison;
    }

    cpplink::Model model_;
    cpplink::Scorer scorer_;
};

// The exact level is adjusted by how common the name is: a shared rare name is
// worth more than a shared common one, and by exactly the log of the ratio of
// their frequencies.
TEST_F(AliasScoring, TheExactLevelIsAdjustedByTheNamesFrequency) {
    ASSERT_TRUE(scorer_.HasAdjustment(0));
    const uint32_t common = comparisons_.Evaluate(0, 1);
    const uint32_t rare = comparisons_.Evaluate(4, 5);
    ASSERT_EQ(comparisons_.LevelOf(common, 0), kExact);
    ASSERT_EQ(comparisons_.LevelOf(rare, 0), kExact);
    EXPECT_EQ(scorer_.FrequencyFor(0, common, 0), 3u);
    EXPECT_EQ(scorer_.FrequencyFor(0, rare, 4), 2u);

    const double common_move = scorer_.AdjustmentFor(0, common, 0, 1);
    const double rare_move = scorer_.AdjustmentFor(0, rare, 4, 5);
    EXPECT_NEAR(rare_move - common_move, std::log2(3.0 / 2.0), 1e-12);
    // log2(u / p_v) with u = 0.04 and p_v = 3 / 10.
    EXPECT_NEAR(common_move, std::log2(0.04 * 10.0 / 3.0), 1e-12);
}

// The membership level is not adjusted: which of a row's aliases the other name
// matched has no frequency the level could be priced by.
TEST_F(AliasScoring, TheMembershipLevelIsNotAdjusted) {
    for (const auto& [a, b] :
         std::vector<std::pair<uint64_t, uint64_t>>{{0, 3}, {0, 7}}) {
        const uint32_t gamma = comparisons_.Evaluate(a, b);
        ASSERT_EQ(comparisons_.LevelOf(gamma, 0), kContains);
        EXPECT_EQ(scorer_.AdjustmentFor(0, gamma, a, b), 0.0);
        EXPECT_EQ(scorer_.FrequencyFor(0, gamma, a), 0u);
    }
}

TEST_F(AliasScoring, TheWaterfallStillSumsToTheScore) {
    for (uint64_t a = 0; a < kRows; ++a) {
        for (uint64_t b = a + 1; b < kRows; ++b) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            double running = scorer_.PriorWeight();
            for (size_t c = 0; c < comparisons_.Size(); ++c) {
                running += scorer_.LevelWeight(c, comparisons_.LevelOf(gamma, c));
                running += scorer_.AdjustmentFor(c, gamma, a, b);
            }
            EXPECT_NEAR(running, scorer_.Weight(gamma, a, b), 1e-9)
                << "rows " << a << " and " << b;
        }
    }
}

// The bracket and the ceiling are what predict drops and emits on, so every pair
// must sit inside them for bounded scoring to write exactly what exhaustive
// scoring does.
TEST_F(AliasScoring, TheBracketAndTheCeilingHoldEveryPair) {
    for (uint64_t a = 0; a < kRows; ++a) {
        for (uint64_t b = 0; b < kRows; ++b) {
            if (a == b) continue;
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            const double weight = scorer_.Weight(gamma, a, b);
            EXPECT_GE(weight, scorer_.BaseWeight(gamma) + scorer_.DeltaMin(gamma) - 1e-9);
            EXPECT_LE(weight, scorer_.BaseWeight(gamma) + scorer_.DeltaMax(gamma) + 1e-9);
            EXPECT_GE(scorer_.Ceiling(a, b), weight - 1e-9)
                << "rows " << a << " and " << b;
        }
    }
}

}  // namespace
