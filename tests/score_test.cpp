// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/score.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

// surname is Zipf-shaped on purpose: smith is common, zolnerowich is unique, and
// term-frequency adjustment exists precisely to tell those two apart.
const char* const kSchemaJson = R"({
  "columns": [
    {"name": "surname", "type": "string"},
    {"name": "city", "type": "string"}
  ],
  "comparisons": [
    {"name": "surname", "columns": ["surname"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "city", "columns": ["city"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ]
})";

class ScoreFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        const uint32_t smith = surname.dict.Intern("smith");
        const uint32_t jones = surname.dict.Intern("jones");
        const uint32_t rare = surname.dict.Intern("zolnerowich");
        surname.ids = {smith, smith, smith, smith, smith,
                       smith, jones, jones, rare,  rare};
        const uint32_t north = city.dict.Intern("north");
        const uint32_t south = city.dict.Intern("south");
        city.ids = {north, north, north, north, north, south, south, south, north, north};

        store_->set_num_records(10);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;

        model_.lambda = 0.25;  // prior weight log2(1/3)
        model_.records = 10;
        model_.comparisons = {
            Comparison("surname", {"surname"}, true,
                       {{1e-9, 1e-9}, {0.9, 0.4}, {0.1, 0.6}}),
            Comparison("city", {"city"}, false, {{0.8, 0.5}, {0.2, 0.5}})};
    }

    static cpplink::ModelComparison Comparison(
        const std::string& name, const std::vector<std::string>& columns, bool tf,
        const std::vector<std::pair<double, double>>& levels) {
        cpplink::ModelComparison comparison;
        comparison.name = name;
        comparison.columns = columns;
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

    bool BindScorer(double threshold, bool bounds = true) {
        cpplink::ScoreOptions options;
        options.threshold = threshold;
        options.use_bounds = bounds;
        std::string error;
        const bool ok = scorer_.Bind(model_, comparisons_, *store_, options, &error);
        if (!ok) message_ = error;
        return ok;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
    cpplink::Model model_;
    cpplink::Scorer scorer_;
    std::string message_;
};

TEST(ScoreTest, ProbabilityAndWeightAreInverses) {
    EXPECT_DOUBLE_EQ(cpplink::WeightForProbability(0.5), 0.0);
    EXPECT_DOUBLE_EQ(cpplink::ProbabilityForWeight(0.0), 0.5);
    for (const double weight : {-20.0, -3.5, 0.0, 1.25, 17.0}) {
        EXPECT_NEAR(cpplink::WeightForProbability(cpplink::ProbabilityForWeight(weight)),
                    weight, 1e-9);
    }
}

TEST_F(ScoreFixture, RejectsAModelThatDoesNotMatchTheSchema) {
    model_.comparisons.pop_back();
    EXPECT_FALSE(BindScorer(0.0));
    EXPECT_NE(message_.find("comparisons"), std::string::npos);

    SetUp();
    model_.comparisons[0].name = "surnames";
    EXPECT_FALSE(BindScorer(0.0));

    SetUp();
    model_.comparisons[1].levels.pop_back();
    EXPECT_FALSE(BindScorer(0.0));
    EXPECT_NE(message_.find("levels"), std::string::npos);
}

TEST_F(ScoreFixture, BaseWeightIsThePriorPlusEachLevelsWeight) {
    ASSERT_TRUE(BindScorer(0.0)) << message_;
    // Row 0 and row 1 agree on both: surname exact and city exact.
    const uint32_t gamma = comparisons_.Evaluate(0, 1);
    const double expected =
        std::log2(0.25 / 0.75) + std::log2(0.9 / 0.4) + std::log2(0.8 / 0.5);
    EXPECT_NEAR(scorer_.BaseWeight(gamma), expected, 1e-12);
}

// The whole point of the adjustment: agreeing on a value nobody else has is
// stronger evidence than agreeing on the commonest value in the column.
TEST_F(ScoreFixture, RareValuesScoreHigherThanCommonOnes) {
    ASSERT_TRUE(BindScorer(0.0)) << message_;
    const uint32_t common = comparisons_.Evaluate(0, 1);  // smith, both north
    const uint32_t rare = comparisons_.Evaluate(8, 9);    // zolnerowich, both north
    ASSERT_EQ(common, rare) << "the two pairs must share a pattern for this to test TF";
    EXPECT_GT(scorer_.Weight(rare, 8, 9), scorer_.Weight(common, 0, 1));
}

// Admissibility is the property the drop and emit decisions rest on. If the
// bracket ever fails to contain the true weight, output stops being exact.
TEST_F(ScoreFixture, TheBracketContainsEveryPairsTrueWeight) {
    ASSERT_TRUE(BindScorer(0.0)) << message_;
    for (uint64_t a = 0; a < store_->NumRecords(); ++a) {
        for (uint64_t b = a + 1; b < store_->NumRecords(); ++b) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            const double base = scorer_.BaseWeight(gamma);
            const double exact = scorer_.Weight(gamma, a, b);
            EXPECT_LE(base + scorer_.DeltaMin(gamma), exact + 1e-9)
                << "pair " << a << "," << b;
            EXPECT_GE(base + scorer_.DeltaMax(gamma), exact - 1e-9)
                << "pair " << a << "," << b;
        }
    }
}

TEST_F(ScoreFixture, ZonesFollowTheThreshold) {
    // Far below anything achievable: every reachable pattern is certain to pass.
    ASSERT_TRUE(BindScorer(-1000.0)) << message_;
    EXPECT_EQ(scorer_.PatternsIn(cpplink::Zone::kDrop), 0u);
    EXPECT_EQ(scorer_.PatternsIn(cpplink::Zone::kCheck), 0u);
    EXPECT_EQ(scorer_.PatternsIn(cpplink::Zone::kEmit), scorer_.PatternSpace());

    // Far above: nothing can reach it, whatever the term frequencies say.
    ASSERT_TRUE(BindScorer(1000.0)) << message_;
    EXPECT_EQ(scorer_.PatternsIn(cpplink::Zone::kDrop), scorer_.PatternSpace());
    EXPECT_EQ(scorer_.PatternsIn(cpplink::Zone::kEmit), 0u);
}

// The packed space is larger than the reachable space: three levels occupy two
// bits, and the fourth code names no level. Those patterns must be excluded, not
// indexed.
TEST_F(ScoreFixture, PatternSpaceCountsOnlyReachablePatterns) {
    ASSERT_TRUE(BindScorer(0.0)) << message_;
    EXPECT_EQ(scorer_.PatternSpace(), 3u * 2u);
    EXPECT_EQ(scorer_.PatternsIn(cpplink::Zone::kDrop) +
                  scorer_.PatternsIn(cpplink::Zone::kCheck) +
                  scorer_.PatternsIn(cpplink::Zone::kEmit),
              scorer_.PatternSpace());
}

TEST_F(ScoreFixture, TurningTheBoundsOffChecksEveryPattern) {
    ASSERT_TRUE(BindScorer(0.0, false)) << message_;
    for (uint64_t a = 0; a < store_->NumRecords(); ++a) {
        for (uint64_t b = a + 1; b < store_->NumRecords(); ++b) {
            EXPECT_EQ(scorer_.Classify(comparisons_.Evaluate(a, b)),
                      cpplink::Zone::kCheck);
        }
    }
}

}  // namespace
