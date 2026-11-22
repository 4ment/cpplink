// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// Scoring with null levels held uninformative: a missing field is worth 0 bits,
// and the levels that survive are read given the field is present.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/dictionary.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/init.hpp"
#include "cpplink/model.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"

namespace {

constexpr uint64_t kRecords = 240;

const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "surname", "type": "string"},
    {"name": "city", "type": "string"}
  ],
  "comparisons": [
    {"name": "surname", "columns": ["surname"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "city", "columns": ["city"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "all_pairs"}]
})";

using Edge = std::tuple<uint32_t, uint32_t, uint32_t, double>;

class NullLevelFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        std::vector<uint32_t> names;
        for (int i = 0; i < 30; ++i) {
            names.push_back(surname.dict.Intern("name" + std::to_string(i)));
        }
        std::vector<uint32_t> cities;
        for (int i = 0; i < 5; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        // Missing on different rows in each column, so every pairing of a null
        // level with the other comparison's levels occurs.
        for (uint64_t row = 0; row < kRecords; ++row) {
            const size_t pick = static_cast<size_t>((row * row) % 30) / 2;
            surname.ids.push_back(row % 5 == 0 ? cpplink::kNullId : names[pick]);
            city.ids.push_back(row % 7 == 0 ? cpplink::kNullId : cities[(row * 3) % 5]);
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();
        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;

        model_.lambda = 0.05;
        model_.records = kRecords;
        model_.comparisons = {
            Comparison("surname", true, {{0.15, 0.04}, {0.75, 0.01}, {0.10, 0.95}}),
            Comparison("city", false, {{0.10, 0.13}, {0.80, 0.20}, {0.10, 0.67}})};
        // Every cell nonzero, so zeroing the null row and column is visible.
        cpplink::ModelInteraction term;
        term.left = "surname";
        term.right = "city";
        term.left_levels = 3;
        term.right_levels = 3;
        term.bits = {1.5, 0.7, -0.4, 0.9, 0.8, 0.3, -0.6, 0.5, -1.2};
        model_.interactions = {term};
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

    void Bind(cpplink::Scorer* scorer, bool uninformative, double threshold = 0.0) {
        cpplink::ScoreOptions options;
        options.threshold = threshold;
        options.uninformative_nulls = uninformative;
        std::string error;
        ASSERT_TRUE(scorer->Bind(model_, comparisons_, *store_, options, &error))
            << error;
    }

    std::vector<Edge> Predict(double threshold, bool shortcuts) {
        cpplink::ScoreOptions score;
        score.threshold = threshold;
        score.uninformative_nulls = true;
        score.use_bounds = shortcuts;
        score.use_ceiling = shortcuts;
        cpplink::Scorer scorer;
        std::string error;
        EXPECT_TRUE(scorer.Bind(model_, comparisons_, *store_, score, &error)) << error;
        cpplink::EdgeTable table;
        cpplink::PredictOptions options;
        options.threads = 2;
        options.table = &table;
        cpplink::PredictReport report;
        EXPECT_TRUE(cpplink::Predict(*store_, comparisons_, plan_, scorer, options,
                                     &report, &error))
            << error;
        std::vector<Edge> edges;
        for (size_t i = 0; i < table.Size(); ++i) {
            edges.emplace_back(table.a[i], table.b[i], table.gamma[i], table.weight[i]);
        }
        std::sort(edges.begin(), edges.end());
        return edges;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::BlockingPlan plan_;
    cpplink::ComparisonSet comparisons_;
    cpplink::Model model_;
};

// Off by default, and off means the file's own rates and nothing else.
TEST_F(NullLevelFixture, TheDefaultScoresTheModelAsWritten) {
    cpplink::Scorer scorer;
    Bind(&scorer, false);
    for (size_t c = 0; c < model_.comparisons.size(); ++c) {
        for (size_t l = 0; l < model_.comparisons[c].levels.size(); ++l) {
            const cpplink::ModelLevel& level = model_.comparisons[c].levels[l];
            const auto at = static_cast<uint8_t>(l);
            EXPECT_DOUBLE_EQ(scorer.LevelWeight(c, at), level.Weight());
            EXPECT_DOUBLE_EQ(scorer.LevelM(c, at), level.m);
            EXPECT_DOUBLE_EQ(scorer.LevelU(c, at), level.u);
        }
    }
    EXPECT_TRUE(scorer.UninformativeNulls().empty());
}

// The renormalising form, not the zeroed cell: every surviving level gains the
// same log2((1 - u_null) / (1 - m_null)), and its rates are the observed-field ones.
TEST_F(NullLevelFixture, ANullLevelScoresNothingAndTheRestAreReadGivenTheField) {
    cpplink::Scorer scorer;
    Bind(&scorer, true);
    EXPECT_EQ(scorer.UninformativeNulls().size(), 2u);
    for (size_t c = 0; c < model_.comparisons.size(); ++c) {
        const std::vector<cpplink::ModelLevel>& levels = model_.comparisons[c].levels;
        const double m_null = levels[0].m;
        const double u_null = levels[0].u;
        EXPECT_EQ(scorer.LevelWeight(c, 0), 0.0);
        EXPECT_TRUE(std::isnan(scorer.LevelM(c, 0)));
        EXPECT_TRUE(std::isnan(scorer.LevelU(c, 0)));
        const double shift = std::log2((1.0 - u_null) / (1.0 - m_null));
        double m_sum = 0.0;
        double u_sum = 0.0;
        for (size_t l = 1; l < levels.size(); ++l) {
            const auto at = static_cast<uint8_t>(l);
            EXPECT_NEAR(scorer.LevelWeight(c, at), levels[l].Weight() + shift, 1e-12);
            EXPECT_NEAR(scorer.LevelM(c, at), levels[l].m / (1.0 - m_null), 1e-12);
            EXPECT_NEAR(scorer.LevelU(c, at), levels[l].u / (1.0 - u_null), 1e-12);
            m_sum += scorer.LevelM(c, at);
            u_sum += scorer.LevelU(c, at);
        }
        // Conditional on the field being there, the levels are a distribution.
        EXPECT_NEAR(m_sum, 1.0, 1e-12);
        EXPECT_NEAR(u_sum, 1.0, 1e-12);
    }
}

// A comparison on its null level is not evaluated, so no association it takes
// part in is either; the observed cells keep their bits.
TEST_F(NullLevelFixture, InteractionCellsTouchingANullLevelAreZero) {
    cpplink::Scorer on;
    cpplink::Scorer off;
    Bind(&on, true);
    Bind(&off, false);
    ASSERT_EQ(on.InteractionCount(), 1u);
    const std::vector<double>& bits = model_.interactions[0].bits;
    bool saw_null = false;
    bool saw_observed = false;
    for (uint64_t a = 0; a < kRecords; ++a) {
        for (uint64_t b = a + 1; b < kRecords; b += 13) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            const uint8_t left = comparisons_.LevelOf(gamma, 0);
            const uint8_t right = comparisons_.LevelOf(gamma, 1);
            const double cell = bits[left * 3 + right];
            ASSERT_DOUBLE_EQ(off.InteractionBits(0, gamma), cell);
            if (left == 0 || right == 0) {
                ASSERT_EQ(on.InteractionBits(0, gamma), 0.0);
                saw_null = true;
            } else {
                ASSERT_DOUBLE_EQ(on.InteractionBits(0, gamma), cell);
                saw_observed = true;
            }
        }
    }
    EXPECT_TRUE(saw_null);
    EXPECT_TRUE(saw_observed);
}

// The waterfall has to add up to the score whichever way the nulls are read, and
// with them uninformative a null step is exactly zero.
TEST_F(NullLevelFixture, TheWaterfallSumsToTheScoreUnderBothReadings) {
    for (const bool uninformative : {false, true}) {
        cpplink::Scorer scorer;
        Bind(&scorer, uninformative);
        size_t null_steps = 0;
        for (uint64_t a = 0; a < 60; ++a) {
            for (uint64_t b = a + 1; b < 60; ++b) {
                const cpplink::PairWaterfall w = cpplink::BuildPairWaterfall(
                    *store_, comparisons_, scorer, a, b, &model_);
                double running = w.prior;
                for (const cpplink::WaterfallStep& step : w.steps) {
                    running += step.bits + step.tf;
                    ASSERT_NEAR(running, step.running, 1e-12);
                    if (step.level != 0) continue;
                    ++null_steps;
                    if (uninformative) {
                        ASSERT_EQ(step.bits, 0.0);
                        ASSERT_EQ(step.tf, 0.0);
                    } else {
                        ASSERT_NE(step.bits, 0.0);
                    }
                }
                for (const cpplink::WaterfallInteraction& term : w.interactions) {
                    running += term.bits;
                }
                ASSERT_NEAR(running, w.weight, 1e-9) << a << ", " << b;
                ASSERT_NEAR(w.weight, scorer.Weight(w.gamma, a, b), 1e-12);
            }
        }
        EXPECT_GT(null_steps, 0u);
    }
}

// The JSON reports the rates the scorer read, and a null level read as
// uninformative has none, so they are absent rather than the file's.
TEST_F(NullLevelFixture, TheJsonLedgerCarriesTheObservedFieldRates) {
    cpplink::Scorer scorer;
    Bind(&scorer, true);
    // Row 0 has neither field, row 1 has both.
    const cpplink::PairWaterfall w =
        cpplink::BuildPairWaterfall(*store_, comparisons_, scorer, 0, 1, &model_);
    const nlohmann::json root = nlohmann::json::parse(cpplink::PairWaterfallJson(w));
    ASSERT_EQ(root["steps"].size(), 2u);
    for (const nlohmann::json& step : root["steps"]) {
        EXPECT_EQ(step["level"], 0);
        EXPECT_EQ(step["bits"], 0.0);
        EXPECT_FALSE(step.contains("m"));
        EXPECT_FALSE(step.contains("u"));
    }
    const cpplink::PairWaterfall both =
        cpplink::BuildPairWaterfall(*store_, comparisons_, scorer, 1, 2, &model_);
    for (size_t c = 0; c < both.steps.size(); ++c) {
        const cpplink::WaterfallStep& step = both.steps[c];
        ASSERT_NE(step.level, 0);
        EXPECT_DOUBLE_EQ(step.m, scorer.LevelM(c, step.level));
        EXPECT_DOUBLE_EQ(step.u, scorer.LevelU(c, step.level));
        EXPECT_LT(step.m, 1.0);
    }
}

// The ceiling and the bracket stay admissible under the option: the shortcuts
// emit exactly the predictions scoring every pair does, at every threshold.
TEST_F(NullLevelFixture, TheShortcutsEmitExactlyWhatExhaustiveScoringDoes) {
    for (const double threshold : {-4.0, 0.0, 3.0, 6.0, 9.0}) {
        const std::vector<Edge> fast = Predict(threshold, true);
        const std::vector<Edge> exhaustive = Predict(threshold, false);
        EXPECT_EQ(fast, exhaustive) << "threshold " << threshold;
        if (threshold <= 0.0) EXPECT_FALSE(exhaustive.empty()) << threshold;
    }
}

// Per comparison, three ways: named by the run, declared by the schema, or
// neither, and a comparison left alone keeps the file's weights exactly.
TEST_F(NullLevelFixture, ARunCanNameTheComparisonsWhoseNullSaysNothing) {
    cpplink::ScoreOptions options;
    options.uninformative_null_comparisons = {"city"};
    cpplink::Scorer scorer;
    std::string error;
    ASSERT_TRUE(scorer.Bind(model_, comparisons_, *store_, options, &error)) << error;
    EXPECT_EQ(scorer.UninformativeNulls(), std::vector<std::string>{"city"});
    const std::vector<cpplink::ModelLevel>& surname = model_.comparisons[0].levels;
    for (uint8_t l = 0; l < 3; ++l) {
        EXPECT_DOUBLE_EQ(scorer.LevelWeight(0, l), surname[l].Weight());
    }
    EXPECT_EQ(scorer.LevelWeight(1, 0), 0.0);
    const std::vector<cpplink::ModelLevel>& city = model_.comparisons[1].levels;
    EXPECT_NEAR(scorer.LevelWeight(1, 1),
                city[1].Weight() + std::log2((1.0 - city[0].u) / (1.0 - city[0].m)),
                1e-12);
    // Only the null cells of the named comparison leave the interaction.
    const std::vector<double>& bits = model_.interactions[0].bits;
    for (uint64_t a = 0; a < kRecords; ++a) {
        const uint32_t gamma = comparisons_.Evaluate(a, a + 1 < kRecords ? a + 1 : 0);
        const uint8_t left = comparisons_.LevelOf(gamma, 0);
        const uint8_t right = comparisons_.LevelOf(gamma, 1);
        const double expected = right == 0 ? 0.0 : bits[left * 3 + right];
        ASSERT_DOUBLE_EQ(scorer.InteractionBits(0, gamma), expected);
    }
}

TEST_F(NullLevelFixture, AnUnknownComparisonIsRefused) {
    for (const std::string& name : {std::string("cty"), std::string("nowhere")}) {
        cpplink::ScoreOptions options;
        options.uninformative_null_comparisons = {name};
        cpplink::Scorer scorer;
        std::string error;
        EXPECT_FALSE(scorer.Bind(model_, comparisons_, *store_, options, &error));
        EXPECT_NE(error.find("does not declare"), std::string::npos) << error;
    }
}

// The schema's declaration is the splink one, on the level, and it holds whatever
// the run passes; it survives the schema writer.
TEST_F(NullLevelFixture, TheSchemaCanDeclareANullLevelUninformative) {
    std::string text = kSchemaJson;
    const std::string city_null = R"("columns": ["city"],
     "levels": [{"type": "null"})";
    const size_t at = text.find(city_null);
    ASSERT_NE(at, std::string::npos);
    text.replace(at, city_null.size(), R"("columns": ["city"],
     "levels": [{"type": "null", "informative": false})");
    cpplink::Schema declared;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(text, &declared, &error)) << error;
    EXPECT_TRUE(declared.comparisons[0].levels[0].informative);
    EXPECT_FALSE(declared.comparisons[1].levels[0].informative);

    cpplink::Schema reread;
    ASSERT_TRUE(cpplink::ParseSchema(cpplink::SchemaToJson(declared), &reread, &error))
        << error;
    EXPECT_FALSE(reread.comparisons[1].levels[0].informative);
    EXPECT_EQ(cpplink::SchemaToJson(schema_).find("informative"), std::string::npos);

    cpplink::ComparisonSet bound;
    ASSERT_TRUE(bound.Bind(declared, *store_, &error)) << error;
    cpplink::Scorer scorer;
    ASSERT_TRUE(scorer.Bind(model_, bound, *store_, cpplink::ScoreOptions(), &error))
        << error;
    EXPECT_EQ(scorer.UninformativeNulls(), std::vector<std::string>{"city"});
    EXPECT_NE(scorer.LevelWeight(0, 0), 0.0);
    EXPECT_EQ(scorer.LevelWeight(1, 0), 0.0);
}

TEST(NullLevelSchema, InformativeIsABooleanOnANullLevelOnly) {
    const char* const on_exact = R"({"unique_id": "id",
      "columns": [{"name": "a", "type": "string"}],
      "comparisons": [{"name": "a", "columns": ["a"], "levels": [
        {"type": "exact", "informative": false}, {"type": "else"}]}]})";
    const char* const not_bool = R"({"unique_id": "id",
      "columns": [{"name": "a", "type": "string"}],
      "comparisons": [{"name": "a", "columns": ["a"], "levels": [
        {"type": "null", "informative": "no"}, {"type": "else"}]}]})";
    cpplink::Schema schema;
    std::string error;
    EXPECT_FALSE(cpplink::ParseSchema(on_exact, &schema, &error));
    EXPECT_NE(error.find("only a null level"), std::string::npos) << error;
    EXPECT_FALSE(cpplink::ParseSchema(not_bool, &schema, &error));
    EXPECT_NE(error.find("true or false"), std::string::npos) << error;
}

}  // namespace
