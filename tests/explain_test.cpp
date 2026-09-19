// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/explain.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"

namespace {

constexpr uint64_t kRecords = 200;

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
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "city"}]
})";

class ExplainFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        // A skewed surname distribution, so a common value and a rare one exist and
        // the term-frequency move has something to say.
        std::vector<uint32_t> names;
        for (int i = 0; i < 20; ++i) {
            names.push_back(surname.dict.Intern("name" + std::to_string(i)));
        }
        std::vector<uint32_t> cities;
        for (int i = 0; i < 4; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        for (uint64_t row = 0; row < kRecords; ++row) {
            const size_t pick = static_cast<size_t>((row * row) % 20) / 2;
            surname.ids.push_back(names[pick]);
            city.ids.push_back(cities[(row * 7) % 4]);
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;

        model_.lambda = 0.05;
        model_.records = kRecords;
        model_.comparisons = {
            Comparison("surname", true, {{1e-9, 1e-9}, {0.9, 0.02}, {0.1, 0.98}}),
            Comparison("city", false, {{0.8, 0.25}, {0.2, 0.75}})};
        cpplink::ScoreOptions options;
        options.threshold = 1.0;
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

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
    cpplink::Model model_;
    cpplink::Scorer scorer_;
};

// The waterfall is only worth printing if it adds up to the number scoring uses.
// Anything else is a plausible-looking explanation of a different calculation.
TEST_F(ExplainFixture, TheStepsSumToTheScore) {
    for (uint64_t a = 0; a < 40; ++a) {
        for (uint64_t b = a + 1; b < 40; ++b) {
            const cpplink::PairWaterfall w = cpplink::BuildPairWaterfall(
                *store_, comparisons_, scorer_, a, b, &model_);
            double running = w.prior;
            for (const cpplink::WaterfallStep& step : w.steps) {
                running += step.bits + step.tf;
                ASSERT_NEAR(running, step.running, 1e-12);
            }
            for (const cpplink::WaterfallInteraction& term : w.interactions) {
                running += term.bits;
                ASSERT_NEAR(running, term.running, 1e-12);
            }
            ASSERT_NEAR(running, w.weight, 1e-9) << "rows " << a << " and " << b;
            ASSERT_NEAR(w.weight, scorer_.Weight(w.gamma, a, b), 1e-12);
            ASSERT_EQ(w.gamma, comparisons_.Evaluate(a, b));
        }
    }
}

// The JSON object carries the same ledger as the text, so a tool drawing it draws
// the run's own arithmetic. The rates come from the model the report was given.
TEST_F(ExplainFixture, JsonCarriesTheSameLedger) {
    const cpplink::PairWaterfall w =
        cpplink::BuildPairWaterfall(*store_, comparisons_, scorer_, 3, 11, &model_);
    const nlohmann::json root = nlohmann::json::parse(cpplink::PairWaterfallJson(w));
    EXPECT_EQ(root["id_a"], "r3");
    EXPECT_EQ(root["id_b"], "r11");
    EXPECT_EQ(root["gamma"].get<uint32_t>(), w.gamma);
    EXPECT_DOUBLE_EQ(root["prior"].get<double>(), w.prior);
    EXPECT_DOUBLE_EQ(root["weight"].get<double>(), w.weight);
    EXPECT_DOUBLE_EQ(root["probability"].get<double>(), w.probability);
    EXPECT_EQ(root["threshold"].get<double>(), 1.0);
    EXPECT_EQ(root["emitted"].get<bool>(), w.weight >= 1.0);
    ASSERT_EQ(root["steps"].size(), 2u);
    ASSERT_EQ(root["bracket"].size(), 2u);
    double running = w.prior;
    for (size_t i = 0; i < 2; ++i) {
        const nlohmann::json& step = root["steps"][i];
        EXPECT_EQ(step["name"], w.steps[i].name);
        EXPECT_EQ(step["label"], w.steps[i].label);
        EXPECT_EQ(step["level"].get<int>(), w.steps[i].level);
        EXPECT_EQ(step["values"].size(), 2u);
        running += step["bits"].get<double>() + step["tf"].get<double>();
        EXPECT_DOUBLE_EQ(step["running"].get<double>(), running);
        const cpplink::ModelLevel& learned =
            model_.comparisons[i].levels[w.steps[i].level];
        EXPECT_DOUBLE_EQ(step["m"].get<double>(), learned.m);
        EXPECT_DOUBLE_EQ(step["u"].get<double>(), learned.u);
        // The frequency is reported exactly where a term-frequency move is.
        EXPECT_EQ(step.contains("frequency"), step["tf"].get<double>() != 0.0);
    }
    EXPECT_DOUBLE_EQ(running, w.weight);

    // Without a model the rates are absent rather than null.
    const cpplink::PairWaterfall bare =
        cpplink::BuildPairWaterfall(*store_, comparisons_, scorer_, 3, 11);
    const nlohmann::json plain = nlohmann::json::parse(cpplink::PairWaterfallJson(bare));
    EXPECT_FALSE(plain["steps"][0].contains("m"));
    EXPECT_DOUBLE_EQ(plain["weight"].get<double>(), w.weight);
}

TEST_F(ExplainFixture, AdjustmentAppliesOnlyToAnAdjustedLevel) {
    // Any pair whose surnames disagree lands on `else`, where no term-frequency
    // move applies however rare either value is.
    bool checked = false;
    for (uint64_t b = 1; b < kRecords; ++b) {
        const uint32_t gamma = comparisons_.Evaluate(0, b);
        if (comparisons_.LevelOf(gamma, 0) == 1) continue;
        EXPECT_DOUBLE_EQ(scorer_.AdjustmentFor(0, gamma, 0, b), 0.0) << b;
        checked = true;
    }
    ASSERT_TRUE(checked) << "the fixture must contain a disagreeing pair";

    // A comparison without term_frequency never moves, whatever level it hits.
    for (uint64_t b = 1; b < 40; ++b) {
        const uint32_t gamma = comparisons_.Evaluate(0, b);
        EXPECT_DOUBLE_EQ(scorer_.AdjustmentFor(1, gamma, 0, b), 0.0) << b;
    }
}

TEST_F(ExplainFixture, RareValuesEarnMoreBitsThanCommonOnes) {
    // Find two rows agreeing exactly on surname, for a common value and a rare one.
    uint32_t rare_moves = 0;
    double rarest = -1e9;
    double commonest = 1e9;
    for (uint64_t a = 0; a < kRecords; ++a) {
        for (uint64_t b = a + 1; b < kRecords; ++b) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            if (comparisons_.LevelOf(gamma, 0) != 1) continue;
            const double move = scorer_.AdjustmentFor(0, gamma, a, b);
            rarest = std::max(rarest, move);
            commonest = std::min(commonest, move);
            ++rare_moves;
        }
    }
    ASSERT_GT(rare_moves, 0u);
    EXPECT_GT(rarest, commonest) << "a skewed column must produce a spread";
    // And the spread is bracketed by what the pattern's bounds promised.
    const uint32_t any = comparisons_.Evaluate(0, 0);
    EXPECT_LE(rarest, scorer_.DeltaMax(any) + 1e-9);
}

TEST_F(ExplainFixture, ReportShowsTheRunningTotalAndTheDecision) {
    uint64_t a = 0;
    uint64_t b = 0;
    for (uint64_t i = 0; i < kRecords && b == 0; ++i) {
        for (uint64_t j = i + 1; j < kRecords; ++j) {
            if (comparisons_.LevelOf(comparisons_.Evaluate(i, j), 0) == 1) {
                a = i;
                b = j;
                break;
            }
        }
    }
    ASSERT_GT(b, 0u);

    std::ostringstream out;
    cpplink::PrintPairWaterfall(*store_, comparisons_, scorer_, a, b, out);
    const std::string text = out.str();
    EXPECT_NE(text.find("(prior)"), std::string::npos);
    EXPECT_NE(text.find("Match weight"), std::string::npos);
    EXPECT_NE(text.find("Term frequency"), std::string::npos);
    EXPECT_NE(text.find("Pattern bracket"), std::string::npos);

    const uint32_t gamma = comparisons_.Evaluate(a, b);
    const double weight = scorer_.Weight(gamma, a, b);
    const bool emitted = weight >= scorer_.threshold();
    EXPECT_NE(text.find(emitted ? "This pair would be emitted."
                                : "This pair would not be emitted."),
              std::string::npos)
        << text;
    EXPECT_NE(text.find(cpplink::ZoneName(scorer_.Classify(gamma))), std::string::npos);
}

TEST_F(ExplainFixture, NoTermFrequencySectionWhenNothingAgreedExactly) {
    uint64_t a = 0;
    uint64_t b = 0;
    for (uint64_t j = 1; j < kRecords; ++j) {
        if (comparisons_.LevelOf(comparisons_.Evaluate(0, j), 0) != 1) {
            b = j;
            break;
        }
    }
    ASSERT_GT(b, 0u);
    std::ostringstream out;
    cpplink::PrintPairWaterfall(*store_, comparisons_, scorer_, a, b, out);
    EXPECT_EQ(out.str().find("Term frequency"), std::string::npos) << out.str();
}

}  // namespace
