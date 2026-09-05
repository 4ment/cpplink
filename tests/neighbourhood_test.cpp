// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/neighbourhood.hpp"

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
#include "cpplink/score.hpp"

namespace {

const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "surname", "type": "string"},
    {"name": "town", "type": "string"}
  ],
  "comparisons": [
    {"name": "surname", "columns": ["surname"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"},
                {"type": "levenshtein", "threshold": 1},
                {"type": "jaro_winkler", "threshold": 0.85},
                {"type": "else"}]},
    {"name": "town", "columns": ["town"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "town"}]
})";

// A skewed surname column with real near-neighbours in it, so the balls are not
// all singletons and the masses have something to say.
const std::vector<std::string>& Surnames() {
    static const std::vector<std::string> names = {
        "smith",  "smyth",  "smithe",  "jones",  "jonas", "brown", "browne",     "taylor",
        "tailor", "wilson", "willson", "davies", "davis", "evans", "zolnerowich"};
    return names;
}

class NeighbourhoodFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& town = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        const std::vector<std::string>& names = Surnames();
        // A skewed distribution: name i appears (15 - i) times, so the rarest
        // value is a singleton and the commonest is fifteen records.
        uint64_t rows = 0;
        for (size_t i = 0; i < names.size(); ++i) {
            for (size_t k = 0; k + i < names.size(); ++k) {
                surname.ids.push_back(surname.dict.Intern(names[i]));
                town.ids.push_back(town.dict.Intern("t" + std::to_string(rows % 3)));
                store_->mutable_ids().Append("r" + std::to_string(rows));
                ++rows;
            }
        }
        records_ = rows;
        store_->set_num_records(rows);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
    uint64_t records_ = 0;
};

// The whole claim of the self-join: it counts what enumerating every ordered pair
// of records would have counted, without enumerating any of them.
TEST_F(NeighbourhoodFixture, LevelUMatchesEnumeratingEveryOrderedPair) {
    cpplink::BallMassTable table;
    std::string reason;
    cpplink::BallOptions options;
    options.threads = 2;
    ASSERT_TRUE(table.Build(comparisons_, 0, records_, options, &reason)) << reason;

    const size_t levels = schema_.comparisons[0].levels.size();
    std::vector<uint64_t> counted(levels, 0);
    for (uint64_t a = 0; a < records_; ++a) {
        for (uint64_t b = 0; b < records_; ++b) {
            ++counted[comparisons_.EvaluateOne(0, a, b)];
        }
    }
    const double denominator = static_cast<double>(records_) * records_;
    for (size_t l = 0; l < levels; ++l) {
        if (!table.Covers(l)) continue;
        EXPECT_NEAR(table.LevelU(l), static_cast<double>(counted[l]) / denominator, 1e-12)
            << "level " << l;
    }
}

// And the masses are the same thing per value: how many records fall in the
// ball of the value this row carries.
TEST_F(NeighbourhoodFixture, MassCountsTheRecordsInEachValuesBall) {
    cpplink::BallMassTable table;
    std::string reason;
    cpplink::BallOptions options;
    options.threads = 2;
    ASSERT_TRUE(table.Build(comparisons_, 0, records_, options, &reason)) << reason;

    const auto& surname = std::get<cpplink::StringColumn>(store_->column(0));
    for (uint64_t a = 0; a < records_; ++a) {
        std::vector<uint64_t> reached(schema_.comparisons[0].levels.size(), 0);
        for (uint64_t b = 0; b < records_; ++b) {
            ++reached[comparisons_.EvaluateOne(0, a, b)];
        }
        for (size_t l = 0; l < reached.size(); ++l) {
            if (!table.Covers(l)) continue;
            EXPECT_NEAR(table.Mass(l, surname.ids[a]),
                        static_cast<double>(reached[l]) / static_cast<double>(records_),
                        1e-12)
                << "row " << a << " level " << l;
        }
    }
}

TEST_F(NeighbourhoodFixture, RefusesADictionaryOverTheBudget) {
    cpplink::BallMassTable table;
    std::string reason;
    cpplink::BallOptions options;
    options.budget = 4;  // fifteen values is a hundred and five pairs
    EXPECT_FALSE(table.Build(comparisons_, 0, records_, options, &reason));
    EXPECT_NE(reason.find("over the budget"), std::string::npos) << reason;

    // And a comparison with no fuzzy level has nothing to compute: its ball is
    // the value itself, which the term frequencies already say.
    EXPECT_FALSE(table.Build(comparisons_, 1, records_, cpplink::BallOptions(), &reason));
    EXPECT_NE(reason.find("no fuzzy level"), std::string::npos) << reason;
}

// The predicate the self-join runs must be the predicate the pair path runs, or
// the masses describe a different model from the one scoring uses.
TEST_F(NeighbourhoodFixture, ValueLevelsAgreeWithRowLevels) {
    const auto& surname = std::get<cpplink::StringColumn>(store_->column(0));
    for (uint64_t a = 0; a < records_; a += 3) {
        for (uint64_t b = 0; b < records_; b += 5) {
            EXPECT_EQ(comparisons_.LevelForValues(0, surname.ids[a], surname.ids[b]),
                      comparisons_.EvaluateOne(0, a, b))
                << a << " " << b;
        }
    }
}

// The bracket is what the drop and emit decisions rest on, and a fuzzy level's
// adjustment has to sit inside it exactly as an exact level's does.
TEST_F(NeighbourhoodFixture, TheBracketContainsEveryFuzzyAdjustedWeight) {
    cpplink::BallTables balls;
    cpplink::BallOptions options;
    options.threads = 2;
    balls.Build(comparisons_, records_, options);
    ASSERT_TRUE(balls.Has(0));

    cpplink::Model model;
    model.lambda = 0.01;
    model.records = records_;
    for (size_t c = 0; c < comparisons_.Size(); ++c) {
        cpplink::ModelComparison comparison;
        comparison.name = schema_.comparisons[c].name;
        comparison.sessions = 2;
        const size_t levels = schema_.comparisons[c].levels.size();
        for (size_t l = 0; l < levels; ++l) {
            cpplink::ModelLevel level;
            // Anything monotone will do; the bracket is a property of the masses,
            // not of the parameters.
            level.m = 0.9 / static_cast<double>(l + 1);
            level.u = 0.02 / static_cast<double>(l + 1);
            level.m_estimated = true;
            comparison.levels.push_back(level);
        }
        model.comparisons.push_back(comparison);
    }

    cpplink::ScoreOptions score;
    score.threshold = 0.0;
    cpplink::Scorer scorer;
    std::string error;
    ASSERT_TRUE(scorer.Bind(model, comparisons_, *store_, score, &error, &balls))
        << error;
    EXPECT_TRUE(scorer.AdjustsFuzzyLevels());

    bool saw_fuzzy_move = false;
    for (uint64_t a = 0; a < records_; ++a) {
        for (uint64_t b = a + 1; b < records_; ++b) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            const double base = scorer.BaseWeight(gamma);
            const double weight = scorer.Weight(gamma, a, b);
            EXPECT_LE(weight, base + scorer.DeltaMax(gamma) + 1e-9)
                << "rows " << a << " " << b;
            EXPECT_GE(weight, base + scorer.DeltaMin(gamma) - 1e-9)
                << "rows " << a << " " << b;
            const uint8_t level = comparisons_.LevelOf(gamma, 0);
            if (level == 2 || level == 3) {
                if (std::abs(weight - base) > 1e-9) saw_fuzzy_move = true;
            }
        }
    }
    EXPECT_TRUE(saw_fuzzy_move) << "the fixture must contain a fuzzy agreement";
}

// Without the tables the scorer is what it always was, so turning the feature on
// and off has to be visible in exactly one place.
TEST_F(NeighbourhoodFixture, WithoutTablesOnlyTheExactLevelMoves) {
    cpplink::Model model;
    model.lambda = 0.01;
    model.records = records_;
    for (size_t c = 0; c < comparisons_.Size(); ++c) {
        cpplink::ModelComparison comparison;
        comparison.name = schema_.comparisons[c].name;
        const size_t levels = schema_.comparisons[c].levels.size();
        for (size_t l = 0; l < levels; ++l) {
            cpplink::ModelLevel level;
            level.m = 0.9 / static_cast<double>(l + 1);
            level.u = 0.02 / static_cast<double>(l + 1);
            comparison.levels.push_back(level);
        }
        model.comparisons.push_back(comparison);
    }
    cpplink::ScoreOptions score;
    cpplink::Scorer scorer;
    std::string error;
    ASSERT_TRUE(scorer.Bind(model, comparisons_, *store_, score, &error)) << error;
    EXPECT_FALSE(scorer.AdjustsFuzzyLevels());
    for (uint64_t a = 0; a < records_; a += 7) {
        for (uint64_t b = a + 1; b < records_; b += 11) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            if (comparisons_.LevelOf(gamma, 0) == 1) continue;  // exact
            EXPECT_DOUBLE_EQ(scorer.Weight(gamma, a, b), scorer.BaseWeight(gamma));
        }
    }
}

}  // namespace
