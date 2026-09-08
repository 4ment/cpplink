// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/interaction.hpp"

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"

namespace {

constexpr uint64_t kRecords = 4000;
constexpr uint64_t kPlanted = 400;  // duplicate pairs at rows (2i, 2i + 1)

uint32_t Draw(uint64_t row, uint64_t salt, uint32_t pool) {
    uint64_t value = row * 0x9E3779B97F4A7C15ull + salt;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return static_cast<uint32_t>((value ^ (value >> 31)) % pool);
}

// `surname_copy` is `surname` written out a second time, which is the dependence
// this whole mechanism exists for: the two columns agree together always, under
// both classes, and a model multiplying their weights counts one signal twice.
// `city` and `postcode` are drawn independently and must stay uncorrected.
const char* const kSchemaJson = R"({
  "columns": [
    {"name": "email", "type": "string"},
    {"name": "surname", "type": "string"},
    {"name": "surname_copy", "type": "string"},
    {"name": "city", "type": "string"},
    {"name": "postcode", "type": "string"}
  ],
  "comparisons": [
    {"name": "email", "columns": ["email"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "surname", "columns": ["surname"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "surname_copy", "columns": ["surname_copy"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "city", "columns": ["city"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "postcode", "columns": ["postcode"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "email"},
    {"type": "exact_value", "column": "city"},
    {"type": "exact_value", "column": "postcode"}
  ]
})";

class InteractionFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& email = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        auto& copy = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(3));
        auto& postcode = std::get<cpplink::StringColumn>(store_->mutable_column(4));

        std::vector<uint32_t> surnames;
        std::vector<uint32_t> copies;
        for (uint32_t i = 0; i < 300; ++i) {
            const std::string name = "name" + std::to_string(i);
            surnames.push_back(surname.dict.Intern(name));
            copies.push_back(copy.dict.Intern(name));
        }
        std::vector<uint32_t> cities;
        for (uint32_t i = 0; i < 12; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        std::vector<uint32_t> postcodes;
        for (uint32_t i = 0; i < 40; ++i) {
            postcodes.push_back(postcode.dict.Intern("pc" + std::to_string(i)));
        }

        email.ids.assign(kRecords, 0);
        surname.ids.assign(kRecords, 0);
        copy.ids.assign(kRecords, 0);
        city.ids.assign(kRecords, 0);
        postcode.ids.assign(kRecords, 0);
        const auto write = [&](uint64_t row, uint32_t name, uint32_t town,
                               uint32_t code) {
            surname.ids[row] = surnames[name];
            copy.ids[row] = copies[name];
            city.ids[row] = cities[town];
            postcode.ids[row] = postcodes[code];
        };
        for (uint64_t i = 0; i < kPlanted; ++i) {
            const uint64_t a = 2 * i;
            const uint64_t b = 2 * i + 1;
            const uint32_t shared = email.dict.Intern("dup" + std::to_string(i));
            email.ids[a] = shared;
            email.ids[b] = shared;
            // A duplicate keeps the surname three times in four, and the copy
            // follows it exactly, which is the dependence to be found.
            const uint32_t name = static_cast<uint32_t>(i % 300);
            write(a, name, static_cast<uint32_t>(i % 12), static_cast<uint32_t>(i % 40));
            write(b, i % 4 == 0 ? static_cast<uint32_t>((i + 7) % 300) : name,
                  static_cast<uint32_t>(i % 3 == 0 ? (i + 1) % 12 : i % 12),
                  static_cast<uint32_t>(i % 5 == 0 ? (i + 3) % 40 : i % 40));
        }
        for (uint64_t row = 2 * kPlanted; row < kRecords; ++row) {
            email.ids[row] = email.dict.Intern("solo" + std::to_string(row));
            write(row, Draw(row, 1, 300), Draw(row, 2, 12), Draw(row, 3, 40));
        }

        store_->set_num_records(kRecords);
        store_->Finalize();
        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;

        options_.u_sample = 400000;
        options_.threads = 2;
        options_.seed = 11;
        options_.interactions.enabled = true;
        options_.interactions.max_terms = 4;
    }

    bool Run() {
        std::string error;
        const bool ok = cpplink::Estimate(*store_, comparisons_, plan_, options_, &model_,
                                          &report_, &error);
        if (!ok) message_ = error;
        return ok;
    }

    const cpplink::InteractionCandidate& Candidate(const std::string& left,
                                                   const std::string& right) const {
        for (const cpplink::InteractionCandidate& entry :
             report_.interactions.candidates) {
            if (entry.left == left && entry.right == right) return entry;
        }
        ADD_FAILURE() << "no candidate " << left << " x " << right;
        return report_.interactions.candidates.front();
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::BlockingPlan plan_;
    cpplink::ComparisonSet comparisons_;
    cpplink::EstimateOptions options_;
    cpplink::Model model_;
    cpplink::EstimateReport report_;
    std::string message_;
};

// IPF's whole point: it moves the margins and leaves every odds ratio where it
// was. The correction is the odds ratios, so a fit that disturbed them would be
// measuring something other than dependence.
TEST(FitToMargins, MovesTheMarginsAndNotTheOddsRatios) {
    std::vector<double> table = {4.0, 1.0, 2.0, 8.0};
    const double before = (table[0] * table[3]) / (table[1] * table[2]);
    const std::vector<double> rows = {0.3, 0.7};
    const std::vector<double> columns = {0.55, 0.45};
    cpplink::FitToMargins(rows, columns, &table);

    EXPECT_NEAR(table[0] + table[1], 0.3, 1e-9);
    EXPECT_NEAR(table[2] + table[3], 0.7, 1e-9);
    EXPECT_NEAR(table[0] + table[2], 0.55, 1e-9);
    EXPECT_NEAR(table[1] + table[3], 0.45, 1e-9);
    const double after = (table[0] * table[3]) / (table[1] * table[2]);
    EXPECT_NEAR(after, before, 1e-9);
}

// A column written out twice is the strongest dependence a schema can carry, and
// the pair has to be found and corrected downwards: the model was crediting two
// independent agreements where there is one.
TEST_F(InteractionFixture, FindsAColumnDuplicatedUnderAnotherName) {
    ASSERT_TRUE(Run()) << message_;
    ASSERT_FALSE(model_.interactions.empty());
    const cpplink::ModelInteraction& first = model_.interactions.front();
    EXPECT_EQ(first.left, "surname");
    EXPECT_EQ(first.right, "surname_copy");
    EXPECT_LT(first.match_bits, -1.0);
    // Level 1 is `exact` in both, and that is the cell the plain model counts
    // twice: two columns holding the same string agree together always, so the
    // second agreement carries no evidence the first did not.
    EXPECT_LT(first.Bits(1, 1), -1.0);
}

// And the independent pair must be left alone. Finding dependence everywhere is
// the failure mode this whole estimator is exposed to, so the null case is the
// half of the test that matters.
TEST_F(InteractionFixture, LeavesTwoIndependentColumnsUncorrected) {
    ASSERT_TRUE(Run()) << message_;
    const cpplink::InteractionCandidate& pair = Candidate("city", "postcode");
    EXPECT_FALSE(pair.admitted);
    EXPECT_LT(std::fabs(pair.match_bits), 0.25);
}

// The ceiling is what lets `predict` skip a pair before comparing it, and it is
// only allowed to do that because it can never fall below the true weight. A
// correction makes the weight non-separable, so this is the property most at risk.
TEST_F(InteractionFixture, TheCeilingStaysAboveEveryWeightItBounds) {
    ASSERT_TRUE(Run()) << message_;
    ASSERT_FALSE(model_.interactions.empty());

    cpplink::Scorer scorer;
    cpplink::ScoreOptions options;
    std::string error;
    ASSERT_TRUE(scorer.Bind(model_, comparisons_, *store_, options, &error)) << error;
    for (uint64_t a = 0; a < 400; ++a) {
        for (uint64_t b = a + 1; b < 400; ++b) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            EXPECT_GE(scorer.Ceiling(a, b) + 1e-9, scorer.Weight(gamma, a, b))
                << "rows " << a << " and " << b;
        }
    }
}

// The waterfall has to total what the scorer computes. A report that plausibly
// explains a different sum from the one that runs is worse than no report.
TEST_F(InteractionFixture, TheWaterfallStillSumsToTheWeight) {
    ASSERT_TRUE(Run()) << message_;
    ASSERT_FALSE(model_.interactions.empty());

    cpplink::Scorer scorer;
    cpplink::ScoreOptions options;
    std::string error;
    ASSERT_TRUE(scorer.Bind(model_, comparisons_, *store_, options, &error)) << error;
    for (uint64_t a = 0; a < 40; ++a) {
        const uint64_t b = a + 1;
        const uint32_t gamma = comparisons_.Evaluate(a, b);
        double running = scorer.PriorWeight();
        for (size_t c = 0; c < comparisons_.Size(); ++c) {
            running += scorer.LevelWeight(c, comparisons_.LevelOf(gamma, c));
            running += scorer.AdjustmentFor(c, gamma, a, b);
        }
        for (size_t i = 0; i < scorer.InteractionCount(); ++i) {
            running += scorer.InteractionBits(i, gamma);
        }
        EXPECT_NEAR(running, scorer.Weight(gamma, a, b), 1e-9);
    }
    std::ostringstream out;
    cpplink::PrintPairWaterfall(*store_, comparisons_, scorer, 0, 1, out);
    EXPECT_NE(out.str().find("(interaction)"), std::string::npos);
}

// Turning the terms off has to score the plain model out of the same file, which
// is how the two are measured against each other.
TEST_F(InteractionFixture, ScoringWithoutTheTermsIgnoresThem) {
    ASSERT_TRUE(Run()) << message_;
    ASSERT_FALSE(model_.interactions.empty());

    cpplink::ScoreOptions off;
    off.use_interactions = false;
    cpplink::Scorer plain;
    cpplink::Scorer corrected;
    std::string error;
    ASSERT_TRUE(plain.Bind(model_, comparisons_, *store_, off, &error)) << error;
    ASSERT_TRUE(
        corrected.Bind(model_, comparisons_, *store_, cpplink::ScoreOptions(), &error))
        << error;
    EXPECT_EQ(plain.InteractionCount(), 0u);
    // The two scorers differ by exactly the terms and by nothing else, whichever
    // way a given pattern's correction happens to point.
    bool moved = false;
    for (uint64_t a = 0; a < 60; ++a) {
        const uint32_t gamma = comparisons_.Evaluate(a, a + 1);
        double correction = 0.0;
        for (size_t i = 0; i < corrected.InteractionCount(); ++i) {
            correction += corrected.InteractionBits(i, gamma);
        }
        EXPECT_NEAR(corrected.BaseWeight(gamma), plain.BaseWeight(gamma) + correction,
                    1e-9);
        moved = moved || correction != 0.0;
    }
    EXPECT_TRUE(moved);
}

// A term is a claim about the data and has to survive the file it is written to.
TEST_F(InteractionFixture, TheTermsSurviveTheModelFile) {
    ASSERT_TRUE(Run()) << message_;
    ASSERT_FALSE(model_.interactions.empty());

    cpplink::Model reloaded;
    std::string error;
    ASSERT_TRUE(cpplink::ParseModelJson(cpplink::ModelJson(model_), &reloaded, &error))
        << error;
    ASSERT_EQ(reloaded.interactions.size(), model_.interactions.size());
    for (size_t i = 0; i < model_.interactions.size(); ++i) {
        EXPECT_EQ(reloaded.interactions[i].left, model_.interactions[i].left);
        EXPECT_EQ(reloaded.interactions[i].right, model_.interactions[i].right);
        ASSERT_EQ(reloaded.interactions[i].bits.size(),
                  model_.interactions[i].bits.size());
        for (size_t c = 0; c < model_.interactions[i].bits.size(); ++c) {
            EXPECT_DOUBLE_EQ(reloaded.interactions[i].bits[c],
                             model_.interactions[i].bits[c]);
        }
    }
}

// gamma is meaningless without the levels that made it, so a term fitted over one
// level count must be refused against a schema declaring another rather than
// silently indexing a different cell.
TEST_F(InteractionFixture, RefusesATermWhoseLevelCountsNoLongerMatch) {
    ASSERT_TRUE(Run()) << message_;
    ASSERT_FALSE(model_.interactions.empty());
    model_.interactions.front().left_levels += 1;

    cpplink::Scorer scorer;
    std::string error;
    EXPECT_FALSE(
        scorer.Bind(model_, comparisons_, *store_, cpplink::ScoreOptions(), &error));
    EXPECT_NE(error.find("was fitted over"), std::string::npos) << error;
}

}  // namespace
