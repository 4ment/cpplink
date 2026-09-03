// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/estimate.hpp"

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

constexpr uint64_t kRecords = 1200;
constexpr uint64_t kPlanted = 100;  // duplicate pairs at rows (2i, 2i + 1)
constexpr uint32_t kSurnames = 20;
constexpr uint32_t kCities = 4;
constexpr uint32_t kPostcodes = 50;

// A surname disagrees on every fifth planted pair and a city on every third, so
// the agreement rates the estimator has to recover are fixed by construction and
// counted below rather than written down as magic numbers.
bool SurnameAgrees(uint64_t pair) { return pair % 5 != 0; }
bool CityAgrees(uint64_t pair) { return pair % 3 != 0; }
bool PostcodeAgrees(uint64_t pair) { return pair % 7 != 0; }

// Singleton values are drawn through a mixing function rather than from row * k
// modulo the pool size: with plain multiples the residues stay in lockstep, so
// surname would determine city and the comparisons would not be conditionally
// independent -- which is the one assumption the Fellegi-Sunter model makes.
uint32_t Draw(uint64_t row, uint64_t salt, uint32_t pool) {
    uint64_t value = row * 0x9E3779B97F4A7C15ull + salt;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return static_cast<uint32_t>((value ^ (value >> 31)) % pool);
}

const char* const kSchemaJson = R"({
  "columns": [
    {"name": "email", "type": "string"},
    {"name": "surname", "type": "string"},
    {"name": "city", "type": "string"},
    {"name": "postcode", "type": "string"}
  ],
  "comparisons": [
    {"name": "email", "columns": ["email"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "surname", "columns": ["surname"],
     "levels": [{"type": "null"}, {"type": "exact"},
                {"type": "jaro_winkler", "threshold": 0.9}, {"type": "else"}]},
    {"name": "city", "columns": ["city"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "postcode", "columns": ["postcode"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "email"},
    {"type": "exact_value", "column": "city"}
  ]
})";

class EstimateFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& email = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        auto& postcode = std::get<cpplink::StringColumn>(store_->mutable_column(3));

        // Distinct names, not "surname0".."surname19": a shared prefix would put
        // every pair over the Jaro-Winkler threshold and leave the exact level
        // with nothing to separate.
        static const char* const kNames[kSurnames] = {
            "smith",  "jones",   "brown", "taylor", "wilson", "davies", "evans",
            "thomas", "walker",  "green", "hall",   "wood",   "harris", "martin",
            "clarke", "jackson", "white", "moore",  "bell",   "cooper"};
        std::vector<uint32_t> surnames;
        for (uint32_t i = 0; i < kSurnames; ++i) {
            surnames.push_back(surname.dict.Intern(kNames[i]));
        }
        std::vector<uint32_t> cities;
        for (uint32_t i = 0; i < kCities; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        std::vector<uint32_t> postcodes;
        for (uint32_t i = 0; i < kPostcodes; ++i) {
            postcodes.push_back(postcode.dict.Intern("pc" + std::to_string(i)));
        }

        email.ids.assign(kRecords, 0);
        surname.ids.assign(kRecords, 0);
        city.ids.assign(kRecords, 0);
        postcode.ids.assign(kRecords, 0);
        for (uint64_t i = 0; i < kPlanted; ++i) {
            const uint64_t a = 2 * i;
            const uint64_t b = 2 * i + 1;
            const uint32_t shared = email.dict.Intern("dup" + std::to_string(i));
            email.ids[a] = shared;
            email.ids[b] = shared;
            surname.ids[a] = surnames[i % kSurnames];
            surname.ids[b] =
                surnames[SurnameAgrees(i) ? i % kSurnames : (i + 7) % kSurnames];
            city.ids[a] = cities[i % kCities];
            city.ids[b] = cities[CityAgrees(i) ? i % kCities : (i + 1) % kCities];
            postcode.ids[a] = postcodes[i % kPostcodes];
            postcode.ids[b] =
                postcodes[PostcodeAgrees(i) ? i % kPostcodes : (i + 11) % kPostcodes];
        }
        for (uint64_t row = 2 * kPlanted; row < kRecords; ++row) {
            email.ids[row] = email.dict.Intern("solo" + std::to_string(row));
            surname.ids[row] = surnames[Draw(row, 1, kSurnames)];
            city.ids[row] = cities[Draw(row, 2, kCities)];
            postcode.ids[row] = postcodes[Draw(row, 3, kPostcodes)];
        }

        store_->set_num_records(kRecords);
        store_->Finalize();
        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;

        options_.u_sample = 200000;
        options_.threads = 2;
        options_.seed = 7;
    }

    bool Run() {
        std::string error;
        const bool ok = cpplink::Estimate(*store_, comparisons_, plan_, options_, &model_,
                                          &report_, &error);
        if (!ok) message_ = error;
        return ok;
    }

    const cpplink::ModelComparison& Comparison(const std::string& name) const {
        for (const cpplink::ModelComparison& entry : model_.comparisons) {
            if (entry.name == name) return entry;
        }
        ADD_FAILURE() << "no comparison named " << name;
        return model_.comparisons.front();
    }

    const cpplink::SessionReport& Session(const std::string& column) const {
        for (const cpplink::SessionReport& session : report_.sessions) {
            if (session.column == column) return session;
        }
        ADD_FAILURE() << "no session on " << column;
        return report_.sessions.front();
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

// The closed form is the whole reason u needs no sampling at this cardinality:
// it is the term frequencies' second moment, and it must be exact, not close.
TEST_F(EstimateFixture, ExactLevelUComesFromTheTermFrequenciesExactly) {
    ASSERT_TRUE(Run()) << message_;

    const auto& email = std::get<cpplink::StringColumn>(store_->column(0));
    double expected = 0.0;
    for (const uint32_t frequency : email.tf) {
        const double share = static_cast<double>(frequency) / kRecords;
        expected += share * share;
    }
    const cpplink::ModelLevel& level = Comparison("email").levels[1];
    EXPECT_TRUE(level.u_exact);
    EXPECT_NEAR(level.u, expected, 1e-15);
    // 100 values seen twice and 1000 seen once, over 1200 records.
    EXPECT_NEAR(level.u, (100 * 4.0 + 1000 * 1.0) / (1200.0 * 1200.0), 1e-15);
}

TEST_F(EstimateFixture, EveryComparisonsLevelsSumToOne) {
    ASSERT_TRUE(Run()) << message_;
    for (const cpplink::ModelComparison& comparison : model_.comparisons) {
        double total = 0.0;
        for (const cpplink::ModelLevel& level : comparison.levels) total += level.u;
        EXPECT_NEAR(total, 1.0, 1e-9) << comparison.name;
    }
}

// The data has no missing value anywhere, so the null level is not a probability
// to be sampled: it is zero, and the closed form has to say so.
TEST_F(EstimateFixture, NullLevelUIsCountedNotSampled) {
    ASSERT_TRUE(Run()) << message_;
    const cpplink::ModelLevel& level = Comparison("email").levels[0];
    EXPECT_TRUE(level.u_exact);
    EXPECT_LT(level.u, 1e-11);
}

// Blocking on email reaches only the planted pairs, so the session is close to a
// deterministic rule and its m is the agreement rate among known duplicates.
TEST_F(EstimateFixture, EmailSessionRecoversTheAgreementRateAmongDuplicates) {
    ASSERT_TRUE(Run()) << message_;
    const cpplink::SessionReport& session = Session("email");
    EXPECT_EQ(session.enumerated, kPlanted);
    ASSERT_TRUE(session.merged);

    uint64_t agreed = 0;
    for (uint64_t i = 0; i < kPlanted; ++i) {
        if (SurnameAgrees(i)) ++agreed;
    }
    EXPECT_GT(session.lambda, 0.9);
    EXPECT_NEAR(Comparison("surname").levels[1].m, static_cast<double>(agreed) / kPlanted,
                0.06);
}

// Blocking on city buries 66 matches in roughly 180,000 pairs, which is the case
// EM actually exists for: the split has to come out of the pattern counts.
TEST_F(EstimateFixture, CitySessionSeparatesMatchesFromABlockOfMostlyNonMatches) {
    ASSERT_TRUE(Run()) << message_;
    const cpplink::SessionReport& session = Session("city");
    EXPECT_TRUE(session.merged);
    EXPECT_TRUE(session.converged);
    EXPECT_GT(session.enumerated, 100000u);

    uint64_t reachable = 0;
    for (uint64_t i = 0; i < kPlanted; ++i) {
        if (CityAgrees(i)) ++reachable;
    }
    // EM's match mass is a posterior, so it undercounts on purpose: a pair that
    // agrees only on email is worth 10 bits against a 12-bit prior, which is not
    // enough to call it a match. That is why lambda is reported as a lower bound
    // and not as a count.
    EXPECT_LT(session.implied_matches, static_cast<double>(reachable) * 1.1);
    EXPECT_GT(session.implied_matches, static_cast<double>(reachable) * 0.6);
    // Every planted pair shares an email, and no other pair in the block does.
    EXPECT_GT(Comparison("email").levels[1].m, 0.95);
}

// A session cannot estimate the comparison it blocked on: gamma is constant there
// by construction. Every column therefore has to be covered by another session.
TEST_F(EstimateFixture, EachSessionHoldsOutTheColumnItBlockedOn) {
    ASSERT_TRUE(Run()) << message_;
    EXPECT_EQ(Session("email").excluded, std::vector<std::string>{"email"});
    EXPECT_EQ(Session("city").excluded, std::vector<std::string>{"city"});
    EXPECT_EQ(Comparison("email").sessions, 1u);
    EXPECT_EQ(Comparison("city").sessions, 1u);
    EXPECT_EQ(Comparison("surname").sessions, 2u);
    EXPECT_EQ(Comparison("postcode").sessions, 2u);
}

// A blocked session's match rate is not lambda: it is the match rate inside the
// block, which blocking raised on purpose. lambda has to be put back on the full
// pair count.
TEST_F(EstimateFixture, LambdaIsPutBackOnTheFullPairCount) {
    ASSERT_TRUE(Run()) << message_;
    const double all_pairs = kRecords * (kRecords - 1.0) / 2.0;
    EXPECT_LT(model_.lambda, 1e-3);
    EXPECT_NEAR(model_.lambda, kPlanted / all_pairs, 0.5 * kPlanted / all_pairs);
    EXPECT_NE(model_.lambda_basis.find("lower bound"), std::string::npos);
    EXPECT_LT(model_.PriorWeight(), 0.0);
}

TEST_F(EstimateFixture, GivenLambdaOverridesTheDerivedOne) {
    options_.lambda = 1e-5;
    ASSERT_TRUE(Run()) << message_;
    EXPECT_DOUBLE_EQ(model_.lambda, 1e-5);
    EXPECT_EQ(model_.lambda_basis, "given on the command line");
}

// Agreement must be worth positive evidence and disagreement negative, or the
// model is inverted and nothing downstream of it means anything.
TEST_F(EstimateFixture, AgreementWeighsMoreThanDisagreement) {
    ASSERT_TRUE(Run()) << message_;
    for (const cpplink::ModelComparison& comparison : model_.comparisons) {
        const cpplink::ModelLevel& last = comparison.levels.back();
        for (size_t l = 0; l + 1 < comparison.levels.size(); ++l) {
            // A level nothing reached has a floored m, not an estimate, and the
            // report flags it rather than pretending it carries evidence.
            if (comparison.levels[l].m_support == 0.0) continue;
            EXPECT_GT(comparison.levels[l].Weight(), last.Weight()) << comparison.name;
        }
        EXPECT_LT(last.Weight(), 0.0) << comparison.name;
    }
}

TEST_F(EstimateFixture, RefusesToEstimateWithoutComparisons) {
    cpplink::ComparisonSet empty;
    cpplink::Model model;
    cpplink::EstimateReport report;
    std::string error;
    EXPECT_FALSE(
        cpplink::Estimate(*store_, empty, plan_, options_, &model, &report, &error));
    EXPECT_NE(error.find("no comparisons"), std::string::npos);
}

TEST_F(EstimateFixture, ReportNamesTheSessionsAndWhatTheyHeldOut) {
    ASSERT_TRUE(Run()) << message_;
    std::ostringstream out;
    cpplink::PrintEstimateReport(report_, out);
    const std::string text = out.str();
    EXPECT_NE(text.find("Session email"), std::string::npos);
    EXPECT_NE(text.find("held out"), std::string::npos);
    EXPECT_NE(text.find("closed form"), std::string::npos);
}

}  // namespace
