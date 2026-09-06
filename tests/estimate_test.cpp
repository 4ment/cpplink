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
// it is a count over the term frequencies, and it must be exact, not close.
//
// The count is of ordered pairs of *distinct* rows, which is what a run can see
// and what `SampleRandomPairs` draws. Taking it over every ordered draw instead
// puts each row's agreement with itself in the numerator and denominator both,
// which is `1/records` of u -- nothing while u is large, and the whole of u for a
// near-unique column. This fixture is that column: 1,100 distinct emails over
// 1,200 rows, where the wrong denominator reads u seven times too large and costs
// the strongest comparison in the schema 2.8 bits of weight.
TEST_F(EstimateFixture, ExactLevelUComesFromTheTermFrequenciesExactly) {
    ASSERT_TRUE(Run()) << message_;

    const auto& email = std::get<cpplink::StringColumn>(store_->column(0));
    double expected = 0.0;
    for (const uint32_t frequency : email.tf) {
        const double count = static_cast<double>(frequency);
        expected += count * (count - 1.0);
    }
    expected /= static_cast<double>(kRecords) * (kRecords - 1);
    const cpplink::ModelLevel& level = Comparison("email").levels[1];
    EXPECT_TRUE(level.u_exact);
    EXPECT_NEAR(level.u, expected, 1e-15);
    // 100 values seen twice and 1000 seen once: 200 ordered agreeing pairs out of
    // the 1200 * 1199 ordered pairs of distinct rows, and not one pair more.
    EXPECT_NEAR(level.u, 200.0 / (1200.0 * 1199.0), 1e-15);
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
    // EM's match mass is a posterior, so it lands near the count rather than on
    // it. How near turns on what an email agreement is worth: over the ordered
    // pairs of distinct rows it is 12.8 bits against this block's 11.4-bit prior,
    // which is enough to call such a pair a match, and the implied count comes out
    // within a tenth of the truth. Over every ordered draw it was 10.0 bits, not
    // enough, and the same session read 52 matches where there are 66.
    EXPECT_LT(session.implied_matches, static_cast<double>(reachable) * 1.15);
    EXPECT_GT(session.implied_matches, static_cast<double>(reachable) * 0.9);
    // Every planted pair shares an email and no other pair in the block does, so
    // the truth is 1. EM does not reach it, and the gap is its own: finding the
    // matches it was missing also pulls in the pairs agreeing on surname and
    // postcode by chance, and those carry no email agreement to credit. The same
    // trade shows up as m for surname and postcode moving *towards* their known
    // rates, which is the direction that matters.
    EXPECT_GT(Comparison("email").levels[1].m, 0.8);
    EXPECT_NEAR(Comparison("surname").levels[1].m, 0.8, 0.05);  // 4 pairs in 5
    EXPECT_NEAR(Comparison("postcode").levels[0].m, 6.0 / 7.0, 0.05);
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

// ---------------------------------------------------------------------------
// A column that contains another, which is the shape that breaks a session.

constexpr uint64_t kTieRecords = 2000;
constexpr uint64_t kTiePairs = 200;  // planted at rows (2i, 2i + 1)
constexpr uint32_t kFirsts = 20;
constexpr uint32_t kLasts = 200;
constexpr uint32_t kPlaces = 50;
constexpr uint32_t kJobs = 10;

bool FirstAgrees(uint64_t pair) { return pair % 5 != 0; }  // 0.8
bool LastAgrees(uint64_t pair) { return pair % 10 >= 3; }  // 0.7
bool PlaceAgrees(uint64_t pair) { return pair % 5 >= 2; }  // 0.6
bool JobAgrees(uint64_t pair) { return pair % 10 != 0; }   // 0.9

// `full` is "first_last", so `first` occurs inside it on every row and a pair
// agreeing on `first` is already most of the way to the fuzzy level of `full`:
// "john_smith" against "john_brown" is a Jaro-Winkler of about 0.80 on a shared
// prefix alone. Blocking on `first` therefore conditions on `full`, and the u the
// model holds for `full` is nothing like the rate seen inside that session.
const char* const kTieSchemaJson = R"({
  "columns": [
    {"name": "first", "type": "string"},
    {"name": "last", "type": "string"},
    {"name": "full", "type": "string"},
    {"name": "place", "type": "string"},
    {"name": "job", "type": "string"}
  ],
  "comparisons": [
    {"name": "first", "columns": ["first"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "last", "columns": ["last"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "full", "columns": ["full"],
     "levels": [{"type": "exact"},
                {"type": "jaro_winkler", "threshold": 0.8}, {"type": "else"}]},
    {"name": "place", "columns": ["place"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "job", "columns": ["job"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "first"},
    {"type": "exact_value", "column": "place"}
  ]
})";

class TieFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kTieSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);
        std::vector<uint32_t> first(kTieRecords);
        std::vector<uint32_t> last(kTieRecords);
        std::vector<uint32_t> place(kTieRecords);
        std::vector<uint32_t> job(kTieRecords);
        for (uint64_t row = 0; row < kTieRecords; ++row) {
            first[row] = Draw(row, 11, kFirsts);
            last[row] = Draw(row, 22, kLasts);
            place[row] = Draw(row, 33, kPlaces);
            job[row] = Draw(row, 44, kJobs);
        }
        for (uint64_t i = 0; i < kTiePairs; ++i) {
            const uint64_t a = 2 * i;
            const uint64_t b = 2 * i + 1;
            first[b] = FirstAgrees(i) ? first[a] : (first[a] + 3) % kFirsts;
            last[b] = LastAgrees(i) ? last[a] : (last[a] + 7) % kLasts;
            place[b] = PlaceAgrees(i) ? place[a] : (place[a] + 5) % kPlaces;
            job[b] = JobAgrees(i) ? job[a] : (job[a] + 1) % kJobs;
        }
        auto& c0 = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& c1 = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        auto& c2 = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        auto& c3 = std::get<cpplink::StringColumn>(store_->mutable_column(3));
        auto& c4 = std::get<cpplink::StringColumn>(store_->mutable_column(4));
        for (uint64_t row = 0; row < kTieRecords; ++row) {
            const std::string one = "f" + std::to_string(first[row]);
            const std::string two = "l" + std::to_string(last[row]);
            c0.ids.push_back(c0.dict.Intern(one));
            c1.ids.push_back(c1.dict.Intern(two));
            c2.ids.push_back(c2.dict.Intern(one + "_" + two));
            c3.ids.push_back(c3.dict.Intern("p" + std::to_string(place[row])));
            c4.ids.push_back(c4.dict.Intern("j" + std::to_string(job[row])));
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kTieRecords);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
        ASSERT_TRUE(plan_.Build(schema_, *store_, cpplink::PairMode::kAll, &error))
            << error;
    }

    cpplink::EstimateReport Run(bool exclude_tied) {
        cpplink::EstimateOptions options;
        options.threads = 1;
        options.exclude_tied = exclude_tied;
        cpplink::Model model;
        cpplink::EstimateReport report;
        std::string error;
        EXPECT_TRUE(
            Estimate(*store_, comparisons_, plan_, options, &model, &report, &error))
            << error;
        return report;
    }

    const cpplink::SessionReport& Session(const cpplink::EstimateReport& report,
                                          const std::string& column) const {
        for (const cpplink::SessionReport& session : report.sessions) {
            if (session.column == column) return session;
        }
        ADD_FAILURE() << "no session for " << column;
        return report.sessions.front();
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
    cpplink::BlockingPlan plan_;
};

// The session blocking on `first` conditions on `full`, because `first` occurs
// inside it, so `full` has to be held out as well as the column the session names.
TEST_F(TieFixture, ASessionHoldsOutWhatItsBlockingConditionsOn) {
    const cpplink::EstimateReport report = Run(true);
    const cpplink::SessionReport& session = Session(report, "first");
    ASSERT_EQ(session.excluded.size(), 1u);
    EXPECT_EQ(session.excluded[0], "first");
    ASSERT_EQ(session.tied.size(), 1u);
    EXPECT_EQ(session.tied[0], "full");
    // And a column tied to nothing is not held out of anything.
    EXPECT_TRUE(Session(report, "place").tied.empty());
}

// What the hold-out is for. Without it, the agreement `full` shows inside this
// session is nothing like its u over the whole file, EM reads the difference as
// evidence of matching, and the match rate runs away from the truth.
TEST_F(TieFixture, WithoutItTheMatchRateRunsAway) {
    const cpplink::EstimateReport with = Run(true);
    const cpplink::SessionReport& held = Session(with, "first");
    // 200 planted pairs agree on `first` four times in five, and they are the only
    // matches this block can hold.
    const double truth =
        0.8 * static_cast<double>(kTiePairs) / static_cast<double>(held.enumerated);
    EXPECT_GT(held.lambda, 0.5 * truth);
    EXPECT_LT(held.lambda, 2.0 * truth);
    EXPECT_TRUE(held.merged) << (held.warnings.empty() ? "" : held.warnings[0]);

    const cpplink::EstimateReport without = Run(false);
    const cpplink::SessionReport& loose = Session(without, "first");
    EXPECT_TRUE(loose.tied.empty());
    // It does not merely drift: it decides every pair in the block is a match, and
    // nothing in the mixture's own likelihood says otherwise.
    EXPECT_GT(loose.lambda, 0.5);
    EXPECT_GT(loose.lambda, 100.0 * held.lambda);
}

// The tie pass is the profile's pairwise pass and reads no candidate pair, so a
// schema whose columns are independent loses nothing to it. The fixture above this
// one is built to be conditionally independent, and this is the regression guard
// that it stays that way.
TEST_F(EstimateFixture, IndependentColumnsAreNeverTiedOut) {
    ASSERT_TRUE(Run()) << message_;
    for (const cpplink::SessionReport& session : report_.sessions) {
        EXPECT_TRUE(session.tied.empty())
            << session.column << " tied out " << session.tied.front();
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
