// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/search.hpp"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/comparison.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"

namespace {

constexpr uint64_t kRecords = 500;

// Four comparisons of three different shapes: two the searcher can tabulate over
// a dictionary, one over a derived column it has to compute for the query, and a
// date, which is not a function of a value id and so keeps the pair path's own
// evaluation. A search that gets the same answer either way has to get both right.
const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "surname", "type": "string"},
    {"name": "city", "type": "string"},
    {"name": "dob", "type": "date"},
    {"name": "surname_key", "derive": {"from": "surname", "transform": "soundex"}}
  ],
  "comparisons": [
    {"name": "surname", "columns": ["surname"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"},
                {"type": "jaro_winkler", "threshold": 0.85}, {"type": "else"}]},
    {"name": "city", "columns": ["city"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "dob", "columns": ["dob"],
     "levels": [{"type": "null"}, {"type": "exact"},
                {"type": "date_within", "threshold": 400}, {"type": "else"}]},
    {"name": "surname_key", "columns": ["surname_key"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "city"}]
})";

cpplink::ModelComparison Comparison(const std::string& name, bool tf,
                                    const std::vector<std::pair<double, double>>& rates) {
    cpplink::ModelComparison comparison;
    comparison.name = name;
    comparison.term_frequency = tf;
    for (const auto& entry : rates) {
        cpplink::ModelLevel level;
        level.m = entry.first;
        level.u = entry.second;
        level.m_estimated = true;
        comparison.levels.push_back(level);
    }
    return comparison;
}

// The same ordering the searcher uses: the best weight first, and where two tie,
// the lower row.
struct Scored {
    uint64_t row = 0;
    uint32_t gamma = 0;
    double weight = 0.0;
};

bool BetterThan(const Scored& left, const Scored& right) {
    if (left.weight != right.weight) return left.weight > right.weight;
    return left.row < right.row;
}

class SearchFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        auto& dob = std::get<cpplink::DateColumn>(store_->mutable_column(2));
        // A skewed surname distribution, so the term-frequency adjustment has a
        // spread and the bracket is not degenerate; names one edit apart, so the
        // fuzzy level fires on something.
        std::vector<uint32_t> names;
        for (int i = 0; i < 30; ++i) {
            names.push_back(surname.dict.Intern("sander" + std::to_string(i % 10) +
                                                (i < 10 ? "" : "s")));
        }
        std::vector<uint32_t> cities;
        for (int i = 0; i < 5; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        for (uint64_t row = 0; row < kRecords; ++row) {
            const size_t pick = static_cast<size_t>((row * row) % 30) / 2;
            // Every seventh record is missing its surname, so the null level and
            // the derived column's empty result are both exercised.
            surname.ids.push_back(row % 7 == 3 ? cpplink::kNullId : names[pick]);
            city.ids.push_back(cities[(row * 11) % 5]);
            dob.values.push_back(row % 11 == 5
                                     ? cpplink::kNullDate
                                     : static_cast<int32_t>(4000 + (row * 37) % 900));
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();

        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
        model_.lambda = 0.02;
        model_.records = kRecords;
        model_.comparisons = {
            Comparison("surname", true,
                       {{1e-9, 1e-9}, {0.75, 0.03}, {0.15, 0.05}, {0.10, 0.92}}),
            Comparison("city", false, {{0.7, 0.2}, {0.3, 0.8}}),
            Comparison("dob", false,
                       {{1e-9, 1e-9}, {0.6, 0.002}, {0.25, 0.4}, {0.15, 0.598}}),
            Comparison("surname_key", false, {{1e-9, 1e-9}, {0.8, 0.06}, {0.2, 0.94}})};
        cpplink::ScoreOptions score;
        ASSERT_TRUE(scorer_.Bind(model_, comparisons_, *store_, score, &error)) << error;
    }

    // Every row scored the expensive way against the query the searcher installed,
    // which is what the top-k answer has to agree with.
    std::vector<Scored> ScoreEveryRow(uint64_t query_row) const {
        std::vector<Scored> all;
        all.reserve(kRecords);
        for (uint64_t row = 0; row < kRecords; ++row) {
            Scored one;
            one.row = row;
            one.gamma = comparisons_.Evaluate(row, query_row);
            one.weight = scorer_.Weight(one.gamma, row, query_row);
            all.push_back(one);
        }
        std::sort(all.begin(), all.end(), BetterThan);
        return all;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
    cpplink::Model model_;
    cpplink::Scorer scorer_;
};

cpplink::QueryRecord Query() {
    cpplink::QueryRecord query;
    query.Set("surname", "sander3s");
    query.Set("city", "city2");
    query.Set("dob", "1981-01-24");
    return query;
}

// The claim the whole design rests on: the k rows the two-phase search returns are
// the k rows scoring every pair exactly would return, in the same order. The
// bound it prunes on is admissible or this fails.
TEST_F(SearchFixture, MatchesScoringEveryRow) {
    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;

    cpplink::SearchOptions options;
    options.k = 12;
    cpplink::SearchReport report;
    ASSERT_TRUE(searcher.Search(Query(), options, &report, &error)) << error;

    const std::vector<Scored> all = ScoreEveryRow(searcher.QueryRow());
    ASSERT_EQ(report.hits.size(), options.k);
    for (size_t i = 0; i < report.hits.size(); ++i) {
        EXPECT_EQ(report.hits[i].row, all[i].row) << "at rank " << i;
        EXPECT_EQ(report.hits[i].gamma, all[i].gamma) << "at rank " << i;
        EXPECT_DOUBLE_EQ(report.hits[i].weight, all[i].weight) << "at rank " << i;
        EXPECT_EQ(report.hits[i].id, "r" + std::to_string(all[i].row));
    }
    // Three comparisons are a function of one value id and are answered from a
    // table; the date is not, and keeps the pair path's evaluation.
    EXPECT_EQ(report.tabled, 3u);
    EXPECT_EQ(report.evaluated, 1u);
    EXPECT_EQ(report.constant, 0u);
    // The bracket is what makes the exact weight optional, so most rows never
    // reach it.
    EXPECT_LT(report.rescored, kRecords);
}

// And the level every row lands on is the level the pair path assigns it, not
// only for the winners: a table that disagreed anywhere would be a different
// model, whether or not it changed the top of the list.
TEST_F(SearchFixture, EveryRowsPatternIsThePairPaths) {
    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;

    cpplink::SearchOptions options;
    options.k = kRecords;
    cpplink::SearchReport report;
    ASSERT_TRUE(searcher.Search(Query(), options, &report, &error)) << error;
    ASSERT_EQ(report.hits.size(), kRecords);
    for (const cpplink::SearchHit& hit : report.hits) {
        EXPECT_EQ(hit.gamma, comparisons_.Evaluate(hit.row, searcher.QueryRow()))
            << "row " << hit.row;
    }
    // Every row was scored exactly, so nothing was pruned and nothing was missed.
    EXPECT_EQ(report.rescored, kRecords);
}

// A record already in the store is its own best answer, which is the sanity check
// that the query's values reach the dictionary ids the store holds.
TEST_F(SearchFixture, AStoredRecordFindsItself) {
    const auto& surname = std::get<cpplink::StringColumn>(store_->column(0));
    const auto& city = std::get<cpplink::StringColumn>(store_->column(1));
    uint64_t subject = 0;
    while (surname.ids[subject] == cpplink::kNullId) ++subject;

    cpplink::QueryRecord query;
    query.Set("surname", std::string(surname.dict.Value(surname.ids[subject])));
    query.Set("city", std::string(city.dict.Value(city.ids[subject])));

    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;
    cpplink::SearchOptions options;
    options.k = 5;
    cpplink::SearchReport report;
    ASSERT_TRUE(searcher.Search(query, options, &report, &error)) << error;
    ASSERT_FALSE(report.hits.empty());
    // The surname agrees exactly and so does its key and the city, which is the
    // best any record can do against this query.
    EXPECT_EQ(comparisons_.Evaluate(subject, searcher.QueryRow()),
              report.hits.front().gamma);
    EXPECT_GT(report.hits.front().weight, 0.0);
    EXPECT_EQ(report.values_adopted, 0u);
}

// A value the store never held is still a value: the fuzzy level needs its text
// and its signature, both of which only a value id addresses.
TEST_F(SearchFixture, AValueTheStoreNeverHeldIsStillCompared) {
    const auto& surname = std::get<cpplink::StringColumn>(store_->column(0));
    const uint32_t before = surname.dict.Size();

    cpplink::QueryRecord query;
    query.Set("surname", "sander3x");  // one edit from "sander3", which is here
    query.Set("city", "city2");

    std::string error;
    {
        cpplink::Searcher searcher;
        ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error))
            << error;
        cpplink::SearchOptions options;
        options.k = 5;
        cpplink::SearchReport report;
        ASSERT_TRUE(searcher.Search(query, options, &report, &error)) << error;
        // The surname is new; its soundex key is not, because a key is what two
        // spellings of a name have in common.
        EXPECT_EQ(report.values_adopted, 1u);
        ASSERT_FALSE(report.hits.empty());
        // Nothing agrees exactly, but the fuzzy level does, which is the whole
        // reason a query value has to become a value id at all.
        const uint8_t level = comparisons_.LevelOf(report.hits.front().gamma, 0);
        EXPECT_EQ(level, 2u);
    }
    // And the store is what it was: the adoptions are taken back with the row.
    EXPECT_EQ(surname.dict.Size(), before);
    EXPECT_EQ(surname.ids.size(), kRecords);
    EXPECT_EQ(surname.tf.size(), before);
    EXPECT_EQ(store_->NumRecords(), kRecords);
}

// The query row is removed when the searcher goes, and the pairs the store
// already held score what they scored before it arrived.
TEST_F(SearchFixture, TheStoreIsLeftAsItWasFound) {
    const uint32_t gamma_before = comparisons_.Evaluate(3, 17);
    const double weight_before = scorer_.Weight(gamma_before, 3, 17);
    {
        cpplink::Searcher searcher;
        std::string error;
        ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error))
            << error;
        cpplink::SearchOptions options;
        cpplink::SearchReport report;
        ASSERT_TRUE(searcher.Search(Query(), options, &report, &error)) << error;
    }
    EXPECT_EQ(comparisons_.Evaluate(3, 17), gamma_before);
    EXPECT_DOUBLE_EQ(scorer_.Weight(gamma_before, 3, 17), weight_before);
    const auto& ids = store_->ids();
    EXPECT_EQ(ids.offsets.size(), kRecords + 1);
    EXPECT_EQ(ids.Get(kRecords - 1), "r" + std::to_string(kRecords - 1));
}

// Splitting the rows over threads is a partition of the same scan, so it is the
// same answer or it is a bug.
TEST_F(SearchFixture, ThreadsChangeNothing) {
    std::string error;
    cpplink::SearchReport one;
    cpplink::SearchReport many;
    {
        cpplink::Searcher searcher;
        ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error))
            << error;
        cpplink::SearchOptions options;
        options.k = 9;
        options.threads = 1;
        ASSERT_TRUE(searcher.Search(Query(), options, &one, &error)) << error;
        options.threads = 4;
        ASSERT_TRUE(searcher.Search(Query(), options, &many, &error)) << error;
    }
    ASSERT_EQ(one.hits.size(), many.hits.size());
    for (size_t i = 0; i < one.hits.size(); ++i) {
        EXPECT_EQ(one.hits[i].row, many.hits[i].row) << "at rank " << i;
        EXPECT_DOUBLE_EQ(one.hits[i].weight, many.hits[i].weight) << "at rank " << i;
    }
}

// The prior is the one thing search does not inherit from the model, and it moves
// every hit by the same constant: the posterior changes, the ranking does not.
TEST_F(SearchFixture, ThePriorShiftsEveryHitAndReordersNothing) {
    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;

    cpplink::SearchOptions options;
    options.k = 8;
    cpplink::SearchReport plain;
    ASSERT_TRUE(searcher.Search(Query(), options, &plain, &error)) << error;

    options.override_prior = true;
    options.prior_weight = cpplink::PriorWeightForExpected(1.0, kRecords);
    cpplink::SearchReport shifted;
    ASSERT_TRUE(searcher.Search(Query(), options, &shifted, &error)) << error;

    ASSERT_EQ(plain.hits.size(), shifted.hits.size());
    const double move = shifted.prior - plain.prior;
    EXPECT_NE(move, 0.0);
    for (size_t i = 0; i < plain.hits.size(); ++i) {
        EXPECT_EQ(plain.hits[i].row, shifted.hits[i].row) << "at rank " << i;
        EXPECT_NEAR(shifted.hits[i].weight, plain.hits[i].weight + move, 1e-9);
    }
    EXPECT_DOUBLE_EQ(plain.prior, scorer_.PriorWeight());
}

// A hit explains itself through the report every other pair goes through, which
// is what keeps the explanation and the answer one calculation.
TEST_F(SearchFixture, AHitIsExplainedByTheWaterfall) {
    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;
    cpplink::SearchOptions options;
    options.k = 4;
    cpplink::SearchReport report;
    ASSERT_TRUE(searcher.Search(Query(), options, &report, &error)) << error;
    ASSERT_FALSE(report.hits.empty());
    for (const cpplink::SearchHit& hit : report.hits) {
        const cpplink::PairWaterfall waterfall = cpplink::BuildPairWaterfall(
            *store_, comparisons_, scorer_, hit.row, searcher.QueryRow(), &model_);
        EXPECT_EQ(waterfall.gamma, hit.gamma);
        EXPECT_DOUBLE_EQ(waterfall.weight, hit.weight);
        EXPECT_EQ(waterfall.steps.size(), comparisons_.Size());
    }
}

// A column the query does not name is missing, not empty, so every row lands on
// that comparison's null level whatever it holds.
TEST_F(SearchFixture, AnUnnamedColumnIsTheNullLevel) {
    cpplink::QueryRecord query;
    query.Set("city", "city1");

    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;
    cpplink::SearchOptions options;
    options.k = 6;
    cpplink::SearchReport report;
    ASSERT_TRUE(searcher.Search(query, options, &report, &error)) << error;
    ASSERT_FALSE(report.hits.empty());
    for (const cpplink::SearchHit& hit : report.hits) {
        EXPECT_EQ(comparisons_.LevelOf(hit.gamma, 0), 0u);  // surname: null
        EXPECT_EQ(comparisons_.LevelOf(hit.gamma, 2), 0u);  // dob: null
        EXPECT_EQ(comparisons_.LevelOf(hit.gamma, 3), 0u);  // the derived key: null
    }
    // Three of the four comparisons cost nothing at all: the query does not carry
    // their columns, so no row can move them off the null level.
    EXPECT_EQ(report.constant, 3u);
    EXPECT_EQ(report.tabled, 1u);
    EXPECT_EQ(report.evaluated, 0u);
}

// A derived column is filled in for the query from the source the query gave,
// exactly as the loader fills it in for a row.
TEST_F(SearchFixture, ADerivedColumnIsComputedForTheQuery) {
    cpplink::QueryRecord query;
    query.Set("surname", "saunders0");  // not in the store; the same soundex as
                                        // "sander0s", which is
    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;
    cpplink::SearchOptions options;
    options.k = 5;
    cpplink::SearchReport report;
    ASSERT_TRUE(searcher.Search(query, options, &report, &error)) << error;

    const auto& key = std::get<cpplink::StringColumn>(store_->column(3));
    ASSERT_NE(key.ids[searcher.QueryRow()], cpplink::kNullId);
    EXPECT_FALSE(key.dict.Value(key.ids[searcher.QueryRow()]).empty());
}

// The threshold is what makes "nobody here is this person" expressible: an empty
// answer rather than k bad ones.
TEST_F(SearchFixture, AThresholdCanRefuseEveryRow) {
    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;
    cpplink::SearchOptions options;
    options.k = 10;
    options.threshold = 1e6;
    cpplink::SearchReport report;
    ASSERT_TRUE(searcher.Search(Query(), options, &report, &error)) << error;
    EXPECT_TRUE(report.hits.empty());
    std::ostringstream text;
    cpplink::PrintSearchReport(report, text);
    EXPECT_NE(text.str().find("no record here is this one"), std::string::npos);
}

TEST_F(SearchFixture, ADateThatIsNotOneIsRefused) {
    cpplink::QueryRecord query;
    query.Set("dob", "24/01/1981");
    cpplink::Searcher searcher;
    std::string error;
    ASSERT_TRUE(searcher.Bind(store_.get(), &comparisons_, &scorer_, &error)) << error;
    cpplink::SearchOptions options;
    cpplink::SearchReport report;
    EXPECT_FALSE(searcher.Search(query, options, &report, &error));
    EXPECT_NE(error.find("YYYY-MM-DD"), std::string::npos);
    // And the refusal leaves nothing behind, so the next query starts clean.
    // Both sides of the column that refused: the ones already written are taken
    // back, and the ones after it were never written and must not be popped.
    EXPECT_EQ(store_->NumRecords(), kRecords);
    EXPECT_EQ(std::get<cpplink::StringColumn>(store_->column(0)).ids.size(), kRecords);
    EXPECT_EQ(std::get<cpplink::StringColumn>(store_->column(3)).ids.size(), kRecords);
    EXPECT_EQ(store_->ids().offsets.size(), kRecords + 1);
    EXPECT_EQ(store_->ids().Get(kRecords - 1), "r" + std::to_string(kRecords - 1));
    // And the next query works, which it would not if a column had lost a row.
    cpplink::SearchOptions good;
    good.k = 3;
    ASSERT_TRUE(searcher.Search(Query(), good, &report, &error)) << error;
    EXPECT_EQ(report.hits.size(), 3u);
}

// A comparison whose levels read *different* columns -- splink's email, an
// address and the username derived from it -- is not a function of one value id,
// so it is tabulated a level at a time instead. The levels still have to be
// walked in the schema's order, and null still has to pre-empt them.
const char* const kEmailSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "email", "type": "string"},
    {"name": "email_username",
     "derive": {"from": "email", "transform": "email_username"}}
  ],
  "comparisons": [
    {"name": "email", "columns": ["email", "email_username"],
     "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"},
                {"type": "exact", "column": "email_username"},
                {"type": "jaro_winkler", "threshold": 0.93},
                {"type": "jaro_winkler", "threshold": 0.93,
                 "column": "email_username"},
                {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "email"}]
})";

TEST(SearchEmailTest, AComparisonReadingTwoColumnsIsTabulatedALevelAtATime) {
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(kEmailSchemaJson, &schema, &error)) << error;
    cpplink::RecordStore store(schema);
    auto& email = std::get<cpplink::StringColumn>(store.mutable_column(0));
    constexpr uint64_t kRows = 300;
    for (uint64_t row = 0; row < kRows; ++row) {
        // Four shapes: the username with another domain, a near-miss username,
        // an unrelated address, and no address at all.
        std::string value;
        switch (row % 4) {
            case 0:
                value = "user" + std::to_string(row / 4) + "@example.com";
                break;
            case 1:
                value = "user" + std::to_string(row / 4) + "@other.org";
                break;
            case 2:
                value = "user" + std::to_string(row / 4) + "x@example.com";
                break;
            default:
                break;
        }
        email.ids.push_back(value.empty() ? cpplink::kNullId : email.dict.Intern(value));
        store.mutable_ids().Append("r" + std::to_string(row));
    }
    store.set_num_records(kRows);
    store.Finalize();

    cpplink::ComparisonSet comparisons;
    ASSERT_TRUE(comparisons.Bind(schema, store, &error)) << error;
    cpplink::Model model;
    model.lambda = 0.01;
    model.records = kRows;
    model.comparisons = {Comparison(
        "email", true,
        {{1e-9, 1e-9}, {0.5, 1e-5}, {0.2, 1e-4}, {0.1, 1e-3}, {0.1, 1e-2}, {0.1, 0.98}})};
    cpplink::Scorer scorer;
    cpplink::ScoreOptions score;
    ASSERT_TRUE(scorer.Bind(model, comparisons, store, score, &error)) << error;

    cpplink::QueryRecord query;
    query.Set("email", "user7@example.com");

    cpplink::Searcher searcher;
    ASSERT_TRUE(searcher.Bind(&store, &comparisons, &scorer, &error)) << error;
    cpplink::SearchOptions options;
    options.k = kRows;
    cpplink::SearchReport report;
    ASSERT_TRUE(searcher.Search(query, options, &report, &error)) << error;
    EXPECT_EQ(report.tabled, 1u);
    EXPECT_EQ(report.evaluated, 0u);
    ASSERT_EQ(report.hits.size(), kRows);
    // The level-at-a-time table is the pair path or it is a different model.
    for (const cpplink::SearchHit& hit : report.hits) {
        EXPECT_EQ(hit.gamma, comparisons.Evaluate(hit.row, searcher.QueryRow()))
            << "row " << hit.row;
    }
    // And it reaches every one of the shapes: the address exactly (row 28), the
    // username where the domain differs (row 29), and null (row 31).
    const auto level = [&](uint64_t row) {
        for (const cpplink::SearchHit& hit : report.hits) {
            if (hit.row == row) return comparisons.LevelOf(hit.gamma, 0);
        }
        return static_cast<uint8_t>(255);
    };
    EXPECT_EQ(level(28), 1u);  // user7@example.com
    EXPECT_EQ(level(29), 2u);  // user7@other.org: the username alone
    EXPECT_EQ(level(31), 0u);  // no address at all
}

TEST(SearchQueryTest, AFieldIsColumnEqualsValue) {
    cpplink::QueryField field;
    std::string error;
    ASSERT_TRUE(cpplink::ParseQueryField("first_name=john smith", &field, &error));
    EXPECT_EQ(field.column, "first_name");
    EXPECT_EQ(field.value, "john smith");
    EXPECT_FALSE(cpplink::ParseQueryField("first_name", &field, &error));
    EXPECT_FALSE(cpplink::ParseQueryField("=john", &field, &error));
}

// The prior a query faces is not the pair space's: one expected match in 500
// records is a very different number from lambda, and it is meant to be.
TEST(SearchQueryTest, ThePriorIsOddsOverRecordsNotOverPairs) {
    EXPECT_NEAR(cpplink::PriorWeightForExpected(1.0, 1000), std::log2(1.0 / 999.0),
                1e-12);
    EXPECT_GT(cpplink::PriorWeightForExpected(10.0, 1000),
              cpplink::PriorWeightForExpected(1.0, 1000));
}

}  // namespace
