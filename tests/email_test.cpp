// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// splink's email comparison, expressed as a derived column and one comparison
// whose levels read either the address or its username. The mechanism is
// general -- a level may name which of several string columns it reads -- and
// this is the instance that motivated it.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/derive.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"
#include "cpplink/simplify.hpp"
#include "cpplink/string_metrics.hpp"

namespace {

std::string Derived(cpplink::Transform transform, const std::string& value) {
    std::string out;
    cpplink::ApplyTransforms({transform}, value, &out);
    return out;
}

TEST(EmailTransformTest, SplitsAtTheAtSign) {
    EXPECT_EQ(Derived(cpplink::Transform::kEmailUsername, "john.smith@example.com"),
              "john.smith");
    EXPECT_EQ(Derived(cpplink::Transform::kEmailDomain, "john.smith@example.com"),
              "example.com");
    // Without an "@" the whole value is the username and there is no domain,
    // which is what splink's two regular expressions extract.
    EXPECT_EQ(Derived(cpplink::Transform::kEmailUsername, "john.smith"), "john.smith");
    EXPECT_EQ(Derived(cpplink::Transform::kEmailDomain, "john.smith"), "");
    // A second "@" belongs to the domain side of the first and the username side
    // of the last, so the two halves never overlap.
    EXPECT_EQ(Derived(cpplink::Transform::kEmailUsername, "a@b@c"), "a");
    EXPECT_EQ(Derived(cpplink::Transform::kEmailDomain, "a@b@c"), "c");
    // An empty half is a missing value, like every other empty derivation.
    EXPECT_EQ(Derived(cpplink::Transform::kEmailUsername, "@example.com"), "");
    EXPECT_EQ(Derived(cpplink::Transform::kEmailDomain, "john.smith@"), "");
    // Case is left alone; a chain can normalize where it wants to.
    EXPECT_EQ(Derived(cpplink::Transform::kEmailUsername, "John.Smith@Example.com"),
              "John.Smith");
}

TEST(EmailTransformTest, ParsesByName) {
    cpplink::Transform transform;
    ASSERT_TRUE(cpplink::ParseTransform("email_username", &transform));
    EXPECT_EQ(transform, cpplink::Transform::kEmailUsername);
    ASSERT_TRUE(cpplink::ParseTransform("email_domain", &transform));
    EXPECT_EQ(transform, cpplink::Transform::kEmailDomain);
    EXPECT_EQ(cpplink::TransformInput(cpplink::Transform::kEmailUsername),
              cpplink::ColumnType::kString);
    EXPECT_EQ(cpplink::TransformOutput(cpplink::Transform::kEmailDomain),
              cpplink::ColumnType::kString);
}

constexpr const char* kColumns = R"("columns": [
    {"name": "email", "type": "string"},
    {"name": "username", "derive": {"from": "email", "transform": "email_username"}},
    {"name": "surname", "type": "string"},
    {"name": "town", "type": "string"},
    {"name": "job", "type": "string"}])";

// The comparison as splink writes it: exact on the address, exact on the
// username, Jaro-Winkler at 0.88 on the address, the same on the username.
constexpr const char* kEmailComparison = R"({
    "name": "email", "columns": ["email", "username"], "term_frequency": true,
    "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "exact", "column": "username"},
      {"type": "jaro_winkler", "threshold": 0.88},
      {"type": "jaro_winkler", "threshold": 0.88, "column": "username"},
      {"type": "else"}]})";

std::string SchemaJson(const std::string& comparisons, const std::string& blocking) {
    return "{" + std::string(kColumns) + ", \"comparisons\": [" + comparisons + "]" +
           (blocking.empty() ? "" : ", \"blocking\": " + blocking) + "}";
}

class EmailSchema : public ::testing::Test {
   protected:
    bool Parse(const std::string& comparison) {
        return cpplink::ParseSchema(SchemaJson(comparison, ""), &schema_, &error_);
    }
    cpplink::Schema schema_;
    std::string error_;
};

TEST_F(EmailSchema, ALevelMayNameOneOfSeveralStringColumns) {
    ASSERT_TRUE(Parse(kEmailComparison)) << error_;
    ASSERT_EQ(schema_.comparisons.size(), 1u);
    const cpplink::ComparisonSpec& spec = schema_.comparisons[0];
    ASSERT_EQ(spec.levels.size(), 6u);
    EXPECT_EQ(spec.levels[1].column, 0);  // the first column by default
    EXPECT_EQ(spec.levels[2].column, 1);
    EXPECT_EQ(spec.levels[3].column, 0);
    EXPECT_EQ(spec.levels[4].column, 1);
    // Two "exact" rows in a report would be indistinguishable, so a level that
    // names its column is labelled with it unless the schema says otherwise.
    EXPECT_EQ(spec.levels[1].Describe(), "exact");
    EXPECT_EQ(spec.levels[2].Describe(), "exact on username");
    EXPECT_EQ(spec.levels[4].Describe(), "jaro_winkler >= 0.88 on username");
    EXPECT_EQ(spec.bits, 3);
}

TEST_F(EmailSchema, RefusesAColumnTheComparisonDoesNotName) {
    EXPECT_FALSE(Parse(R"({"columns": ["email", "username"], "levels": [
        {"type": "exact", "column": "surname"}, {"type": "else"}]})"));
    EXPECT_NE(error_.find("does not name"), std::string::npos);
}

TEST_F(EmailSchema, RefusesAColumnOnASingleColumnComparison) {
    EXPECT_FALSE(Parse(R"({"columns": ["email"], "levels": [
        {"type": "exact", "column": "email"}, {"type": "else"}]})"));
    EXPECT_NE(error_.find("only a single-column level"), std::string::npos);
}

TEST_F(EmailSchema, RefusesAColumnOnALevelThatReadsThemAll) {
    EXPECT_FALSE(Parse(R"({"columns": ["email", "username"], "levels": [
        {"type": "null", "column": "email"}, {"type": "else"}]})"));
    EXPECT_NE(error_.find("only a single-column level"), std::string::npos);
}

TEST_F(EmailSchema, SeveralColumnsWithoutNamesReadTheFirst) {
    ASSERT_TRUE(Parse(R"({"columns": ["email", "username"], "levels": [
        {"type": "exact"}, {"type": "else"}]})"))
        << error_;
    EXPECT_EQ(schema_.comparisons[0].levels[0].column, 0);
}

// The rows, arranged so that every level of the comparison has a pair that lands
// on it against row 0 and no other level above it fires first.
//
//   0  john.smith@example.com   the reference
//   1  john.smith@example.com   exact
//   2  john.smith@gmail.com     same username, and the address is within 0.88 too,
//                               which the username level pre-empts
//   3  john.smyth@example.com   address within 0.88, username not equal
//   4  jon.smith@other.org      address far, username within 0.88
//   5  mary.jones@example.com   else
//   6  <null>                   null
//   7  @example.com             an empty username, so the comparison is null
//   8  @example.com             the same again, so the addresses agree and the
//                               exact level still cannot fire
constexpr uint64_t kRows = 9;

class EmailFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        const std::string json =
            SchemaJson(std::string(kEmailComparison) + R"(,
            {"name": "surname", "columns": ["surname"],
             "levels": [{"type": "exact"}, {"type": "else"}]},
            {"name": "town", "columns": ["town"],
             "levels": [{"type": "exact"}, {"type": "else"}]},
            {"name": "job", "columns": ["job"],
             "levels": [{"type": "exact"}, {"type": "else"}]})",
                       R"([{"type": "exact_value", "column": "surname"}])");
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(json, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        const std::vector<std::string> emails = {"john.smith@example.com",
                                                 "john.smith@example.com",
                                                 "john.smith@gmail.com",
                                                 "john.smyth@example.com",
                                                 "jon.smith@other.org",
                                                 "mary.jones@example.com",
                                                 "",
                                                 "@example.com",
                                                 "@example.com"};
        auto& email = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        auto& town = std::get<cpplink::StringColumn>(store_->mutable_column(3));
        auto& job = std::get<cpplink::StringColumn>(store_->mutable_column(4));
        for (size_t row = 0; row < emails.size(); ++row) {
            email.ids.push_back(emails[row].empty() ? cpplink::kNullId
                                                    : email.dict.Intern(emails[row]));
            surname.ids.push_back(surname.dict.Intern(row < 5 ? "smith" : "jones"));
            town.ids.push_back(town.dict.Intern("t" + std::to_string(row % 3)));
            job.ids.push_back(job.dict.Intern("j" + std::to_string(row % 2)));
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        ASSERT_EQ(emails.size(), kRows);
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

TEST_F(EmailFixture, TheDerivedColumnHoldsTheUsername) {
    const auto& username = std::get<cpplink::StringColumn>(store_->column(1));
    ASSERT_EQ(username.ids.size(), kRows);
    EXPECT_EQ(username.dict.Value(username.ids[0]), "john.smith");
    EXPECT_EQ(username.ids[0], username.ids[2]);
    EXPECT_NE(username.ids[0], username.ids[3]);
    EXPECT_EQ(username.dict.Value(username.ids[4]), "jon.smith");
    EXPECT_EQ(username.ids[6], cpplink::kNullId);
    EXPECT_EQ(username.ids[7], cpplink::kNullId);
}

// The premise the fixture rests on, checked with the project's own metric rather
// than assumed: row 4's address is far from row 0's and its username is close.
TEST_F(EmailFixture, TheFixtureSeparatesTheTwoFuzzyLevels) {
    EXPECT_FALSE(cpplink::JaroWinklerAtLeast("john.smith@example.com",
                                             "jon.smith@other.org", 0.88));
    EXPECT_TRUE(cpplink::JaroWinklerAtLeast("john.smith", "jon.smith", 0.88));
    EXPECT_TRUE(cpplink::JaroWinklerAtLeast("john.smith@example.com",
                                            "john.smith@gmail.com", 0.88));
}

TEST_F(EmailFixture, EachLevelReadsTheColumnItNames) {
    EXPECT_EQ(Level(0, 1), 1);  // exact on the address
    EXPECT_EQ(Level(0, 2), 2);  // exact on the username, above the fuzzy address
    EXPECT_EQ(Level(0, 3), 3);  // the address within 0.88
    EXPECT_EQ(Level(0, 4), 4);  // only the username within 0.88
    EXPECT_EQ(Level(0, 5), 5);  // else
    EXPECT_EQ(Level(0, 6), 0);  // null
    EXPECT_EQ(Level(4, 0), 4);  // and the order of the pair does not matter
}

TEST_F(EmailFixture, AnEmptyUsernameMakesTheComparisonNull) {
    // "@example.com" has an address and no username. A level reading the
    // username would have nothing on one side, so the whole comparison is
    // missing rather than falling through to "else".
    EXPECT_EQ(Level(0, 7), 0);
    EXPECT_EQ(Level(7, 5), 0);
    EXPECT_EQ(Level(7, 8), 0);  // agreeing addresses, and still null
    EXPECT_TRUE(comparisons_.IsNullValue(0, 7));
    EXPECT_FALSE(comparisons_.IsNullValue(0, 0));
}

TEST_F(EmailFixture, LevelPossibleIsNeverFalseWhereTheLevelFires) {
    for (uint64_t a = 0; a < kRows; ++a) {
        for (uint64_t b = a + 1; b < kRows; ++b) {
            const uint8_t level = Level(a, b);
            EXPECT_TRUE(comparisons_.LevelPossible(0, level, a, b))
                << "rows " << a << " and " << b << " at level " << int{level};
        }
    }
}

TEST_F(EmailFixture, TheSignatureFilterChangesNoPattern) {
    cpplink::ComparisonSet unfiltered;
    std::string error;
    ASSERT_TRUE(unfiltered.Bind(schema_, *store_, &error, false)) << error;
    // Both slots of the filtered set carry a table, because a fuzzy level reads
    // each; the unfiltered set carries none.
    ASSERT_EQ(comparisons_.at(0).slots.size(), 2u);
    EXPECT_NE(comparisons_.at(0).slots[0].signatures, nullptr);
    EXPECT_NE(comparisons_.at(0).slots[1].signatures, nullptr);
    EXPECT_EQ(unfiltered.at(0).slots[1].signatures, nullptr);
    for (uint64_t a = 0; a < kRows; ++a) {
        for (uint64_t b = a + 1; b < kRows; ++b) {
            EXPECT_EQ(comparisons_.Evaluate(a, b), unfiltered.Evaluate(a, b))
                << "rows " << a << " and " << b;
        }
    }
}

// The exact level's u is the one number here that has to come from a closed
// form: an address is the column whose collision rate sits near 1/N, where a
// sampled u sees nothing. The comparison reads two columns, so it cannot come off
// the term frequencies alone -- the rows that can agree are the rows where both
// halves are present -- and the count over those rows must equal the count over
// the pairs.
TEST_F(EmailFixture, TheExactLevelsUIsClosedFormAndExact) {
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
            if (level == 0) ++null_pairs;
            if (level == 1) ++exact_pairs;
        }
    }
    ASSERT_EQ(pairs, kRows * (kRows - 1) / 2);
    const double space = static_cast<double>(pairs);
    const cpplink::ModelComparison& learned = model.comparisons[0];
    ASSERT_EQ(learned.name, "email");
    EXPECT_TRUE(learned.levels[0].u_exact);
    EXPECT_TRUE(learned.levels[1].u_exact);
    EXPECT_FALSE(learned.levels[2].u_exact);  // below another exact level
    EXPECT_NEAR(learned.levels[0].u, static_cast<double>(null_pairs) / space, 1e-12);
    EXPECT_NEAR(learned.levels[1].u, static_cast<double>(exact_pairs) / space, 1e-12);
    // Rows 7 and 8 share an address and have no username, so they are not rows
    // the exact level can see: a closed form off the address's own term
    // frequencies would have counted their pair and read two.
    EXPECT_EQ(exact_pairs, 1u);
}

class EmailScoring : public EmailFixture {
   protected:
    void SetUp() override {
        EmailFixture::SetUp();
        cpplink::Model& model = model_;
        model.lambda = 0.05;
        model.records = kRows;
        model.comparisons = {Comparison("email", true,
                                        {{1e-9, 1e-9},
                                         {0.6, 0.01},
                                         {0.2, 0.02},
                                         {0.1, 0.05},
                                         {0.05, 0.1},
                                         {0.05, 0.82}}),
                             Comparison("surname", false, {{0.9, 0.3}, {0.1, 0.7}}),
                             Comparison("town", false, {{0.9, 0.3}, {0.1, 0.7}}),
                             Comparison("job", false, {{0.9, 0.5}, {0.1, 0.5}})};
        cpplink::ScoreOptions options;
        options.threshold = 0.0;
        std::string error;
        ASSERT_TRUE(scorer_.Bind(model, comparisons_, *store_, options, &error)) << error;
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

// Each exact level is adjusted by the frequency of the value it read: the
// address's on the address level, the username's on the username level.
TEST_F(EmailScoring, EachExactLevelIsAdjustedByItsOwnColumn) {
    const auto& email = std::get<cpplink::StringColumn>(store_->column(0));
    const auto& username = std::get<cpplink::StringColumn>(store_->column(1));
    ASSERT_EQ(email.tf[email.ids[0]], 2u);        // john.smith@example.com, twice
    ASSERT_EQ(username.tf[username.ids[0]], 3u);  // john.smith, three times

    const uint32_t address = comparisons_.Evaluate(0, 1);
    ASSERT_EQ(comparisons_.LevelOf(address, 0), 1);
    EXPECT_EQ(scorer_.FrequencyFor(0, address, 0), 2u);
    EXPECT_NE(scorer_.AdjustmentFor(0, address, 0, 1), 0.0);

    const uint32_t user = comparisons_.Evaluate(0, 2);
    ASSERT_EQ(comparisons_.LevelOf(user, 0), 2);
    EXPECT_EQ(scorer_.FrequencyFor(0, user, 0), 3u);
    EXPECT_NE(scorer_.AdjustmentFor(0, user, 0, 2), 0.0);

    // The two moves differ, because the columns' frequencies and the levels' u
    // do; a single adjustment read off the first column could not tell them
    // apart.
    EXPECT_NE(scorer_.AdjustmentFor(0, address, 0, 1),
              scorer_.AdjustmentFor(0, user, 0, 2));

    // The fuzzy levels get none of either kind.
    const uint32_t fuzzy = comparisons_.Evaluate(0, 4);
    ASSERT_EQ(comparisons_.LevelOf(fuzzy, 0), 4);
    EXPECT_EQ(scorer_.AdjustmentFor(0, fuzzy, 0, 4), 0.0);
}

TEST_F(EmailScoring, TheWaterfallStillSumsToTheScore) {
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

TEST_F(EmailScoring, TheBracketHoldsEveryPair) {
    for (uint64_t a = 0; a < kRows; ++a) {
        for (uint64_t b = a + 1; b < kRows; ++b) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            const double weight = scorer_.Weight(gamma, a, b);
            EXPECT_GE(weight, scorer_.BaseWeight(gamma) + scorer_.DeltaMin(gamma) - 1e-9);
            EXPECT_LE(weight, scorer_.BaseWeight(gamma) + scorer_.DeltaMax(gamma) + 1e-9);
            EXPECT_GE(scorer_.Ceiling(a, b), weight - 1e-9)
                << "rows " << a << " and " << b;
        }
    }
}

// A merge is expressible only where the stronger level's predicate implies the
// weaker one's, and two levels reading different columns imply nothing about
// each other here: an exact address does imply an exact username, but only
// through the derivation, which simplify does not read. With merging switched
// off -- an alpha no p-value reaches and no gap -- the report lists every
// adjacent pair of the comparison with its refusal, and the three cross-column
// pairs are refused for their columns while the username's fuzzy level could
// still fall into "else".
TEST_F(EmailScoring, SimplifyRefusesToMergeAcrossColumns) {
    cpplink::BlockingPlan plan;
    std::string error;
    ASSERT_TRUE(plan.Build(schema_, *store_, cpplink::PairMode::kAll, &error)) << error;
    cpplink::SimplifyOptions options;
    options.threads = 1;
    options.alpha = 2.0;
    options.min_gap = 0.0;
    const cpplink::SimplifyReport report =
        BuildSimplify(*store_, comparisons_, plan, model_, options);
    ASSERT_FALSE(report.comparisons.empty());
    const cpplink::ComparisonSimplify& item = report.comparisons.front();
    ASSERT_EQ(item.name, "email");
    ASSERT_EQ(item.kept.size(), 6u);
    ASSERT_EQ(item.tests.size(), 5u);
    size_t crossed = 0;
    for (const cpplink::LevelMerge& test : item.tests) {
        const cpplink::LevelSpec& upper = schema_.comparisons[0].levels[test.upper];
        const cpplink::LevelSpec& lower = schema_.comparisons[0].levels[test.lower];
        EXPECT_FALSE(test.merged);
        if (lower.type == cpplink::LevelType::kElse) {
            EXPECT_TRUE(test.expressible) << test.upper << " into else";
        } else if (upper.type == cpplink::LevelType::kNull) {
            EXPECT_FALSE(test.expressible);
        } else {
            ASSERT_NE(upper.column, lower.column);
            ++crossed;
            EXPECT_FALSE(test.expressible) << test.upper << " into " << test.lower;
            EXPECT_NE(test.refusal.find("different columns"), std::string::npos);
        }
    }
    EXPECT_EQ(crossed, 3u);
}

}  // namespace
