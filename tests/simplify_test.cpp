// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/simplify.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

// Four columns and four comparisons, with a hand-written model rather than a
// fitted one, so what every level is worth is chosen rather than discovered and
// the merge decisions have arithmetic to be checked against.
//
//   a  five levels, whose two Jaro levels are given *exactly* the same m and u:
//      the pair a likelihood ratio cannot separate at any sample size
//   b  the same five weights with the middle two on different metrics, which is
//      the merge that would change the model rather than shrink it
//   c  the column blocking runs on
//   d  three levels, all far apart
constexpr uint64_t kRecords = 4000;
constexpr uint64_t kPlanted = 1000;
constexpr uint32_t kBlocks = 100;

const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "a", "type": "string"},
    {"name": "b", "type": "string"},
    {"name": "c", "type": "string"},
    {"name": "d", "type": "string"}
  ],
  "comparisons": [
    {"name": "a", "columns": ["a"], "levels": [
      {"type": "null"}, {"type": "exact"},
      {"type": "jaro_winkler", "threshold": 0.9},
      {"type": "jaro_winkler", "threshold": 0.8, "label": "loose"},
      {"type": "else"}]},
    {"name": "b", "columns": ["b"], "levels": [
      {"type": "null"}, {"type": "exact"},
      {"type": "levenshtein", "threshold": 1},
      {"type": "jaro_winkler", "threshold": 0.88},
      {"type": "else"}]},
    {"name": "c", "columns": ["c"], "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "d", "columns": ["d"], "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "c"}]
})";

std::string Value(char prefix, uint64_t row, uint64_t salt) {
    uint64_t hash = row * 0x9E3779B97F4A7C15ull + salt;
    hash ^= hash >> 29;
    hash *= 0xBF58476D1CE4E5B9ull;
    hash ^= hash >> 32;
    std::string out(1, prefix);
    for (int i = 0; i < 12; ++i) {
        out.push_back(static_cast<char>('a' + hash % 26));
        hash /= 26;
        if (hash == 0) hash = row + 1;
    }
    return out;
}

cpplink::ModelLevel Level(const char* label, double m, double u) {
    cpplink::ModelLevel level;
    level.label = label;
    level.m = m;
    level.u = u;
    return level;
}

cpplink::Model MakeModel() {
    cpplink::Model model;
    model.lambda = 1e-4;
    model.records = kRecords;

    // exact is worth log2(0.4 / 0.00002) = 14.29 bits, and both Jaro levels
    // log2(0.2 / 0.0005) = 8.64. The two Jaro levels are identical, so G^2 between
    // them is exactly zero however many pairs there are.
    cpplink::ModelComparison a;
    a.name = "a";
    a.columns = {"a"};
    a.levels = {Level("null", 0.01, 0.01), Level("exact", 0.40, 0.00002),
                Level("jaro_winkler >= 0.90", 0.20, 0.0005), Level("loose", 0.20, 0.0005),
                Level("else", 0.19, 0.98898)};
    cpplink::ModelComparison b = a;
    b.name = "b";
    b.columns = {"b"};
    b.levels[2].label = "levenshtein <= 1";
    b.levels[3].label = "jaro_winkler >= 0.88";

    cpplink::ModelComparison c;
    c.name = "c";
    c.columns = {"c"};
    c.levels = {Level("null", 0.01, 0.01), Level("exact", 0.90, 0.01),
                Level("else", 0.09, 0.98)};
    cpplink::ModelComparison d;
    d.name = "d";
    d.columns = {"d"};
    d.levels = {Level("null", 0.01, 0.01), Level("exact", 0.50, 0.02),
                Level("else", 0.49, 0.97)};

    model.comparisons = {a, b, c, d};
    return model;
}

class SimplifyFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);
        cpplink::StringColumn* column[4];
        for (size_t i = 0; i < 4; ++i) {
            column[i] = &std::get<cpplink::StringColumn>(store_->mutable_column(i));
        }
        for (uint64_t row = 0; row < kRecords; ++row) {
            // Rows 2i and 2i+1 share a block, and the first kPlanted of those
            // pairs agree on everything, which is what puts real mass on the
            // match side of every table.
            const uint64_t source = (row % 2 == 1 && row / 2 < kPlanted) ? row - 1 : row;
            column[0]->ids.push_back(column[0]->dict.Intern(Value('a', source, 11)));
            column[1]->ids.push_back(column[1]->dict.Intern(Value('b', source, 22)));
            column[2]->ids.push_back(
                column[2]->dict.Intern("c" + std::to_string((row / 2) % kBlocks)));
            column[3]->ids.push_back(column[3]->dict.Intern(Value('d', source, 44)));
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
        ASSERT_TRUE(plan_.Build(schema_, *store_, cpplink::PairMode::kAll, &error))
            << error;
        model_ = MakeModel();
    }

    cpplink::SimplifyReport Run(double alpha = 0.01, double min_gap = 0.0) {
        cpplink::SimplifyOptions options;
        options.threads = 1;
        options.alpha = alpha;
        options.min_gap = min_gap;
        return BuildSimplify(*store_, comparisons_, plan_, model_, options);
    }

    const cpplink::ComparisonSimplify& Of(const cpplink::SimplifyReport& report,
                                          const std::string& name) const {
        for (const cpplink::ComparisonSimplify& item : report.comparisons) {
            if (item.name == name) return item;
        }
        ADD_FAILURE() << "no comparison " << name;
        return report.comparisons.front();
    }

    const cpplink::LevelMerge* Test(const cpplink::ComparisonSimplify& item,
                                    size_t upper) const {
        for (const cpplink::LevelMerge& test : item.tests) {
            if (test.upper == upper) return &test;
        }
        return nullptr;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
    cpplink::BlockingPlan plan_;
    cpplink::Model model_;
};

// Two levels given the same m and the same u are the same level, and no amount of
// data says otherwise: G^2 is identically zero, so this is the one merge the test
// makes at any scale.
TEST_F(SimplifyFixture, MergesTwoLevelsWorthTheSameThing) {
    const cpplink::SimplifyReport report = Run();
    const cpplink::ComparisonSimplify& item = Of(report, "a");
    ASSERT_EQ(item.levels.size(), 5u);
    EXPECT_TRUE(item.changed);
    ASSERT_EQ(item.kept.size(), 4u);
    // The loosest predicate is the one that survives: every pair the 0.90 level
    // fired on fires the 0.80 level too.
    EXPECT_EQ(item.kept[0], 0u);
    EXPECT_EQ(item.kept[1], 1u);
    EXPECT_EQ(item.kept[2], 3u);
    EXPECT_EQ(item.kept[3], 4u);
    const cpplink::LevelMerge* merged = Test(item, 2);
    ASSERT_NE(merged, nullptr);
    EXPECT_TRUE(merged->merged);
    EXPECT_NEAR(merged->g2, 0.0, 1e-9);
    EXPECT_NEAR(merged->p, 1.0, 1e-9);
    EXPECT_NEAR(merged->gap, 0.0, 1e-9);
    EXPECT_FALSE(merged->drops_exact);
    // One level fewer is one bit fewer here, and the packed width follows.
    EXPECT_EQ(item.bits, 3);
    EXPECT_EQ(item.proposed_bits, 2);
    EXPECT_EQ(report.width, 10);
    EXPECT_EQ(report.proposed_width, 9);
    EXPECT_EQ(report.merged_levels, 1u);
}

// And levels the run can tell apart are kept, which is the other half of the
// claim: this is not a pass that shrinks every schema it is shown.
TEST_F(SimplifyFixture, KeepsLevelsTheRunCanSeparate) {
    const cpplink::SimplifyReport report = Run();
    const cpplink::ComparisonSimplify& item = Of(report, "a");
    const cpplink::LevelMerge* kept = Test(item, 1);  // exact against the merged Jaro
    ASSERT_NE(kept, nullptr);
    EXPECT_FALSE(kept->merged);
    EXPECT_TRUE(kept->expressible);
    EXPECT_LT(kept->p, 0.01);
    EXPECT_NEAR(kept->gap, 14.29 - 8.64, 0.02);
    for (const cpplink::ComparisonSimplify& other : report.comparisons) {
        if (other.name == "a" || other.name == "b") continue;
        EXPECT_FALSE(other.changed) << other.name;
    }
}

// The merge has to be expressible. Deleting a level and letting the next absorb
// its pairs is the same partition only where the stronger predicate implies the
// weaker one, and `levenshtein <= 1` against `jaro_winkler >= 0.88` implies
// nothing in either direction however alike the two are priced.
TEST_F(SimplifyFixture, RefusesAMergeThatWouldChangeTheModel) {
    const cpplink::SimplifyReport report = Run();
    const cpplink::ComparisonSimplify& item = Of(report, "b");
    const cpplink::LevelMerge* refused = Test(item, 2);
    ASSERT_NE(refused, nullptr);
    EXPECT_FALSE(refused->merged);
    EXPECT_FALSE(refused->expressible);
    EXPECT_NE(refused->refusal.find("different metrics"), std::string::npos);
    // Its two levels are priced identically, so only expressibility stopped it.
    EXPECT_NEAR(refused->gap, 0.0, 1e-9);
    EXPECT_EQ(item.kept.size(), 5u);
    EXPECT_FALSE(item.changed);
}

// Missing is not a degree of agreement, so the null level is never merged into
// what follows it however alike the two look.
TEST_F(SimplifyFixture, NeverMergesTheNullLevel) {
    const cpplink::SimplifyReport report = Run();
    for (const cpplink::ComparisonSimplify& item : report.comparisons) {
        const cpplink::LevelMerge* first = Test(item, 0);
        ASSERT_NE(first, nullptr) << item.name;
        EXPECT_FALSE(first->merged) << item.name;
        EXPECT_FALSE(first->expressible) << item.name;
        EXPECT_NE(first->refusal.find("not a degree of agreement"), std::string::npos);
        EXPECT_EQ(item.kept.front(), 0u) << item.name;
    }
}

// The finding the report is built around: significance is decided by the size of
// the run, not by the size of the difference. The same levels the default keeps
// merge once the criterion is what they are worth.
TEST_F(SimplifyFixture, TheEffectSizeDecidesWhereSignificanceCannot) {
    const cpplink::SimplifyReport strict = Run();
    EXPECT_EQ(Of(strict, "d").kept.size(), 3u);
    const cpplink::LevelMerge* kept = Test(Of(strict, "d"), 1);
    ASSERT_NE(kept, nullptr);
    EXPECT_LT(kept->p, 0.01);

    // d's exact level is 4.64 bits and its else -0.99, so a gap of 6 merges them
    // and a gap of 5 does not.
    const cpplink::SimplifyReport loose = Run(0.01, 6.0);
    ASSERT_EQ(Of(loose, "d").kept.size(), 2u);
    // And once everything below it is one level, the null level is adjacent to
    // the else and still is not merged into it.
    EXPECT_EQ(Of(loose, "d").kept[0], 0u);
    EXPECT_EQ(Of(loose, "d").kept[1], 2u);
    EXPECT_EQ(Run(0.01, 5.0).comparisons[3].kept.size(), 3u);
}

// A run of three levels nothing separates becomes one level, not two: the merge
// is agglomerative and re-tests the pooled group.
TEST_F(SimplifyFixture, MergesARunOfThree) {
    model_.comparisons[0].levels[1] = Level("exact", 0.20, 0.0005);
    const cpplink::SimplifyReport report = Run();
    const cpplink::ComparisonSimplify& item = Of(report, "a");
    ASSERT_EQ(item.kept.size(), 3u);
    EXPECT_EQ(item.kept[0], 0u);  // null
    EXPECT_EQ(item.kept[1], 3u);  // the loosest of the three
    EXPECT_EQ(item.kept[2], 4u);  // else
    EXPECT_EQ(report.merged_levels, 2u);
    // Dropping the exact level from a term-frequency comparison is worth saying,
    // and this one does not ask for the adjustment.
    const cpplink::LevelMerge* merged = Test(item, 1);
    ASSERT_NE(merged, nullptr);
    EXPECT_FALSE(merged->drops_exact);
}

TEST_F(SimplifyFixture, SaysWhenAMergeDropsAnExactLevel) {
    model_.comparisons[0].levels[1] = Level("exact", 0.20, 0.0005);
    schema_.comparisons[0].term_frequency = true;
    std::string error;
    ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    const cpplink::SimplifyReport report = Run();
    const cpplink::LevelMerge* merged = Test(Of(report, "a"), 1);
    ASSERT_NE(merged, nullptr);
    EXPECT_TRUE(merged->merged);
    EXPECT_TRUE(merged->drops_exact);
    std::ostringstream text;
    PrintSimplifyReport(report, text);
    EXPECT_NE(text.str().find("drops the exact level"), std::string::npos);
}

// The rewrite deletes the merged levels and touches nothing else, so what comes
// out is the same file with fewer levels rather than a regenerated one.
TEST_F(SimplifyFixture, RewritesTheSchemaAndKeepsEverythingElse) {
    const cpplink::SimplifyReport report = Run();
    std::string rewritten;
    std::string error;
    ASSERT_TRUE(RewriteSchema(kSchemaJson, report, &rewritten, &error)) << error;

    const nlohmann::json root = nlohmann::json::parse(rewritten);
    ASSERT_EQ(root["comparisons"].size(), 4u);
    const nlohmann::json& a = root["comparisons"][0];
    ASSERT_EQ(a["levels"].size(), 4u);
    EXPECT_EQ(a["levels"][2]["type"], "jaro_winkler");
    EXPECT_EQ(a["levels"][2]["threshold"], 0.8);
    // The surviving level keeps its own entry, label and all.
    EXPECT_EQ(a["levels"][2]["label"], "loose");
    EXPECT_EQ(a["levels"][3]["type"], "else");
    EXPECT_EQ(root["comparisons"][1]["levels"].size(), 5u);
    EXPECT_EQ(root["unique_id"], "id");
    EXPECT_EQ(root["blocking"].size(), 1u);

    // And what it writes is a schema that parses and packs narrower.
    cpplink::Schema reparsed;
    ASSERT_TRUE(cpplink::ParseSchema(rewritten, &reparsed, &error)) << error;
    EXPECT_EQ(reparsed.GammaWidth(), 9);
    EXPECT_EQ(schema_.GammaWidth(), 10);
}

TEST_F(SimplifyFixture, RefusesAModelFittedToAnotherSchema) {
    std::string error;
    EXPECT_TRUE(ModelMatches(model_, comparisons_, &error)) << error;
    cpplink::Model wrong = model_;
    wrong.comparisons[0].levels.pop_back();
    EXPECT_FALSE(ModelMatches(wrong, comparisons_, &error));
    EXPECT_NE(error.find("levels"), std::string::npos);
    wrong = model_;
    wrong.comparisons.pop_back();
    EXPECT_FALSE(ModelMatches(wrong, comparisons_, &error));
    EXPECT_NE(error.find("comparisons"), std::string::npos);
}

}  // namespace
