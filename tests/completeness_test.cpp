// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/completeness.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

// Eight hundred planted pairs, one for each combination of agreeing on key_a,
// key_b and key_c, a hundred pairs per combination. Agreement on the three keys
// is independent by construction and one combination in eight agrees on nothing,
// so the true pair completeness of a plan blocking all three is exactly 87.5% --
// a number the estimator has to find without being told which pairs are pairs.
constexpr uint64_t kPairs = 800;
constexpr uint64_t kRecords = kPairs * 2;

const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "key_a", "type": "string"},
    {"name": "key_b", "type": "string"},
    {"name": "key_c", "type": "string"},
    {"name": "sig", "type": "string"}
  ],
  "comparisons": [
    {"name": "key_a", "columns": ["key_a"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "key_b", "columns": ["key_b"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "key_c", "columns": ["key_c"],
     "levels": [{"type": "exact"}, {"type": "else"}]},
    {"name": "sig", "columns": ["sig"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "key_a"},
    {"type": "exact_value", "column": "key_b"},
    {"type": "exact_value", "column": "key_c"}
  ]
})";

class CompletenessFixture : public ::testing::Test {
   protected:
    void SetUp() override { Build(kSchemaJson); }

    void Build(const std::string& schema_json) {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(schema_json, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);
        auto& key_a = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& key_b = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        auto& key_c = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        auto& sig = std::get<cpplink::StringColumn>(store_->mutable_column(3));

        for (uint64_t pair = 0; pair < kPairs; ++pair) {
            const std::string tag = std::to_string(pair);
            const bool agree[3] = {(pair & 1u) != 0, (pair & 2u) != 0, (pair & 4u) != 0};
            cpplink::StringColumn* keys[3] = {&key_a, &key_b, &key_c};
            for (int side = 0; side < 2; ++side) {
                for (int k = 0; k < 3; ++k) {
                    // Both rows share the value when the pair agrees on that key,
                    // and otherwise carry a value nothing else in the file holds,
                    // so the only candidates are the planted pairs themselves.
                    const std::string value = (side == 0 || agree[k])
                                                  ? "k" + std::to_string(k) + "_" + tag
                                                  : "u" + std::to_string(k) + "_" + tag;
                    keys[k]->ids.push_back(keys[k]->dict.Intern(value));
                }
                sig.ids.push_back(sig.dict.Intern("s" + tag));
                store_->mutable_ids().Append("r" + tag + "_" + std::to_string(side));
            }
        }
        store_->set_num_records(kRecords);
        store_->Finalize();

        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;

        model_.lambda = static_cast<double>(kPairs) /
                        (0.5 * static_cast<double>(kRecords) * (kRecords - 1));
        model_.records = kRecords;
        model_.comparisons.clear();
        for (const char* name : {"key_a", "key_b", "key_c"}) {
            model_.comparisons.push_back(Comparison(name, {{0.5, 1e-4}, {0.5, 0.9999}}));
        }
        // The signal that says "this is a pair", and the only comparison the plan
        // does not block on, so it is what the posterior is really reading.
        model_.comparisons.push_back(Comparison("sig", {{0.999, 1e-9}, {0.001, 1.0}}));
    }

    static cpplink::ModelComparison Comparison(
        const std::string& name, const std::vector<std::pair<double, double>>& levels,
        size_t sessions = 2) {
        cpplink::ModelComparison comparison;
        comparison.name = name;
        comparison.sessions = sessions;
        for (const auto& entry : levels) {
            cpplink::ModelLevel level;
            level.m = entry.first;
            level.u = entry.second;
            level.m_estimated = sessions > 0;
            comparison.levels.push_back(level);
        }
        return comparison;
    }

    cpplink::CompletenessReport Estimate(const cpplink::CompletenessOptions& options,
                                         bool* ok) {
        cpplink::CompletenessReport report;
        std::string error;
        *ok = cpplink::EstimateCompleteness(*store_, comparisons_, plan_, model_, options,
                                            &report, &error);
        if (!*ok) last_error_ = error;
        return report;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::BlockingPlan plan_;
    cpplink::ComparisonSet comparisons_;
    cpplink::Model model_;
    std::string last_error_;
};

// The claim the whole estimator rests on: the pairs blocking never produced can
// be counted from the ones it did.
TEST_F(CompletenessFixture, FindsTheCompletenessItWasNeverShown) {
    cpplink::CompletenessOptions options;
    options.threads = 2;
    bool ok = false;
    const cpplink::CompletenessReport report = Estimate(options, &ok);
    ASSERT_TRUE(ok) << last_error_;
    EXPECT_TRUE(report.trusted);
    EXPECT_TRUE(report.table_available);
    // Three binary keys: eight cells, one of them dark.
    EXPECT_EQ(report.cells, 8u);
    EXPECT_EQ(report.dark_cells, 1u);
    EXPECT_NEAR(report.pc_estimate, 0.875, 0.01);
    EXPECT_NEAR(report.pc_independence, 0.875, 0.01);
}

// The product model reaches the same answer here, because this fixture really is
// conditionally independent. That is the case where the two disagree on real
// data, so it is worth pinning where they agree.
TEST_F(CompletenessFixture, TheProductModelAgreesWhenTheDataAreIndependent) {
    cpplink::CompletenessOptions options;
    options.threads = 2;
    bool ok = false;
    const cpplink::CompletenessReport report = Estimate(options, &ok);
    ASSERT_TRUE(ok) << last_error_;
    EXPECT_NEAR(report.pc_product, 0.875, 0.01);
    // Every source here fires with probability one given exact agreement, so the
    // best source per pattern and the sources combined are the same event and
    // nothing about independence *between sources* is being assumed. What is
    // being assumed is the product over comparisons inside m(gamma), which this
    // fixture satisfies exactly and real data does not.
    EXPECT_NEAR(report.pc_bound, 0.875, 0.01);
    EXPECT_DOUBLE_EQ(report.pc_bound, report.pc_analytic);
}

// Every source blocking one column leaves EM no session that can learn that
// column's m, and the estimate is a sum weighted by exactly those parameters.
TEST_F(CompletenessFixture, RefusesToTrustAModelThatNeverLearnedTheBlockedColumn) {
    model_.comparisons[0].sessions = 0;
    for (cpplink::ModelLevel& level : model_.comparisons[0].levels) {
        level.m_estimated = false;
    }
    cpplink::CompletenessOptions options;
    options.threads = 1;
    bool ok = false;
    const cpplink::CompletenessReport report = Estimate(options, &ok);
    ASSERT_TRUE(ok) << last_error_;
    EXPECT_FALSE(report.trusted);
    ASSERT_EQ(report.unlearned.size(), 1u);
    EXPECT_EQ(report.unlearned[0], "key_a");

    std::ostringstream out;
    cpplink::PrintCompletenessReport(report, out);
    EXPECT_NE(out.str().find("REFUSE"), std::string::npos);
    EXPECT_NE(out.str().find("key_a"), std::string::npos);
}

// With nothing but a window there is no firing probability to compute, and an
// estimator that cannot say why it believes a number should not produce one.
TEST_F(CompletenessFixture, RefusesAPlanWithNoAnalyticSource) {
    std::string json(kSchemaJson);
    const std::string old_blocking = R"("blocking": [
    {"type": "exact_value", "column": "key_a"},
    {"type": "exact_value", "column": "key_b"},
    {"type": "exact_value", "column": "key_c"}
  ])";
    const std::string new_blocking =
        R"("blocking": [{"type": "sorted_neighbourhood", "column": "key_a",
                        "window": 4}])";
    ASSERT_NE(json.find(old_blocking), std::string::npos);
    json.replace(json.find(old_blocking), old_blocking.size(), new_blocking);
    Build(json);

    cpplink::CompletenessOptions options;
    options.threads = 1;
    bool ok = false;
    Estimate(options, &ok);
    EXPECT_FALSE(ok);
    EXPECT_NE(last_error_.find("firing probability"), std::string::npos);
}

// A window on the same column as the only analytic source is not a second look
// at the pair: both fire for the same reason, and reading the overlap that way
// says the window catches everything.
TEST_F(CompletenessFixture, WithholdsTheCorrectionWhenTheSourcesShareAColumn) {
    std::string json(kSchemaJson);
    const std::string old_blocking = R"({"type": "exact_value", "column": "key_b"},
    {"type": "exact_value", "column": "key_c"})";
    const std::string new_blocking =
        R"({"type": "sorted_neighbourhood", "column": "key_a", "window": 4})";
    ASSERT_NE(json.find(old_blocking), std::string::npos);
    json.replace(json.find(old_blocking), old_blocking.size(), new_blocking);
    Build(json);

    cpplink::CompletenessOptions options;
    options.threads = 1;
    bool ok = false;
    const cpplink::CompletenessReport report = Estimate(options, &ok);
    ASSERT_TRUE(ok) << last_error_;
    EXPECT_FALSE(report.correction_available);
    // One blocked column leaves no table either: a dark cell cannot be told from
    // a marginal.
    EXPECT_FALSE(report.table_available);
}

// Sampling the fold changes how many pairs are looked at and not what they say,
// because everything read off the table is a ratio.
TEST_F(CompletenessFixture, SamplingTheFoldDoesNotMoveTheAnswerMuch) {
    cpplink::CompletenessOptions full;
    full.threads = 2;
    cpplink::CompletenessOptions half;
    half.threads = 2;
    half.sample = 0.5;
    bool ok = false;
    const cpplink::CompletenessReport all = Estimate(full, &ok);
    ASSERT_TRUE(ok) << last_error_;
    const cpplink::CompletenessReport some = Estimate(half, &ok);
    ASSERT_TRUE(ok) << last_error_;
    EXPECT_LT(some.observed_pairs, all.observed_pairs);
    EXPECT_NEAR(all.pc_estimate, some.pc_estimate, 0.05);
}

TEST_F(CompletenessFixture, TheReportNamesEverySourceAndItsClass) {
    cpplink::CompletenessOptions options;
    options.threads = 1;
    bool ok = false;
    const cpplink::CompletenessReport report = Estimate(options, &ok);
    ASSERT_TRUE(ok) << last_error_;
    ASSERT_EQ(report.sources.size(), 3u);
    for (const cpplink::SourceCapture& source : report.sources) {
        EXPECT_EQ(source.capture, cpplink::CaptureClass::kAnalytic);
        EXPECT_TRUE(source.bound_to_comparison);
        EXPECT_DOUBLE_EQ(source.fires_given_exact, 1.0);
    }
    std::ostringstream out;
    cpplink::PrintCompletenessReport(report, out);
    EXPECT_NE(out.str().find("key_a"), std::string::npos);
    EXPECT_NE(out.str().find("Pair completeness"), std::string::npos);

    std::ostringstream json;
    cpplink::WriteCompletenessJson(report, json);
    EXPECT_NE(json.str().find("\"pc_estimate\""), std::string::npos);
    EXPECT_NE(json.str().find("\"pc_basis\""), std::string::npos);
}

}  // namespace
