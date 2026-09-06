// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/profile.hpp"

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/app.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

// Twenty thousand rows over six columns whose relationships are all known in
// advance, so every number the profile prints has an answer to be checked
// against rather than a plausible shape to be eyeballed.
//
//   x       2,000 values drawn at random
//   y       2,000 values drawn at random, independently of x: the pair that a
//           plug-in collision estimator gets badly wrong
//   coarse  x's value modulo 50, so x determines it and it does not determine x
//   wide    one value per row, which determines everything by being a key
//   full    x's value with the row number appended, so x occurs inside it
//   flat    one value on all but a handful of rows
//   even    exactly four values, five thousand rows each
constexpr uint64_t kRecords = 20000;
constexpr uint32_t kWideValues = 2000;
constexpr uint32_t kEvenValues = 4;

const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "x", "type": "string"},
    {"name": "y", "type": "string"},
    {"name": "coarse", "type": "string"},
    {"name": "wide", "type": "string"},
    {"name": "full", "type": "string"},
    {"name": "flat", "type": "string"},
    {"name": "even", "type": "string"}
  ]
})";

class ProfileFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);
        cpplink::StringColumn* column[7];
        for (size_t i = 0; i < 7; ++i) {
            column[i] = &std::get<cpplink::StringColumn>(store_->mutable_column(i));
        }
        // Two independent streams, so no arithmetic relation between x and y can
        // sneak in through the row index.
        uint64_t seed_x = 12345;
        uint64_t seed_y = 67890;
        for (uint64_t row = 0; row < kRecords; ++row) {
            seed_x = seed_x * 6364136223846793005ull + 1442695040888963407ull;
            seed_y = seed_y * 6364136223846793005ull + 1442695040888963407ull;
            const uint32_t x = static_cast<uint32_t>((seed_x >> 33) % kWideValues);
            const uint32_t y = static_cast<uint32_t>((seed_y >> 33) % kWideValues);
            const std::string x_value = "x" + std::to_string(x);
            column[0]->ids.push_back(column[0]->dict.Intern(x_value));
            column[1]->ids.push_back(column[1]->dict.Intern("y" + std::to_string(y)));
            column[2]->ids.push_back(
                column[2]->dict.Intern("c" + std::to_string(x % 50)));
            column[3]->ids.push_back(column[3]->dict.Intern("w" + std::to_string(row)));
            column[4]->ids.push_back(
                column[4]->dict.Intern(x_value + "_" + std::to_string(row)));
            column[5]->ids.push_back(
                column[5]->dict.Intern(row < 4 ? "rare" : "dominant"));
            column[6]->ids.push_back(
                column[6]->dict.Intern("e" + std::to_string(row % kEvenValues)));
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();
    }

    // The fixture plants no duplicates, so the match rate is set to the smallest
    // thing it can be rather than to the one-per-record default, which would
    // refuse every joint in the file.
    cpplink::ProfileOptions Options() const {
        cpplink::ProfileOptions options;
        options.expected_matches = 1;
        options.threads = 1;
        return options;
    }

    const cpplink::ColumnProfile& ColumnNamed(const cpplink::ProfileReport& report,
                                              const std::string& name) const {
        for (const cpplink::ColumnProfile& column : report.columns) {
            if (column.name == name) return column;
        }
        ADD_FAILURE() << "no column " << name;
        return report.columns.front();
    }

    const cpplink::ColumnPairProfile& PairNamed(const cpplink::ProfileReport& report,
                                                const std::string& left,
                                                const std::string& right) const {
        for (const cpplink::ColumnPairProfile& pair : report.pairs) {
            if (pair.left_name == left && pair.right_name == right) return pair;
        }
        ADD_FAILURE() << "no pair " << left << " / " << right;
        return report.pairs.front();
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
};

// The exact-match ceiling is a closed form, so it has an exact answer: four
// values of five thousand rows collide at 4 * 5000 * 4999 / (20000 * 19999).
TEST_F(ProfileFixture, ColumnCeilingIsExact) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    const cpplink::ColumnProfile& even = ColumnNamed(report, "even");
    const double expected = static_cast<double>(kEvenValues) * 5000.0 * 4999.0 /
                            (static_cast<double>(kRecords) * (kRecords - 1));
    EXPECT_NEAR(even.collision, expected, 1e-12);
    EXPECT_NEAR(even.effective_values, 1.0 / expected, 1e-9);
    EXPECT_NEAR(even.bits, -std::log2(expected), 1e-9);
    EXPECT_NEAR(even.covered_bits, even.bits, 1e-9);  // no nulls
    EXPECT_FALSE(even.bits_floored);
}

// A column no two rows share collides at a rate the file cannot resolve, which
// bounds its worth from below rather than leaving it unmeasured.
TEST_F(ProfileFixture, UniqueColumnBitsAreAFloor) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    const cpplink::ColumnProfile& wide = ColumnNamed(report, "wide");
    EXPECT_TRUE(wide.scored);
    EXPECT_TRUE(wide.bits_floored);
    const double records = static_cast<double>(kRecords);
    EXPECT_NEAR(wide.bits, std::log2(records * (records - 1)), 1e-9);
}

// The regression test for the estimator this command started with. The plug-in
// sum of squared shares reads 1/rows on a joint table of singletons, where the
// truth is near zero, and calls these two independent columns redundant by about
// 7.6 bits. The collision form reads them as what they are.
TEST_F(ProfileFixture, IndependentColumnsAreNotRedundant) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    const cpplink::ColumnPairProfile& pair = PairNamed(report, "x", "y");
    ASSERT_TRUE(pair.resolved);
    EXPECT_GT(pair.expected_collisions, 32.0);
    EXPECT_NEAR(pair.redundant_bits, 0.0, 0.5);
    EXPECT_FALSE(pair.Suspect());
}

// Determination is directional, and the direction is the whole point: knowing
// that one column is a coarsening of another says which one to drop.
TEST_F(ProfileFixture, DeterminationFindsTheDirection) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    const cpplink::ColumnPairProfile& pair = PairNamed(report, "x", "coarse");
    EXPECT_TRUE(pair.left_informative);
    EXPECT_NEAR(pair.determines_right, 1.0, 1e-12);
    EXPECT_LT(pair.determines_left, 0.5);
    EXPECT_TRUE(pair.Suspect());
    EXPECT_EQ(pair.Verdict(),
              "x determines coarse: drop coarse, or make the two one comparison");
}

// A key determines everything and says nothing by doing so.
TEST_F(ProfileFixture, NearUniqueDeterminantIsRefused) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    const cpplink::ColumnPairProfile& pair = PairNamed(report, "x", "wide");
    EXPECT_NEAR(pair.determines_left, 1.0, 1e-12);
    EXPECT_FALSE(pair.right_informative);
    EXPECT_EQ(pair.RightLift(), 0.0);
}

// So does a column that is almost a constant: mapping anything to its commonest
// value keeps almost every row, whatever the two columns have to do with each
// other.
TEST_F(ProfileFixture, DominantTargetIsRefused) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    const cpplink::ColumnPairProfile& pair = PairNamed(report, "x", "flat");
    EXPECT_GT(pair.determines_right, 0.99);
    EXPECT_GT(pair.baseline_right, 0.99);
    EXPECT_FALSE(pair.left_informative);
}

// The detector for a column that literally contains another, which is the shape
// the design's worst measured dependence pair has.
TEST_F(ProfileFixture, ContainmentNamesTheInnerColumn) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    const cpplink::ColumnPairProfile& pair = PairNamed(report, "x", "full");
    EXPECT_NEAR(pair.containment, 1.0, 1e-12);
    EXPECT_TRUE(pair.left_inside_right);
    EXPECT_TRUE(pair.Suspect());
    EXPECT_EQ(pair.Verdict(), "x occurs inside full: make the two one comparison");
}

// The joint of two high-cardinality columns is rarer than the match rate, so on
// a file with duplicates in it the joint holds the duplicates and not a
// dependence. The report says so instead of reading a number off it.
TEST_F(ProfileFixture, MatchRateRefusesTheJoint) {
    cpplink::ProfileOptions options = Options();
    options.expected_matches = kRecords;  // one duplicate per record
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, options);
    EXPECT_FALSE(PairNamed(report, "x", "y").resolved);
    EXPECT_GT(report.unresolved_pairs, 0u);
    // The low-cardinality joints stay readable, because their independent rate is
    // still well above the match rate; refusing the sparse ones simply keeps the
    // bits they would have invented out of the ledger.
    EXPECT_LT(report.unresolved_pairs, report.pairs.size());
    EXPECT_NEAR(report.redundant_bits, 0.0, 0.1);
}

// A sample is a Bernoulli draw over rows and every quantity read off it is a
// ratio, so the answers move by sampling noise and not by a factor.
TEST_F(ProfileFixture, SamplingKeepsTheAnswers) {
    cpplink::ProfileOptions options = Options();
    options.sample_rows = kRecords / 4;
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, options);
    EXPECT_TRUE(report.sampled);
    EXPECT_NEAR(static_cast<double>(report.sampled_rows),
                static_cast<double>(kRecords) / 4.0, kRecords / 40.0);
    const cpplink::ColumnPairProfile& pair = PairNamed(report, "x", "coarse");
    EXPECT_NEAR(pair.determines_right, 1.0, 1e-12);
    EXPECT_NEAR(PairNamed(report, "x", "full").containment, 1.0, 1e-12);
}

TEST_F(ProfileFixture, LedgerAddsUp) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    EXPECT_NEAR(report.margin_bits,
                report.prior_bits + report.available_bits + report.redundant_bits, 1e-9);
    EXPECT_LT(report.prior_bits, 0.0);
    EXPECT_LE(report.redundant_bits, 0.0);
    double available = 0.0;
    for (const cpplink::ColumnProfile& column : report.columns) {
        if (column.scored) available += column.covered_bits;
    }
    EXPECT_NEAR(report.available_bits, available, 1e-9);
}

TEST_F(ProfileFixture, ReportsPrintAndParse) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    std::ostringstream text;
    PrintProfileReport(report, text);
    EXPECT_NE(text.str().find("Evidence ledger"), std::string::npos);
    EXPECT_NE(text.str().find("Suspects"), std::string::npos);

    std::ostringstream json;
    WriteProfileJson(report, json);
    EXPECT_NE(json.str().find("\"margin_bits\""), std::string::npos);
    EXPECT_NE(json.str().find("\"determines_right\""), std::string::npos);
}

// --no-pairs asks for the column half alone, which touches no row.
TEST_F(ProfileFixture, WithoutPairsNothingIsWalked) {
    cpplink::ProfileOptions options = Options();
    options.pairs = false;
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, options);
    EXPECT_FALSE(report.walked);
    EXPECT_TRUE(report.pairs.empty());
    EXPECT_GT(report.available_bits, 0.0);
}

TEST(ProfileCommand, NeedsSchemaAndData) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"profile"}, out, err), 1);
    EXPECT_NE(err.str().find("--schema"), std::string::npos);
}

TEST(ProfileCommand, RejectsUnknownOption) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"profile", "--nope"}, out, err), 1);
    EXPECT_NE(err.str().find("unknown option"), std::string::npos);
}

}  // namespace
