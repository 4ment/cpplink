// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/profile.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/app.hpp"
#include "cpplink/recall.hpp"
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

// Thirty thousand rows: fifteen thousand originals and a duplicate of each,
// corrupted column by column at rates this test knows, so m has a planted answer
// to be checked against rather than a plausible shape to be eyeballed.
//
//   key1/key2/key3  3,000 values each, redrawn on the duplicate at 10, 20 and 30%
//   soft              200 values, redrawn at 40%
//   linked            400 values, redrawn on exactly the rows key3 was, which is a
//                     dependence among matches that the u side cannot see at all
//   coarse          key1 modulo 40, recomputed after the corruption, so key1
//                   determines it and no session may anchor on one and learn the
//                   other
constexpr uint64_t kPlantedPairs = 15000;
constexpr uint32_t kKeyValues = 3000;
constexpr uint32_t kSoftValues = 200;
constexpr uint32_t kLinkedValues = 400;
constexpr uint32_t kCoarseClasses = 40;

const char* const kAnchorSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "key1", "type": "string"},
    {"name": "key2", "type": "string"},
    {"name": "key3", "type": "string"},
    {"name": "soft", "type": "string"},
    {"name": "linked", "type": "string"},
    {"name": "coarse", "type": "string"}
  ]
})";

class AnchorFixture : public ::testing::Test {
   protected:
    // key1, key2, key3, soft, linked. coarse is derived and not carried.
    using Record = std::array<uint32_t, 5>;

    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kAnchorSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        std::vector<Record> planted(kPlantedPairs);
        for (uint64_t i = 0; i < kPlantedPairs; ++i) {
            planted[i] = {Draw(kKeyValues), Draw(kKeyValues), Draw(kKeyValues),
                          Draw(kSoftValues), Draw(kLinkedValues)};
        }
        for (const Record& record : planted) Emit(record);
        for (const Record& record : planted) {
            Record copy = record;
            if (Draw(100) < 10) copy[0] = Draw(kKeyValues);
            if (Draw(100) < 20) copy[1] = Draw(kKeyValues);
            if (Draw(100) < 30) {
                copy[2] = Draw(kKeyValues);
                copy[4] = Draw(kLinkedValues);
            }
            if (Draw(100) < 40) copy[3] = Draw(kSoftValues);
            Emit(copy);
        }
        store_->set_num_records(2 * kPlantedPairs);
        store_->Finalize();
    }

    uint32_t Draw(uint32_t range) {
        seed_ = seed_ * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>((seed_ >> 33) % range);
    }

    void Emit(const Record& record) {
        static const char* const kPrefix[5] = {"a", "b", "c", "s", "l"};
        for (size_t i = 0; i < 5; ++i) {
            cpplink::StringColumn& column =
                std::get<cpplink::StringColumn>(store_->mutable_column(i));
            column.ids.push_back(
                column.dict.Intern(kPrefix[i] + std::to_string(record[i])));
        }
        cpplink::StringColumn& coarse =
            std::get<cpplink::StringColumn>(store_->mutable_column(5));
        coarse.ids.push_back(
            coarse.dict.Intern("g" + std::to_string(record[0] % kCoarseClasses)));
        store_->mutable_ids().Append("r" + std::to_string(emitted_++));
    }

    // One planted duplicate per original, which is what the prior odds have to be
    // told: the default of one per record would double the match rate.
    cpplink::ProfileOptions Options() const {
        cpplink::ProfileOptions options;
        options.expected_matches = kPlantedPairs;
        options.threads = 1;
        return options;
    }

    const cpplink::ColumnMatchProfile& MatchNamed(const cpplink::ProfileReport& report,
                                                  const std::string& name) const {
        for (const cpplink::ColumnMatchProfile& match : report.matches) {
            if (match.name == name) return match;
        }
        ADD_FAILURE() << "no m for " << name;
        return report.matches.front();
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

    // Originals are emitted first and their duplicates after, so row i and row
    // i + kPlantedPairs are the pair the fixture planted. That is exactly what
    // `LoadTruthPairs` would resolve an "id_a,id_b" file to, without the file.
    const cpplink::TruthPairs& Planted() const {
        if (truth_.rows.empty()) {
            for (uint64_t i = 0; i < kPlantedPairs; ++i) {
                truth_.rows.emplace_back(static_cast<uint32_t>(i),
                                         static_cast<uint32_t>(i + kPlantedPairs));
            }
            truth_.lines = kPlantedPairs;
        }
        return truth_;
    }

    uint64_t emitted_ = 0;
    uint64_t seed_ = 987654321;
    mutable cpplink::TruthPairs truth_;
    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
};

// The estimate this whole half of the command exists for. A column redrawn on a
// tenth of the duplicates has m of 0.9 plus the chance the redraw lands on the
// same value, and the anchor pairs have to find that without a model, without a
// truth file and without ever being told which rows are duplicates of which.
TEST_F(AnchorFixture, AnchorMFindsThePlantedRate) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    ASSERT_TRUE(report.anchored) << report.anchor_refusal;
    EXPECT_NEAR(MatchNamed(report, "key1").m, 0.9 + 0.1 / kKeyValues, 0.02);
    EXPECT_NEAR(MatchNamed(report, "key2").m, 0.8 + 0.2 / kKeyValues, 0.02);
    EXPECT_NEAR(MatchNamed(report, "key3").m, 0.7 + 0.3 / kKeyValues, 0.02);
    EXPECT_NEAR(MatchNamed(report, "soft").m, 0.6 + 0.4 / kSoftValues, 0.02);
    EXPECT_NEAR(MatchNamed(report, "linked").m, 0.7 + 0.3 / kLinkedValues, 0.02);
    EXPECT_NEAR(MatchNamed(report, "coarse").m, 0.9 + 0.1 / kCoarseClasses, 0.02);
}

// The truth side is the second reading the anchor estimate is scored against, and
// this fixture is the one place the two can be checked at once: the planted rates
// are known, so a truth m that misses them would mean the reading and not the
// estimate is broken. Every anchor m sits at or above its truth m, which is the
// bias the report says runs one way.
TEST_F(AnchorFixture, TruthMReadsThePlantedRateTheAnchorEstimates) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options(), &Planted());
    ASSERT_TRUE(report.anchored) << report.anchor_refusal;
    ASSERT_TRUE(report.truthed);
    EXPECT_EQ(report.truth_pairs, kPlantedPairs);
    EXPECT_NEAR(MatchNamed(report, "key1").truth_m, 0.9 + 0.1 / kKeyValues, 0.02);
    EXPECT_NEAR(MatchNamed(report, "key3").truth_m, 0.7 + 0.3 / kKeyValues, 0.02);
    EXPECT_NEAR(MatchNamed(report, "soft").truth_m, 0.6 + 0.4 / kSoftValues, 0.02);
    for (const cpplink::ColumnMatchProfile& match : report.matches) {
        if (!match.estimated || !match.truthed) continue;
        EXPECT_GE(match.m, match.truth_m - 0.02) << match.name;
    }
    // Every column of this fixture is corrupted on its own coin, so an anchor
    // selects nothing about the rest and the two margins land on each other.
    EXPECT_NEAR(report.estimated_margin_bits, report.truth_margin_bits, 1.0);
}

// Without a truth file nothing above changes and nothing below is reported, which
// is what "scored beside it, never fitted to" has to mean.
TEST_F(AnchorFixture, TheTruthReadingChangesNoEstimate) {
    const cpplink::ProfileReport without =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    const cpplink::ProfileReport with =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options(), &Planted());
    EXPECT_FALSE(without.truthed);
    EXPECT_EQ(without.matches.size(), with.matches.size());
    for (size_t i = 0; i < without.matches.size(); ++i) {
        EXPECT_DOUBLE_EQ(without.matches[i].m, with.matches[i].m);
    }
    EXPECT_DOUBLE_EQ(without.estimated_margin_bits, with.estimated_margin_bits);
}

// An anchor forces its own columns to agree, and forces every column it determines
// to agree with them, so a session that read either back would be reporting its own
// selection as a measurement.
TEST_F(AnchorFixture, NoSessionLearnsWhatItsAnchorHasAlreadySaid) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    ASSERT_FALSE(report.sessions.empty());
    bool anchored_on_key1 = false;
    for (const cpplink::AnchorSession& session : report.sessions) {
        for (size_t column : session.anchor) {
            EXPECT_EQ(std::count(session.learns.begin(), session.learns.end(), column),
                      0);
        }
        const bool holds_key1 =
            std::find(session.anchor_names.begin(), session.anchor_names.end(), "key1") !=
            session.anchor_names.end();
        anchored_on_key1 = anchored_on_key1 || holds_key1;
        if (!holds_key1) continue;
        for (size_t column : session.learns) {
            EXPECT_NE(report.columns[column].name, "coarse")
                << "coarse is key1 modulo 40 and this session anchors on key1";
        }
    }
    EXPECT_TRUE(anchored_on_key1);
}

// What the M side is for. `linked` is corrupted on exactly the duplicates `key3`
// was, so among matches the two agree together far more than agreeing apart would
// predict: log2(0.7 / (0.7 * 0.7)) is 0.51 bits. The u side cannot see it, because
// the two columns are drawn independently and share no value.
TEST_F(AnchorFixture, MSideFindsTheDependenceTheUSideCannot) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    ASSERT_TRUE(report.anchored) << report.anchor_refusal;
    const cpplink::ColumnPairProfile& coupled = PairNamed(report, "key3", "linked");
    ASSERT_TRUE(coupled.m_resolved);
    EXPECT_NEAR(coupled.m_redundant_bits, -std::log2(0.7), 0.06);
    // And invents none where there is none: soft is corrupted on its own coin.
    const cpplink::ColumnPairProfile& apart = PairNamed(report, "key2", "soft");
    ASSERT_TRUE(apart.m_resolved);
    EXPECT_NEAR(apart.m_redundant_bits, 0.0, 0.06);
}

// m <= 1, so log2(m/u) can never exceed the column's own ceiling. The ledger's
// second half is a tighter bound than its first and never a looser one.
TEST_F(AnchorFixture, EstimatedWeightNeverExceedsTheCeiling) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    ASSERT_TRUE(report.anchored) << report.anchor_refusal;
    for (const cpplink::ColumnMatchProfile& match : report.matches) {
        if (!match.estimated) continue;
        EXPECT_LE(match.weight, report.columns[match.column].bits + 1e-9) << match.name;
        EXPECT_GT(match.m, 0.0);
        EXPECT_LE(match.m, 1.0);
    }
    EXPECT_EQ(report.estimated_columns, report.matches.size());
}

// The pairwise pass is what separates an anchor from a column it determines, so
// without it the M side is refused rather than run on an anchor that has already
// answered the question.
TEST_F(AnchorFixture, WithoutThePairwisePassThereIsNoMSide) {
    cpplink::ProfileOptions options = Options();
    options.pairs = false;
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, options);
    EXPECT_FALSE(report.anchored);
    EXPECT_FALSE(report.anchor_refusal.empty());
    EXPECT_TRUE(report.matches.empty());
    EXPECT_EQ(report.expected_bits, 0.0);
}

// Turning it off leaves the ceiling half of the report exactly as it was.
TEST_F(AnchorFixture, AnchorsCanBeTurnedOffWithoutASound) {
    cpplink::ProfileOptions options = Options();
    options.anchors = false;
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, options);
    EXPECT_FALSE(report.anchored);
    EXPECT_TRUE(report.anchor_refusal.empty());
    EXPECT_TRUE(report.sessions.empty());
    EXPECT_GT(report.available_bits, 0.0);
}

// An anchor on key1 and key2 selects the planted pairs that survived both
// corruptions, which is 15,000 * 0.9 * 0.8, and it is a test rather than an
// observation because it is the one number that says the enumeration found the
// duplicates rather than something else that collides.
TEST_F(AnchorFixture, AnchorSelectsThePlantedPairs) {
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, Options());
    ASSERT_TRUE(report.anchored) << report.anchor_refusal;
    const cpplink::AnchorSession* found = nullptr;
    for (const cpplink::AnchorSession& session : report.sessions) {
        if (session.anchor_names.size() == 2 && session.anchor_names[0] == "key1" &&
            session.anchor_names[1] == "key2") {
            found = &session;
        }
    }
    ASSERT_NE(found, nullptr);
    EXPECT_TRUE(found->used);
    EXPECT_FALSE(found->capped);
    EXPECT_NEAR(static_cast<double>(found->pairs), 0.9 * 0.8 * kPlantedPairs, 400.0);
    EXPECT_GT(found->posterior_bits, 0.0);
}

// A budget stops the walk rather than the estimate: a run that reaches it still
// reports m, and says it was capped.
TEST_F(AnchorFixture, PairBudgetCapsTheWalk) {
    cpplink::ProfileOptions options = Options();
    options.anchor_pairs = 500;
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, options);
    ASSERT_TRUE(report.anchored) << report.anchor_refusal;
    for (const cpplink::AnchorSession& session : report.sessions) {
        EXPECT_LE(session.pairs, 500u);
        EXPECT_TRUE(session.capped);
    }
    EXPECT_NEAR(MatchNamed(report, "key1").m, 0.9, 0.06);
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
