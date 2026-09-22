// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// A boolean column is the smallest countable column there is: two values and a
// null. What it buys is exactly what an interned column buys at two values --
// an integer exact level, a closed-form u, a term-frequency adjustment that says
// agreeing on the value one row in ten carries is evidence and agreeing on the
// other is nearly none -- so these tests check that it gets each of those, and
// that the one thing it cannot do, be sorted, is refused rather than done.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/model.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/profile.hpp"
#include "cpplink/recall.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"
#include "tests/process_id.hpp"
#include "tests/temp_dir.hpp"

namespace {

constexpr uint64_t kRecords = 600;
constexpr uint64_t kPlanted = 50;  // duplicate pairs at rows (2i, 2i + 1)
constexpr uint32_t kSurnames = 20;
constexpr uint32_t kDays = 30;

// The flag is set on one row in ten (one planted pair in ten, so the rate holds
// among the duplicates too), and a planted pair disagrees on it every tenth pair,
// so both rates the estimator has to recover are fixed here.
bool FlagSet(uint64_t index) { return index % 10 == 3; }
bool FlagAgrees(uint64_t pair) { return pair % 10 != 0; }
bool SurnameAgrees(uint64_t pair) { return pair % 5 != 0; }
bool DobAgrees(uint64_t pair) { return pair % 4 != 0; }

uint32_t Draw(uint64_t row, uint64_t salt, uint32_t pool) {
    uint64_t value = row * 0x9E3779B97F4A7C15ull + salt;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return static_cast<uint32_t>((value ^ (value >> 31)) % pool);
}

uint64_t PairsIn(uint64_t group) { return group * (group - 1) / 2; }

// email is blocked on and never compared, so the one session holds nothing out
// and every comparison is free in it.
constexpr const char* kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "email", "type": "string"},
    {"name": "surname", "type": "string"},
    {"name": "dob", "type": "date"},
    {"name": "deceased", "type": "boolean"}
  ],
  "comparisons": [
    {"name": "surname", "columns": ["surname"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "dob", "columns": ["dob"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "deceased", "columns": ["deceased"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "email"}
  ]
})";

// Rows 0 and 1 carry the flag, 2 and 3 do not, and 4 is missing it. Everything
// after that is the planted pairs and the singletons above.
class BooleanFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& email = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        auto& dob = std::get<cpplink::DateColumn>(store_->mutable_column(2));
        auto& deceased = std::get<cpplink::BooleanColumn>(store_->mutable_column(3));

        std::vector<uint32_t> surnames;
        for (uint32_t i = 0; i < kSurnames; ++i) {
            surnames.push_back(surname.dict.Intern("surname" + std::to_string(i)));
        }
        email.ids.assign(kRecords, 0);
        surname.ids.assign(kRecords, 0);
        dob.values.assign(kRecords, 0);
        deceased.values.assign(kRecords, 0);
        for (uint64_t i = 0; i < kPlanted; ++i) {
            const uint64_t a = 2 * i;
            const uint64_t b = 2 * i + 1;
            const uint32_t shared = email.dict.Intern("dup" + std::to_string(i));
            email.ids[a] = shared;
            email.ids[b] = shared;
            surname.ids[a] = surnames[i % kSurnames];
            surname.ids[b] =
                surnames[SurnameAgrees(i) ? i % kSurnames : (i + 7) % kSurnames];
            dob.values[a] = static_cast<int32_t>(i % kDays);
            dob.values[b] =
                static_cast<int32_t>(DobAgrees(i) ? i % kDays : (i + 11) % kDays);
            const int8_t flag = FlagSet(i) ? 1 : 0;
            deceased.values[a] = flag;
            deceased.values[b] = FlagAgrees(i) ? flag : static_cast<int8_t>(1 - flag);
        }
        for (uint64_t row = 2 * kPlanted; row < kRecords; ++row) {
            email.ids[row] = email.dict.Intern("solo" + std::to_string(row));
            surname.ids[row] = surnames[Draw(row, 1, kSurnames)];
            dob.values[row] = static_cast<int32_t>(Draw(row, 2, kDays));
            deceased.values[row] = FlagSet(row) ? 1 : 0;
        }
        // The first rows are the ones the pair tests read by number.
        deceased.values[0] = 1;
        deceased.values[1] = 1;
        deceased.values[2] = 0;
        deceased.values[3] = 0;
        deceased.values[4] = cpplink::kNullBoolean;
        for (uint64_t row = 0; row < kRecords; ++row) {
            store_->mutable_ids().Append("r" + std::to_string(row));
        }

        store_->set_num_records(kRecords);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    const cpplink::BooleanColumn& Flags() const {
        return std::get<cpplink::BooleanColumn>(store_->column(3));
    }

    // u for the exact level, in the form the estimator and the profile both
    // mean: ordered pairs of distinct rows that agree, over ordered pairs of
    // distinct rows.
    double ClosedFormU() const {
        double agreeing = 0.0;
        for (const uint32_t count : Flags().tf) {
            agreeing += static_cast<double>(count) * (static_cast<double>(count) - 1.0);
        }
        return agreeing / (static_cast<double>(kRecords) * (kRecords - 1.0));
    }

    // Replaces the plan with one over the given sources.
    bool BuildPlan(const std::string& blocking, std::string* error) {
        cpplink::Schema schema;
        std::string json(kSchemaJson);
        const size_t at = json.find("\"blocking\"");
        json = json.substr(0, at) + "\"blocking\": " + blocking + "\n}";
        if (!cpplink::ParseSchema(json, &schema, error)) return false;
        return plan_.Build(schema, *store_, error);
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
    cpplink::BlockingPlan plan_;
};

// -- Schema -----------------------------------------------------------------------

TEST(BooleanSchemaTest, ParsesTheTypeAndCountsIt) {
    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(R"({"columns":[{"name":"flag","type":"boolean"}]})",
                                     &schema, &error))
        << error;
    ASSERT_EQ(schema.columns.size(), 1u);
    EXPECT_EQ(schema.columns[0].type, cpplink::ColumnType::kBoolean);
    EXPECT_STREQ(cpplink::ColumnTypeName(cpplink::ColumnType::kBoolean), "boolean");
    EXPECT_TRUE(cpplink::HasTermFrequencies(cpplink::ColumnType::kBoolean));
}

// An exact level is an integer equality on any countable column. The string
// metrics and the date window read something a boolean does not hold, and that
// is refused before a file is opened.
TEST(BooleanSchemaTest, AcceptsExactAndRefusesWhatNeedsCharactersOrDays) {
    auto parse = [](const std::string& levels, std::string* error) {
        cpplink::Schema schema;
        const std::string json = R"({"columns":[{"name":"flag","type":"boolean"}],
            "comparisons":[{"name":"flag","columns":["flag"],"levels":)" +
                                 levels + "}]}";
        return cpplink::ParseSchema(json, &schema, error);
    };
    std::string error;
    EXPECT_TRUE(parse(R"([{"type":"null"},{"type":"exact"},{"type":"else"}])", &error))
        << error;
    EXPECT_FALSE(
        parse(R"([{"type":"jaro_winkler","threshold":0.9},{"type":"else"}])", &error));
    EXPECT_NE(error.find("cannot read"), std::string::npos) << error;
    EXPECT_FALSE(
        parse(R"([{"type":"levenshtein","threshold":1},{"type":"else"}])", &error));
    EXPECT_NE(error.find("cannot read"), std::string::npos) << error;
    EXPECT_FALSE(
        parse(R"([{"type":"date_within","threshold":3},{"type":"else"}])", &error));
    EXPECT_NE(error.find("cannot read"), std::string::npos) << error;
}

// No transform reads a boolean, so a derivation from one is a type error at
// parse time like any other mismatched chain.
TEST(BooleanSchemaTest, NoTransformDerivesFromABoolean) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_FALSE(cpplink::ParseSchema(
        R"({"columns":[{"name":"flag","type":"boolean"},
            {"name":"key","derive":{"from":"flag","transform":"soundex"}}]})",
        &schema, &error));
    EXPECT_FALSE(error.empty());
}

// -- Store ------------------------------------------------------------------------

TEST_F(BooleanFixture, TermFrequenciesAreTwoCountsAndNullIsNeither) {
    uint32_t set = 0;
    uint32_t clear = 0;
    uint64_t missing = 0;
    for (const int8_t value : Flags().values) {
        if (value == cpplink::kNullBoolean) {
            ++missing;
        } else if (value == 1) {
            ++set;
        } else {
            ++clear;
        }
    }
    ASSERT_EQ(Flags().tf.size(), 2u);
    EXPECT_EQ(Flags().tf[0], clear);
    EXPECT_EQ(Flags().tf[1], set);
    EXPECT_EQ(set + clear + missing, kRecords);
    EXPECT_EQ(missing, 1u);
    EXPECT_EQ(store_->NullCount(3), 1u);
    EXPECT_EQ(store_->DistinctValues(3), 2u);
    EXPECT_GT(clear, 5 * set)
        << "the flag has to be rare for the TF tests to mean anything";
}

// Distinct is the values seen, not the two the table always has room for.
TEST(BooleanStoreTest, DistinctCountsOnlyTheValuesSeen) {
    cpplink::Schema schema;
    schema.columns = {{"flag", cpplink::ColumnType::kBoolean}};
    cpplink::RecordStore store(schema);
    auto& flag = std::get<cpplink::BooleanColumn>(store.mutable_column(0));
    flag.values = {1, 1, cpplink::kNullBoolean, 1};
    store.set_num_records(4);
    store.Finalize();
    ASSERT_EQ(flag.tf.size(), 2u);
    EXPECT_EQ(flag.tf[0], 0u);
    EXPECT_EQ(flag.tf[1], 3u);
    EXPECT_EQ(store.DistinctValues(0), 1u);
    EXPECT_EQ(store.NullCount(0), 1u);

    cpplink::RecordStore empty(schema);
    std::get<cpplink::BooleanColumn>(empty.mutable_column(0))
        .values.assign(3, cpplink::kNullBoolean);
    empty.set_num_records(3);
    empty.Finalize();
    EXPECT_EQ(empty.DistinctValues(0), 0u);
    EXPECT_EQ(empty.NullCount(0), 3u);
}

TEST_F(BooleanFixture, MemoryReportChargesOneByteARecord) {
    const cpplink::MemoryReport report = store_->Memory();
    bool seen = false;
    for (const cpplink::MemoryLine& line : report.lines) {
        if (line.structure != "deceased values") continue;
        seen = true;
        EXPECT_GE(line.bytes, kRecords);
        EXPECT_LT(line.bytes, 2 * kRecords);
    }
    EXPECT_TRUE(seen);
}

// -- Loader -----------------------------------------------------------------------

class BooleanLoadFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_boolean_" + cpplink_test::ProcessId());
        cpplink_test::RemoveAll(dir_);
        std::filesystem::create_directories(dir_);
        path_ = (dir_ / "flags.parquet").string();
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    // Two row groups, so the loader appends chunk by chunk as it does at scale.
    void Write(const std::shared_ptr<arrow::Array>& flags) {
        arrow::StringBuilder ids;
        for (int64_t i = 0; i < flags->length(); ++i) {
            ASSERT_TRUE(ids.Append("r" + std::to_string(i)).ok());
        }
        std::shared_ptr<arrow::Array> id_array;
        ASSERT_TRUE(ids.Finish(&id_array).ok());
        auto schema = arrow::schema(
            {arrow::field("id", arrow::utf8()), arrow::field("flag", flags->type())});
        auto table = arrow::Table::Make(schema, {id_array, flags});
        auto sink = arrow::io::FileOutputStream::Open(path_);
        ASSERT_TRUE(sink.ok());
        ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                               *sink,
                                               /*chunk_size=*/3)
                        .ok());
        // `WriteTable` does not close a stream it was handed.
        ASSERT_TRUE((*sink)->Close().ok());
    }

    std::filesystem::path dir_;
    std::string path_;
};

TEST_F(BooleanLoadFixture, LoadsTrueFalseAndNullAsThreeDistinctValues) {
    arrow::BooleanBuilder builder;
    ASSERT_TRUE(builder.AppendValues(std::vector<bool>{true, false}).ok());
    ASSERT_TRUE(builder.AppendNull().ok());
    ASSERT_TRUE(builder.AppendValues(std::vector<bool>{true, true}).ok());
    std::shared_ptr<arrow::Array> flags;
    ASSERT_TRUE(builder.Finish(&flags).ok());
    Write(flags);

    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(
        R"({"unique_id":"id","columns":[{"name":"flag","type":"boolean"}]})", &schema,
        &error))
        << error;
    cpplink::RecordStore store(schema);
    ASSERT_TRUE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error)) << error;

    const auto& column = std::get<cpplink::BooleanColumn>(store.column(0));
    const std::vector<int8_t> expected = {1, 0, cpplink::kNullBoolean, 1, 1};
    EXPECT_EQ(column.values, expected);
    EXPECT_EQ(column.tf[0], 1u);
    EXPECT_EQ(column.tf[1], 3u);
    EXPECT_EQ(store.NullCount(0), 1u);
}

// A column declared boolean has to be one in the file: a string of "yes" and "no"
// is not silently read as anything.
TEST_F(BooleanLoadFixture, RefusesAColumnThatIsNotBooleanInTheFile) {
    arrow::StringBuilder builder;
    ASSERT_TRUE(builder.AppendValues(std::vector<std::string>{"yes", "no", "yes"}).ok());
    std::shared_ptr<arrow::Array> flags;
    ASSERT_TRUE(builder.Finish(&flags).ok());
    Write(flags);

    cpplink::Schema schema;
    std::string error;
    ASSERT_TRUE(cpplink::ParseSchema(
        R"({"unique_id":"id","columns":[{"name":"flag","type":"boolean"}]})", &schema,
        &error))
        << error;
    cpplink::RecordStore store(schema);
    EXPECT_FALSE(cpplink::LoadParquet(path_, schema, &store, nullptr, &error));
    EXPECT_NE(error.find("expected a boolean column"), std::string::npos) << error;
}

// -- Comparison -------------------------------------------------------------------

TEST_F(BooleanFixture, ExactLevelFiresOnTheSameValueOnly) {
    constexpr size_t kFlag = 2;  // the comparison's index
    EXPECT_EQ(comparisons_.EvaluateOne(kFlag, 0, 1), 1) << "true, true";
    EXPECT_EQ(comparisons_.EvaluateOne(kFlag, 2, 3), 1) << "false, false";
    EXPECT_EQ(comparisons_.EvaluateOne(kFlag, 0, 2), 2) << "true, false is else";
    EXPECT_EQ(comparisons_.EvaluateOne(kFlag, 4, 0), 0) << "null on one side";
    EXPECT_EQ(comparisons_.EvaluateOne(kFlag, 4, 4), 0) << "null on both";
    EXPECT_TRUE(comparisons_.IsNullValue(kFlag, 4));
    EXPECT_FALSE(comparisons_.IsNullValue(kFlag, 2));
}

// The ceiling is admissible only if the cheap test never says "no" where the
// level would fire. For a boolean the cheap test is the level, so they agree
// everywhere, and this pins that down over every pair of the first rows.
TEST_F(BooleanFixture, LevelPossibleAgreesWithTheLevel) {
    constexpr size_t kFlag = 2;
    const size_t levels = comparisons_.at(kFlag).spec->levels.size();
    for (uint64_t a = 0; a < 40; ++a) {
        for (uint64_t b = a + 1; b < 40; ++b) {
            const uint8_t fired = comparisons_.EvaluateOne(kFlag, a, b);
            for (size_t level = 0; level < levels; ++level) {
                if (level == fired) {
                    EXPECT_TRUE(comparisons_.LevelPossible(kFlag, level, a, b))
                        << a << "," << b << " level " << level;
                }
            }
        }
    }
}

// The explain report is how a level assignment is checked by eye, so the value
// it shows has to be the value, not a byte.
TEST_F(BooleanFixture, ExplainPrintsTheValueAsAWord) {
    std::ostringstream out;
    cpplink::PrintPairExplanation(*store_, comparisons_, 0, 2, out);
    EXPECT_NE(out.str().find("true"), std::string::npos) << out.str();
    EXPECT_NE(out.str().find("false"), std::string::npos) << out.str();
    std::ostringstream missing;
    cpplink::PrintPairExplanation(*store_, comparisons_, 4, 0, missing);
    EXPECT_NE(missing.str().find("<null>"), std::string::npos) << missing.str();
}

// -- Blocking ---------------------------------------------------------------------

// Two values make two groups, and the count of what that costs comes from the two
// term frequencies with no enumeration -- which the enumeration then confirms.
TEST_F(BooleanFixture, ExactValueBlocksTheTwoHalvesAndPricesThemExactly) {
    std::string error;
    ASSERT_TRUE(BuildPlan(R"([{"type":"exact_value","column":"deceased"}])", &error))
        << error;
    const uint64_t expected = PairsIn(Flags().tf[0]) + PairsIn(Flags().tf[1]);
    EXPECT_EQ(plan_.CountPairs(0), expected);
    EXPECT_EQ(plan_.CountUnion(), expected);
    EXPECT_EQ(plan_.LargestGroup(0), std::max(Flags().tf[0], Flags().tf[1]));
    EXPECT_TRUE(plan_.Produces(0, 0, 1)) << "true with true";
    EXPECT_TRUE(plan_.Produces(0, 2, 3)) << "false with false";
    EXPECT_FALSE(plan_.Produces(0, 0, 2)) << "true with false";
    EXPECT_EQ(plan_.KeyOf(0, 0), 1u);
    EXPECT_EQ(plan_.KeyOf(0, 2), 0u);
}

TEST_F(BooleanFixture, ANullFlagNeverBlocks) {
    std::string error;
    ASSERT_TRUE(BuildPlan(R"([{"type":"exact_value","column":"deceased"}])", &error))
        << error;
    EXPECT_EQ(plan_.KeyOf(0, 4), cpplink::kNoKey);
    for (uint64_t row = 0; row < 8; ++row) {
        EXPECT_FALSE(plan_.Produces(0, 4, row)) << "row " << row;
    }
}

// A cap between the two counts keeps the rare half and drops the common one,
// which is the only shape in which blocking on a flag is not most of the file.
TEST_F(BooleanFixture, RareValueKeepsOnlyTheRareHalf) {
    const uint32_t cap = (Flags().tf[0] + Flags().tf[1]) / 2;
    ASSERT_GT(cap, Flags().tf[1]);
    ASSERT_LT(cap, Flags().tf[0]);
    std::string error;
    ASSERT_TRUE(
        BuildPlan(R"([{"type":"rare_value","column":"deceased","max_frequency":)" +
                      std::to_string(cap) + "}]",
                  &error))
        << error;
    EXPECT_EQ(plan_.CountPairs(0), PairsIn(Flags().tf[1]));
    EXPECT_EQ(plan_.CountUnion(), PairsIn(Flags().tf[1]));
    EXPECT_TRUE(plan_.Produces(0, 0, 1));
    EXPECT_FALSE(plan_.Produces(0, 2, 3));
}

// A boolean has no order a window could exploit, so the source is refused with a
// message that names what to use instead.
TEST_F(BooleanFixture, SortedNeighbourhoodIsRefused) {
    std::string error;
    EXPECT_FALSE(BuildPlan(
        R"([{"type":"sorted_neighbourhood","column":"deceased","window":3}])", &error));
    EXPECT_NE(error.find("deceased"), std::string::npos) << error;
    EXPECT_NE(error.find("exact_value"), std::string::npos) << error;
}

// The miss diagnostic reads the same keys and frequencies the plan does, so a
// pair disagreeing on the flag is a disagreement and a pair past the cap is too
// common, not an unknown column type.
TEST_F(BooleanFixture, MissDiagnosticReadsTheFlag) {
    const uint32_t cap = (Flags().tf[0] + Flags().tf[1]) / 2;
    std::string error;
    ASSERT_TRUE(
        BuildPlan(R"([{"type":"rare_value","column":"deceased","max_frequency":)" +
                      std::to_string(cap) + "}]",
                  &error))
        << error;
    cpplink::TruthPairs truth;
    truth.rows = {{0, 2}, {2, 3}, {4, 0}};  // disagreed, too common, null
    cpplink::MissReport report;
    cpplink::DiagnoseMisses(plan_, comparisons_, truth, &report);
    EXPECT_EQ(report.missed, 3u);
    ASSERT_EQ(report.reasons.size(), 1u);
    using cpplink::MissReason;
    EXPECT_EQ(report.reasons[0][static_cast<size_t>(MissReason::kDisagreed)], 1u);
    EXPECT_EQ(report.reasons[0][static_cast<size_t>(MissReason::kTooCommon)], 1u);
    EXPECT_EQ(report.reasons[0][static_cast<size_t>(MissReason::kNull)], 1u);
    EXPECT_EQ(report.fixable_by_cap, 1u);
}

// -- Estimation -------------------------------------------------------------------

TEST_F(BooleanFixture, ExactLevelUIsClosedForm) {
    std::string error;
    ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
    cpplink::EstimateOptions options;
    options.u_sample = 20000;
    options.threads = 1;
    options.seed = 3;
    cpplink::Model model;
    cpplink::EstimateReport report;
    ASSERT_TRUE(
        cpplink::Estimate(*store_, comparisons_, plan_, options, &model, &report, &error))
        << error;

    const cpplink::ModelComparison* flag = nullptr;
    for (const cpplink::ModelComparison& comparison : model.comparisons) {
        if (comparison.name == "deceased") flag = &comparison;
    }
    ASSERT_NE(flag, nullptr);
    ASSERT_EQ(flag->levels.size(), 3u);
    EXPECT_TRUE(flag->levels[0].u_exact);
    EXPECT_TRUE(flag->levels[1].u_exact);
    EXPECT_NEAR(flag->levels[1].u, ClosedFormU(), 1e-15);
    // One row of six hundred is missing the flag, so the null level fires on the
    // ordered pairs that row is one side of.
    EXPECT_NEAR(flag->levels[0].u, 2.0 * (kRecords - 1) / (kRecords * (kRecords - 1.0)),
                1e-15);
    double total = 0.0;
    for (const cpplink::ModelLevel& level : flag->levels) total += level.u;
    EXPECT_NEAR(total, 1.0, 1e-9);
    // The planted pairs agree on the flag nine times in ten, and blocking on
    // email reaches every one of them.
    EXPECT_TRUE(flag->levels[1].m_estimated);
    EXPECT_NEAR(flag->levels[1].m, 0.9, 0.05);
}

// -- Scoring ----------------------------------------------------------------------

class BooleanScoreFixture : public BooleanFixture {
   protected:
    void SetUp() override {
        BooleanFixture::SetUp();
        model_.lambda = 0.05;
        model_.records = kRecords;
        for (const cpplink::ComparisonSpec& spec : schema_.comparisons) {
            cpplink::ModelComparison comparison;
            comparison.name = spec.name;
            comparison.columns = spec.columns;
            comparison.term_frequency = spec.term_frequency;
            // Null is a hair, exact is where the mass is, else takes the rest.
            const double u_exact = spec.name == "deceased" ? ClosedFormU() : 0.05;
            const std::vector<std::pair<double, double>> levels = {
                {1e-6, 1e-6}, {0.9, u_exact}, {0.1 - 1e-6, 1.0 - 1e-6 - u_exact}};
            for (const auto& entry : levels) {
                cpplink::ModelLevel level;
                level.m = entry.first;
                level.u = entry.second;
                level.m_estimated = true;
                comparison.levels.push_back(level);
            }
            model_.comparisons.push_back(comparison);
        }
        cpplink::ScoreOptions options;
        options.threshold = 0.0;
        std::string error;
        ASSERT_TRUE(scorer_.Bind(model_, comparisons_, *store_, options, &error))
            << error;
    }

    cpplink::Model model_;
    cpplink::Scorer scorer_;
};

// This is what the two counts are for. Agreeing on the value one row in ten
// carries is worth more than agreeing on the value the other nine carry, and the
// pattern is the same either way, so the whole difference is the adjustment.
TEST_F(BooleanScoreFixture, AgreeingOnTheRareValueScoresHigher) {
    // The two pairs need not share a pattern on the other comparisons, so the
    // flag's contribution is read as the move away from the pattern's base
    // weight, which is where the adjustment and nothing else lives.
    const uint32_t rare = comparisons_.Evaluate(0, 1);
    const uint32_t common = comparisons_.Evaluate(2, 3);
    ASSERT_EQ(comparisons_.EvaluateOne(2, 0, 1), 1);
    ASSERT_EQ(comparisons_.EvaluateOne(2, 2, 3), 1);
    const double rare_delta = scorer_.Weight(rare, 0, 1) - scorer_.BaseWeight(rare);
    const double common_delta = scorer_.Weight(common, 2, 3) - scorer_.BaseWeight(common);
    EXPECT_GT(rare_delta, common_delta);
    // And the adjustment is log2(u / p) exactly, with p the shared value's share
    // of the rows: for the rare value that is well above zero bits, and for the
    // common one below.
    const double u = ClosedFormU();
    const double p_true = static_cast<double>(Flags().tf[1]) / kRecords;
    const double p_false = static_cast<double>(Flags().tf[0]) / kRecords;
    EXPECT_NEAR(rare_delta, std::log2(u / p_true), 1e-9);
    EXPECT_NEAR(common_delta, std::log2(u / p_false), 1e-9);
    EXPECT_GT(rare_delta, 0.0);
    EXPECT_LT(common_delta, 0.0);
}

// The bracket has to hold every pair, or dropping a pattern on its bound emits
// a different edge set from scoring every pair. Two values means two possible
// adjustments, and the bracket is exactly those two.
TEST_F(BooleanScoreFixture, TheBracketIsTheTwoValues) {
    for (uint64_t a = 0; a < 60; ++a) {
        for (uint64_t b = a + 1; b < 60; ++b) {
            const uint32_t gamma = comparisons_.Evaluate(a, b);
            const double base = scorer_.BaseWeight(gamma);
            const double exact = scorer_.Weight(gamma, a, b);
            EXPECT_LE(base + scorer_.DeltaMin(gamma), exact + 1e-9) << a << "," << b;
            EXPECT_GE(base + scorer_.DeltaMax(gamma), exact - 1e-9) << a << "," << b;
        }
    }
    const uint32_t rare = comparisons_.Evaluate(0, 1);
    const double u = ClosedFormU();
    const double p_true = static_cast<double>(Flags().tf[1]) / kRecords;
    const double p_false = static_cast<double>(Flags().tf[0]) / kRecords;
    // Neither surname nor dob carries an adjustment, so the flag's is the whole
    // bracket on this pattern.
    EXPECT_NEAR(scorer_.DeltaMax(rare), std::log2(u / p_true), 1e-9);
    EXPECT_NEAR(scorer_.DeltaMin(rare), std::log2(u / p_false), 1e-9);
}

// The waterfall explains the same number the scorer produces, and names the
// frequency the move came from.
TEST_F(BooleanScoreFixture, WaterfallSumsToTheWeightAndShowsTheCount) {
    std::ostringstream out;
    cpplink::PrintPairWaterfall(*store_, comparisons_, scorer_, 0, 1, out);
    const std::string text = out.str();
    EXPECT_NE(text.find("deceased"), std::string::npos) << text;
    EXPECT_NE(text.find(std::to_string(Flags().tf[1])), std::string::npos) << text;
}

// -- Profile ----------------------------------------------------------------------

// The profile's ceiling for a column is the collision rate its term frequencies
// give, and a boolean has one like any other scalar. The pairwise pass folds it
// against the other columns, and the M side reads its agreement rate off the
// anchor pairs like any other column's.
TEST_F(BooleanFixture, ProfileScoresTheFlagAsAScalarColumn) {
    cpplink::ProfileOptions options;
    options.threads = 1;
    options.expected_matches = kPlanted;
    const cpplink::ProfileReport report =
        BuildProfile(*store_, cpplink::PairMode::kAll, options);

    const cpplink::ColumnProfile* flag = nullptr;
    for (const cpplink::ColumnProfile& column : report.columns) {
        if (column.name == "deceased") flag = &column;
    }
    ASSERT_NE(flag, nullptr);
    EXPECT_EQ(flag->type, cpplink::ColumnType::kBoolean);
    EXPECT_TRUE(flag->scored);
    EXPECT_EQ(flag->distinct, 2u);
    EXPECT_EQ(flag->nulls, 1u);
    // Its ceiling is the same closed form over the present rows.
    const double present = static_cast<double>(kRecords - 1);
    double agreeing = 0.0;
    for (const uint32_t count : Flags().tf) {
        agreeing += static_cast<double>(count) * (static_cast<double>(count) - 1.0);
    }
    EXPECT_NEAR(flag->collision, agreeing / (present * (present - 1.0)), 1e-12);
    EXPECT_LT(flag->bits, 1.0) << "two values are under a bit of evidence";

    bool folded = false;
    for (const cpplink::ColumnPairProfile& pair : report.pairs) {
        if (pair.left_name != "deceased" && pair.right_name != "deceased") continue;
        folded = true;
        EXPECT_EQ(pair.rows, kRecords - 1) << pair.left_name << " / " << pair.right_name;
    }
    EXPECT_TRUE(folded) << "the pairwise pass skipped the boolean";
}

}  // namespace
