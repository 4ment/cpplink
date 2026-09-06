// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/levels.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "cpplink/app.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

// Twelve thousand originals and a duplicate of each, corrupted in a way that puts
// the answer on the similarity axis where the test can point at it.
//
//   key1/key2  4,000 values each, never corrupted, so the anchor is exact and
//              selects every planted pair
//   word       500 six-letter values. The duplicate keeps it 60% of the time,
//              substitutes the *first* character 25% of the time, and takes a
//              different word the remaining 15%. A substitution in the first
//              position earns no Winkler prefix bonus, so its similarity is the bare
//              Jaro of (5/6 + 5/6 + 1)/3 = 0.8889: a quarter of every matching pair
//              lands just under the schema's hand-written 0.90 and is invisible to
//              it.
//   code       300 five-digit values, two digits substituted on 30% of the
//              duplicates, against a schema that declares levenshtein <= 1.
constexpr uint64_t kPlanted = 12000;
constexpr uint32_t kKeyValues = 4000;
constexpr uint32_t kWordValues = 500;
constexpr uint32_t kCodeValues = 300;
constexpr double kWordKept = 0.60;
constexpr double kWordNudged = 0.25;
constexpr double kCodeKept = 0.70;

const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "key1", "type": "string"},
    {"name": "key2", "type": "string"},
    {"name": "word", "type": "string"},
    {"name": "code", "type": "string"}
  ],
  "comparisons": [
    {"name": "key1", "columns": ["key1"],
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "word", "columns": ["word"],
     "levels": [{"type": "null"}, {"type": "exact"},
                {"type": "jaro_winkler", "threshold": 0.9},
                {"type": "jaro_winkler", "threshold": 0.8},
                {"type": "else"}]},
    {"name": "code", "columns": ["code"],
     "levels": [{"type": "null"}, {"type": "exact"},
                {"type": "levenshtein", "threshold": 1},
                {"type": "else"}]}
  ]
})";

// The same data, with one comparison written in two metrics at once.
const char* const kMixedSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "key1", "type": "string"},
    {"name": "key2", "type": "string"},
    {"name": "word", "type": "string"},
    {"name": "code", "type": "string"}
  ],
  "comparisons": [
    {"name": "word", "columns": ["word"],
     "levels": [{"type": "null"}, {"type": "exact"},
                {"type": "levenshtein", "threshold": 1},
                {"type": "jaro_winkler", "threshold": 0.8},
                {"type": "else"}]}
  ]
})";

class LevelsFixture : public ::testing::Test {
   protected:
    // key1, key2, word, code.
    using Record = std::array<uint32_t, 4>;

    void Build(const char* schema_json) {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(schema_json, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        // Two alphabets, drawn once so the dictionaries are what the test says.
        for (uint32_t w = 0; w < kWordValues; ++w) {
            std::string value;
            for (int c = 0; c < 6; ++c) value += static_cast<char>('a' + Draw(26));
            words_.push_back(value);
        }
        for (uint32_t c = 0; c < kCodeValues; ++c) {
            std::string value;
            for (int d = 0; d < 5; ++d) value += static_cast<char>('0' + Draw(10));
            codes_.push_back(value);
        }

        std::vector<Record> planted(kPlanted);
        for (uint64_t i = 0; i < kPlanted; ++i) {
            planted[i] = {Draw(kKeyValues), Draw(kKeyValues), Draw(kWordValues),
                          Draw(kCodeValues)};
        }
        for (const Record& record : planted) Emit(record, "", "");
        for (const Record& record : planted) {
            const uint32_t roll = Draw(100);
            std::string word;
            if (roll >= 60 && roll < 85) {
                word = words_[record[2]];
                char replacement = static_cast<char>('a' + Draw(26));
                while (replacement == word[0]) {
                    replacement = static_cast<char>('a' + Draw(26));
                }
                word[0] = replacement;
            }
            Record copy = record;
            if (roll >= 85) copy[2] = Draw(kWordValues);
            std::string code;
            if (Draw(100) >= 70) {
                code = codes_[record[3]];
                for (int at : {1, 3}) {
                    char replacement = static_cast<char>('0' + Draw(10));
                    while (replacement == code[at]) {
                        replacement = static_cast<char>('0' + Draw(10));
                    }
                    code[at] = replacement;
                }
            }
            Emit(copy, word, code);
        }
        store_->set_num_records(2 * kPlanted);
        store_->Finalize();
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    uint32_t Draw(uint32_t range) {
        seed_ = seed_ * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>((seed_ >> 33) % range);
    }

    // An empty override means the record's own value; a non-empty one replaces it,
    // which is how a corruption that is not in the alphabet gets written.
    void Emit(const Record& record, const std::string& word, const std::string& code) {
        for (size_t i = 0; i < 2; ++i) {
            cpplink::StringColumn& column =
                std::get<cpplink::StringColumn>(store_->mutable_column(i));
            column.ids.push_back(column.dict.Intern("k" + std::to_string(i) + "_" +
                                                    std::to_string(record[i])));
        }
        cpplink::StringColumn& third =
            std::get<cpplink::StringColumn>(store_->mutable_column(2));
        third.ids.push_back(third.dict.Intern(word.empty() ? words_[record[2]] : word));
        cpplink::StringColumn& fourth =
            std::get<cpplink::StringColumn>(store_->mutable_column(3));
        fourth.ids.push_back(fourth.dict.Intern(code.empty() ? codes_[record[3]] : code));
        store_->mutable_ids().Append("r" + std::to_string(emitted_++));
    }

    cpplink::LevelsOptions Options() const {
        cpplink::LevelsOptions options;
        options.profile.expected_matches = kPlanted;
        options.profile.threads = 1;
        // One thread, so the order the stripes are folded in is fixed and every
        // number below is the same on every run.
        options.threads = 1;
        return options;
    }

    const cpplink::ComparisonLevels& Named(const cpplink::LevelsReport& report,
                                           const std::string& name) const {
        for (const cpplink::ComparisonLevels& item : report.comparisons) {
            if (item.name == name) return item;
        }
        ADD_FAILURE() << "no comparison " << name;
        return report.comparisons.front();
    }

    uint64_t emitted_ = 0;
    uint64_t seed_ = 24681357;
    std::vector<std::string> words_;
    std::vector<std::string> codes_;
    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::ComparisonSet comparisons_;
};

// The whole point. A quarter of every matching pair sits at a Jaro-Winkler of
// 0.8889, and the schema's hand-written 0.90 is above it: that level fires on
// nothing at all and the evidence falls through to a 0.80 whose u carries the whole
// band. The proposal has to find 0.88 from the data, with no model and no truth.
TEST_F(LevelsFixture, ProposalFindsThePlantedThreshold) {
    Build(kSchemaJson);
    const cpplink::LevelsReport report =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    const cpplink::ComparisonLevels& word = Named(report, "word");
    ASSERT_TRUE(word.proposed) << word.refusal;

    // What the schema does now: the 0.90 level is dead.
    ASSERT_EQ(word.current.size(), 4u);
    EXPECT_NEAR(word.current[1].threshold, 0.9, 1e-9);
    EXPECT_LT(word.current[1].m, 0.01);

    ASSERT_EQ(word.best.size(), 4u);
    EXPECT_EQ(word.best[0].type, cpplink::LevelType::kExact);
    EXPECT_EQ(word.best[1].type, cpplink::LevelType::kJaroWinkler);
    EXPECT_NEAR(word.best[1].threshold, 0.88, 1e-9);
    EXPECT_EQ(word.best.back().type, cpplink::LevelType::kElse);
    // And the rates it implies are the ones that were planted.
    EXPECT_NEAR(word.best[0].m, kWordKept, 0.02);
    EXPECT_NEAR(word.best[1].m, kWordNudged, 0.03);
    EXPECT_GE(word.best_bits, word.current_bits);
}

// The same on the other metric. Two digits are substituted, so `levenshtein <= 1`
// cannot reach them and the proposal has to widen rather than move.
TEST_F(LevelsFixture, ProposalWidensAnEditDistanceThatCannotReach) {
    Build(kSchemaJson);
    const cpplink::LevelsReport report =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    const cpplink::ComparisonLevels& code = Named(report, "code");
    ASSERT_TRUE(code.proposed) << code.refusal;
    EXPECT_EQ(code.metric, cpplink::LevelType::kLevenshtein);
    ASSERT_EQ(code.best.size(), 3u);
    EXPECT_EQ(code.best[1].type, cpplink::LevelType::kLevenshtein);
    EXPECT_NEAR(code.best[1].threshold, 2.0, 1e-9);
    EXPECT_NEAR(code.best[0].m, kCodeKept, 0.02);
    EXPECT_GT(code.best_bits, code.current_bits);
}

// The proposal is made at the schema's own count, and the dynamic program returns
// the best partition at that count, so it can never come out behind the partition
// the schema already has. This is the invariant the whole report rests on: without
// it a "gain" would be an artefact of comparing two different-sized models.
TEST_F(LevelsFixture, ProposalIsNeverWorseAtTheSameCount) {
    Build(kSchemaJson);
    const cpplink::LevelsReport report =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    size_t seen = 0;
    for (const cpplink::ComparisonLevels& item : report.comparisons) {
        if (!item.proposed) continue;
        ++seen;
        EXPECT_EQ(item.best_count, item.schema_count) << item.name;
        EXPECT_EQ(item.best.size(), item.current.size()) << item.name;
        EXPECT_GE(item.best_bits, item.current_bits - 1e-9) << item.name;
    }
    EXPECT_EQ(seen, 2u);
}

// u is a probability over the bins and has to behave like one, and the exact bin's
// share is the closed form the rest of the pipeline uses for an exact level: the
// rate two distinct rows carry the same value.
TEST_F(LevelsFixture, TheCurveIsAProbabilityAndItsExactBinIsTheClosedForm) {
    Build(kSchemaJson);
    const cpplink::LevelsReport report =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    const cpplink::ComparisonLevels& word = Named(report, "word");
    ASSERT_TRUE(word.proposed) << word.refusal;
    double u = 0.0;
    double m = 0.0;
    for (const cpplink::SimilarityBin& bin : word.bins) {
        EXPECT_GE(bin.u, 0.0);
        EXPECT_GE(bin.m, 0.0);
        u += bin.u;
        m += bin.m;
    }
    EXPECT_NEAR(u, 1.0, 1e-9);
    EXPECT_NEAR(m, 1.0, 1e-9);

    const cpplink::StringColumn& column =
        std::get<cpplink::StringColumn>(store_->column(2));
    double collisions = 0.0;
    for (uint32_t count : column.tf) {
        collisions += static_cast<double>(count) * (static_cast<double>(count) - 1.0);
    }
    const double rows = static_cast<double>(store_->NumRecords());
    EXPECT_NEAR(word.bins[0].u, collisions / (rows * (rows - 1.0)), 1e-12);
}

// A column with nothing fuzzy to place, and a column written in two metrics at
// once, are refused with a reason rather than given a partition of an axis that
// does not exist.
TEST_F(LevelsFixture, ComparisonsWithoutOneMetricAreRefused) {
    Build(kSchemaJson);
    const cpplink::LevelsReport report =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    const cpplink::ComparisonLevels& key = Named(report, "key1");
    EXPECT_FALSE(key.proposed);
    EXPECT_NE(key.refusal.find("no fuzzy level"), std::string::npos) << key.refusal;

    Build(kMixedSchemaJson);
    const cpplink::LevelsReport mixed =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    const cpplink::ComparisonLevels& word = Named(mixed, "word");
    EXPECT_FALSE(word.proposed);
    EXPECT_NE(word.refusal.find("mix"), std::string::npos) << word.refusal;
}

// Forcing a count gives that count, and more levels can only be worth more bits,
// because merging two bins is a partition the wider search already contains.
TEST_F(LevelsFixture, MoreLevelsAreNeverWorthLess) {
    Build(kSchemaJson);
    double previous = -1.0;
    for (size_t count = 2; count <= 6; ++count) {
        cpplink::LevelsOptions options = Options();
        options.levels = count;
        const cpplink::LevelsReport report =
            BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, options);
        const cpplink::ComparisonLevels& word = Named(report, "word");
        ASSERT_TRUE(word.proposed) << word.refusal;
        EXPECT_EQ(word.best_count, count);
        EXPECT_EQ(word.best.size(), count);
        EXPECT_GE(word.best_bits, previous - 1e-9);
        previous = word.best_bits;
    }
}

// The rewritten schema is a file that runs: it parses, it keeps everything the
// original carried, and its levels are the proposed ones.
TEST_F(LevelsFixture, RewrittenSchemaParsesAndCarriesTheProposal) {
    Build(kSchemaJson);
    const cpplink::LevelsReport report =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    std::string rewritten;
    std::string error;
    ASSERT_TRUE(cpplink::RewriteSchema(kSchemaJson, report, &rewritten, &error)) << error;

    cpplink::Schema parsed;
    ASSERT_TRUE(cpplink::ParseSchema(rewritten, &parsed, &error)) << error;
    ASSERT_EQ(parsed.columns.size(), schema_.columns.size());
    ASSERT_EQ(parsed.comparisons.size(), schema_.comparisons.size());
    // The refused comparison is untouched.
    EXPECT_EQ(parsed.comparisons[0].levels.size(), schema_.comparisons[0].levels.size());
    const cpplink::ComparisonLevels& word = Named(report, "word");
    const std::vector<cpplink::LevelSpec>& levels = parsed.comparisons[1].levels;
    ASSERT_EQ(levels.size(), word.best.size() + 1);  // the null level is carried over
    EXPECT_EQ(levels[0].type, cpplink::LevelType::kNull);
    for (size_t i = 0; i < word.best.size(); ++i) {
        EXPECT_EQ(levels[i + 1].type, word.best[i].type);
        EXPECT_NEAR(levels[i + 1].threshold, word.best[i].threshold, 1e-9);
    }
}

// Known pairs are scored against and never fitted to, so the partition is the one
// the anchor curve chose either way.
TEST_F(LevelsFixture, TruthIsScoredAgainstAndNotFittedTo) {
    Build(kSchemaJson);
    cpplink::TruthPairs truth;
    for (uint32_t i = 0; i < kPlanted; ++i) {
        truth.rows.emplace_back(i, static_cast<uint32_t>(i + kPlanted));
    }
    const cpplink::LevelsReport plain =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    const cpplink::LevelsReport scored =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options(), &truth);
    ASSERT_TRUE(scored.truthed);
    const cpplink::ComparisonLevels& before = Named(plain, "word");
    const cpplink::ComparisonLevels& after = Named(scored, "word");
    ASSERT_EQ(before.best.size(), after.best.size());
    for (size_t i = 0; i < before.best.size(); ++i) {
        EXPECT_EQ(before.best[i].type, after.best[i].type);
        EXPECT_NEAR(before.best[i].threshold, after.best[i].threshold, 1e-12);
    }
    EXPECT_NEAR(before.best_bits, after.best_bits, 1e-12);
    // The anchor here is exact, so the two curves are the same population and the
    // truth reading lands on the anchor one.
    EXPECT_TRUE(after.truthed);
    EXPECT_NEAR(after.best_truth_bits, after.best_bits, 0.05);
    EXPECT_GE(after.best_truth_bits, after.current_truth_bits - 1e-9);
}

TEST_F(LevelsFixture, ReportsPrintAndParse) {
    Build(kSchemaJson);
    const cpplink::LevelsReport report =
        BuildLevels(*store_, comparisons_, cpplink::PairMode::kAll, Options());
    std::ostringstream text;
    PrintLevelsReport(report, text);
    EXPECT_NE(text.str().find("Proposed"), std::string::npos);
    EXPECT_NE(text.str().find("no fuzzy level"), std::string::npos);

    std::ostringstream json;
    WriteLevelsJson(report, json);
    EXPECT_NO_THROW(nlohmann::json::parse(json.str()));
}

TEST(LevelsCommand, NeedsSchemaAndData) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"levels"}, out, err), 1);
    EXPECT_NE(err.str().find("--schema"), std::string::npos);
}

TEST(LevelsCommand, RejectsUnknownOption) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"levels", "--nope"}, out, err), 1);
    EXPECT_NE(err.str().find("unknown option"), std::string::npos);
}

}  // namespace
