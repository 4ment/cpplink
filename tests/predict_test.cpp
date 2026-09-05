// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/predict.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"

namespace {

constexpr uint64_t kRecords = 400;

const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "surname", "type": "string"},
    {"name": "city", "type": "string"}
  ],
  "comparisons": [
    {"name": "surname", "columns": ["surname"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "city", "columns": ["city"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "city"}]
})";

struct Edge {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t gamma = 0;
    double weight = 0.0;
    bool operator<(const Edge& other) const {
        return std::tie(a, b, gamma) < std::tie(other.a, other.b, other.gamma);
    }
    bool operator==(const Edge& other) const {
        return a == other.a && b == other.b && gamma == other.gamma &&
               weight == other.weight;
    }
};

class PredictFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / "cpplink_predict";
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);

        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        // A skewed surname distribution so the term-frequency adjustment has a
        // real spread to work with, and four cities so blocking leaves work to do.
        std::vector<uint32_t> names;
        for (int i = 0; i < 40; ++i) {
            names.push_back(surname.dict.Intern("name" + std::to_string(i)));
        }
        std::vector<uint32_t> cities;
        for (int i = 0; i < 4; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        for (uint64_t row = 0; row < kRecords; ++row) {
            // row % 40 is uniform; squaring the index concentrates the low names.
            const size_t pick = static_cast<size_t>((row * row) % 40) / 2;
            surname.ids.push_back(names[pick]);
            city.ids.push_back(cities[(row * 7) % 4]);
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();

        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;

        model_.lambda = 0.05;
        model_.records = kRecords;
        model_.comparisons = {
            Comparison("surname", true, {{1e-9, 1e-9}, {0.9, 0.02}, {0.1, 0.98}}),
            Comparison("city", false, {{0.8, 0.25}, {0.2, 0.75}})};
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
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

    // Runs prediction into a fresh subdirectory and returns the edges, sorted.
    std::vector<Edge> Run(const std::string& name, double threshold, bool bounds,
                          unsigned threads, cpplink::PredictReport* report) {
        cpplink::ScoreOptions score;
        score.threshold = threshold;
        score.use_bounds = bounds;
        // Both shortcuts move together here: "bounds off" is the exhaustive run
        // every admissibility claim is checked against.
        score.use_ceiling = bounds;
        cpplink::Scorer scorer;
        std::string error;
        EXPECT_TRUE(scorer.Bind(model_, comparisons_, *store_, score, &error)) << error;

        cpplink::PredictOptions options;
        options.out_dir = (dir_ / name).string();
        options.threads = threads;
        cpplink::PredictReport local;
        cpplink::PredictReport* target = report != nullptr ? report : &local;
        EXPECT_TRUE(cpplink::Predict(*store_, comparisons_, plan_, scorer, options,
                                     target, &error))
            << error;
        return ReadShards(options.out_dir);
    }

    static std::vector<Edge> ReadShards(const std::string& directory) {
        std::vector<Edge> edges;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            std::ifstream file(entry.path(), std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            EXPECT_GE(bytes.size(), 8u) << entry.path();
            EXPECT_EQ(bytes.compare(0, 8, "CPPLNKE1"), 0) << entry.path();
            const size_t body = bytes.size() - 8;
            EXPECT_EQ(body % 20, 0u) << entry.path();
            for (size_t at = 8; at + 20 <= bytes.size(); at += 20) {
                Edge edge;
                std::memcpy(&edge.a, bytes.data() + at, 4);
                std::memcpy(&edge.b, bytes.data() + at + 4, 4);
                std::memcpy(&edge.gamma, bytes.data() + at + 8, 4);
                std::memcpy(&edge.weight, bytes.data() + at + 12, 8);
                edges.push_back(edge);
            }
        }
        std::sort(edges.begin(), edges.end());
        return edges;
    }

    std::filesystem::path dir_;
    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::BlockingPlan plan_;
    cpplink::ComparisonSet comparisons_;
    cpplink::Model model_;
};

// The claim the whole three-way bracket rests on. If dropping a pattern on its
// bound ever loses an edge, the speed is bought with wrong answers.
TEST_F(PredictFixture, BoundedScoringEmitsExactlyWhatExhaustiveScoringDoes) {
    for (const double threshold : {-8.0, -2.0, 0.0, 1.5, 3.0, 20.0}) {
        cpplink::PredictReport bounded;
        cpplink::PredictReport exhaustive;
        const std::vector<Edge> fast =
            Run("fast" + std::to_string(threshold), threshold, true, 4, &bounded);
        const std::vector<Edge> slow =
            Run("slow" + std::to_string(threshold), threshold, false, 4, &exhaustive);
        EXPECT_EQ(fast, slow) << "threshold " << threshold;
        EXPECT_EQ(bounded.edges, exhaustive.edges) << "threshold " << threshold;
        EXPECT_EQ(bounded.enumerated, exhaustive.enumerated);
        EXPECT_EQ(exhaustive.dropped, 0u);
    }
}

// The ceiling on its own: same bracket, same threshold, the only difference is
// whether the pair is bounded before the comparison runs or after.
TEST_F(PredictFixture, TheCeilingSkipsWorkWithoutChangingTheEdges) {
    for (const double threshold : {-8.0, 0.0, 1.5, 3.0, 4.5}) {
        std::vector<Edge> with;
        std::vector<Edge> without;
        cpplink::PredictReport ceiling;
        cpplink::PredictReport flat;
        for (int pass = 0; pass < 2; ++pass) {
            cpplink::ScoreOptions score;
            score.threshold = threshold;
            score.use_ceiling = pass == 0;
            cpplink::Scorer scorer;
            std::string error;
            ASSERT_TRUE(scorer.Bind(model_, comparisons_, *store_, score, &error))
                << error;
            cpplink::PredictOptions options;
            options.out_dir =
                (dir_ / ("ceil" + std::to_string(pass) + std::to_string(threshold)))
                    .string();
            options.threads = 4;
            cpplink::PredictReport* report = pass == 0 ? &ceiling : &flat;
            ASSERT_TRUE(cpplink::Predict(*store_, comparisons_, plan_, scorer, options,
                                         report, &error))
                << error;
            (pass == 0 ? with : without) = ReadShards(options.out_dir);
        }
        EXPECT_EQ(with, without) << "threshold " << threshold;
        EXPECT_EQ(flat.skipped, 0u);
        EXPECT_EQ(ceiling.enumerated, flat.enumerated);
        // The comparisons the ceiling avoided are exactly the ones the bracket
        // would have dropped afterwards, so nothing else can have moved.
        EXPECT_EQ(ceiling.skipped + ceiling.dropped, flat.dropped);
        EXPECT_EQ(ceiling.checked, flat.checked);
        EXPECT_EQ(ceiling.certain, flat.certain);
    }
}

// At a threshold no pattern in this fixture can reach, the ceiling refuses every
// pair and not one comparison runs.
TEST_F(PredictFixture, AnUnreachableThresholdSkipsEveryCandidate) {
    cpplink::PredictReport report;
    const std::vector<Edge> edges = Run("unreachable", 40.0, true, 2, &report);
    EXPECT_TRUE(edges.empty());
    EXPECT_GT(report.enumerated, 0u);
    EXPECT_EQ(report.skipped, report.enumerated);
}

TEST_F(PredictFixture, ThreadCountDoesNotChangeTheEdges) {
    const std::vector<Edge> one = Run("t1", 0.0, true, 1, nullptr);
    EXPECT_EQ(one, Run("t4", 0.0, true, 4, nullptr));
    EXPECT_EQ(one, Run("t8", 0.0, true, 8, nullptr));
    EXPECT_FALSE(one.empty());
}

TEST_F(PredictFixture, ZoneCountsAddUpToTheCandidatesEnumerated) {
    cpplink::PredictReport report;
    Run("zones", 1.0, true, 4, &report);
    EXPECT_EQ(report.skipped + report.dropped + report.checked + report.certain,
              report.enumerated);
    EXPECT_EQ(report.enumerated, plan_.CountPairs(0));
    EXPECT_EQ(report.tf_lookups, report.checked + report.certain);
    EXPECT_EQ(report.patterns_drop + report.patterns_check + report.patterns_certain,
              report.pattern_space);
}

// Every emitted weight must clear the threshold, and every dropped pair must not.
TEST_F(PredictFixture, EveryEmittedEdgeClearsTheThreshold) {
    const double threshold = 1.0;
    const std::vector<Edge> edges = Run("above", threshold, true, 2, nullptr);
    ASSERT_FALSE(edges.empty());
    for (const Edge& edge : edges) {
        EXPECT_GE(edge.weight, threshold);
        EXPECT_LT(edge.a, edge.b);
    }
}

TEST_F(PredictFixture, RaisingTheThresholdOnlyRemovesEdges) {
    const std::vector<Edge> low = Run("low", -2.0, true, 2, nullptr);
    const std::vector<Edge> high = Run("high", 2.0, true, 2, nullptr);
    EXPECT_LE(high.size(), low.size());
    for (const Edge& edge : high) {
        EXPECT_NE(std::find(low.begin(), low.end(), edge), low.end());
    }
}

TEST_F(PredictFixture, TheEdgeLimitStopsAndSaysSo) {
    cpplink::ScoreOptions score;
    score.threshold = 0.0;
    cpplink::Scorer scorer;
    std::string error;
    ASSERT_TRUE(scorer.Bind(model_, comparisons_, *store_, score, &error)) << error;

    cpplink::PredictOptions options;
    options.out_dir = (dir_ / "limited").string();
    options.threads = 2;
    options.max_edges = 5;
    cpplink::PredictReport report;
    ASSERT_TRUE(
        cpplink::Predict(*store_, comparisons_, plan_, scorer, options, &report, &error))
        << error;
    EXPECT_TRUE(report.truncated);
    EXPECT_LE(report.edges, 5u);
    EXPECT_EQ(ReadShards(options.out_dir).size(), report.edges);
}

TEST_F(PredictFixture, CsvCarriesRecordIdsAndAProbability) {
    cpplink::ScoreOptions score;
    score.threshold = 1.0;
    cpplink::Scorer scorer;
    std::string error;
    ASSERT_TRUE(scorer.Bind(model_, comparisons_, *store_, score, &error)) << error;

    cpplink::PredictOptions options;
    options.out_dir = (dir_ / "csv").string();
    options.format = cpplink::EdgeFormat::kCsv;
    options.threads = 1;
    cpplink::PredictReport report;
    ASSERT_TRUE(
        cpplink::Predict(*store_, comparisons_, plan_, scorer, options, &report, &error))
        << error;

    std::ifstream file(options.out_dir + "/shard-000.csv");
    std::string header;
    std::getline(file, header);
    EXPECT_EQ(header, "id_a,id_b,gamma,match_weight,match_probability");
    std::string line;
    uint64_t rows = 0;
    while (std::getline(file, line)) {
        ++rows;
        EXPECT_EQ(line.compare(0, 1, "r"), 0) << line;
    }
    EXPECT_EQ(rows, report.edges);
}

TEST_F(PredictFixture, ReportNamesTheZonesAndTheShards) {
    cpplink::ScoreOptions score;
    score.threshold = 1.0;
    cpplink::Scorer scorer;
    std::string error;
    ASSERT_TRUE(scorer.Bind(model_, comparisons_, *store_, score, &error)) << error;
    cpplink::PredictOptions options;
    options.out_dir = (dir_ / "report").string();
    options.threads = 2;
    cpplink::PredictReport report;
    ASSERT_TRUE(
        cpplink::Predict(*store_, comparisons_, plan_, scorer, options, &report, &error));

    std::ostringstream out;
    cpplink::PrintPredictReport(report, scorer, out);
    const std::string text = out.str();
    EXPECT_NE(text.find("drop"), std::string::npos);
    EXPECT_NE(text.find("check"), std::string::npos);
    EXPECT_NE(text.find("admissible"), std::string::npos);
    EXPECT_NE(text.find("shard-000.bin"), std::string::npos);
}

}  // namespace
