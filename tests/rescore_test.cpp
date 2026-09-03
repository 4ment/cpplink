// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/rescore.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"
#include "cpplink/spill.hpp"

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

class RescoreFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / "cpplink_rescore";
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);

        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        std::vector<uint32_t> names;
        for (int i = 0; i < 40; ++i) {
            names.push_back(surname.dict.Intern("name" + std::to_string(i)));
        }
        std::vector<uint32_t> cities;
        for (int i = 0; i < 4; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        for (uint64_t row = 0; row < kRecords; ++row) {
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

    cpplink::Scorer Bind(const cpplink::Model& model, double threshold) {
        cpplink::ScoreOptions score;
        score.threshold = threshold;
        cpplink::Scorer scorer;
        std::string error;
        EXPECT_TRUE(scorer.Bind(model, comparisons_, *store_, score, &error)) << error;
        return scorer;
    }

    std::vector<Edge> Predict(const std::string& name, const cpplink::Model& model,
                              double threshold, const std::string& spill, double sample,
                              cpplink::PredictReport* report) {
        const cpplink::Scorer scorer = Bind(model, threshold);
        cpplink::PredictOptions options;
        options.out_dir = (dir_ / name).string();
        options.threads = 4;
        if (!spill.empty()) options.spill_dir = (dir_ / spill).string();
        options.spill_sample = sample;
        cpplink::PredictReport local;
        cpplink::PredictReport* target = report != nullptr ? report : &local;
        std::string error;
        EXPECT_TRUE(cpplink::Predict(*store_, comparisons_, plan_, scorer, options,
                                     target, &error))
            << error;
        return ReadShards(options.out_dir);
    }

    std::vector<Edge> Rescore(const std::string& name, const cpplink::Model& model,
                              double threshold, const std::string& spill,
                              cpplink::RescoreReport* report, std::string* error) {
        const cpplink::Scorer scorer = Bind(model, threshold);
        cpplink::RescoreOptions options;
        options.spill_dir = (dir_ / spill).string();
        options.out_dir = (dir_ / name).string();
        options.threads = 4;
        cpplink::RescoreReport local;
        cpplink::RescoreReport* target = report != nullptr ? report : &local;
        std::string problem;
        std::string* sink = error != nullptr ? error : &problem;
        if (!cpplink::Rescore(*store_, comparisons_, scorer, options, target, sink)) {
            return {};
        }
        return ReadShards(options.out_dir);
    }

    static std::vector<Edge> ReadShards(const std::string& directory) {
        std::vector<Edge> edges;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            std::ifstream file(entry.path(), std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
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

// The property the whole feature rests on: gamma is everything the comparison
// produced, so re-reading it must give back exactly what scoring the pairs gave.
TEST_F(RescoreFixture, ReproducesPredictExactlyUnderTheSameModel) {
    for (const double threshold : {-8.0, -2.0, 0.0, 1.5, 3.0}) {
        const std::string tag = std::to_string(threshold);
        cpplink::PredictReport predicted;
        const std::vector<Edge> direct =
            Predict("p" + tag, model_, threshold, "s" + tag, 0.0, &predicted);
        cpplink::RescoreReport rescored;
        const std::vector<Edge> replayed =
            Rescore("r" + tag, model_, threshold, "s" + tag, &rescored, nullptr);
        EXPECT_EQ(direct, replayed) << "threshold " << threshold;
        EXPECT_EQ(rescored.edges, predicted.edges);
        EXPECT_EQ(rescored.pairs, predicted.spilled);
        EXPECT_FALSE(rescored.below_spill_threshold);
    }
}

TEST_F(RescoreFixture, SpillHoldsExactlyTheEdgesWhenNotSampling) {
    cpplink::PredictReport report;
    Predict("p", model_, 0.0, "s", 0.0, &report);
    EXPECT_EQ(report.spilled, report.edges);
    EXPECT_GT(report.spilled, 0u);

    cpplink::SpillManifest manifest;
    std::string error;
    ASSERT_TRUE(cpplink::ReadSpillManifest((dir_ / "s").string(), &manifest, &error))
        << error;
    EXPECT_EQ(manifest.records, kRecords);
    EXPECT_EQ(manifest.candidates, report.enumerated);
    EXPECT_EQ(manifest.above_threshold, report.edges);
    EXPECT_DOUBLE_EQ(manifest.sample_rate, 0.0);
    EXPECT_EQ(manifest.layout, cpplink::GammaLayout(comparisons_));
}

TEST_F(RescoreFixture, SamplingKeepsPairsBelowTheThresholdToo) {
    cpplink::PredictReport report;
    Predict("p", model_, 0.0, "s", 0.25, &report);
    EXPECT_GT(report.spilled, report.edges) << "the sample reaches discarded pairs";
    // Roughly a quarter of the pairs that did not become edges, plus every edge.
    const double expected =
        static_cast<double>(report.enumerated - report.edges) * 0.25 + report.edges;
    EXPECT_NEAR(static_cast<double>(report.spilled), expected, expected * 0.1);
}

// Raising the threshold on a spill is exact, because everything that could
// qualify is already there.
TEST_F(RescoreFixture, RaisingTheThresholdMatchesAFullRun) {
    Predict("p", model_, 0.0, "s", 0.0, nullptr);
    for (const double threshold : {0.5, 2.0, 4.0}) {
        const std::string tag = std::to_string(threshold);
        const std::vector<Edge> full =
            Predict("full" + tag, model_, threshold, "", 0.0, nullptr);
        cpplink::RescoreReport report;
        const std::vector<Edge> replayed =
            Rescore("re" + tag, model_, threshold, "s", &report, nullptr);
        EXPECT_EQ(full, replayed) << "threshold " << threshold;
        EXPECT_FALSE(report.below_spill_threshold);
    }
}

// Lowering it is not, and the report has to say so rather than look like a run.
TEST_F(RescoreFixture, LoweringTheThresholdIsFlaggedAsIncomplete) {
    Predict("p", model_, 2.0, "s", 0.0, nullptr);
    cpplink::RescoreReport report;
    Rescore("re", model_, -5.0, "s", &report, nullptr);
    EXPECT_TRUE(report.below_spill_threshold);

    std::ostringstream out;
    cpplink::PrintRescoreReport(report, Bind(model_, -5.0), out);
    EXPECT_NE(out.str().find("tuning, not as a run"), std::string::npos) << out.str();
}

TEST_F(RescoreFixture, ANewModelChangesTheWeightsWithoutComparingAgain) {
    Predict("p", model_, -20.0, "s", 0.0, nullptr);
    cpplink::Model stronger = model_;
    stronger.comparisons[1].levels[0].m = 0.99;
    stronger.comparisons[1].levels[0].u = 0.01;

    cpplink::RescoreReport before;
    const std::vector<Edge> original = Rescore("a", model_, -20.0, "s", &before, nullptr);
    cpplink::RescoreReport after;
    const std::vector<Edge> updated = Rescore("b", stronger, -20.0, "s", &after, nullptr);

    ASSERT_EQ(original.size(), updated.size()) << "the same pairs, re-weighted";
    bool moved = false;
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_EQ(original[i].a, updated[i].a);
        EXPECT_EQ(original[i].gamma, updated[i].gamma);
        if (original[i].weight != updated[i].weight) moved = true;
    }
    EXPECT_TRUE(moved) << "a different model must produce different weights";
}

TEST_F(RescoreFixture, RefusesASpillFromADifferentSchema) {
    Predict("p", model_, 0.0, "s", 0.0, nullptr);
    cpplink::SpillManifest manifest;
    std::string error;
    ASSERT_TRUE(cpplink::ReadSpillManifest((dir_ / "s").string(), &manifest, &error));
    manifest.layout = "surname:9,city:9";
    ASSERT_TRUE(cpplink::WriteSpillManifest((dir_ / "s").string(), manifest, &error));

    Rescore("re", model_, 0.0, "s", nullptr, &error);
    EXPECT_NE(error.find("gamma means something else"), std::string::npos) << error;
}

TEST_F(RescoreFixture, RefusesASpillOverADifferentRowCount) {
    Predict("p", model_, 0.0, "s", 0.0, nullptr);
    cpplink::SpillManifest manifest;
    std::string error;
    ASSERT_TRUE(cpplink::ReadSpillManifest((dir_ / "s").string(), &manifest, &error));
    manifest.records = kRecords + 1;
    ASSERT_TRUE(cpplink::WriteSpillManifest((dir_ / "s").string(), manifest, &error));

    Rescore("re", model_, 0.0, "s", nullptr, &error);
    EXPECT_NE(error.find("but the data has"), std::string::npos) << error;
}

TEST_F(RescoreFixture, RefusesATruncatedSpill) {
    Predict("p", model_, -20.0, "s", 0.0, nullptr);
    for (const auto& entry : std::filesystem::directory_iterator(dir_ / "s")) {
        if (entry.path().extension() != ".bin") continue;
        const auto size = std::filesystem::file_size(entry.path());
        if (size > 8 + 12) {
            std::filesystem::resize_file(entry.path(), size - 5);
            break;
        }
    }
    std::string error;
    Rescore("re", model_, -20.0, "s", nullptr, &error);
    EXPECT_NE(error.find("truncated mid-pair"), std::string::npos) << error;
}

TEST_F(RescoreFixture, RefusesADirectoryThatIsNotASpill) {
    std::string error;
    Rescore("re", model_, 0.0, "nothing", nullptr, &error);
    EXPECT_NE(error.find("is that a spill directory?"), std::string::npos) << error;
}

}  // namespace
