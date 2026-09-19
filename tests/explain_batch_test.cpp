// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// `cpplink explain` answering many pairs from one load: the batch modes, the
// JSON object, and the wide file of every prediction's waterfall.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/reader.h>
#include <unistd.h>

#include "cpplink/app.hpp"
#include "cpplink/model.hpp"
#include "cpplink/sample_data.hpp"

namespace {

constexpr const char* kSchema = R"({
  "unique_id": "id",
  "columns": [
    {"name": "first_name", "type": "string"},
    {"name": "last_name", "type": "string"},
    {"name": "postcode", "type": "string"}
  ],
  "comparisons": [
    {"name": "last_name", "columns": ["last_name"], "term_frequency": true, "levels": [
      {"type": "null"}, {"type": "exact"},
      {"type": "jaro_winkler", "threshold": 0.85}, {"type": "else"}]},
    {"name": "first_name", "columns": ["first_name"], "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "postcode", "columns": ["postcode"], "term_frequency": true, "levels": [
      {"type": "null"}, {"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "postcode"}]
})";

cpplink::ModelComparison Learned(const std::string& name, bool tf,
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

class ExplainBatch : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_explain_batch_" + std::to_string(::getpid()));
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);
        data_ = (dir_ / "sample.parquet").string();
        schema_ = (dir_ / "schema.json").string();
        model_ = (dir_ / "model.json").string();
        pairs_ = (dir_ / "pairs.txt").string();

        cpplink::SampleOptions options;
        options.rows = 500;
        options.row_group_size = 100;
        options.duplicate_rate = 0.2;
        std::string error;
        ASSERT_TRUE(cpplink::WriteSampleParquet(data_, options, &error)) << error;
        std::ofstream(schema_) << kSchema;

        cpplink::Model model;
        model.lambda = 0.01;
        model.records = 500;
        model.comparisons = {
            Learned("last_name", true,
                    {{1e-3, 1e-3}, {0.8, 0.01}, {0.1, 0.02}, {0.099, 0.969}}),
            Learned("first_name", false, {{1e-3, 1e-3}, {0.85, 0.05}, {0.149, 0.949}}),
            Learned("postcode", true, {{1e-3, 1e-3}, {0.9, 0.005}, {0.099, 0.994}})};
        ASSERT_TRUE(cpplink::WriteModelJson(model, model_, &error)) << error;
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    int Explain(std::vector<std::string> extra, std::string* out, std::string* err) {
        std::vector<std::string> args = {"explain", "--schema",    schema_, "--model",
                                         model_,    "--threshold", "2"};
        args.insert(args.end(), extra.begin(), extra.end());
        args.push_back(data_);
        std::ostringstream o;
        std::ostringstream e;
        const int code = cpplink::Run(args, o, e);
        *out = o.str();
        *err = e.str();
        return code;
    }

    static std::vector<nlohmann::json> Lines(const std::string& text) {
        std::vector<nlohmann::json> lines;
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) lines.push_back(nlohmann::json::parse(line));
        }
        return lines;
    }

    std::filesystem::path dir_;
    std::string data_;
    std::string schema_;
    std::string model_;
    std::string pairs_;
};

TEST_F(ExplainBatch, OnePairGivesOneObject) {
    std::string out;
    std::string err;
    ASSERT_EQ(Explain({"--json", "--pair", "r1,r2"}, &out, &err), 0) << err;
    const std::vector<nlohmann::json> lines = Lines(out);
    ASSERT_EQ(lines.size(), 1u) << out;
    const nlohmann::json& w = lines[0];
    EXPECT_EQ(w["id_a"], "r1");
    EXPECT_EQ(w["id_b"], "r2");
    EXPECT_EQ(w["row_a"].get<int>(), 1);
    EXPECT_EQ(w["row_b"].get<int>(), 2);
    EXPECT_EQ(w["records"].get<int>(), 500);
    EXPECT_EQ(w["threshold"].get<double>(), 2.0);
    ASSERT_EQ(w["steps"].size(), 3u);
    EXPECT_EQ(w["steps"][0]["name"], "last_name");
    EXPECT_TRUE(w["steps"][0].contains("m"));
    // The ledger ends where the weight is.
    const nlohmann::json& last = w["steps"][2];
    EXPECT_NEAR(last["running"].get<double>(), w["weight"].get<double>(), 1e-9);
}

TEST_F(ExplainBatch, BatchAnswersEveryLineAndSurvivesABadOne) {
    std::ofstream(pairs_) << "r1,r2\nr3,nope\n\nr5,r6\n";
    std::string out;
    std::string err;
    ASSERT_EQ(Explain({"--json", "--pairs", pairs_}, &out, &err), 0) << err;
    const std::vector<nlohmann::json> lines = Lines(out);
    ASSERT_EQ(lines.size(), 3u) << out;
    EXPECT_EQ(lines[0]["id_a"], "r1");
    EXPECT_EQ(lines[1]["pair"], "r3,nope");
    EXPECT_NE(lines[1]["error"].get<std::string>().find("nope"), std::string::npos);
    EXPECT_EQ(lines[2]["id_b"], "r6");

    // The same pair answered alone is the same object.
    std::string one;
    ASSERT_EQ(Explain({"--json", "--pair", "r5,r6"}, &one, &err), 0) << err;
    EXPECT_EQ(Lines(one)[0], lines[2]);
}

TEST_F(ExplainBatch, RowPairsNameRowsAndAgreeWithIds) {
    std::ofstream(pairs_) << "5,6\n999,1\n";
    std::string out;
    std::string err;
    ASSERT_EQ(Explain({"--json", "--row-pairs", pairs_}, &out, &err), 0) << err;
    const std::vector<nlohmann::json> lines = Lines(out);
    ASSERT_EQ(lines.size(), 2u) << out;
    EXPECT_EQ(lines[0]["id_a"], "r5");
    EXPECT_EQ(lines[0]["id_b"], "r6");
    EXPECT_NE(lines[1]["error"].get<std::string>().find("out of range"),
              std::string::npos);

    std::string by_id;
    ASSERT_EQ(Explain({"--json", "--pair", "r5,r6"}, &by_id, &err), 0) << err;
    EXPECT_EQ(Lines(by_id)[0], lines[0]);
}

TEST_F(ExplainBatch, TextBatchPrintsOneReportPerPair) {
    std::ofstream(pairs_) << "r1,r2\nr5,r6\n";
    std::string out;
    std::string err;
    ASSERT_EQ(Explain({"--pairs", pairs_}, &out, &err), 0) << err;
    size_t reports = 0;
    for (size_t at = out.find("Match weight"); at != std::string::npos;
         at = out.find("Match weight", at + 1)) {
        ++reports;
    }
    EXPECT_EQ(reports, 2u) << out;
}

// The wide file: one row per prediction, whose columns add up to the weight the
// prediction file holds, in either format.
TEST_F(ExplainBatch, WaterfallFileExplainsEveryPrediction) {
    const std::string predictions = (dir_ / "predictions.parquet").string();
    std::ostringstream out;
    std::ostringstream err;
    ASSERT_EQ(
        cpplink::Run({"predict", "--schema", schema_, "--model", model_, "--threshold",
                      "2", "--out", predictions, "--threads", "2", data_},
                     out, err),
        0)
        << err.str();

    std::map<std::pair<std::string, std::string>, std::pair<uint32_t, double>> wrote;
    {
        auto input = arrow::io::ReadableFile::Open(predictions);
        ASSERT_TRUE(input.ok());
        auto reader = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
        ASSERT_TRUE(reader.ok());
        std::shared_ptr<arrow::Table> table;
        ASSERT_TRUE((*reader)->ReadTable(&table).ok());
        table = table->CombineChunks().ValueOrDie();
        const auto a = std::static_pointer_cast<arrow::StringArray>(
            table->GetColumnByName("id_a")->chunk(0));
        const auto b = std::static_pointer_cast<arrow::StringArray>(
            table->GetColumnByName("id_b")->chunk(0));
        const auto gamma = std::static_pointer_cast<arrow::UInt32Array>(
            table->GetColumnByName("gamma")->chunk(0));
        const auto weight = std::static_pointer_cast<arrow::DoubleArray>(
            table->GetColumnByName("match_weight")->chunk(0));
        for (int64_t i = 0; i < table->num_rows(); ++i) {
            wrote[{a->GetString(i), b->GetString(i)}] = {gamma->Value(i),
                                                         weight->Value(i)};
        }
    }
    ASSERT_GT(wrote.size(), 10u) << "the fixture must produce predictions";

    const std::string waterfalls = (dir_ / "waterfalls.parquet").string();
    std::string text;
    std::string trouble;
    ASSERT_EQ(
        Explain({"--predictions", predictions, "--out", waterfalls}, &text, &trouble), 0)
        << trouble;
    EXPECT_NE(text.find("wrote " + std::to_string(wrote.size()) + " waterfalls"),
              std::string::npos)
        << text;
    EXPECT_EQ(text.find("no longer produce"), std::string::npos) << text;

    auto input = arrow::io::ReadableFile::Open(waterfalls);
    ASSERT_TRUE(input.ok());
    auto reader = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
    ASSERT_TRUE(reader.ok());
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*reader)->ReadTable(&table).ok());
    table = table->CombineChunks().ValueOrDie();
    ASSERT_EQ(static_cast<size_t>(table->num_rows()), wrote.size());
    const std::vector<std::string> names = {"last_name", "first_name", "postcode"};
    for (const std::string& name : names) {
        for (const char* suffix : {"_level", "_bits", "_tf", "_frequency"}) {
            ASSERT_NE(table->GetColumnByName(name + suffix), nullptr) << name << suffix;
        }
    }
    const auto column = [&](const char* name) {
        return table->GetColumnByName(name)->chunk(0);
    };
    const auto a = std::static_pointer_cast<arrow::StringArray>(column("id_a"));
    const auto b = std::static_pointer_cast<arrow::StringArray>(column("id_b"));
    const auto gamma = std::static_pointer_cast<arrow::UInt32Array>(column("gamma"));
    const auto prior = std::static_pointer_cast<arrow::DoubleArray>(column("prior"));
    const auto weight =
        std::static_pointer_cast<arrow::DoubleArray>(column("match_weight"));
    const auto threshold =
        std::static_pointer_cast<arrow::DoubleArray>(column("threshold"));
    const auto emitted = std::static_pointer_cast<arrow::BooleanArray>(column("emitted"));
    std::vector<std::shared_ptr<arrow::DoubleArray>> bits;
    std::vector<std::shared_ptr<arrow::DoubleArray>> tf;
    std::vector<std::shared_ptr<arrow::UInt32Array>> frequency;
    for (const std::string& name : names) {
        bits.push_back(std::static_pointer_cast<arrow::DoubleArray>(
            column((name + "_bits").c_str())));
        tf.push_back(
            std::static_pointer_cast<arrow::DoubleArray>(column((name + "_tf").c_str())));
        frequency.push_back(std::static_pointer_cast<arrow::UInt32Array>(
            column((name + "_frequency").c_str())));
    }
    for (int64_t i = 0; i < table->num_rows(); ++i) {
        const auto found = wrote.find({a->GetString(i), b->GetString(i)});
        ASSERT_NE(found, wrote.end()) << a->GetString(i) << "," << b->GetString(i);
        EXPECT_EQ(gamma->Value(i), found->second.first);
        // The weight is the run's, to the last bit, and the row adds up to it.
        EXPECT_EQ(weight->Value(i), found->second.second);
        double running = prior->Value(i);
        for (size_t c = 0; c < names.size(); ++c) {
            running += bits[c]->Value(i) + tf[c]->Value(i);
            EXPECT_EQ(tf[c]->Value(i) != 0.0, frequency[c]->Value(i) != 0)
                << names[c] << " row " << i;
        }
        EXPECT_NEAR(running, weight->Value(i), 1e-9);
        EXPECT_EQ(threshold->Value(i), 2.0);
        EXPECT_EQ(emitted->Value(i), weight->Value(i) >= 2.0);
    }

    // The csv holds the same rows: a header naming every column, one line each.
    const std::string csv = (dir_ / "waterfalls.csv").string();
    ASSERT_EQ(Explain({"--predictions", predictions, "--out", csv}, &text, &trouble), 0)
        << trouble;
    std::ifstream file(csv);
    std::string line;
    ASSERT_TRUE(std::getline(file, line));
    EXPECT_EQ(line.rfind("id_a,id_b,gamma,prior,match_weight,", 0), 0u) << line;
    EXPECT_NE(line.find(",postcode_frequency"), std::string::npos) << line;
    const size_t width = static_cast<size_t>(std::count(line.begin(), line.end(), ','));
    size_t rows = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        ++rows;
        // Every row has the header's width, or a reader would misalign columns.
        ASSERT_EQ(static_cast<size_t>(std::count(line.begin(), line.end(), ',')), width)
            << line;
        std::string_view id_a;
        std::string_view id_b;
        const size_t first = line.find(',');
        const size_t second = line.find(',', first + 1);
        id_a = std::string_view(line).substr(0, first);
        id_b = std::string_view(line).substr(first + 1, second - first - 1);
        const auto found = wrote.find({std::string(id_a), std::string(id_b)});
        ASSERT_NE(found, wrote.end()) << line;
        // The csv rounds to six decimals, as the prediction csv does.
        const size_t third = line.find(',', second + 1);
        const size_t fourth = line.find(',', third + 1);
        const size_t fifth = line.find(',', fourth + 1);
        const double weight = std::stod(line.substr(fourth + 1, fifth - fourth - 1));
        EXPECT_NEAR(weight, found->second.second, 1e-6) << line;
    }
    EXPECT_EQ(rows, wrote.size());
}

TEST_F(ExplainBatch, WaterfallFileSkipsAndReportsUnknownIds) {
    const std::string predictions = (dir_ / "predictions.csv").string();
    std::ofstream(predictions) << "id_a,id_b,gamma,match_weight,match_probability\n"
                               << "r1,r2,0,1.0,0.5\nr3,nope,0,1.0,0.5\n";
    const std::string waterfalls = (dir_ / "waterfalls.csv").string();
    std::string text;
    std::string trouble;
    ASSERT_EQ(
        Explain({"--predictions", predictions, "--out", waterfalls}, &text, &trouble), 0)
        << trouble;
    EXPECT_NE(text.find("wrote 1 waterfalls"), std::string::npos) << text;
    EXPECT_NE(text.find("1 predictions name an id no record holds"), std::string::npos)
        << text;

    // A waterfall file needs a model and an output, and --out goes with it.
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"explain", "--schema", schema_, "--predictions", predictions,
                            "--out", waterfalls, data_},
                           out, err),
              1);
    EXPECT_NE(err.str().find("--model"), std::string::npos);
    err.str("");
    EXPECT_EQ(cpplink::Run({"explain", "--schema", schema_, "--model", model_,
                            "--predictions", predictions, data_},
                           out, err),
              1);
    EXPECT_NE(err.str().find("--out"), std::string::npos);
    err.str("");
    EXPECT_EQ(cpplink::Run({"explain", "--schema", schema_, "--model", model_, "--pair",
                            "r1,r2", "--out", waterfalls, data_},
                           out, err),
              1);
    EXPECT_NE(err.str().find("--predictions"), std::string::npos);
}

TEST_F(ExplainBatch, JsonNeedsAModelAndOnePairSource) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(
        cpplink::Run({"explain", "--schema", schema_, "--json", "--pair", "r1,r2", data_},
                     out, err),
        1);
    EXPECT_NE(err.str().find("--model"), std::string::npos);

    err.str("");
    EXPECT_EQ(cpplink::Run({"explain", "--schema", schema_, "--pair", "r1,r2", "--pairs",
                            pairs_, data_},
                           out, err),
              1);
    EXPECT_NE(err.str().find("exactly one"), std::string::npos);
}

}  // namespace
