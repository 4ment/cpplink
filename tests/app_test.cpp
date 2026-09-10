// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/app.hpp"

#include <sstream>

#include <gtest/gtest.h>

namespace {

TEST(RunTest, ReportsVersion) {
    std::ostringstream out, err;
    EXPECT_EQ(cpplink::Run({"--version"}, out, err), 0);
    EXPECT_EQ(out.str(), std::string(cpplink::kVersion) + "\n");
    EXPECT_TRUE(err.str().empty());
}

TEST(RunTest, PrintsUsageForHelp) {
    std::ostringstream out, err;
    EXPECT_EQ(cpplink::Run({"--help"}, out, err), 0);
    EXPECT_NE(out.str().find("usage: cpplink"), std::string::npos);
    EXPECT_NE(out.str().find("inspect"), std::string::npos);
    EXPECT_NE(out.str().find("gen-sample"), std::string::npos);
}

TEST(RunTest, RejectsUnknownCommand) {
    std::ostringstream out, err;
    EXPECT_EQ(cpplink::Run({"--nope"}, out, err), 1);
    EXPECT_NE(err.str().find("unknown command"), std::string::npos);
}

TEST(RunTest, InspectNeedsSchemaAndFile) {
    std::ostringstream out, err;
    EXPECT_EQ(cpplink::Run({"inspect"}, out, err), 1);
    EXPECT_NE(err.str().find("--schema"), std::string::npos);
}

TEST(RunTest, OptionsNeedValues) {
    std::ostringstream out, err;
    EXPECT_EQ(cpplink::Run({"inspect", "--schema"}, out, err), 1);
    EXPECT_NE(err.str().find("needs a value"), std::string::npos);
}

}  // namespace

TEST(RunTest, EstimateNeedsSchemaAndFile) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"estimate"}, out, err), 1);
    EXPECT_NE(err.str().find("--schema"), std::string::npos);
}

TEST(RunTest, EstimateRejectsUnknownOptions) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"estimate", "--u-samples", "10"}, out, err), 1);
    EXPECT_NE(err.str().find("unknown option"), std::string::npos);
}

TEST(RunTest, UsageListsEveryCommand) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"--help"}, out, err), 0);
    for (const char* command :
         {"inspect", "explain", "explain-blocking", "recall", "estimate", "predict",
          "rescore", "cluster", "merge-predictions", "gen-sample"}) {
        EXPECT_NE(out.str().find(command), std::string::npos) << command;
    }
}

TEST(RunTest, PredictNeedsSchemaModelOutAndThreshold) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"predict"}, out, err), 1);
    EXPECT_NE(err.str().find("--model"), std::string::npos);
}

TEST(RunTest, PredictRejectsABadProbability) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"predict", "--probability", "1.5"}, out, err), 1);
    EXPECT_NE(err.str().find("(0, 1)"), std::string::npos);
}

TEST(RunTest, PredictRejectsAnUnknownFormat) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"predict", "--format", "parquet"}, out, err), 1);
    EXPECT_NE(err.str().find("bin or csv"), std::string::npos);
}

TEST(RunTest, ExplainRejectsAnUnknownOption) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"explain", "--waterfall"}, out, err), 1);
    EXPECT_NE(err.str().find("unknown option"), std::string::npos);
}

TEST(RunTest, RescoreNeedsASpillAndAModel) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"rescore"}, out, err), 1);
    EXPECT_NE(err.str().find("--spill"), std::string::npos);
}

TEST(RunTest, PredictRejectsABadSpillSample) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"predict", "--spill-sample", "0"}, out, err), 1);
    EXPECT_NE(err.str().find("(0, 1]"), std::string::npos);
}

TEST(RunTest, ClusterNeedsSchemaPredictionsAndData) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"cluster"}, out, err), 1);
    EXPECT_NE(err.str().find("--predictions"), std::string::npos);
}

TEST(RunTest, ClusterRejectsABadProbability) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"cluster", "--probability", "0"}, out, err), 1);
    EXPECT_NE(err.str().find("(0, 1)"), std::string::npos);
}

TEST(RunTest, ClusterRejectsAnUnknownOption) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"cluster", "--parent-array"}, out, err), 1);
    EXPECT_NE(err.str().find("unknown option"), std::string::npos);
}

TEST(RunTest, MergePredictionsNeedsShardsAndOut) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"merge-predictions", "--shards", "edges"}, out, err), 1);
    EXPECT_NE(err.str().find("--out"), std::string::npos);
}

TEST(RunTest, MergePredictionsRejectsAnUnknownFormat) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run(
                  {"merge-predictions", "--shards", "e", "--out", "o", "--format", "orc"},
                  out, err),
              1);
    EXPECT_NE(err.str().find("csv or parquet"), std::string::npos);
}

TEST(RunTest, MergePredictionsWantsBothHalvesOfTheIdLookup) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"merge-predictions", "--shards", "e", "--out", "o.csv",
                            "--schema", "s.json"},
                           out, err),
              1);
    EXPECT_NE(err.str().find("parquet input"), std::string::npos);
}

TEST(RunTest, PredictRefusesAShardFormatForASingleFileOut) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(cpplink::Run({"predict", "--schema", "s.json", "--model", "m.json", "--out",
                            "edges.parquet", "--threshold", "20", "--format", "csv",
                            "data.parquet"},
                           out, err),
              1);
    EXPECT_NE(err.str().find("single file"), std::string::npos) << err.str();
}
