// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/model.hpp"

#include <cmath>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace {

cpplink::Model SmallModel() {
    cpplink::Model model;
    model.lambda = 1e-6;
    model.lambda_basis = "given on the command line";
    model.records = 1000;

    cpplink::ModelComparison comparison;
    comparison.name = "email";
    comparison.columns = {"email"};
    comparison.term_frequency = true;
    comparison.sessions = 2;

    cpplink::ModelLevel exact;
    exact.label = "exact";
    exact.m = 0.8;
    exact.u = 1e-7;
    exact.u_exact = true;
    exact.m_estimated = true;
    exact.m_support = 1234.5;
    comparison.levels.push_back(exact);

    cpplink::ModelLevel rest;
    rest.label = "else";
    rest.m = 0.2;
    rest.u = 1.0 - 1e-7;
    rest.m_estimated = true;
    rest.u_observed = 999999;
    comparison.levels.push_back(rest);

    model.comparisons.push_back(comparison);
    return model;
}

TEST(ModelTest, WeightIsTheLogTwoBayesFactor) {
    cpplink::ModelLevel level;
    level.m = 0.5;
    level.u = 0.125;
    EXPECT_DOUBLE_EQ(level.Weight(), 2.0);

    cpplink::Model model;
    model.lambda = 0.5;
    EXPECT_DOUBLE_EQ(model.PriorWeight(), 0.0);
    model.lambda = 0.2;
    EXPECT_NEAR(model.PriorWeight(), std::log2(0.25), 1e-12);
}

// A degenerate parameter must not become an infinite weight: scoring adds these
// up and one infinity takes the whole score with it.
TEST(ModelTest, ZeroParametersDoNotProduceAnInfiniteWeight) {
    cpplink::ModelLevel level;
    level.m = 0.0;
    level.u = 0.5;
    EXPECT_DOUBLE_EQ(level.Weight(), 0.0);

    cpplink::Model model;
    model.lambda = 0.0;
    EXPECT_DOUBLE_EQ(model.PriorWeight(), 0.0);
    model.lambda = 1.0;
    EXPECT_DOUBLE_EQ(model.PriorWeight(), 0.0);
}

TEST(ModelTest, JsonRoundTripsEveryField) {
    const cpplink::Model original = SmallModel();
    cpplink::Model parsed;
    std::string error;
    ASSERT_TRUE(cpplink::ParseModelJson(cpplink::ModelJson(original), &parsed, &error))
        << error;

    EXPECT_DOUBLE_EQ(parsed.lambda, original.lambda);
    EXPECT_EQ(parsed.lambda_basis, original.lambda_basis);
    EXPECT_EQ(parsed.records, original.records);
    ASSERT_EQ(parsed.comparisons.size(), 1u);
    EXPECT_EQ(parsed.comparisons[0].name, "email");
    EXPECT_EQ(parsed.comparisons[0].columns, original.comparisons[0].columns);
    EXPECT_TRUE(parsed.comparisons[0].term_frequency);
    EXPECT_EQ(parsed.comparisons[0].sessions, 2u);
    ASSERT_EQ(parsed.comparisons[0].levels.size(), 2u);
    EXPECT_DOUBLE_EQ(parsed.comparisons[0].levels[0].m, 0.8);
    EXPECT_DOUBLE_EQ(parsed.comparisons[0].levels[0].u, 1e-7);
    EXPECT_TRUE(parsed.comparisons[0].levels[0].u_exact);
    EXPECT_DOUBLE_EQ(parsed.comparisons[0].levels[0].m_support, 1234.5);
    EXPECT_EQ(parsed.comparisons[0].levels[1].u_observed, 999999u);
}

TEST(ModelTest, RejectsAModelMissingItsParameters) {
    cpplink::Model model;
    std::string error;
    EXPECT_FALSE(cpplink::ParseModelJson("not json", &model, &error));
    EXPECT_FALSE(cpplink::ParseModelJson(R"({"comparisons":[]})", &model, &error));
    EXPECT_FALSE(cpplink::ParseModelJson(R"({"lambda":0.1})", &model, &error));
    EXPECT_FALSE(cpplink::ParseModelJson(
        R"({"lambda":0.1,"comparisons":[{"levels":[{"m":0.5}]}]})", &model, &error));
    EXPECT_FALSE(error.empty());
}

TEST(ModelTest, PrintsAWeightPerLevel) {
    std::ostringstream out;
    cpplink::PrintModel(SmallModel(), out);
    const std::string text = out.str();
    EXPECT_NE(text.find("email"), std::string::npos);
    EXPECT_NE(text.find("u exact"), std::string::npos);
    EXPECT_NE(text.find("given on the command line"), std::string::npos);
    // log2(0.8 / 1e-7) is about 22.9 bits.
    EXPECT_NE(text.find("22.9"), std::string::npos);
}

}  // namespace
