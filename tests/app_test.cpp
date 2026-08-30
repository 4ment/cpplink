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
