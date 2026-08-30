// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/app.hpp"

#include <gtest/gtest.h>

#include <sstream>

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
}

TEST(RunTest, RejectsUnknownArgument) {
    std::ostringstream out, err;
    EXPECT_EQ(cpplink::Run({"--nope"}, out, err), 1);
    EXPECT_NE(err.str().find("unknown argument"), std::string::npos);
}

}  // namespace
