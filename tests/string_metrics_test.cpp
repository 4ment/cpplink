// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/string_metrics.hpp"

#include <string>

#include <gtest/gtest.h>

namespace {

TEST(LevenshteinTest, CountsSingleEdits) {
    EXPECT_EQ(cpplink::BoundedLevenshtein("smith", "smith", 3), 0);
    EXPECT_EQ(cpplink::BoundedLevenshtein("smith", "smyth", 3), 1);   // substitution
    EXPECT_EQ(cpplink::BoundedLevenshtein("smith", "smiths", 3), 1);  // insertion
    EXPECT_EQ(cpplink::BoundedLevenshtein("smith", "sith", 3), 1);    // deletion
    EXPECT_EQ(cpplink::BoundedLevenshtein("smith", "smtih", 3), 2);   // transposition
}

TEST(LevenshteinTest, IsSymmetric) {
    EXPECT_EQ(cpplink::BoundedLevenshtein("kitten", "sitting", 5),
              cpplink::BoundedLevenshtein("sitting", "kitten", 5));
    EXPECT_EQ(cpplink::BoundedLevenshtein("kitten", "sitting", 5), 3);
}

// The bound is the contract: anything past it reports max + 1 rather than the true
// distance, because callers only ever ask "is this within k".
TEST(LevenshteinTest, ReportsBoundExceededRatherThanTheTrueDistance) {
    EXPECT_EQ(cpplink::BoundedLevenshtein("abcdef", "uvwxyz", 2), 3);
    EXPECT_EQ(cpplink::BoundedLevenshtein("abcdef", "uvwxyz", 6), 6);
}

TEST(LevenshteinTest, LengthGapAloneExceedsTheBound) {
    EXPECT_EQ(cpplink::BoundedLevenshtein("ab", "abcdefghij", 3), 4);
    EXPECT_EQ(cpplink::BoundedLevenshtein("", "abcd", 10), 4);
    EXPECT_EQ(cpplink::BoundedLevenshtein("", "", 2), 0);
}

TEST(LevenshteinTest, HandlesStringsLongerThanTheStackBuffer) {
    const std::string a(400, 'a');
    std::string b(400, 'a');
    b[200] = 'b';
    EXPECT_EQ(cpplink::BoundedLevenshtein(a, b, 3), 1);
}

TEST(JaroTest, MatchesThePublishedReferenceValues) {
    EXPECT_NEAR(cpplink::Jaro("MARTHA", "MARHTA"), 0.944444, 1e-5);
    EXPECT_NEAR(cpplink::Jaro("DIXON", "DICKSONX"), 0.766667, 1e-5);
    EXPECT_NEAR(cpplink::Jaro("JELLYFISH", "SMELLYFISH"), 0.896296, 1e-5);
}

TEST(JaroTest, HandlesEmptyAndDisjointStrings) {
    EXPECT_DOUBLE_EQ(cpplink::Jaro("", ""), 1.0);
    EXPECT_DOUBLE_EQ(cpplink::Jaro("abc", ""), 0.0);
    EXPECT_DOUBLE_EQ(cpplink::Jaro("abc", "xyz"), 0.0);
    EXPECT_DOUBLE_EQ(cpplink::Jaro("a", "a"), 1.0);
}

TEST(JaroWinklerTest, BoostsSharedPrefixesOnlyAboveTheThreshold) {
    EXPECT_NEAR(cpplink::JaroWinkler("MARTHA", "MARHTA"), 0.961111, 1e-5);
    EXPECT_NEAR(cpplink::JaroWinkler("DIXON", "DICKSONX"), 0.813333, 1e-5);
    // Below the boost threshold the prefix earns nothing.
    const double jaro = cpplink::Jaro("abcde", "abxyz");
    ASSERT_LT(jaro, 0.7);
    EXPECT_DOUBLE_EQ(cpplink::JaroWinkler("abcde", "abxyz"), jaro);
}

TEST(HaversineTest, MeasuresKnownDistances) {
    // Sydney to Melbourne, about 713 km.
    EXPECT_NEAR(cpplink::HaversineKm(-33.8688, 151.2093, -37.8136, 144.9631), 713.0, 5.0);
    EXPECT_DOUBLE_EQ(cpplink::HaversineKm(10.0, 20.0, 10.0, 20.0), 0.0);
    // A degree of latitude is about 111 km anywhere.
    EXPECT_NEAR(cpplink::HaversineKm(0.0, 0.0, 1.0, 0.0), 111.2, 0.5);
    EXPECT_NEAR(cpplink::HaversineKm(-60.0, 30.0, -59.0, 30.0), 111.2, 0.5);
}

}  // namespace
