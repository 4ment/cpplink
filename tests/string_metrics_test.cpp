// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/string_metrics.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/signature.hpp"

namespace {

// The textbook full-matrix edit distance, kept deliberately dumb: it is the thing
// the bit-parallel implementation has to agree with, so it must not share a line
// of code with it.
int ReferenceLevenshtein(const std::string& a, const std::string& b) {
    std::vector<int> previous(b.size() + 1);
    std::vector<int> current(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) previous[j] = static_cast<int>(j);
    for (size_t i = 1; i <= a.size(); ++i) {
        current[0] = static_cast<int>(i);
        for (size_t j = 1; j <= b.size(); ++j) {
            current[j] = std::min({previous[j] + 1, current[j - 1] + 1,
                                   previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
        }
        previous = current;
    }
    return previous[b.size()];
}

// The textbook Jaro, kept as deliberately dumb as the reference edit distance
// above and for the same reason: it is what the bit-parallel implementation has to
// agree with, so it shares no line of code with it.
double ReferenceJaro(const std::string& a, const std::string& b) {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;
    const size_t n = a.size();
    const size_t m = b.size();
    const size_t half = std::max(n, m) / 2;
    const size_t reach = half > 0 ? half - 1 : 0;
    std::vector<bool> hit_a(n, false);
    std::vector<bool> hit_b(m, false);
    size_t matches = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t lo = i > reach ? i - reach : 0;
        const size_t hi = std::min(i + reach + 1, m);
        for (size_t j = lo; j < hi; ++j) {
            if (hit_b[j] || a[i] != b[j]) continue;
            hit_a[i] = true;
            hit_b[j] = true;
            ++matches;
            break;
        }
    }
    if (matches == 0) return 0.0;
    size_t transpositions = 0;
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!hit_a[i]) continue;
        while (!hit_b[k]) ++k;
        if (a[i] != b[k]) ++transpositions;
        ++k;
    }
    transpositions /= 2;
    const double matched = static_cast<double>(matches);
    return (matched / static_cast<double>(n) + matched / static_cast<double>(m) +
            (matched - static_cast<double>(transpositions)) / matched) /
           3.0;
}

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

// Jaro has two implementations -- one word per match set where both values fit a
// word, the byte arrays where they do not -- and they are one metric, so the
// lengths here straddle that boundary deliberately. A disagreement in the last bit
// is a level decision that moves, not a rounding curiosity.
TEST(JaroTest, BothPathsAgreeWithTheReferenceOverRandomPairs) {
    std::mt19937_64 rng(13);
    const std::string alphabet = "abcdefgh";
    std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
    for (const std::pair<size_t, size_t>& span :
         {std::pair<size_t, size_t>{0, 14}, std::pair<size_t, size_t>{55, 72}}) {
        std::uniform_int_distribution<size_t> length(span.first, span.second);
        for (int trial = 0; trial < 20000; ++trial) {
            std::string a;
            std::string b;
            for (size_t i = length(rng); i > 0; --i) a.push_back(alphabet[pick(rng)]);
            for (size_t i = length(rng); i > 0; --i) b.push_back(alphabet[pick(rng)]);
            ASSERT_DOUBLE_EQ(cpplink::Jaro(a, b), ReferenceJaro(a, b))
                << "'" << a << "' vs '" << b << "'";
        }
    }
}

// The threshold reaches the metric, so the match count can end a call before a
// transposition is counted. That is only sound while it never changes the answer,
// which is the whole of what this asserts -- at thresholds either side of the
// prefix boost, over pairs near enough to sit on the decision.
TEST(JaroWinklerTest, AtLeastAgreesWithTheMetricItScreens) {
    std::mt19937_64 rng(14);
    const std::string alphabet = "abcd";
    std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
    std::uniform_int_distribution<size_t> length(1, 12);
    for (int trial = 0; trial < 20000; ++trial) {
        std::string a;
        std::string b;
        for (size_t i = length(rng); i > 0; --i) a.push_back(alphabet[pick(rng)]);
        // Mostly a corruption of `a`, so the pairs land near the thresholds rather
        // than being rejected on length or on a mask before the metric runs.
        b = a;
        if (!b.empty()) b[pick(rng) % b.size()] = alphabet[pick(rng)];
        if (trial % 3 == 0) {
            for (size_t i = length(rng); i > 0; --i) b.push_back(alphabet[pick(rng)]);
        }
        for (const double threshold : {0.5, 0.68, 0.7, 0.8, 0.88, 0.9, 0.95, 1.0}) {
            ASSERT_EQ(cpplink::JaroWinklerAtLeast(a, b, threshold),
                      cpplink::JaroWinkler(a, b) >= threshold)
                << "'" << a << "' vs '" << b << "' at " << threshold;
        }
    }
}

TEST(HaversineTest, MeasuresKnownDistances) {
    // Sydney to Melbourne, about 713 km.
    EXPECT_NEAR(cpplink::HaversineKm(-33.8688, 151.2093, -37.8136, 144.9631), 713.0, 5.0);
    EXPECT_DOUBLE_EQ(cpplink::HaversineKm(10.0, 20.0, 10.0, 20.0), 0.0);
    // A degree of latitude is about 111 km anywhere.
    EXPECT_NEAR(cpplink::HaversineKm(0.0, 0.0, 1.0, 0.0), 111.2, 0.5);
    EXPECT_NEAR(cpplink::HaversineKm(-60.0, 30.0, -59.0, 30.0), 111.2, 0.5);
}

// Myers' algorithm is a rewrite of the inner loop, not a new definition of the
// metric, so the only test that means anything is agreement with the reference on
// a large number of pairs -- including the alphabet sizes where bit collisions and
// the word boundary are most likely to bite.
TEST(LevenshteinTest, BitParallelAgreesWithTheFullMatrix) {
    std::mt19937_64 rng(20260903);
    const std::string alphabets[] = {"ab", "abcde",
                                     "abcdefghijklmnopqrstuvwxyz0123456789"};
    for (const std::string& alphabet : alphabets) {
        std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
        std::uniform_int_distribution<size_t> length(0, 70);
        for (int trial = 0; trial < 4000; ++trial) {
            std::string a;
            std::string b;
            for (size_t i = length(rng); i > 0; --i) a.push_back(alphabet[pick(rng)]);
            for (size_t i = length(rng); i > 0; --i) b.push_back(alphabet[pick(rng)]);
            const int expected = ReferenceLevenshtein(a, b);
            for (const int bound : {0, 1, 2, 5, 100}) {
                const int got = cpplink::BoundedLevenshtein(a, b, bound);
                const int want = expected > bound ? bound + 1 : expected;
                ASSERT_EQ(got, want) << "'" << a << "' vs '" << b << "' at k=" << bound;
            }
        }
    }
}

TEST(LevenshteinTest, BitParallelHandlesTheWordBoundary) {
    // 63, 64 and 65 characters: one below the word, exactly it, and one past it,
    // which is where the shift-based recurrence stops being usable.
    for (const size_t length : {63u, 64u, 65u}) {
        const std::string a(length, 'a');
        std::string b = a;
        b[length / 2] = 'b';
        EXPECT_EQ(cpplink::BoundedLevenshtein(a, b, 3), 1) << length;
        EXPECT_EQ(cpplink::BoundedLevenshtein(a, a, 3), 0) << length;
        EXPECT_EQ(cpplink::BoundedLevenshtein(a, a.substr(1), 3), 1) << length;
    }
}

TEST(SignatureTest, MaskCarriesEveryCharacterPresent) {
    const uint64_t mask = cpplink::CharacterMask("abc");
    EXPECT_NE(mask & (uint64_t{1} << ('a' & 63)), 0u);
    EXPECT_NE(mask & (uint64_t{1} << ('c' & 63)), 0u);
    EXPECT_EQ(mask & (uint64_t{1} << ('z' & 63)), 0u);
    EXPECT_EQ(cpplink::CharacterMask(""), 0u);
    EXPECT_EQ(cpplink::CharacterMask("aaa"), cpplink::CharacterMask("a"));
}

// The whole value of the filter rests on this: it may reject only pairs the metric
// itself would have rejected. A single counterexample makes it a bug rather than
// an optimization, so it is checked over the same population the metric is.
TEST(SignatureTest, JaroBoundNeverFallsBelowTheTrueSimilarity) {
    std::mt19937_64 rng(11);
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
    std::uniform_int_distribution<size_t> length(1, 14);
    for (int trial = 0; trial < 40000; ++trial) {
        std::string a;
        std::string b;
        for (size_t i = length(rng); i > 0; --i) a.push_back(alphabet[pick(rng)]);
        for (size_t i = length(rng); i > 0; --i) b.push_back(alphabet[pick(rng)]);
        const double bound = cpplink::JaroWinklerUpperBound(
            cpplink::CharacterMask(a), static_cast<uint32_t>(a.size()),
            cpplink::CharacterMask(b), static_cast<uint32_t>(b.size()));
        ASSERT_GE(bound, cpplink::JaroWinkler(a, b) - 1e-12)
            << "'" << a << "' vs '" << b << "'";
    }
}

TEST(SignatureTest, EditBoundNeverExceedsTheTrueDistance) {
    std::mt19937_64 rng(12);
    const std::string alphabet = "abcdefgh";
    std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
    std::uniform_int_distribution<size_t> length(0, 12);
    for (int trial = 0; trial < 40000; ++trial) {
        std::string a;
        std::string b;
        for (size_t i = length(rng); i > 0; --i) a.push_back(alphabet[pick(rng)]);
        for (size_t i = length(rng); i > 0; --i) b.push_back(alphabet[pick(rng)]);
        const int bound = cpplink::LevenshteinLowerBound(
            cpplink::CharacterMask(a), static_cast<uint32_t>(a.size()),
            cpplink::CharacterMask(b), static_cast<uint32_t>(b.size()));
        ASSERT_LE(bound, ReferenceLevenshtein(a, b)) << "'" << a << "' vs '" << b << "'";
    }
}

TEST(SignatureTest, IdenticalValuesAreNeverRejected) {
    for (const char* value : {"smith", "", "a", "Zolnerowich", "07700 900123"}) {
        const uint64_t mask = cpplink::CharacterMask(value);
        const uint32_t length = static_cast<uint32_t>(std::string(value).size());
        EXPECT_GE(cpplink::JaroWinklerUpperBound(mask, length, mask, length), 1.0)
            << value;
        EXPECT_EQ(cpplink::LevenshteinLowerBound(mask, length, mask, length), 0) << value;
    }
}

TEST(SignatureTest, TableIsIndexedByValueId) {
    cpplink::Dictionary dict;
    const uint32_t smith = dict.Intern("smith");
    const uint32_t jones = dict.Intern("jones");
    cpplink::SignatureTable table;
    table.Build(dict);
    EXPECT_EQ(table.Length(smith), 5u);
    EXPECT_EQ(table.Mask(jones), cpplink::CharacterMask("jones"));
    EXPECT_FALSE(table.Empty());
    EXPECT_EQ(table.BytesUsed(), 2 * (sizeof(uint64_t) + sizeof(uint32_t)));
}

}  // namespace
