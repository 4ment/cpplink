// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/minhash.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

double Jaccard(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    std::vector<uint64_t> shared;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(shared));
    const size_t together = a.size() + b.size() - shared.size();
    return together == 0
               ? 0.0
               : static_cast<double>(shared.size()) / static_cast<double>(together);
}

TEST(ShingleTest, ProducesSortedUniqueNGrams) {
    std::vector<uint64_t> out;
    cpplink::Shingles("abcd", 3, &out);
    EXPECT_EQ(out.size(), 2u);  // abc, bcd
    EXPECT_TRUE(std::is_sorted(out.begin(), out.end()));

    // A repeated n-gram is one shingle, not two.
    cpplink::Shingles("ababab", 2, &out);
    EXPECT_EQ(out.size(), 2u);  // ab, ba
}

// A value shorter than the n-gram must still participate rather than vanish.
TEST(ShingleTest, ShortValuesYieldThemselves) {
    std::vector<uint64_t> out;
    cpplink::Shingles("ab", 3, &out);
    EXPECT_EQ(out.size(), 1u);
    std::vector<uint64_t> same;
    cpplink::Shingles("ab", 3, &same);
    EXPECT_EQ(out, same);

    cpplink::Shingles("", 3, &out);
    EXPECT_TRUE(out.empty());
}

// The point of MinHash: the fraction of agreeing signature entries estimates
// Jaccard similarity. Without this the banding probabilities mean nothing.
TEST(MinHashTest, SignatureAgreementEstimatesJaccard) {
    constexpr size_t kHashes = 256;
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"johnathan smith", "johnathan smith"},
        {"johnathan smith", "jonathan smith"},
        {"johnathan smith", "johnathan smyth"},
        {"johnathan smith", "wilhelmina brackenridge"},
    };
    for (const auto& pair : cases) {
        std::vector<uint64_t> a, b;
        cpplink::Shingles(pair.first, 3, &a);
        cpplink::Shingles(pair.second, 3, &b);
        std::vector<uint32_t> sig_a(kHashes), sig_b(kHashes);
        cpplink::MinHashSignature(a, 42, kHashes, sig_a.data());
        cpplink::MinHashSignature(b, 42, kHashes, sig_b.data());

        size_t agree = 0;
        for (size_t i = 0; i < kHashes; ++i) {
            if (sig_a[i] == sig_b[i]) ++agree;
        }
        const double estimated = static_cast<double>(agree) / kHashes;
        EXPECT_NEAR(estimated, Jaccard(a, b), 0.08)
            << pair.first << " vs " << pair.second;
    }
}

TEST(MinHashTest, IdenticalValuesAlwaysShareEveryBandKey) {
    std::vector<uint64_t> shingles;
    cpplink::Shingles("kowalczyk", 3, &shingles);
    std::vector<uint32_t> signature(40);
    cpplink::MinHashSignature(shingles, 3, 40, signature.data());
    std::vector<uint64_t> first(10), second(10);
    cpplink::BandKeys(signature.data(), 10, 4, first.data());
    cpplink::BandKeys(signature.data(), 10, 4, second.data());
    EXPECT_EQ(first, second);
    // Different bands of the same signature are different keys.
    EXPECT_NE(first[0], first[1]);
}

TEST(BandingTest, RecallFollowsTheSCurve) {
    // P = 1 - (1 - s^r)^b, the property that lets recall be computed before a run.
    EXPECT_NEAR(cpplink::BandingRecall(1.0, 10, 4), 1.0, 1e-12);
    EXPECT_NEAR(cpplink::BandingRecall(0.0, 10, 4), 0.0, 1e-12);
    EXPECT_NEAR(cpplink::BandingRecall(0.5, 10, 4), 1.0 - std::pow(1.0 - 0.0625, 10),
                1e-12);
    // More bands can only help; more rows per band can only hurt.
    EXPECT_GT(cpplink::BandingRecall(0.6, 20, 4), cpplink::BandingRecall(0.6, 10, 4));
    EXPECT_LT(cpplink::BandingRecall(0.6, 10, 6), cpplink::BandingRecall(0.6, 10, 4));
}

// The measured hit rate must match the S-curve, or the configuration advice the
// report gives would be wrong.
TEST(BandingTest, MeasuredHitRateMatchesTheFormula) {
    constexpr size_t kBands = 8;
    constexpr size_t kRows = 3;
    constexpr size_t kHashes = kBands * kRows;
    constexpr int kTrials = 400;

    int hits = 0;
    double similarity_sum = 0.0;
    for (int trial = 0; trial < kTrials; ++trial) {
        // Two strings sharing a prefix, differing in a suffix: a spread of Jaccards.
        const std::string base = "abcdefghij" + std::to_string(trial);
        const std::string other = base.substr(0, 7) + "xyz" + std::to_string(trial);
        std::vector<uint64_t> a, b;
        cpplink::Shingles(base, 3, &a);
        cpplink::Shingles(other, 3, &b);
        similarity_sum += Jaccard(a, b);

        std::vector<uint32_t> sig_a(kHashes), sig_b(kHashes);
        cpplink::MinHashSignature(a, static_cast<uint64_t>(trial), kHashes, sig_a.data());
        cpplink::MinHashSignature(b, static_cast<uint64_t>(trial), kHashes, sig_b.data());
        std::vector<uint64_t> key_a(kBands), key_b(kBands);
        cpplink::BandKeys(sig_a.data(), kBands, kRows, key_a.data());
        cpplink::BandKeys(sig_b.data(), kBands, kRows, key_b.data());
        for (size_t band = 0; band < kBands; ++band) {
            if (key_a[band] == key_b[band]) {
                ++hits;
                break;
            }
        }
    }
    const double measured = static_cast<double>(hits) / kTrials;
    const double predicted =
        cpplink::BandingRecall(similarity_sum / kTrials, kBands, kRows);
    EXPECT_NEAR(measured, predicted, 0.12);
}

}  // namespace
