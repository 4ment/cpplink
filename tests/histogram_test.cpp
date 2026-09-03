// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/histogram.hpp"

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

TEST(PatternHistogramTest, CountsAndMergesInDenseMode) {
    cpplink::PatternHistogram histogram(8);
    ASSERT_TRUE(histogram.Dense());
    histogram.Add(3);
    histogram.Add(3);
    histogram.Add(17, 5);
    EXPECT_EQ(histogram.TotalPairs(), 7u);
    EXPECT_EQ(histogram.DistinctPatterns(), 2u);

    cpplink::PatternHistogram other(8);
    other.Add(3, 10);
    other.Add(200);
    histogram.Merge(other);
    EXPECT_EQ(histogram.TotalPairs(), 18u);
    EXPECT_EQ(histogram.DistinctPatterns(), 3u);

    const std::vector<cpplink::PatternCount> entries = histogram.Entries();
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].gamma, 3u);
    EXPECT_EQ(entries[0].count, 12u);
}

// Above kDenseWidth the container becomes an open-addressing map. The counts it
// keeps must be identical -- the switch is a memory decision, not a model one.
TEST(PatternHistogramTest, SparseModeAgreesWithDense) {
    cpplink::PatternHistogram dense(cpplink::PatternHistogram::kDenseWidth);
    cpplink::PatternHistogram sparse(cpplink::PatternHistogram::kDenseWidth + 1);
    ASSERT_TRUE(dense.Dense());
    ASSERT_FALSE(sparse.Dense());

    for (uint32_t i = 0; i < 5000; ++i) {
        const uint32_t gamma = (i * 2654435761u) >> 12;
        dense.Add(gamma, i + 1);
        sparse.Add(gamma, i + 1);
    }
    EXPECT_EQ(dense.TotalPairs(), sparse.TotalPairs());
    EXPECT_EQ(dense.DistinctPatterns(), sparse.DistinctPatterns());

    std::map<uint32_t, uint64_t> from_dense;
    std::map<uint32_t, uint64_t> from_sparse;
    for (const cpplink::PatternCount& entry : dense.Entries()) {
        from_dense[entry.gamma] = entry.count;
    }
    for (const cpplink::PatternCount& entry : sparse.Entries()) {
        from_sparse[entry.gamma] = entry.count;
    }
    EXPECT_EQ(from_dense, from_sparse);
}

// The all-ones pattern is a legal gamma at the full packed width and is also what
// the sparse table uses to mark a free slot. It must still be counted.
TEST(PatternHistogramTest, SparseModeKeepsTheAllOnesPattern) {
    cpplink::PatternHistogram histogram(32);
    ASSERT_FALSE(histogram.Dense());
    histogram.Add(0xFFFFFFFFu, 3);
    histogram.Add(7u);
    histogram.Add(0xFFFFFFFFu, 4);
    EXPECT_EQ(histogram.TotalPairs(), 8u);
    EXPECT_EQ(histogram.DistinctPatterns(), 2u);

    const std::vector<cpplink::PatternCount> entries = histogram.Entries();
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].gamma, 0xFFFFFFFFu);
    EXPECT_EQ(entries[0].count, 7u);

    cpplink::PatternHistogram merged(32);
    merged.Merge(histogram);
    EXPECT_EQ(merged.TotalPairs(), 8u);
    EXPECT_EQ(merged.DistinctPatterns(), 2u);
}

// Rows 0..5 share a surname, so they are one group of six; rows 6..8 share
// another. Only the first three of each also share a city.
class FoldFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        const std::string json = R"({
            "columns":[{"name":"surname","type":"string"},
                       {"name":"city","type":"string"}],
            "comparisons":[
                {"name":"surname","columns":["surname"],
                 "levels":[{"type":"exact"},{"type":"else"}]},
                {"name":"city","columns":["city"],
                 "levels":[{"type":"exact"},{"type":"else"}]}],
            "blocking":[{"type":"exact_value","column":"surname"}]})";
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(json, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        const uint32_t a = surname.dict.Intern("a");
        const uint32_t b = surname.dict.Intern("b");
        surname.ids = {a, a, a, a, a, a, b, b, b};
        const uint32_t north = city.dict.Intern("north");
        const uint32_t south = city.dict.Intern("south");
        city.ids = {north, north, north, south, south, south, north, north, south};

        store_->set_num_records(9);
        store_->Finalize();
        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
    }

    std::map<uint32_t, uint64_t> FoldWith(unsigned threads) {
        cpplink::HistogramOptions options;
        options.threads = threads;
        cpplink::PatternHistogram histogram(comparisons_.Width());
        cpplink::BuildHistogram(plan_, comparisons_, plan_.AllSources(), options,
                                &histogram, nullptr);
        std::map<uint32_t, uint64_t> counts;
        for (const cpplink::PatternCount& entry : histogram.Entries()) {
            counts[entry.gamma] = entry.count;
        }
        return counts;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::BlockingPlan plan_;
    cpplink::ComparisonSet comparisons_;
};

// The histogram must total exactly what the blocking report priced, or one of the
// two is lying and there is no way to tell which from the output alone.
TEST_F(FoldFixture, TotalsMatchTheExactPairCount) {
    const std::map<uint32_t, uint64_t> counts = FoldWith(1);
    uint64_t total = 0;
    for (const auto& entry : counts) total += entry.second;
    EXPECT_EQ(total, plan_.CountPairs(0));
    EXPECT_EQ(total, 18u);  // 6 choose 2 plus 3 choose 2
}

TEST_F(FoldFixture, ThreadCountDoesNotChangeTheCounts) {
    const std::map<uint32_t, uint64_t> one = FoldWith(1);
    EXPECT_EQ(one, FoldWith(4));
    EXPECT_EQ(one, FoldWith(8));
}

TEST_F(FoldFixture, PatternsCarryTheLevelsTheComparisonsAssign) {
    const std::map<uint32_t, uint64_t> counts = FoldWith(1);
    // Level 0 is exact and level 1 is else, so agreeing on both packs to zero.
    // Rows 0..2 and 3..5 give 3 + 3 pairs agreeing on surname and city; rows
    // 6..7 give one more.
    ASSERT_EQ(counts.count(0u), 1u);
    EXPECT_EQ(counts.at(0u), 7u);
}

// Sampling is what makes a session affordable, and it must fold strictly fewer
// pairs than it enumerates while still enumerating all of them.
TEST_F(FoldFixture, SamplingFoldsASubsetOfWhatItEnumerates) {
    cpplink::HistogramOptions options;
    options.threads = 1;
    options.pair_cap = 4;
    cpplink::PatternHistogram histogram(comparisons_.Width());
    cpplink::HistogramStats stats;
    cpplink::BuildHistogram(plan_, comparisons_, plan_.AllSources(), options, &histogram,
                            &stats);
    EXPECT_EQ(stats.enumerated, 18u);
    EXPECT_LT(stats.rate, 1.0);
    EXPECT_LT(stats.folded, stats.enumerated);
    EXPECT_EQ(stats.folded, histogram.TotalPairs());
}

}  // namespace
