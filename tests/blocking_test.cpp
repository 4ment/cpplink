// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/blocking.hpp"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/recall.hpp"
#include "cpplink/schema.hpp"

namespace {

// surname values by row, with a null at row 6:
//   smith smith smith jones jones brown <null> smythe
constexpr const char* kColumns = R"("columns":[
    {"name":"surname","type":"string"},
    {"name":"dob","type":"date"},
    {"name":"lat","type":"double"}])";

class BlockingFixture : public ::testing::Test {
   protected:
    void Build(const std::string& blocking) {
        const std::string json =
            "{" + std::string(kColumns) + ",\"blocking\":" + blocking + "}";
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(json, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);

        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        const uint32_t smith = surname.dict.Intern("smith");
        const uint32_t jones = surname.dict.Intern("jones");
        const uint32_t brown = surname.dict.Intern("brown");
        const uint32_t smythe = surname.dict.Intern("smythe");
        surname.ids = {smith, smith, smith, jones, jones, brown, cpplink::kNullId,
                       smythe};

        auto& dob = std::get<cpplink::DateColumn>(store_->mutable_column(1));
        dob.values = {100, 100, 200, 200, 300, 300, cpplink::kNullDate, 400};

        store_->set_num_records(8);
        store_->Finalize();
        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
    }

    // Every pair the plan emits, which must contain no duplicates.
    std::vector<std::pair<uint32_t, uint32_t>> Emitted() const {
        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        plan_.ForEachPair([&pairs](uint32_t a, uint32_t b) {
            pairs.emplace_back(std::min(a, b), std::max(a, b));
        });
        return pairs;
    }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::BlockingPlan plan_;
};

TEST_F(BlockingFixture, ExactValueEmitsEveryPairSharingAValue) {
    Build(R"([{"type":"exact_value","column":"surname"}])");
    // smith x3 -> 3 pairs, jones x2 -> 1 pair. brown and smythe are singletons.
    EXPECT_EQ(plan_.CountPairs(0), 4u);
    EXPECT_EQ(Emitted().size(), 4u);
    EXPECT_TRUE(plan_.Produces(0, 0, 1));
    EXPECT_TRUE(plan_.Produces(0, 3, 4));
    EXPECT_FALSE(plan_.Produces(0, 0, 3));
}

// A null is not a value. Two rows that are both missing must never be candidates.
TEST_F(BlockingFixture, NullsNeverBlockTogether) {
    Build(R"([{"type":"exact_value","column":"surname"},
              {"type":"exact_value","column":"dob"}])");
    EXPECT_EQ(plan_.KeyOf(0, 6), cpplink::kNoKey);
    EXPECT_EQ(plan_.KeyOf(1, 6), cpplink::kNoKey);
    EXPECT_FALSE(plan_.Produces(0, 6, 6));
    for (uint32_t row = 0; row < 8; ++row) {
        EXPECT_FALSE(plan_.Produces(0, 6, row)) << "row " << row;
        EXPECT_FALSE(plan_.Produces(1, 6, row)) << "row " << row;
    }
}

TEST_F(BlockingFixture, RareValueExcludesValuesPastTheCap) {
    Build(R"([{"type":"rare_value","column":"surname","max_frequency":2}])");
    // "smith" appears three times, so it is excluded; only the jones pair survives.
    EXPECT_EQ(plan_.CountPairs(0), 1u);
    EXPECT_EQ(plan_.LargestGroup(0), 2u);
    EXPECT_FALSE(plan_.Produces(0, 0, 1));
    EXPECT_TRUE(plan_.Produces(0, 3, 4));
}

// The whole value of explain-blocking is that the count is exact without
// enumerating. If these ever disagree, the report is lying.
TEST_F(BlockingFixture, CountMatchesEnumerationForEverySourceAlone) {
    const std::vector<std::string> configs = {
        R"([{"type":"exact_value","column":"surname"}])",
        R"([{"type":"rare_value","column":"surname","max_frequency":2}])",
        R"([{"type":"exact_value","column":"dob"}])",
        R"([{"type":"sorted_neighbourhood","column":"surname","window":3}])",
        R"([{"type":"minhash","column":"surname","bands":6,"rows_per_band":2,
             "ngram":2}])",
        R"([{"type":"all_pairs"}])",
    };
    for (const std::string& config : configs) {
        Build(config);
        for (size_t s = 0; s < plan_.Size(); ++s) {
            uint64_t enumerated = 0;
            for (uint32_t a = 0; a < 8; ++a) {
                for (uint32_t b = a + 1; b < 8; ++b) {
                    if (plan_.Produces(s, a, b)) ++enumerated;
                }
            }
            EXPECT_EQ(plan_.CountPairs(s), enumerated)
                << config << " source " << plan_.at(s).name;
        }
    }
}

// The source that does no blocking. Every pair of the store is a candidate,
// including the ones every other source refuses: the null row pairs here, because
// there is no key for it to be missing from.
TEST_F(BlockingFixture, AllPairsEmitsEveryPairOfTheStore) {
    Build(R"([{"type":"all_pairs"}])");
    EXPECT_EQ(plan_.CountPairs(0), 28u);  // 8 * 7 / 2
    EXPECT_EQ(Emitted().size(), 28u);
    EXPECT_EQ(plan_.LargestGroup(0), 8u);
    EXPECT_TRUE(plan_.Produces(0, 0, 3));
    EXPECT_TRUE(plan_.Produces(0, 6, 0));   // the null row
    EXPECT_FALSE(plan_.Produces(0, 4, 4));  // a row is not a pair with itself
    // It conditions on nothing, which is the empty column subset: EM can hold
    // nothing out because there is nothing to hold out.
    EXPECT_TRUE(plan_.at(0).em_safe);
    EXPECT_TRUE(plan_.at(0).column.empty());
}

// An unblocked source produces everything, so the union is what it produces and
// every later source is redundant -- emitting nothing rather than emitting twice.
TEST_F(BlockingFixture, AllPairsSubsumesEveryOtherSource) {
    Build(R"([{"type":"all_pairs"},
              {"type":"exact_value","column":"surname"},
              {"type":"sorted_neighbourhood","column":"surname","window":3}])");
    const auto pairs = Emitted();
    const std::set<std::pair<uint32_t, uint32_t>> unique(pairs.begin(), pairs.end());
    EXPECT_EQ(pairs.size(), 28u);
    EXPECT_EQ(unique.size(), 28u);
    EXPECT_EQ(plan_.CountUnion(), 28u);
}

TEST_F(BlockingFixture, SortedNeighbourhoodEmitsWithinTheWindowOnly) {
    Build(R"([{"type":"sorted_neighbourhood","column":"surname","window":2}])");
    // Seven non-null rows in a window of two: 6 + 5 = 11 pairs.
    EXPECT_EQ(plan_.CountPairs(0), 11u);
    EXPECT_EQ(Emitted().size(), 11u);
    // Sorted by text, so "brown" sits next to "jones" and far from "smythe".
    const cpplink::BoundSource& source = plan_.at(0);
    EXPECT_EQ(source.rank[6], cpplink::kNoRank);  // the null row is not ordered
    EXPECT_EQ(source.order.size(), 7u);
}

TEST_F(BlockingFixture, MinHashAlwaysGroupsIdenticalValues) {
    Build(R"([{"type":"minhash","column":"surname","bands":8,"rows_per_band":3,
              "ngram":2}])");
    ASSERT_EQ(plan_.Size(), 8u);  // one source per band
    for (size_t band = 0; band < plan_.Size(); ++band) {
        EXPECT_TRUE(plan_.Produces(band, 0, 1)) << "band " << band;
        EXPECT_FALSE(plan_.Produces(band, 6, 0)) << "band " << band;  // null row
    }
}

// The union is deduplicated by asking whether an earlier source already produced
// the pair, which replaces a global DISTINCT over a materialized pair table.
TEST_F(BlockingFixture, UnionEmitsEachPairExactlyOnce) {
    Build(R"([{"type":"exact_value","column":"surname"},
              {"type":"exact_value","column":"dob"},
              {"type":"sorted_neighbourhood","column":"surname","window":3}])");
    const auto pairs = Emitted();
    const std::set<std::pair<uint32_t, uint32_t>> unique(pairs.begin(), pairs.end());
    EXPECT_EQ(pairs.size(), unique.size()) << "a pair was emitted more than once";
    EXPECT_EQ(plan_.CountUnion(), pairs.size());

    // Every emitted pair is produced by some source, and every pair produced by
    // some source is emitted: the deduplication loses nothing.
    std::set<std::pair<uint32_t, uint32_t>> reachable;
    for (uint32_t a = 0; a < 8; ++a) {
        for (uint32_t b = a + 1; b < 8; ++b) {
            if (plan_.ProducedByAny(a, b)) reachable.emplace(a, b);
        }
    }
    EXPECT_EQ(unique, reachable);
}

TEST_F(BlockingFixture, SumOverSourcesBoundsTheUnionFromAbove) {
    Build(R"([{"type":"exact_value","column":"surname"},
              {"type":"exact_value","column":"dob"},
              {"type":"sorted_neighbourhood","column":"surname","window":3}])");
    uint64_t sum = 0;
    for (size_t s = 0; s < plan_.Size(); ++s) sum += plan_.CountPairs(s);
    EXPECT_GE(sum, plan_.CountUnion());
}

TEST_F(BlockingFixture, EveryPerColumnSourceIsSafeForEstimatingM) {
    Build(R"([{"type":"exact_value","column":"surname"},
              {"type":"minhash","column":"surname","bands":4,"rows_per_band":2,
               "ngram":2},
              {"type":"sorted_neighbourhood","column":"dob","window":2}])");
    for (size_t s = 0; s < plan_.Size(); ++s) {
        EXPECT_TRUE(plan_.at(s).em_safe) << plan_.at(s).name;
    }
}

// The one source that names no column, so naming one is a mistake rather than a
// value it quietly ignores.
TEST(BlockingConfigTest, RejectsAColumnOnAllPairs) {
    cpplink::Schema schema;
    std::string error;
    const std::string json = "{" + std::string(kColumns) +
                             R"(,"blocking":[{"type":"all_pairs","column":"surname"}]})";
    EXPECT_FALSE(cpplink::ParseSchema(json, &schema, &error));
    EXPECT_NE(error.find("all_pairs"), std::string::npos) << error;
}

TEST(BlockingConfigTest, RejectsBlockingOnADoubleColumn) {
    cpplink::Schema schema;
    std::string error;
    const std::string json = "{" + std::string(kColumns) +
                             R"(,"blocking":[{"type":"exact_value","column":"lat"}]})";
    EXPECT_FALSE(cpplink::ParseSchema(json, &schema, &error));
    EXPECT_NE(error.find("discrete agreement"), std::string::npos);
}

TEST(BlockingConfigTest, RejectsAnUndeclaredColumnAndUnknownType) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_FALSE(cpplink::ParseSchema(
        "{" + std::string(kColumns) +
            R"(,"blocking":[{"type":"exact_value","column":"nope"}]})",
        &schema, &error));
    EXPECT_NE(error.find("not declared"), std::string::npos);

    EXPECT_FALSE(cpplink::ParseSchema(
        "{" + std::string(kColumns) +
            R"(,"blocking":[{"type":"telepathy","column":"surname"}]})",
        &schema, &error));
    EXPECT_NE(error.find("unknown blocking type"), std::string::npos);
}

}  // namespace

// The sweep in bench/sweep_blocking.py ranks methods by these numbers, so what
// has to hold is that they describe the stream the pipeline would actually
// evaluate -- not merely that they are self-consistent.
TEST_F(BlockingFixture, RecallMetricsDescribeTheEmittedStream) {
    Build(R"([{"type":"exact_value","column":"surname"},
              {"type":"exact_value","column":"dob"},
              {"type":"sorted_neighbourhood","column":"dob","window":2}])");

    // Two pairs a source reaches, one it does not: row 6 is null on both columns
    // and no source can produce it.
    cpplink::TruthPairs truth;
    truth.rows = {{0, 1}, {3, 4}, {5, 6}};

    const cpplink::RecallMetrics metrics =
        cpplink::MeasureRecall(plan_, *store_, truth, /*count_union=*/true);

    EXPECT_EQ(metrics.truth_pairs, 3u);
    EXPECT_EQ(metrics.union_found, 2u);
    EXPECT_EQ(metrics.pair_space, store_->PairSpace(plan_.mode()));

    // The deduplicated union is the metric the frontier is plotted against, so
    // it has to equal what enumeration emits. Any drift here silently reprices
    // every method in the sweep.
    EXPECT_EQ(metrics.candidate_union, Emitted().size());
    EXPECT_LE(metrics.candidate_union, metrics.candidate_sum);

    // Marginal credit partitions the reached pairs: a pair is credited to the
    // one source that would emit it, so the column sums to the union rather
    // than over-counting the pairs two sources both reach.
    uint64_t first_to = 0;
    uint64_t candidate_sum = 0;
    ASSERT_EQ(metrics.sources.size(), plan_.Size());
    for (size_t s = 0; s < metrics.sources.size(); ++s) {
        EXPECT_LE(metrics.sources[s].first_to, metrics.sources[s].found);
        EXPECT_EQ(metrics.sources[s].candidate_pairs, plan_.CountPairs(s));
        first_to += metrics.sources[s].first_to;
        candidate_sum += metrics.sources[s].candidate_pairs;
    }
    EXPECT_EQ(first_to, metrics.union_found);
    EXPECT_EQ(candidate_sum, metrics.candidate_sum);
}
