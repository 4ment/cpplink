// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/model.hpp"
#include "cpplink/pair_stream.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

using Pair = std::pair<uint32_t, uint32_t>;
using PairSet = std::set<Pair>;

constexpr uint64_t kRows = 12;
constexpr uint64_t kSplit = 7;  // rows 0..6 came from the first input, 7..11 the second

// Twelve rows across two inputs, arranged so that every shape a group can take is
// present: one straddling the split, one wholly inside the first input, one wholly
// inside the second, a singleton, and a null.
//
//   row       0     1     2      3      4      5     6   | 7     8     9      10     11
//   surname   smith smith jones  green  green  brown -   | smith smith jones  white white
//   dob       100   100   200    200    300    300   -   | 100   200   300    400    400
constexpr const char* kColumns = R"("columns":[
    {"name":"surname","type":"string"},
    {"name":"dob","type":"date"},
    {"name":"city","type":"string"}])";

class LinkFixture : public ::testing::Test {
   protected:
    void SetUp() override { MakeStore(); }

    void MakeStore() {
        const std::string json = "{" + std::string(kColumns) +
                                 R"(,"blocking":[{"type":"exact_value",)"
                                 R"("column":"surname"}]})";
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(json, &schema_, &error)) << error;
        Fill();
    }

    // Rebuilds the store under a different schema, keeping the same rows.
    void Reschema(const std::string& json) {
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(json, &schema_, &error)) << error;
        Fill();
    }

    void Fill() {
        store_ = std::make_unique<cpplink::RecordStore>(schema_);
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        const uint32_t smith = surname.dict.Intern("smith");
        const uint32_t jones = surname.dict.Intern("jones");
        const uint32_t green = surname.dict.Intern("green");
        const uint32_t brown = surname.dict.Intern("brown");
        const uint32_t white = surname.dict.Intern("white");
        surname.ids = {smith, smith, jones, green, green, brown, cpplink::kNullId,
                       smith, smith, jones, white, white};

        auto& dob = std::get<cpplink::DateColumn>(store_->mutable_column(1));
        dob.values = {100, 100, 200, 200, 300, 300, cpplink::kNullDate,
                      100, 200, 300, 400, 400};

        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(2));
        const uint32_t c0 = city.dict.Intern("c0");
        const uint32_t c1 = city.dict.Intern("c1");
        city.ids = {c0, c1, c0, c1, c0, c1, c0, c0, c1, c0, c1, c0};

        store_->set_num_records(kRows);
        store_->set_datasets({0, kSplit, kRows});
        store_->Finalize();
    }

    // Builds a plan over the current store in one mode.
    void Plan(const std::string& blocking, cpplink::PairMode mode,
              cpplink::BlockingPlan* plan) {
        const std::string json =
            "{" + std::string(kColumns) + ",\"blocking\":" + blocking + "}";
        cpplink::Schema schema;
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(json, &schema, &error)) << error;
        ASSERT_TRUE(plan->Build(schema, *store_, mode, &error)) << error;
    }

    static PairSet Emitted(const cpplink::BlockingPlan& plan) {
        PairSet pairs;
        plan.ForEachPair([&pairs](uint32_t a, uint32_t b) {
            pairs.emplace(std::min(a, b), std::max(a, b));
        });
        return pairs;
    }

    static bool Crosses(uint32_t a, uint32_t b) { return (a < kSplit) != (b < kSplit); }

    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
};

const std::vector<std::string>& Configurations() {
    static const std::vector<std::string> configs = {
        R"([{"type":"exact_value","column":"surname"}])",
        R"([{"type":"exact_value","column":"dob"}])",
        R"([{"type":"rare_value","column":"surname","max_frequency":2}])",
        R"([{"type":"sorted_neighbourhood","column":"surname","window":3}])",
        R"([{"type":"minhash","column":"surname","bands":6,"rows_per_band":2,
             "ngram":2}])",
        R"([{"type":"exact_value","column":"surname"},
             {"type":"exact_value","column":"dob"},
             {"type":"sorted_neighbourhood","column":"surname","window":4}])",
        R"([{"type":"all_pairs"}])",
    };
    return configs;
}

// The whole claim of the link path in one assertion: the cross-dataset stream is
// the dedup stream with the within-input pairs removed, and nothing else. It is
// asserted rather than argued because the enumeration does not filter -- it jumps
// to a contiguous partner range -- so a bug there loses pairs silently.
TEST_F(LinkFixture, CrossDatasetIsExactlyTheDedupStreamWithoutTheWithinInputPairs) {
    for (const std::string& config : Configurations()) {
        cpplink::BlockingPlan all;
        cpplink::BlockingPlan cross;
        Plan(config, cpplink::PairMode::kAll, &all);
        Plan(config, cpplink::PairMode::kCrossDataset, &cross);

        PairSet expected;
        for (const Pair& pair : Emitted(all)) {
            if (Crosses(pair.first, pair.second)) expected.insert(pair);
        }
        EXPECT_EQ(Emitted(cross), expected) << config;
        EXPECT_GT(expected.size(), 0u) << config;
    }
}

// CountPairs is exact in dedup mode from the term frequencies alone. Those pool
// the inputs, so link mode counts a different way -- and the equivalence with
// enumeration has to hold there too or the cost report is lying.
TEST_F(LinkFixture, CountMatchesEnumerationForEverySourceInLinkMode) {
    for (const std::string& config : Configurations()) {
        cpplink::BlockingPlan plan;
        Plan(config, cpplink::PairMode::kCrossDataset, &plan);
        for (size_t s = 0; s < plan.Size(); ++s) {
            uint64_t enumerated = 0;
            for (uint32_t a = 0; a < kRows; ++a) {
                for (uint32_t b = a + 1; b < kRows; ++b) {
                    if (plan.Produces(s, a, b)) ++enumerated;
                }
            }
            EXPECT_EQ(plan.CountPairs(s), enumerated)
                << config << " source " << plan.at(s).name;
        }
    }
}

// A task is rows [begin, end) of one group, so the partner cursor has to be
// correct for a range that does not start at the group's first row. Every split
// point is checked, because only the first row of a task exercises that setup.
TEST_F(LinkFixture, AGroupSplitIntoRowRangesEmitsTheSamePairs) {
    cpplink::BlockingPlan plan;
    Plan(R"([{"type":"exact_value","column":"surname"}])",
         cpplink::PairMode::kCrossDataset, &plan);
    const std::vector<size_t> selected = plan.AllSources();
    cpplink::SourceGroups groups;
    plan.BuildGroups(0, &groups);
    ASSERT_GT(groups.GroupCount(), 0u);

    for (uint64_t g = 0; g < groups.GroupCount(); ++g) {
        const uint64_t size = groups.Size(g);
        PairSet whole;
        plan.EnumerateGroupRange(selected, 0, groups, g, 0, size,
                                 [&whole](uint32_t a, uint32_t b) {
                                     whole.emplace(std::min(a, b), std::max(a, b));
                                 });
        for (uint64_t cut = 0; cut <= size; ++cut) {
            PairSet split;
            const auto sink = [&split](uint32_t a, uint32_t b) {
                split.emplace(std::min(a, b), std::max(a, b));
            };
            plan.EnumerateGroupRange(selected, 0, groups, g, 0, cut, sink);
            plan.EnumerateGroupRange(selected, 0, groups, g, cut, size, sink);
            EXPECT_EQ(split, whole) << "group " << g << " cut at " << cut;
        }
    }
}

// Linking a small input without blocking is what the unblocked source is for, and
// the whole store is one group there -- so the partner cursor is walked over every
// row of the store rather than over a handful, and the count has to be the cross
// product exactly.
TEST_F(LinkFixture, AllPairsInLinkModeIsTheCrossProduct) {
    cpplink::BlockingPlan plan;
    Plan(R"([{"type":"all_pairs"}])", cpplink::PairMode::kCrossDataset, &plan);
    const uint64_t expected = kSplit * (kRows - kSplit);
    EXPECT_EQ(plan.CountPairs(0), expected);
    const PairSet pairs = Emitted(plan);
    EXPECT_EQ(pairs.size(), expected);
    for (const Pair& pair : pairs) {
        EXPECT_TRUE(Crosses(pair.first, pair.second)) << pair.first << "," << pair.second;
    }

    cpplink::BlockingPlan dedup;
    Plan(R"([{"type":"all_pairs"}])", cpplink::PairMode::kAll, &dedup);
    EXPECT_EQ(dedup.CountPairs(0), kRows * (kRows - 1) / 2);
}

TEST_F(LinkFixture, ThreadedWalkEmitsTheSamePairsAsTheSerialOne) {
    cpplink::BlockingPlan plan;
    Plan(R"([{"type":"exact_value","column":"surname"},
             {"type":"exact_value","column":"dob"},
             {"type":"sorted_neighbourhood","column":"surname","window":4}])",
         cpplink::PairMode::kCrossDataset, &plan);

    std::vector<PairSet> sinks(4);
    cpplink::ForEachPairParallel(plan, plan.AllSources(), 4, [&sinks](unsigned t) {
        return [&sinks, t](uint32_t a, uint32_t b) {
            sinks[t].emplace(std::min(a, b), std::max(a, b));
        };
    });
    PairSet threaded;
    for (const PairSet& sink : sinks) threaded.insert(sink.begin(), sink.end());
    EXPECT_EQ(threaded, Emitted(plan));
}

// Every stage above blocking asks the plan whether a pair is a candidate rather
// than watching the stream: the recall harness, the earlier-source predicate and
// the miss diagnostic all go through Produces. It has to agree with what is
// emitted, or the harness measures a stream that never ran.
TEST_F(LinkFixture, NoSourceProducesAPairInsideOneInput) {
    cpplink::BlockingPlan plan;
    Plan(R"([{"type":"exact_value","column":"surname"},
             {"type":"sorted_neighbourhood","column":"surname","window":4}])",
         cpplink::PairMode::kCrossDataset, &plan);
    for (uint32_t a = 0; a < kRows; ++a) {
        for (uint32_t b = a + 1; b < kRows; ++b) {
            if (Crosses(a, b)) continue;
            EXPECT_FALSE(plan.ProducedByAny(a, b)) << a << "," << b;
            for (size_t s = 0; s < plan.Size(); ++s) {
                EXPECT_FALSE(plan.Produces(s, a, b)) << a << "," << b;
            }
        }
    }
    // And the pairs it does produce are all in the stream.
    for (const Pair& pair : Emitted(plan)) {
        EXPECT_TRUE(plan.ProducedByAny(pair.first, pair.second));
    }
}

TEST_F(LinkFixture, LinkModeNeedsMoreThanOneInput) {
    cpplink::RecordStore single(schema_);
    auto& surname = std::get<cpplink::StringColumn>(single.mutable_column(0));
    surname.ids = {surname.dict.Intern("smith"), surname.dict.Intern("smith")};
    auto& dob = std::get<cpplink::DateColumn>(single.mutable_column(1));
    dob.values = {100, 100};
    auto& city = std::get<cpplink::StringColumn>(single.mutable_column(2));
    city.ids = {city.dict.Intern("c0"), city.dict.Intern("c0")};
    single.set_num_records(2);
    single.Finalize();

    cpplink::BlockingPlan plan;
    std::string error;
    EXPECT_FALSE(plan.Build(schema_, single, cpplink::PairMode::kCrossDataset, &error));
    EXPECT_NE(error.find("two inputs"), std::string::npos) << error;
    EXPECT_TRUE(plan.Build(schema_, single, cpplink::PairMode::kAll, &error)) << error;
}

TEST_F(LinkFixture, PairSpaceIsTheCrossProductNotTheTriangle) {
    EXPECT_EQ(store_->NumDatasets(), 2u);
    EXPECT_EQ(store_->DatasetOf(0), 0u);
    EXPECT_EQ(store_->DatasetOf(kSplit), 1u);
    EXPECT_EQ(store_->DatasetEndFor(3), kSplit);
    EXPECT_EQ(store_->DatasetEndFor(9), kRows);
    EXPECT_DOUBLE_EQ(store_->PairSpace(cpplink::PairMode::kAll), 12.0 * 11.0 / 2.0);
    EXPECT_DOUBLE_EQ(store_->PairSpace(cpplink::PairMode::kCrossDataset), 7.0 * 5.0);
}

// u is what a random admissible pair does, and in link mode the admissible pairs
// are the cross-product. There are exactly N_a x N_b ordered cross draws and none
// of them is a row against itself, so the closed form is not an approximation
// here: it must equal the enumeration exactly.
TEST_F(LinkFixture, ClosedFormUIsTheCrossPairRateExactly) {
    Reschema(R"({"columns":[
        {"name":"surname","type":"string"},
        {"name":"dob","type":"date"},
        {"name":"city","type":"string"}],
      "comparisons":[
        {"name":"surname","columns":["surname"],
         "levels":[{"type":"null"},{"type":"exact"},{"type":"else"}]},
        {"name":"city","columns":["city"],
         "levels":[{"type":"exact"},{"type":"else"}]}],
      "blocking":[{"type":"exact_value","column":"dob"}]})");

    cpplink::BlockingPlan plan;
    std::string error;
    ASSERT_TRUE(plan.Build(schema_, *store_, cpplink::PairMode::kCrossDataset, &error))
        << error;
    cpplink::ComparisonSet comparisons;
    ASSERT_TRUE(comparisons.Bind(schema_, *store_, &error)) << error;

    cpplink::EstimateOptions options;
    options.u_sample = 20000;
    options.threads = 2;
    options.seed = 11;
    cpplink::Model model;
    cpplink::EstimateReport report;
    ASSERT_TRUE(
        cpplink::Estimate(*store_, comparisons, plan, options, &model, &report, &error))
        << error;

    // Every cross pair, by hand, at every level of every comparison.
    std::vector<std::vector<uint64_t>> counts(comparisons.Size());
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        counts[c].assign(comparisons.at(c).spec->levels.size(), 0);
    }
    uint64_t pairs = 0;
    for (uint32_t a = 0; a < kSplit; ++a) {
        for (uint32_t b = kSplit; b < kRows; ++b) {
            const uint32_t gamma = comparisons.Evaluate(a, b);
            for (size_t c = 0; c < comparisons.Size(); ++c) {
                ++counts[c][comparisons.LevelOf(gamma, c)];
            }
            ++pairs;
        }
    }
    ASSERT_EQ(pairs, kSplit * (kRows - kSplit));

    // Both comparisons are made only of levels u is closed form for, so every
    // level is exact and the whole distribution can be checked, not just one.
    for (size_t c = 0; c < model.comparisons.size(); ++c) {
        for (size_t l = 0; l < model.comparisons[c].levels.size(); ++l) {
            const double expected =
                static_cast<double>(counts[c][l]) / static_cast<double>(pairs);
            EXPECT_NEAR(model.comparisons[c].levels[l].u, expected, 1e-12)
                << model.comparisons[c].name << " level " << l;
        }
    }
}

// The dedup path must be untouched by all of this: a store with one input carries
// no boundaries at all, and kAll over two inputs is link-and-dedup, which is the
// same enumeration a single concatenated file would have given.
TEST_F(LinkFixture, DedupOverTwoInputsIsTheWholeTriangle) {
    cpplink::BlockingPlan plan;
    Plan(R"([{"type":"exact_value","column":"surname"}])", cpplink::PairMode::kAll,
         &plan);
    PairSet expected;
    for (uint32_t a = 0; a < kRows; ++a) {
        for (uint32_t b = a + 1; b < kRows; ++b) {
            if (plan.Produces(0, a, b)) expected.emplace(a, b);
        }
    }
    const PairSet emitted = Emitted(plan);
    EXPECT_EQ(emitted, expected);
    // smith is {0,1} and {7,8}: kAll keeps the four within-input pairs that link
    // mode drops.
    EXPECT_EQ(emitted.count({0, 1}), 1u);
    EXPECT_EQ(emitted.count({10, 11}), 1u);
}

}  // namespace
