// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/cluster.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "cpplink/predict.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

constexpr uint64_t kRecords = 12;

struct RawEdge {
    uint32_t a;
    uint32_t b;
    double weight;
};

class ClusterFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        // One directory per process, because the test binary is run once per case:
        // a fixed name has concurrent cases of this fixture writing the same files
        // and deleting the directory under one another.
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_cluster_" + std::to_string(::getpid()));
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);

        cpplink::Schema schema;
        schema.unique_id = "id";
        store_ = std::make_unique<cpplink::RecordStore>(schema);
        for (uint64_t row = 0; row < kRecords; ++row) {
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();
    }

    void TearDown() override { std::filesystem::remove_all(dir_); }

    // Writes one shard in the exact form predict emits.
    std::string WriteShard(const std::string& name,
                           const std::vector<RawEdge>& edges) const {
        const std::string path = (dir_ / name).string();
        std::ofstream file(path, std::ios::binary);
        file.write(cpplink::kEdgeMagic, sizeof(cpplink::kEdgeMagic));
        for (const RawEdge& edge : edges) {
            char record[cpplink::kEdgeBytes];
            const uint32_t gamma = 7;
            std::memcpy(record, &edge.a, 4);
            std::memcpy(record + 4, &edge.b, 4);
            std::memcpy(record + 8, &gamma, 4);
            std::memcpy(record + 12, &edge.weight, 8);
            file.write(record, sizeof(record));
        }
        return path;
    }

    cpplink::ClusterOptions Options() const {
        cpplink::ClusterOptions options;
        options.edge_dir = dir_.string();
        return options;
    }

    std::filesystem::path dir_;
    std::unique_ptr<cpplink::RecordStore> store_;
};

TEST(UnionFindTest, UnionIsIdempotentAndTracksComponents) {
    cpplink::UnionFind uf(5);
    EXPECT_EQ(uf.Components(), 5u);
    EXPECT_TRUE(uf.Union(0, 1));
    EXPECT_FALSE(uf.Union(1, 0)) << "the second time the edge merges nothing";
    EXPECT_EQ(uf.Components(), 4u);
    EXPECT_EQ(uf.Find(0), uf.Find(1));
    EXPECT_NE(uf.Find(0), uf.Find(2));
}

TEST(UnionFindTest, ChainsCollapseToOneRoot) {
    cpplink::UnionFind uf(64);
    for (uint32_t i = 1; i < 64; ++i) EXPECT_TRUE(uf.Union(i - 1, i));
    EXPECT_EQ(uf.Components(), 1u);
    const uint32_t root = uf.Find(0);
    for (uint32_t i = 0; i < 64; ++i) EXPECT_EQ(uf.Find(i), root);
}

TEST_F(ClusterFixture, TransitiveClosureJoinsAChain) {
    // 0-1 and 1-2 were scored; 0-2 never was, and clustering asserts it anyway.
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 30.0}, {5, 6, 30.0}});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error))
        << error;

    EXPECT_TRUE(assignment.SameCluster(0, 2));
    EXPECT_FALSE(assignment.SameCluster(0, 5));
    EXPECT_EQ(assignment.clusters, 2u);
    EXPECT_EQ(assignment.largest, 3u);
    EXPECT_EQ(assignment.clustered, 5u);
    EXPECT_EQ(assignment.singletons, kRecords - 5);
    // C(3,2) + C(2,2) = 3 + 1
    EXPECT_EQ(assignment.implied_pairs, 4u);
    EXPECT_EQ(report.edges_read, 3u);
    EXPECT_EQ(report.merges, 3u);
}

TEST_F(ClusterFixture, ShardsAreReadAsOneStream) {
    // The two halves of a cluster arrive in different shards, which is what
    // happens whenever a component straddles two threads' output.
    WriteShard("shard-000.bin", {{0, 1, 30.0}});
    WriteShard("shard-001.bin", {{1, 2, 30.0}});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error))
        << error;
    EXPECT_EQ(report.shards.size(), 2u);
    EXPECT_EQ(assignment.clusters, 1u);
    EXPECT_TRUE(assignment.SameCluster(0, 2));
}

TEST_F(ClusterFixture, RedundantEdgesMergeNothing) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 30.0}, {0, 2, 30.0}});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error))
        << error;
    EXPECT_EQ(report.edges_used, 3u);
    EXPECT_EQ(report.merges, 2u) << "three edges over three records, one is a cycle";
}

TEST_F(ClusterFixture, ThresholdSplitsTheWeakLink) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 5.0}});
    std::string error;

    cpplink::ClusterAssignment loose;
    cpplink::ClusterReport loose_report;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &loose, &loose_report, &error));
    EXPECT_TRUE(loose.SameCluster(0, 2));

    cpplink::ClusterOptions strict = Options();
    strict.threshold = 20.0;
    cpplink::ClusterAssignment tight;
    cpplink::ClusterReport tight_report;
    ASSERT_TRUE(cpplink::Cluster(kRecords, strict, &tight, &tight_report, &error));
    EXPECT_TRUE(tight.SameCluster(0, 1));
    EXPECT_FALSE(tight.SameCluster(0, 2))
        << "re-clustering at a higher threshold needs no re-scoring";
    EXPECT_EQ(tight_report.edges_read, 2u);
    EXPECT_EQ(tight_report.edges_used, 1u);
    EXPECT_DOUBLE_EQ(tight_report.min_weight, 30.0);
    EXPECT_DOUBLE_EQ(tight_report.max_weight, 30.0);
}

TEST_F(ClusterFixture, RaisingTheThresholdOnlyRefines) {
    WriteShard("shard-000.bin",
               {{0, 1, 40.0}, {1, 2, 25.0}, {3, 4, 12.0}, {5, 6, 8.0}, {7, 8, 30.0}});
    std::string error;
    cpplink::ClusterAssignment previous;
    cpplink::ClusterReport report;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &previous, &report, &error));
    for (const double threshold : {10.0, 20.0, 28.0, 35.0, 50.0}) {
        cpplink::ClusterOptions options = Options();
        options.threshold = threshold;
        cpplink::ClusterAssignment current;
        ASSERT_TRUE(cpplink::Cluster(kRecords, options, &current, &report, &error));
        for (uint32_t a = 0; a < kRecords; ++a) {
            for (uint32_t b = a + 1; b < kRecords; ++b) {
                if (current.SameCluster(a, b)) {
                    EXPECT_TRUE(previous.SameCluster(a, b))
                        << "raising the threshold split nothing apart that should "
                           "have stayed together, at "
                        << threshold;
                }
            }
        }
        previous = current;
    }
}

TEST_F(ClusterFixture, BucketsAccountForEveryRecord) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 30.0}, {5, 6, 30.0}});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error));
    uint64_t records = 0;
    uint64_t clusters = 0;
    for (const cpplink::SizeBucket& bucket : report.buckets) {
        records += bucket.records;
        clusters += bucket.clusters;
    }
    EXPECT_EQ(records, kRecords);
    EXPECT_EQ(clusters, report.clusters + report.singletons);
}

TEST_F(ClusterFixture, QualityIsMeasuredOverTheClosure) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 30.0}});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error));

    cpplink::TruthPairs truth;
    truth.rows = {{0, 1}, {0, 2}, {1, 2}, {7, 8}};
    const cpplink::ClusterQuality quality = cpplink::MeasureClusters(assignment, truth);
    EXPECT_EQ(quality.truth_pairs, 4u);
    EXPECT_EQ(quality.implied_pairs, 3u);
    EXPECT_EQ(quality.recovered, 3u) << "0-2 counts even though it was never scored";
    EXPECT_DOUBLE_EQ(quality.precision, 1.0);
    EXPECT_DOUBLE_EQ(quality.recall, 0.75);
}

// Being a duplicate is transitive, so a truth file that lists (a,b) and (b,c) is
// asserting (a,c) as well. Measuring a partition against the un-closed list counts
// real pairs as false positives and flatters recall.
TEST_F(ClusterFixture, TruthIsClosedBeforeItIsCompared) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error));

    cpplink::TruthPairs truth;
    truth.rows = {{0, 1}, {1, 2}};
    const cpplink::ClusterQuality quality = cpplink::MeasureClusters(assignment, truth);
    EXPECT_EQ(quality.listed_pairs, 2u);
    EXPECT_EQ(quality.truth_pairs, 3u) << "0-1, 1-2 and the implied 0-2";
    EXPECT_EQ(quality.truth_clusters, 1u);
    EXPECT_EQ(quality.largest_truth_cluster, 3u);
    EXPECT_EQ(quality.recovered, 1u) << "only 0-1 was clustered together";
    EXPECT_EQ(quality.implied_pairs, 1u);
    EXPECT_DOUBLE_EQ(quality.precision, 1.0);
    EXPECT_NEAR(quality.recall, 1.0 / 3.0, 1e-12);
}

TEST_F(ClusterFixture, ChainingCostsPrecision) {
    // Two true pairs plus one wrong edge joining them: every cross pair the merged
    // cluster asserts is false, and only the closure shows it.
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {2, 3, 30.0}, {1, 2, 30.0}});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error));

    cpplink::TruthPairs truth;
    truth.rows = {{0, 1}, {2, 3}};
    const cpplink::ClusterQuality quality = cpplink::MeasureClusters(assignment, truth);
    EXPECT_EQ(quality.implied_pairs, 6u) << "one cluster of four";
    EXPECT_EQ(quality.recovered, 2u);
    EXPECT_DOUBLE_EQ(quality.recall, 1.0);
    EXPECT_NEAR(quality.precision, 2.0 / 6.0, 1e-12);
}

TEST_F(ClusterFixture, WritesTheRepresentativeAndSize) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 30.0}, {5, 6, 30.0}});
    cpplink::ClusterOptions options = Options();
    options.out_path = (dir_ / "clusters.csv").string();
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, options, &assignment, &report, &error));
    uint64_t written = 0;
    ASSERT_TRUE(cpplink::WriteClusters(assignment, *store_, options, &written, &error))
        << error;
    EXPECT_EQ(written, 5u) << "only records in a cluster of two or more";

    std::ifstream file(options.out_path);
    std::string header;
    std::getline(file, header);
    EXPECT_EQ(header, "unique_id,cluster_id,cluster_size");
    std::vector<std::string> lines;
    for (std::string line; std::getline(file, line);) lines.push_back(line);
    ASSERT_EQ(lines.size(), 5u);
    // Rows 0, 1 and 2 share a representative and carry the size of their cluster.
    const std::string root = lines[0].substr(lines[0].find(',') + 1);
    EXPECT_EQ(root, lines[1].substr(lines[1].find(',') + 1));
    EXPECT_EQ(root, lines[2].substr(lines[2].find(',') + 1));
    EXPECT_NE(std::string::npos, lines[0].find(",3"));
}

TEST_F(ClusterFixture, MinSizeSelectsLargerClustersOnly) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 30.0}, {5, 6, 30.0}});
    cpplink::ClusterOptions options = Options();
    options.out_path = (dir_ / "clusters.csv").string();
    options.min_size = 3;
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, options, &assignment, &report, &error));
    uint64_t written = 0;
    ASSERT_TRUE(cpplink::WriteClusters(assignment, *store_, options, &written, &error));
    EXPECT_EQ(written, 3u) << "the pair is below min-size, the triple is not";
}

TEST_F(ClusterFixture, RefusesAForeignFile) {
    const std::string path = (dir_ / "shard-000.bin").string();
    std::ofstream file(path, std::ios::binary);
    file << "not a cpplink shard at all";
    file.close();
    cpplink::ClusterAssignment assignment;
    std::string error;
    EXPECT_FALSE(cpplink::Cluster(kRecords, Options(), &assignment, nullptr, &error));
    EXPECT_NE(error.find("not a cpplink edge shard"), std::string::npos) << error;
}

TEST_F(ClusterFixture, RefusesAFileTruncatedMidEdge) {
    const std::string path = WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 30.0}});
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 6);
    cpplink::ClusterAssignment assignment;
    std::string error;
    EXPECT_FALSE(cpplink::Cluster(kRecords, Options(), &assignment, nullptr, &error));
    EXPECT_NE(error.find("truncated"), std::string::npos) << error;
}

TEST_F(ClusterFixture, RefusesAnEdgeOutsideTheData) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, kRecords + 3, 30.0}});
    cpplink::ClusterAssignment assignment;
    std::string error;
    EXPECT_FALSE(cpplink::Cluster(kRecords, Options(), &assignment, nullptr, &error));
    EXPECT_NE(error.find("but the data has"), std::string::npos) << error;
}

TEST_F(ClusterFixture, RefusesADirectoryWithNoShards) {
    cpplink::ClusterAssignment assignment;
    std::string error;
    EXPECT_FALSE(cpplink::Cluster(kRecords, Options(), &assignment, nullptr, &error));
    EXPECT_NE(error.find("no shard-*.bin"), std::string::npos) << error;

    cpplink::ClusterOptions missing = Options();
    missing.edge_dir = (dir_ / "nowhere").string();
    EXPECT_FALSE(cpplink::Cluster(kRecords, missing, &assignment, nullptr, &error));
    EXPECT_NE(error.find("not a directory"), std::string::npos) << error;
}

TEST_F(ClusterFixture, NoEdgesLeavesEveryRecordAlone) {
    WriteShard("shard-000.bin", {});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error));
    EXPECT_EQ(assignment.clusters, 0u);
    EXPECT_EQ(assignment.singletons, kRecords);
    EXPECT_EQ(assignment.implied_pairs, 0u);
}

TEST_F(ClusterFixture, ReportNamesWhatItDid) {
    WriteShard("shard-000.bin", {{0, 1, 30.0}, {1, 2, 30.0}, {5, 6, 12.0}});
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(kRecords, Options(), &assignment, &report, &error));
    std::ostringstream out;
    cpplink::PrintClusterReport(report, out);
    const std::string text = out.str();
    EXPECT_NE(text.find("clusters of two or more"), std::string::npos);
    EXPECT_NE(text.find("Largest cluster 3"), std::string::npos);
    EXPECT_NE(text.find("1 (singleton)"), std::string::npos);
}

}  // namespace
