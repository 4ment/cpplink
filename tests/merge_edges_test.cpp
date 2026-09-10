// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/merge_edges.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>
#include <unistd.h>

#include "cpplink/predict.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

constexpr uint64_t kRecords = 12;

struct RawEdge {
    uint32_t a;
    uint32_t b;
    uint32_t gamma;
    double weight;
};

const std::vector<RawEdge> kFirst = {{0, 1, 7, 30.5}, {1, 2, 9, 12.25}, {3, 4, 7, -4.5}};
const std::vector<RawEdge> kSecond = {{5, 6, 11, 44.0}, {7, 11, 3, 8.125}};

class MergeFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        // One directory per process, because the test binary is run once per case:
        // a fixed name has concurrent cases of this fixture writing the same files
        // and deleting the directory under one another.
        root_ = std::filesystem::temp_directory_path() /
                ("cpplink_merge_" + std::to_string(::getpid()));
        std::filesystem::remove_all(root_);
        std::filesystem::create_directories(root_);

        cpplink::Schema schema;
        schema.unique_id = "id";
        store_ = std::make_unique<cpplink::RecordStore>(schema);
        for (uint64_t row = 0; row < kRecords; ++row) {
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();
    }

    void TearDown() override { std::filesystem::remove_all(root_); }

    std::filesystem::path MakeDir(const std::string& name) const {
        const std::filesystem::path dir = root_ / name;
        std::filesystem::create_directories(dir);
        return dir;
    }

    // Writes one shard through the writer predict itself uses, so the test cannot
    // drift from the format the pipeline emits.
    void WriteShard(const std::filesystem::path& dir, const std::string& name,
                    cpplink::EdgeFormat format, const std::vector<RawEdge>& edges) const {
        cpplink::EdgeShardWriter writer;
        ASSERT_TRUE(writer.Open((dir / name).string(), format));
        for (const RawEdge& edge : edges) {
            if (format == cpplink::EdgeFormat::kBinary) {
                writer.WriteBinary(edge.a, edge.b, edge.gamma, edge.weight);
            } else {
                writer.WriteCsv(store_->ids().Get(edge.a), store_->ids().Get(edge.b),
                                edge.gamma, edge.weight);
            }
        }
        ASSERT_TRUE(writer.Close());
    }

    std::filesystem::path BothShardSets(const std::string& name,
                                        cpplink::EdgeFormat format) const {
        const std::filesystem::path dir = MakeDir(name);
        WriteShard(dir, "shard-000" + Suffix(format), format, kFirst);
        WriteShard(dir, "shard-001" + Suffix(format), format, kSecond);
        return dir;
    }

    static std::string Suffix(cpplink::EdgeFormat format) {
        return format == cpplink::EdgeFormat::kBinary ? ".bin" : ".csv";
    }

    static std::vector<std::string> ReadLines(const std::string& path) {
        std::ifstream file(path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) lines.push_back(line);
        return lines;
    }

    cpplink::MergeOptions Options(const std::filesystem::path& dir,
                                  const std::string& out) const {
        cpplink::MergeOptions options;
        options.edge_dir = dir.string();
        options.out_path = (root_ / out).string();
        return options;
    }

    std::filesystem::path root_;
    std::unique_ptr<cpplink::RecordStore> store_;
};

TEST_F(MergeFixture, BinaryShardsBecomeOneCsvOfIds) {
    const auto dir = BothShardSets("bin", cpplink::EdgeFormat::kBinary);
    cpplink::MergeReport report;
    std::string error;
    const auto options = Options(dir, "merged.csv");
    ASSERT_TRUE(cpplink::MergeEdges(store_.get(), options, &report, &error)) << error;

    EXPECT_EQ(report.shards.size(), 2u);
    EXPECT_TRUE(report.ignored.empty());
    EXPECT_FALSE(report.row_indices);
    EXPECT_EQ(report.read, kFirst.size() + kSecond.size());
    EXPECT_EQ(report.written, report.read);

    const std::vector<std::string> lines = ReadLines(options.out_path);
    ASSERT_EQ(lines.size(), 1 + kFirst.size() + kSecond.size());
    EXPECT_EQ(lines[0], "id_a,id_b,gamma,match_weight,match_probability");
    // Shards are read in name order, so the merged file is the run concatenated.
    EXPECT_EQ(lines[1].rfind("r0,r1,7,30.500000,", 0), 0u);
    EXPECT_EQ(lines[4].rfind("r5,r6,11,44.000000,", 0), 0u);
}

TEST_F(MergeFixture, CsvShardsMergeToTheSameFileAsBinaryOnes) {
    const auto bin = BothShardSets("bin", cpplink::EdgeFormat::kBinary);
    const auto csv = BothShardSets("csv", cpplink::EdgeFormat::kCsv);
    cpplink::MergeReport report;
    std::string error;
    const auto from_bin = Options(bin, "from_bin.csv");
    const auto from_csv = Options(csv, "from_csv.csv");
    ASSERT_TRUE(cpplink::MergeEdges(store_.get(), from_bin, &report, &error)) << error;
    // A csv shard already carries its ids, so no store is needed to read one.
    ASSERT_TRUE(cpplink::MergeEdges(nullptr, from_csv, &report, &error)) << error;
    EXPECT_EQ(report.source, cpplink::MergeSource::kCsv);
    EXPECT_FALSE(report.row_indices);

    EXPECT_EQ(ReadLines(from_bin.out_path), ReadLines(from_csv.out_path))
        << "the two shard formats hold the same edges";
}

TEST_F(MergeFixture, AutoPrefersBinaryAndLeavesTheCsvShardsAlone) {
    const auto dir = MakeDir("both");
    WriteShard(dir, "shard-000.bin", cpplink::EdgeFormat::kBinary, kFirst);
    WriteShard(dir, "shard-000.csv", cpplink::EdgeFormat::kCsv, kFirst);
    cpplink::MergeReport report;
    std::string error;
    const auto options = Options(dir, "merged.csv");
    ASSERT_TRUE(cpplink::MergeEdges(store_.get(), options, &report, &error)) << error;

    EXPECT_EQ(report.source, cpplink::MergeSource::kBinary);
    EXPECT_EQ(report.shards.size(), 1u);
    EXPECT_EQ(report.ignored.size(), 1u);
    // The two shards are one run written twice, not two runs.
    EXPECT_EQ(report.read, kFirst.size());

    cpplink::MergeOptions wanted = options;
    wanted.source = cpplink::MergeSource::kCsv;
    wanted.out_path = (root_ / "from_csv.csv").string();
    ASSERT_TRUE(cpplink::MergeEdges(store_.get(), wanted, &report, &error)) << error;
    EXPECT_EQ(report.source, cpplink::MergeSource::kCsv);
    EXPECT_EQ(report.read, kFirst.size());
}

TEST_F(MergeFixture, ThresholdDropsTheEdgesBelowIt) {
    const auto dir = BothShardSets("bin", cpplink::EdgeFormat::kBinary);
    cpplink::MergeOptions options = Options(dir, "merged.csv");
    options.threshold = 10.0;
    cpplink::MergeReport report;
    std::string error;
    ASSERT_TRUE(cpplink::MergeEdges(store_.get(), options, &report, &error)) << error;

    EXPECT_EQ(report.read, 5u);
    EXPECT_EQ(report.written, 3u) << "-4.5 and 8.125 are below 10 bits";
    EXPECT_EQ(ReadLines(options.out_path).size(), 1 + 3u);
}

TEST_F(MergeFixture, WithoutAStoreBinaryRowsKeepTheirIndices) {
    const auto dir = BothShardSets("bin", cpplink::EdgeFormat::kBinary);
    const auto options = Options(dir, "merged.csv");
    cpplink::MergeReport report;
    std::string error;
    ASSERT_TRUE(cpplink::MergeEdges(nullptr, options, &report, &error)) << error;

    EXPECT_TRUE(report.row_indices);
    const std::vector<std::string> lines = ReadLines(options.out_path);
    ASSERT_EQ(lines.size(), 1 + kFirst.size() + kSecond.size());
    EXPECT_EQ(lines[1].rfind("0,1,7,30.500000,", 0), 0u);
}

TEST_F(MergeFixture, ParquetHoldsTheSameEdgesAsTheCsv) {
    const auto dir = BothShardSets("bin", cpplink::EdgeFormat::kBinary);
    cpplink::MergeOptions options = Options(dir, "merged.parquet");
    options.format = cpplink::MergeFormat::kParquet;
    options.batch_rows = 2;  // more than one row group, from five edges
    cpplink::MergeReport report;
    std::string error;
    ASSERT_TRUE(cpplink::MergeEdges(store_.get(), options, &report, &error)) << error;
    EXPECT_EQ(report.written, kFirst.size() + kSecond.size());

    auto input = arrow::io::ReadableFile::Open(options.out_path);
    ASSERT_TRUE(input.ok()) << input.status().message();
    auto reader = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
    ASSERT_TRUE(reader.ok()) << reader.status().message();
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*reader)->ReadTable(&table).ok());
    ASSERT_EQ(table->num_rows(), 5);
    ASSERT_EQ(table->num_columns(), 5);
    EXPECT_EQ(table->schema()->field(0)->name(), "id_a");
    EXPECT_EQ(table->schema()->field(2)->type()->id(), arrow::Type::UINT32);

    const auto ids =
        std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
    EXPECT_EQ(ids->GetString(0), "r0");
    const auto weights =
        std::static_pointer_cast<arrow::DoubleArray>(table->column(3)->chunk(0));
    EXPECT_DOUBLE_EQ(weights->Value(0), 30.5) << "parquet keeps the full double";
}

TEST_F(MergeFixture, RefusesADirectoryHoldingNoShards) {
    const auto dir = MakeDir("empty");
    cpplink::MergeReport report;
    std::string error;
    EXPECT_FALSE(
        cpplink::MergeEdges(store_.get(), Options(dir, "merged.csv"), &report, &error));
    EXPECT_NE(error.find("no shard-"), std::string::npos) << error;

    cpplink::MergeOptions wanted =
        Options(BothShardSets("bin2", cpplink::EdgeFormat::kBinary), "merged.csv");
    wanted.source = cpplink::MergeSource::kCsv;
    EXPECT_FALSE(cpplink::MergeEdges(store_.get(), wanted, &report, &error));
    EXPECT_NE(error.find("shard-*.csv"), std::string::npos) << error;
}

TEST_F(MergeFixture, RefusesAShardThatIsNotOne) {
    const auto dir = MakeDir("bad");
    std::ofstream file((dir / "shard-000.bin").string(), std::ios::binary);
    file << "not a prediction shard at all";
    file.close();
    cpplink::MergeReport report;
    std::string error;
    EXPECT_FALSE(
        cpplink::MergeEdges(store_.get(), Options(dir, "merged.csv"), &report, &error));
    EXPECT_NE(error.find("not a cpplink prediction shard"), std::string::npos) << error;
}

}  // namespace
