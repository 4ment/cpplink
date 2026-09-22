// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// A record's id is unique within its input and not across inputs: two files that
// both number their rows from 1 are the common shape linking is for. Every path
// that turns an id back into a row -- the merged prediction file, the truth file,
// `explain --pair` -- has to know which input the id belongs to, and every file
// that names a record has to say. These tests build two inputs sharing every id
// and check that nothing guesses.

#include "cpplink/id_index.hpp"

#include <cstdint>
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
#include <parquet/arrow/writer.h>

#include "cpplink/app.hpp"
#include "cpplink/cluster.hpp"
#include "cpplink/merge_edges.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/recall.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "tests/process_id.hpp"
#include "tests/temp_dir.hpp"

namespace {

// Rows 0-2 are input `left` with ids 1, 2, 3; rows 3-5 are input `right` with
// ids 1, 2, 4. So 1 and 2 are shared, 3 and 4 are each held once.
std::unique_ptr<cpplink::RecordStore> SharedIdStore() {
    cpplink::Schema schema;
    schema.unique_id = "id";
    auto store = std::make_unique<cpplink::RecordStore>(schema);
    for (const char* id : {"1", "2", "3", "1", "2", "4"}) {
        store->mutable_ids().Append(id);
    }
    store->set_num_records(6);
    store->set_datasets({0, 3, 6});
    store->set_dataset_names({"left", "right"});
    store->Finalize();
    return store;
}

TEST(DatasetNames, AreTheFileStemsMadeDistinctAndCsvSafe) {
    EXPECT_TRUE(cpplink::DatasetNamesFor({"only.parquet"}).empty());
    const std::vector<std::string> names = cpplink::DatasetNamesFor(
        {"/data/a.parquet", "b/a.parquet", "x,y.parquet", "/other/b.parquet"});
    ASSERT_EQ(names.size(), 4u);
    EXPECT_EQ(names[0], "a");
    EXPECT_EQ(names[1], "a#1");
    EXPECT_EQ(names[2], "x_y");
    EXPECT_EQ(names[3], "b");
}

// The other character a name may not carry, checked where a file can carry it.
// Windows forbids a colon in a filename and its path parser reads one in a path
// as punctuation rather than as part of the stem, so there the rule guards
// against a name no file can have.
#ifndef _WIN32
TEST(DatasetNames, AColonInAStemIsReplaced) {
    const std::vector<std::string> names =
        cpplink::DatasetNamesFor({"x:y.parquet", "b.parquet"});
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "x_y") << "a name is written into `dataset:id`";
}
#endif

TEST(DatasetNames, ASingleInputHasNoNameAndNoQualifier) {
    cpplink::Schema schema;
    schema.unique_id = "id";
    cpplink::RecordStore store(schema);
    store.mutable_ids().Append("a:1");
    store.set_num_records(1);
    store.Finalize();
    EXPECT_EQ(store.DatasetName(0), "");
    size_t dataset = 0;
    std::string_view id;
    store.SplitQualifiedId("a:1", &dataset, &id);
    EXPECT_EQ(dataset, cpplink::kAnyDataset);
    EXPECT_EQ(id, "a:1");
    EXPECT_EQ(cpplink::QualifiedId(store, 0), "a:1");
    uint64_t row = 9;
    EXPECT_EQ(cpplink::FindRowById(store, "a:1", &row), cpplink::IdLookup::kFound);
    EXPECT_EQ(row, 0u);
}

TEST(DatasetNames, TheLongestNamedPrefixQualifies) {
    cpplink::Schema schema;
    schema.unique_id = "id";
    cpplink::RecordStore store(schema);
    for (const char* id : {"7", "7"}) store.mutable_ids().Append(id);
    store.set_num_records(2);
    store.set_datasets({0, 1, 2});
    store.set_dataset_names({"a", "a.b"});
    store.Finalize();
    size_t dataset = 0;
    std::string_view id;
    store.SplitQualifiedId("a.b:7", &dataset, &id);
    EXPECT_EQ(dataset, 1u);
    EXPECT_EQ(id, "7");
    store.SplitQualifiedId("a:x:7", &dataset, &id);
    EXPECT_EQ(dataset, 0u);
    EXPECT_EQ(id, "x:7");
    store.SplitQualifiedId("c:7", &dataset, &id);
    EXPECT_EQ(dataset, cpplink::kAnyDataset);
    EXPECT_EQ(id, "c:7");
    // A store built without names is named by position.
    store.set_dataset_names({});
    EXPECT_EQ(store.DatasetName(1), "1");
    EXPECT_EQ(cpplink::QualifiedId(store, 1), "1:7");
}

TEST(IdIndex, ASharedIdIsAmbiguousUntilItsDatasetIsNamed) {
    const auto store = SharedIdStore();
    const cpplink::IdIndex index(*store);
    std::string error;
    EXPECT_TRUE(index.Unique(&error)) << error;
    EXPECT_TRUE(index.Shared());
    EXPECT_EQ(index.shared_ids(), 2u);

    uint32_t row = 99;
    EXPECT_EQ(index.Find("1", &row), cpplink::IdLookup::kAmbiguous);
    EXPECT_EQ(index.Find("3", &row), cpplink::IdLookup::kFound);
    EXPECT_EQ(row, 2u);
    EXPECT_EQ(index.Find("4", &row), cpplink::IdLookup::kFound);
    EXPECT_EQ(row, 5u);
    EXPECT_EQ(index.Find("5", &row), cpplink::IdLookup::kMissing);

    EXPECT_EQ(index.Find(0, "1", &row), cpplink::IdLookup::kFound);
    EXPECT_EQ(row, 0u);
    EXPECT_EQ(index.Find(1, "1", &row), cpplink::IdLookup::kFound);
    EXPECT_EQ(row, 3u);
    EXPECT_EQ(index.Find(1, "3", &row), cpplink::IdLookup::kMissing);

    EXPECT_EQ(index.FindQualified("right:2", &row), cpplink::IdLookup::kFound);
    EXPECT_EQ(row, 4u);
    EXPECT_EQ(index.FindQualified("left:4", &row), cpplink::IdLookup::kMissing);
    // Not a dataset name, so the colon is part of the id, and no record has it.
    EXPECT_EQ(index.FindQualified("middle:2", &row), cpplink::IdLookup::kMissing);
}

TEST(IdIndex, TheLinearScanAgreesWithTheIndex) {
    const auto store = SharedIdStore();
    const cpplink::IdIndex index(*store);
    for (const char* text : {"1", "2", "3", "4", "5", "left:1", "right:1", "left:4",
                             "right:4", "middle:2"}) {
        uint32_t indexed = 0;
        uint64_t scanned = 0;
        const cpplink::IdLookup a = index.FindQualified(text, &indexed);
        const cpplink::IdLookup b = cpplink::FindRowById(*store, text, &scanned);
        EXPECT_EQ(a, b) << text;
        if (a == cpplink::IdLookup::kFound) EXPECT_EQ(indexed, scanned) << text;
    }
    EXPECT_EQ(cpplink::QualifiedId(*store, 0), "left:1");
    EXPECT_EQ(cpplink::QualifiedId(*store, 5), "right:4");
}

TEST(IdIndex, ARepeatedIdInsideOneInputIsRefused) {
    cpplink::Schema schema;
    schema.unique_id = "id";
    cpplink::RecordStore store(schema);
    for (const char* id : {"1", "2", "2", "1"}) store.mutable_ids().Append(id);
    store.set_num_records(4);
    store.set_datasets({0, 3, 4});
    store.set_dataset_names({"left", "right"});
    store.Finalize();
    const cpplink::IdIndex index(store);
    std::string error;
    EXPECT_FALSE(index.Unique(&error));
    EXPECT_NE(error.find("\"2\""), std::string::npos) << error;
    EXPECT_NE(error.find("left"), std::string::npos) << error;
    // The shared "1" is still counted as shared, not as a duplicate.
    EXPECT_EQ(index.shared_ids(), 1u);
    uint32_t row = 0;
    EXPECT_EQ(index.Find(0, "2", &row), cpplink::IdLookup::kAmbiguous);
    uint64_t scanned = 0;
    EXPECT_EQ(cpplink::FindRowById(store, "left:2", &scanned),
              cpplink::IdLookup::kAmbiguous);
}

// The prediction and cluster files over two inputs that share ids.
class SharedIdFiles : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_shared_ids_" + cpplink_test::ProcessId());
        cpplink_test::RemoveAll(dir_);
        std::filesystem::create_directories(dir_);
        store_ = SharedIdStore();
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    struct RawEdge {
        uint32_t a;
        uint32_t b;
        double weight;
    };

    void WriteShard(const std::string& name, cpplink::EdgeFormat format,
                    const std::vector<RawEdge>& edges) const {
        cpplink::EdgeShardWriter writer;
        ASSERT_TRUE(writer.Open((dir_ / name).string(), format, true));
        for (const RawEdge& edge : edges) {
            if (format == cpplink::EdgeFormat::kBinary) {
                writer.WriteBinary(edge.a, edge.b, 3, edge.weight);
            } else {
                writer.WriteCsv(*store_, edge.a, edge.b, 3, edge.weight);
            }
        }
        ASSERT_TRUE(writer.Close());
    }

    static std::vector<std::string> ReadLines(const std::string& path) {
        std::ifstream file(path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) lines.push_back(line);
        return lines;
    }

    // left:1-right:1 and left:2-right:2 are cross-input matches; left:3-right:4
    // is a third edge. Nothing here joins two rows of one input.
    const std::vector<RawEdge> edges_ = {{0, 3, 20.0}, {1, 4, 18.0}, {2, 5, 9.0}};
    std::filesystem::path dir_;
    std::unique_ptr<cpplink::RecordStore> store_;
};

TEST_F(SharedIdFiles, ACsvShardNamesEachRecordByDatasetAndId) {
    WriteShard("shard-000.csv", cpplink::EdgeFormat::kCsv, edges_);
    const auto lines = ReadLines((dir_ / "shard-000.csv").string());
    ASSERT_EQ(lines.size(), 4u);
    EXPECT_EQ(lines[0],
              "dataset_a,id_a,dataset_b,id_b,gamma,match_weight,match_probability");
    EXPECT_EQ(lines[1].rfind("left,1,right,1,3,20.000000,", 0), 0u) << lines[1];
    EXPECT_EQ(lines[3].rfind("left,3,right,4,3,9.000000,", 0), 0u) << lines[3];

    cpplink::EdgeCsvLayout layout;
    ASSERT_TRUE(cpplink::ParseEdgeCsvHeader(lines[0], &layout));
    EXPECT_TRUE(layout.datasets);
    cpplink::EdgeRow row;
    ASSERT_TRUE(cpplink::ParseEdgeCsvLine(lines[1], layout, &row));
    EXPECT_EQ(row.dataset_a, "left");
    EXPECT_EQ(row.id_a, "1");
    EXPECT_EQ(row.dataset_b, "right");
    EXPECT_EQ(row.id_b, "1");
    EXPECT_EQ(row.gamma, 3u);
    EXPECT_DOUBLE_EQ(row.weight, 20.0);
    // The single-input header still parses to the shape it always had.
    ASSERT_TRUE(cpplink::ParseEdgeCsvHeader(cpplink::EdgeCsvHeader(false), &layout));
    EXPECT_FALSE(layout.datasets);
    const std::string bare = "r0,r1,7,30.000000,1.000000000";
    ASSERT_TRUE(cpplink::ParseEdgeCsvLine(bare, layout, &row));
    EXPECT_EQ(row.id_a, "r0");
    EXPECT_EQ(row.id_b, "r1");
    EXPECT_TRUE(row.dataset_a.empty());
}

TEST_F(SharedIdFiles, MergedFilesCarryTheDatasetsAndClusterLikeTheShards) {
    WriteShard("shard-000.bin", cpplink::EdgeFormat::kBinary, edges_);
    std::string error;
    cpplink::ClusterOptions from_shards;
    from_shards.edge_path = dir_.string();
    cpplink::ClusterAssignment expected;
    cpplink::ClusterReport shard_report;
    ASSERT_TRUE(cpplink::Cluster(*store_, from_shards, &expected, &shard_report, &error))
        << error;
    EXPECT_TRUE(expected.SameCluster(0, 3));
    EXPECT_TRUE(expected.SameCluster(1, 4));
    EXPECT_FALSE(expected.SameCluster(0, 1));

    for (const char* name : {"merged.csv", "merged.parquet"}) {
        cpplink::MergeOptions merge;
        merge.edge_dir = dir_.string();
        merge.out_path = (dir_ / name).string();
        ASSERT_TRUE(cpplink::MergedFormatOf(merge.out_path, &merge.format));
        cpplink::MergeReport merged;
        ASSERT_TRUE(cpplink::MergeEdges(store_.get(), merge, &merged, &error)) << error;
        EXPECT_TRUE(merged.datasets) << name;

        if (merge.format == cpplink::MergeFormat::kCsv) {
            const auto lines = ReadLines(merge.out_path);
            ASSERT_EQ(lines.size(), 4u);
            EXPECT_EQ(lines[0], cpplink::EdgeCsvHeader(true));
            EXPECT_EQ(lines[2].rfind("left,2,right,2,", 0), 0u) << lines[2];
        } else {
            auto input = arrow::io::ReadableFile::Open(merge.out_path);
            ASSERT_TRUE(input.ok());
            auto reader = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
            ASSERT_TRUE(reader.ok()) << reader.status().message();
            std::shared_ptr<arrow::Table> table;
            ASSERT_TRUE((*reader)->ReadTable(&table).ok());
            ASSERT_EQ(table->num_columns(), 7);
            EXPECT_EQ(table->schema()->field(0)->name(), "dataset_a");
            EXPECT_EQ(table->schema()->field(1)->name(), "id_a");
            EXPECT_EQ(table->schema()->field(2)->name(), "dataset_b");
            EXPECT_EQ(table->schema()->field(3)->name(), "id_b");
        }

        cpplink::ClusterOptions from_file;
        from_file.edge_path = merge.out_path;
        cpplink::ClusterAssignment actual;
        cpplink::ClusterReport report;
        ASSERT_TRUE(cpplink::Cluster(*store_, from_file, &actual, &report, &error))
            << error;
        EXPECT_EQ(report.unresolved, 0u) << name;
        EXPECT_EQ(report.ambiguous, 0u) << name;
        EXPECT_EQ(report.edges_read, 3u) << name;
        EXPECT_EQ(actual.root, expected.root) << name;
    }
}

TEST_F(SharedIdFiles, CsvShardsMergeToTheSameFileAsBinaryOnes) {
    const std::filesystem::path bin = dir_ / "bin";
    const std::filesystem::path csv = dir_ / "csv";
    std::filesystem::create_directories(bin);
    std::filesystem::create_directories(csv);
    {
        cpplink::EdgeShardWriter writer;
        ASSERT_TRUE(writer.Open((bin / "shard-000.bin").string(),
                                cpplink::EdgeFormat::kBinary, true));
        for (const RawEdge& edge : edges_)
            writer.WriteBinary(edge.a, edge.b, 3, edge.weight);
        ASSERT_TRUE(writer.Close());
        cpplink::EdgeShardWriter text;
        ASSERT_TRUE(
            text.Open((csv / "shard-000.csv").string(), cpplink::EdgeFormat::kCsv, true));
        for (const RawEdge& edge : edges_) {
            text.WriteCsv(*store_, edge.a, edge.b, 3, edge.weight);
        }
        ASSERT_TRUE(text.Close());
    }
    std::string error;
    std::vector<std::string> outputs;
    for (const auto& source : {bin, csv}) {
        cpplink::MergeOptions merge;
        merge.edge_dir = source.string();
        merge.out_path = (dir_ / (source.filename().string() + ".csv")).string();
        cpplink::MergeReport merged;
        // The csv shards carry their own datasets and need no store.
        const cpplink::RecordStore* store = source == bin ? store_.get() : nullptr;
        ASSERT_TRUE(cpplink::MergeEdges(store, merge, &merged, &error)) << error;
        EXPECT_TRUE(merged.datasets);
        outputs.push_back(merge.out_path);
    }
    EXPECT_EQ(ReadLines(outputs[0]), ReadLines(outputs[1]));
}

TEST_F(SharedIdFiles, AFileWithoutDatasetsCannotNameASharedId) {
    const std::string path = (dir_ / "bare.csv").string();
    {
        std::ofstream file(path);
        file << cpplink::EdgeCsvHeader(false) << "\n"
             << "1,2,3,20.000000,1.000000000\n"   // both shared: ambiguous
             << "3,4,3,20.000000,1.000000000\n"   // each held once: resolves
             << "3,9,3,20.000000,1.000000000\n";  // 9 is nowhere: missing
    }
    cpplink::ClusterOptions options;
    options.edge_path = path;
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(*store_, options, &assignment, &report, &error))
        << error;
    EXPECT_EQ(report.edges_read, 3u);
    EXPECT_EQ(report.unresolved, 2u);
    EXPECT_EQ(report.ambiguous, 1u);
    EXPECT_TRUE(assignment.SameCluster(2, 5));
    EXPECT_FALSE(assignment.SameCluster(0, 3));
    std::ostringstream out;
    cpplink::PrintClusterReport(report, out);
    EXPECT_NE(out.str().find("more than one input"), std::string::npos) << out.str();
}

TEST_F(SharedIdFiles, ClusteringAMergedFileRefusesARepeatedIdWithinAnInput) {
    cpplink::Schema schema;
    schema.unique_id = "id";
    cpplink::RecordStore store(schema);
    for (const char* id : {"1", "1", "2"}) store.mutable_ids().Append(id);
    store.set_num_records(3);
    store.set_datasets({0, 2, 3});
    store.Finalize();
    const std::string path = (dir_ / "any.csv").string();
    {
        std::ofstream file(path);
        file << cpplink::EdgeCsvHeader(false) << "\n1,2,3,20.000000,1.000000000\n";
    }
    cpplink::ClusterOptions options;
    options.edge_path = path;
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    EXPECT_FALSE(cpplink::Cluster(store, options, &assignment, &report, &error));
    EXPECT_NE(error.find("unique within an input"), std::string::npos) << error;
    // Shards name rows and never resolve an id, so they are unaffected.
    WriteShard("shard-000.bin", cpplink::EdgeFormat::kBinary, {{0, 2, 20.0}});
    options.edge_path = dir_.string();
    EXPECT_TRUE(cpplink::Cluster(store, options, &assignment, &report, &error)) << error;
}

TEST_F(SharedIdFiles, ClusterOutputNamesTheDatasetAndQualifiesTheClusterId) {
    WriteShard("shard-000.bin", cpplink::EdgeFormat::kBinary, edges_);
    cpplink::ClusterOptions options;
    options.edge_path = dir_.string();
    options.out_path = (dir_ / "clusters.csv").string();
    cpplink::ClusterAssignment assignment;
    cpplink::ClusterReport report;
    std::string error;
    ASSERT_TRUE(cpplink::Cluster(*store_, options, &assignment, &report, &error))
        << error;
    uint64_t written = 0;
    ASSERT_TRUE(cpplink::WriteClusters(assignment, *store_, options, &written, &error))
        << error;
    EXPECT_EQ(written, 6u);
    const auto lines = ReadLines(options.out_path);
    ASSERT_EQ(lines.size(), 7u);
    EXPECT_EQ(lines[0], "dataset,unique_id,cluster_id,cluster_size");
    // left:1 and right:1 share a cluster whose id is one of them, qualified.
    const std::string root = cpplink::QualifiedId(*store_, assignment.root[0]);
    EXPECT_TRUE(root == "left:1" || root == "right:1") << root;
    EXPECT_EQ(lines[1], "left,1," + root + ",2");
    EXPECT_EQ(lines[4], "right,1," + root + ",2");

    options.out_path = (dir_ / "clusters.parquet").string();
    ASSERT_TRUE(cpplink::WriteClusters(assignment, *store_, options, &written, &error))
        << error;
    auto input = arrow::io::ReadableFile::Open(options.out_path);
    ASSERT_TRUE(input.ok());
    auto reader = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
    ASSERT_TRUE(reader.ok()) << reader.status().message();
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*reader)->ReadTable(&table).ok());
    ASSERT_EQ(table->num_columns(), 4);
    EXPECT_EQ(table->schema()->field(0)->name(), "dataset");
    EXPECT_EQ(table->num_rows(), 6);
}

TEST_F(SharedIdFiles, TruthPairsResolveQualifiedIdsAndRefuseToGuessBareOnes) {
    const std::string path = (dir_ / "truth.csv").string();
    {
        std::ofstream file(path);
        file << "id_a,id_b\n"
             << "left:1,right:1\n"  // resolves
             << "left:2,right:2\n"  // resolves
             << "3,4\n"             // bare but each held once: resolves
             << "1,right:2\n"       // 1 is shared: ambiguous
             << "left:3,right:3\n"  // right has no 3: unresolved
             << "left:1,left:1\n";  // a row against itself: unresolved
    }
    cpplink::TruthPairs truth;
    std::string error;
    ASSERT_TRUE(cpplink::LoadTruthPairs(path, *store_, &truth, &error)) << error;
    EXPECT_EQ(truth.lines, 6u);
    ASSERT_EQ(truth.rows.size(), 3u);
    EXPECT_EQ(truth.rows[0], std::make_pair(0u, 3u));
    EXPECT_EQ(truth.rows[1], std::make_pair(1u, 4u));
    EXPECT_EQ(truth.rows[2], std::make_pair(2u, 5u));
    EXPECT_EQ(truth.unresolved, 3u);
    EXPECT_EQ(truth.ambiguous, 1u);
}

// The same two inputs through the command line: `explain --pair` and `recall`
// read qualified ids, and inspect names the datasets.
class SharedIdCommands : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_shared_id_cli_" + cpplink_test::ProcessId());
        cpplink_test::RemoveAll(dir_);
        std::filesystem::create_directories(dir_);
        left_ = (dir_ / "left.parquet").string();
        right_ = (dir_ / "right.parquet").string();
        WriteInput(left_, {"1", "2", "3"}, {"ann@x.org", "bob@x.org", "cy@x.org"});
        WriteInput(right_, {"1", "2", "4"}, {"ann@x.org", "bob@x.org", "dee@x.org"});
        schema_ = (dir_ / "schema.json").string();
        std::ofstream schema(schema_);
        schema << R"({
  "unique_id": "id",
  "columns": [{"name": "email", "type": "string"}],
  "comparisons": [{
    "name": "email", "columns": ["email"],
    "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]
  }],
  "blocking": [{"type": "exact_value", "column": "email"}]
})";
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    static void WriteInput(const std::string& path, const std::vector<std::string>& ids,
                           const std::vector<std::string>& emails) {
        arrow::StringBuilder id_builder;
        arrow::StringBuilder email_builder;
        for (size_t i = 0; i < ids.size(); ++i) {
            ASSERT_TRUE(id_builder.Append(ids[i]).ok());
            ASSERT_TRUE(email_builder.Append(emails[i]).ok());
        }
        std::shared_ptr<arrow::Array> id_array;
        std::shared_ptr<arrow::Array> email_array;
        ASSERT_TRUE(id_builder.Finish(&id_array).ok());
        ASSERT_TRUE(email_builder.Finish(&email_array).ok());
        auto table =
            arrow::Table::Make(arrow::schema({arrow::field("id", arrow::utf8()),
                                              arrow::field("email", arrow::utf8())}),
                               {id_array, email_array});
        auto sink = arrow::io::FileOutputStream::Open(path);
        ASSERT_TRUE(sink.ok());
        ASSERT_TRUE(
            parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink, 1024)
                .ok());
    }

    std::filesystem::path dir_;
    std::string left_;
    std::string right_;
    std::string schema_;
};

TEST_F(SharedIdCommands, ExplainNeedsTheDatasetForASharedId) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(
        cpplink::Run({"explain", "--schema", schema_, "--pair", "1,1", left_, right_},
                     out, err),
        1);
    EXPECT_NE(err.str().find("more than one record has id '1'"), std::string::npos)
        << err.str();
    EXPECT_NE(err.str().find("left, right"), std::string::npos) << err.str();

    out.str("");
    err.str("");
    EXPECT_EQ(cpplink::Run({"explain", "--schema", schema_, "--pair", "left:1,right:1",
                            left_, right_},
                           out, err),
              0)
        << err.str();
    EXPECT_NE(out.str().find("Pair  left:1  /  right:1"), std::string::npos) << out.str();
    EXPECT_NE(out.str().find("exact"), std::string::npos) << out.str();

    // An id one input holds alone needs no qualifier.
    out.str("");
    err.str("");
    EXPECT_EQ(
        cpplink::Run({"explain", "--schema", schema_, "--pair", "3,4", left_, right_},
                     out, err),
        0)
        << err.str();
    EXPECT_NE(out.str().find("Pair  left:3  /  right:4"), std::string::npos) << out.str();
}

TEST_F(SharedIdCommands, RecallReadsQualifiedTruthAndReportsAmbiguousPairs) {
    const std::string truth = (dir_ / "truth.csv").string();
    {
        std::ofstream file(truth);
        file << "id_a,id_b\nleft:1,right:1\nleft:2,right:2\n2,2\n";
    }
    std::ostringstream out;
    std::ostringstream err;
    ASSERT_EQ(
        cpplink::Run({"recall", "--schema", schema_, "--truth", truth, left_, right_},
                     out, err),
        0)
        << err.str();
    EXPECT_NE(out.str().find("Known pairs  2 resolved, 1 unresolved"), std::string::npos)
        << out.str();
    EXPECT_NE(out.str().find("write it as dataset:id"), std::string::npos) << out.str();
}

TEST_F(SharedIdCommands, InspectNamesTheDatasets) {
    std::ostringstream out;
    std::ostringstream err;
    ASSERT_EQ(cpplink::Run({"inspect", "--schema", schema_, left_, right_}, out, err), 0)
        << err.str();
    EXPECT_NE(out.str().find("(dataset left, 3 rows)"), std::string::npos) << out.str();
    EXPECT_NE(out.str().find("(dataset right, 3 rows)"), std::string::npos) << out.str();
}

}  // namespace
