// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// The in-memory side of a run: predictions kept as a table rather than written
// as shards, clustered from that table or from a prediction table that arrived
// as a C Data stream, and the table exported with the record ids as a
// dictionary over the store's own id arena. Each is held to what the file path
// does, because a front end that returns a frame must return the file's rows.

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <gtest/gtest.h>

#include "cpplink/arrow_export.hpp"
#include "cpplink/blocking.hpp"
#include "cpplink/cluster.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"
#include "tests/process_id.hpp"

namespace {

constexpr uint64_t kRecords = 400;

const char* const kSchemaJson = R"({
  "unique_id": "id",
  "columns": [
    {"name": "surname", "type": "string"},
    {"name": "city", "type": "string"}
  ],
  "comparisons": [
    {"name": "surname", "columns": ["surname"], "term_frequency": true,
     "levels": [{"type": "null"}, {"type": "exact"}, {"type": "else"}]},
    {"name": "city", "columns": ["city"],
     "levels": [{"type": "exact"}, {"type": "else"}]}
  ],
  "blocking": [{"type": "exact_value", "column": "city"}]
})";

struct Edge {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t gamma = 0;
    double weight = 0.0;
    bool operator<(const Edge& other) const {
        return std::tie(a, b, gamma) < std::tie(other.a, other.b, other.gamma);
    }
    bool operator==(const Edge& other) const {
        return a == other.a && b == other.b && gamma == other.gamma &&
               weight == other.weight;
    }
};

class EdgeTableFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("cpplink_edge_table_" + cpplink_test::ProcessId());
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);
        std::string error;
        ASSERT_TRUE(cpplink::ParseSchema(kSchemaJson, &schema_, &error)) << error;
        store_ = std::make_unique<cpplink::RecordStore>(schema_);
        auto& surname = std::get<cpplink::StringColumn>(store_->mutable_column(0));
        auto& city = std::get<cpplink::StringColumn>(store_->mutable_column(1));
        std::vector<uint32_t> names;
        for (int i = 0; i < 40; ++i) {
            names.push_back(surname.dict.Intern("name" + std::to_string(i)));
        }
        std::vector<uint32_t> cities;
        for (int i = 0; i < 4; ++i) {
            cities.push_back(city.dict.Intern("city" + std::to_string(i)));
        }
        for (uint64_t row = 0; row < kRecords; ++row) {
            const size_t pick = static_cast<size_t>((row * row) % 40) / 2;
            surname.ids.push_back(names[pick]);
            city.ids.push_back(cities[(row * 7) % 4]);
            store_->mutable_ids().Append("r" + std::to_string(row));
        }
        store_->set_num_records(kRecords);
        store_->Finalize();
        ASSERT_TRUE(plan_.Build(schema_, *store_, &error)) << error;
        ASSERT_TRUE(comparisons_.Bind(schema_, *store_, &error)) << error;
        model_.lambda = 0.05;
        model_.records = kRecords;
        model_.comparisons = {
            Comparison("surname", true, {{1e-9, 1e-9}, {0.9, 0.02}, {0.1, 0.98}}),
            Comparison("city", false, {{0.8, 0.25}, {0.2, 0.75}})};
        cpplink::ScoreOptions score;
        score.threshold = -2.0;
        ASSERT_TRUE(scorer_.Bind(model_, comparisons_, *store_, score, &error)) << error;
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    static cpplink::ModelComparison Comparison(
        const std::string& name, bool tf,
        const std::vector<std::pair<double, double>>& levels) {
        cpplink::ModelComparison comparison;
        comparison.name = name;
        comparison.term_frequency = tf;
        for (const auto& entry : levels) {
            cpplink::ModelLevel level;
            level.m = entry.first;
            level.u = entry.second;
            level.m_estimated = true;
            comparison.levels.push_back(level);
        }
        return comparison;
    }

    // The same run three ways: shards only, table only, and both at once.
    bool Run(const std::string& shards, cpplink::EdgeTable* table, unsigned threads,
             cpplink::PredictReport* report, std::string* error) {
        cpplink::PredictOptions options;
        options.out_dir = shards.empty() ? "" : (dir_ / shards).string();
        options.table = table;
        options.threads = threads;
        return cpplink::Predict(*store_, comparisons_, plan_, scorer_, options, report,
                                error);
    }

    static std::vector<Edge> ReadShards(const std::string& directory) {
        std::vector<Edge> edges;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            std::ifstream file(entry.path(), std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            for (size_t at = 8; at + 20 <= bytes.size(); at += 20) {
                Edge edge;
                std::memcpy(&edge.a, bytes.data() + at, 4);
                std::memcpy(&edge.b, bytes.data() + at + 4, 4);
                std::memcpy(&edge.gamma, bytes.data() + at + 8, 4);
                std::memcpy(&edge.weight, bytes.data() + at + 12, 8);
                edges.push_back(edge);
            }
        }
        std::sort(edges.begin(), edges.end());
        return edges;
    }

    // The partition with each cluster named by its lowest row, since which row
    // union-find picks as representative depends on the order the edges came.
    static std::vector<uint32_t> Canonical(const cpplink::ClusterAssignment& assignment) {
        std::vector<uint32_t> lowest(assignment.root.size(), UINT32_MAX);
        for (size_t row = 0; row < assignment.root.size(); ++row) {
            lowest[assignment.root[row]] =
                std::min(lowest[assignment.root[row]], static_cast<uint32_t>(row));
        }
        std::vector<uint32_t> canonical(assignment.root.size());
        for (size_t row = 0; row < assignment.root.size(); ++row) {
            canonical[row] = lowest[assignment.root[row]];
        }
        return canonical;
    }

    static std::vector<Edge> Rows(const cpplink::EdgeTable& table) {
        std::vector<Edge> edges;
        for (uint64_t i = 0; i < table.Size(); ++i) {
            edges.push_back({table.a[i], table.b[i], table.gamma[i], table.weight[i]});
        }
        std::sort(edges.begin(), edges.end());
        return edges;
    }

    std::filesystem::path dir_;
    cpplink::Schema schema_;
    std::unique_ptr<cpplink::RecordStore> store_;
    cpplink::BlockingPlan plan_;
    cpplink::ComparisonSet comparisons_;
    cpplink::Model model_;
    cpplink::Scorer scorer_;
};

}  // namespace

TEST_F(EdgeTableFixture, TheTableHoldsWhatTheShardsHold) {
    std::string error;
    cpplink::PredictReport files;
    ASSERT_TRUE(Run("shards", nullptr, 3, &files, &error)) << error;
    const std::vector<Edge> from_files = ReadShards((dir_ / "shards").string());
    ASSERT_FALSE(from_files.empty());

    cpplink::EdgeTable table;
    cpplink::PredictReport memory;
    ASSERT_TRUE(Run("", &table, 3, &memory, &error)) << error;
    EXPECT_TRUE(memory.shards.empty());
    EXPECT_EQ(memory.edges, files.edges);
    EXPECT_EQ(table.Size(), files.edges);
    EXPECT_EQ(Rows(table), from_files);
    EXPECT_FALSE(std::filesystem::exists(dir_ / "shards2"));

    // Both at once: the same rows in both places.
    cpplink::EdgeTable both;
    ASSERT_TRUE(Run("shards2", &both, 2, &memory, &error)) << error;
    EXPECT_EQ(Rows(both), ReadShards((dir_ / "shards2").string()));
    EXPECT_EQ(Rows(both), from_files);

    // Nowhere to put them is refused.
    cpplink::PredictReport nowhere;
    EXPECT_FALSE(Run("", nullptr, 1, &nowhere, &error));
    EXPECT_NE(error.find("nowhere to put"), std::string::npos) << error;
}

TEST_F(EdgeTableFixture, ClusteringTheTableIsClusteringTheShards) {
    std::string error;
    cpplink::PredictReport report;
    cpplink::EdgeTable table;
    ASSERT_TRUE(Run("shards", &table, 2, &report, &error)) << error;

    cpplink::ClusterOptions from_files;
    from_files.edge_path = (dir_ / "shards").string();
    from_files.threshold = 0.0;
    cpplink::ClusterAssignment files;
    cpplink::ClusterReport files_report;
    ASSERT_TRUE(cpplink::Cluster(*store_, from_files, &files, &files_report, &error))
        << error;

    cpplink::ClusterOptions from_table;
    from_table.edges = &table;
    from_table.threshold = 0.0;
    cpplink::ClusterAssignment memory;
    cpplink::ClusterReport memory_report;
    ASSERT_TRUE(cpplink::Cluster(*store_, from_table, &memory, &memory_report, &error))
        << error;
    EXPECT_EQ(Canonical(memory), Canonical(files));
    EXPECT_EQ(memory_report.edges_read, files_report.edges_read);
    EXPECT_EQ(memory_report.edges_used, files_report.edges_used);
    EXPECT_GT(memory.clusters, 0u);

    // A prediction table arriving as a stream: ids as text, weights as doubles,
    // one column the reader does not need, and the rows in a different order.
    cpplink::BatchBuilder builder;
    const int id_a = builder.AddColumn("id_a", cpplink::ExportType::kString);
    const int weight = builder.AddColumn("match_weight", cpplink::ExportType::kDouble);
    const int id_b = builder.AddColumn("id_b", cpplink::ExportType::kString);
    const int extra = builder.AddColumn("note", cpplink::ExportType::kString);
    for (uint64_t i = table.Size(); i-- > 0;) {
        builder.AppendString(id_a, store_->ids().Get(table.a[i]));
        builder.AppendDouble(weight, table.weight[i]);
        builder.AppendString(id_b, store_->ids().Get(table.b[i]));
        builder.AppendString(extra, "x");
    }
    ArrowSchema schema;
    ArrowArray batch;
    builder.ExportSchema(&schema);
    ASSERT_TRUE(builder.ExportBatch(&batch, &error)) << error;
    auto imported = arrow::ImportRecordBatch(&batch, &schema);
    ASSERT_TRUE(imported.ok()) << imported.status().message();
    auto reader = arrow::RecordBatchReader::Make({*imported});
    ASSERT_TRUE(reader.ok());
    ArrowArrayStream stream;
    ASSERT_TRUE(arrow::ExportRecordBatchReader(*reader, &stream).ok());

    cpplink::ClusterOptions from_stream;
    from_stream.stream = &stream;
    from_stream.threshold = 0.0;
    cpplink::ClusterAssignment streamed;
    cpplink::ClusterReport streamed_report;
    ASSERT_TRUE(
        cpplink::Cluster(*store_, from_stream, &streamed, &streamed_report, &error))
        << error;
    EXPECT_EQ(stream.release, nullptr) << "the stream is released";
    EXPECT_EQ(Canonical(streamed), Canonical(files));
    EXPECT_EQ(streamed_report.edges_used, files_report.edges_used);
}

// The record ids leave as a dictionary over the id arena: the arena's own
// offsets and text as the dictionary, the rows as the indices, nothing copied.
TEST_F(EdgeTableFixture, IdsExportAsADictionaryOverTheArena) {
    std::string error;
    cpplink::PredictReport report;
    auto table = std::make_shared<cpplink::EdgeTable>();
    ASSERT_TRUE(Run("", table.get(), 2, &report, &error)) << error;
    ASSERT_GT(table->Size(), 0u);

    const cpplink::IdColumn& ids = store_->ids();
    cpplink::BorrowedColumn id_a;
    id_a.type = cpplink::ExportType::kDictionary;
    id_a.length = static_cast<int64_t>(table->Size());
    id_a.values = table->a.data();
    id_a.dictionary_offsets = reinterpret_cast<const int64_t*>(ids.offsets.data());
    id_a.dictionary_text = ids.text.data();
    id_a.dictionary_size = static_cast<int64_t>(ids.offsets.size() - 1);
    cpplink::BorrowedColumn weight;
    weight.type = cpplink::ExportType::kDouble;
    weight.length = id_a.length;
    weight.values = table->weight.data();

    cpplink::BatchBuilder builder;
    builder.AddBorrowed("id_a", id_a);
    builder.AddBorrowed("match_weight", weight);
    const int note = builder.AddColumn("note", cpplink::ExportType::kString);
    for (uint64_t i = 0; i < table->Size(); ++i) builder.AppendString(note, "n");
    builder.KeepAlive(table);
    EXPECT_EQ(builder.Rows(), id_a.length);

    ArrowSchema schema;
    ArrowArray batch;
    builder.ExportSchema(&schema);
    ASSERT_TRUE(builder.ExportBatch(&batch, &error)) << error;
    auto imported = arrow::ImportRecordBatch(&batch, &schema);
    ASSERT_TRUE(imported.ok()) << imported.status().message();
    const auto& rows = *imported;
    ASSERT_TRUE(rows->ValidateFull().ok()) << rows->ValidateFull().message();
    EXPECT_EQ(rows->schema()->field(0)->type()->ToString(),
              "dictionary<values=large_string, indices=uint32, ordered=0>");
    const auto& dictionary = static_cast<const arrow::DictionaryArray&>(*rows->column(0));
    EXPECT_EQ(dictionary.dictionary()->length(), static_cast<int64_t>(kRecords));
    EXPECT_EQ(dictionary.dictionary()->data()->buffers[2]->data(),
              reinterpret_cast<const uint8_t*>(ids.text.data()))
        << "the dictionary is the arena itself";
    const auto& values =
        static_cast<const arrow::LargeStringArray&>(*dictionary.dictionary());
    const auto& indices = static_cast<const arrow::UInt32Array&>(*dictionary.indices());
    for (int64_t i = 0; i < rows->num_rows(); ++i) {
        EXPECT_EQ(values.GetString(indices.Value(i)), ids.Get(table->a[i]));
    }
    EXPECT_EQ(static_cast<const arrow::DoubleArray&>(*rows->column(1)).Value(0),
              table->weight[0]);

    // The owned column moved out, so the builder is ragged until it is refilled;
    // the borrowed ones are still there, and every live export holds the table.
    ArrowArray again;
    EXPECT_FALSE(builder.ExportBatch(&again, &error));
    EXPECT_NE(error.find("\"note\" has 0 rows"), std::string::npos) << error;
    EXPECT_EQ(builder.Rows(), id_a.length);
    EXPECT_GE(table.use_count(), 3) << "the test, the builder and the live export";
    cpplink::BatchBuilder borrowed_only;
    borrowed_only.AddBorrowed("id_a", id_a);
    borrowed_only.KeepAlive(table);
    ASSERT_TRUE(borrowed_only.ExportBatch(&again, &error)) << error;
    again.release(&again);
    ASSERT_TRUE(borrowed_only.ExportBatch(&again, &error)) << "exported twice";
    EXPECT_EQ(again.children[0]->length, id_a.length);
    again.release(&again);
}
