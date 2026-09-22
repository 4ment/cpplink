// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// The producer side of the C Data seam. A batch built here is what the parquet
// writer takes and what a data frame is handed, so its layout is checked three
// ways: Arrow C++ imports and validates it, the consumer reads it back into a
// store, and a parquet round trip through `parquet_io` returns the same rows.

#include "cpplink/arrow_export.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <gtest/gtest.h>

#include "cpplink/batch_loader.hpp"
#include "cpplink/parquet_io.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "tests/process_id.hpp"
#include "tests/temp_dir.hpp"

namespace {

cpplink::Schema Parse(const std::string& json) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_TRUE(cpplink::ParseSchema(json, &schema, &error)) << error;
    return schema;
}

// Three rows over every export type, the middle one null wherever a null is
// allowed, and a list cell that is empty.
void FillRows(cpplink::BatchBuilder* builder) {
    builder->AppendString(0, "1");
    builder->AppendString(0, "2");
    builder->AppendString(0, "3");
    builder->AppendString(1, "ann");
    builder->AppendNull(1);
    builder->AppendString(1, "ann");
    builder->AppendBoolean(2, true);
    builder->AppendNull(2);
    builder->AppendBoolean(2, false);
    builder->AppendUInt8(3, 7);
    builder->AppendNull(3);
    builder->AppendUInt8(3, 255);
    builder->AppendUInt32(4, 70000);
    builder->AppendNull(4);
    builder->AppendUInt32(4, 1);
    builder->AppendUInt64(5, 1ull << 40);
    builder->AppendNull(5);
    builder->AppendUInt64(5, 2);
    builder->AppendDouble(6, 0.5);
    builder->AppendNull(6);
    builder->AppendDouble(6, -1.5);
    builder->AppendDate32(7, 19000);
    builder->AppendNull(7);
    builder->AppendDate32(7, 19001);
    builder->AppendStringList(8, {"y", "x", "y"});
    builder->AppendStringList(8, {});
    builder->AppendNull(8);
}

cpplink::BatchBuilder EveryType() {
    cpplink::BatchBuilder builder;
    builder.AddColumn("id", cpplink::ExportType::kString);
    builder.AddColumn("name", cpplink::ExportType::kString);
    builder.AddColumn("flag", cpplink::ExportType::kBoolean);
    builder.AddColumn("small", cpplink::ExportType::kUInt8);
    builder.AddColumn("gamma", cpplink::ExportType::kUInt32);
    builder.AddColumn("size", cpplink::ExportType::kUInt64);
    builder.AddColumn("x", cpplink::ExportType::kDouble);
    builder.AddColumn("dob", cpplink::ExportType::kDate32);
    builder.AddColumn("tags", cpplink::ExportType::kStringList);
    return builder;
}

const char* kSchema = R"({"unique_id":"id","columns":[
    {"name":"name","type":"string"},
    {"name":"flag","type":"boolean"},
    {"name":"small","type":"string"},
    {"name":"gamma","type":"string"},
    {"name":"size","type":"string"},
    {"name":"x","type":"double"},
    {"name":"dob","type":"date"},
    {"name":"tags","type":"string_list"}]})";

void ExpectRows(const cpplink::RecordStore& store) {
    ASSERT_EQ(store.NumRecords(), 3u);
    EXPECT_EQ(store.ids().Get(1), "2");
    const auto& name = std::get<cpplink::StringColumn>(store.column(0));
    EXPECT_EQ(name.dict.Value(name.ids[0]), "ann");
    EXPECT_EQ(name.ids[1], cpplink::kNullId);
    EXPECT_EQ(name.ids[0], name.ids[2]);
    const auto& flag = std::get<cpplink::BooleanColumn>(store.column(1));
    EXPECT_EQ(flag.values, (std::vector<int8_t>{1, cpplink::kNullBoolean, 0}));
    const auto& small = std::get<cpplink::StringColumn>(store.column(2));
    EXPECT_EQ(small.dict.Value(small.ids[2]), "255");
    EXPECT_EQ(small.ids[1], cpplink::kNullId);
    const auto& gamma = std::get<cpplink::StringColumn>(store.column(3));
    EXPECT_EQ(gamma.dict.Value(gamma.ids[0]), "70000");
    const auto& size = std::get<cpplink::StringColumn>(store.column(4));
    EXPECT_EQ(size.dict.Value(size.ids[0]), "1099511627776");
    const auto& x = std::get<cpplink::DoubleColumn>(store.column(5));
    EXPECT_EQ(x.values[0], 0.5);
    EXPECT_TRUE(std::isnan(x.values[1]));
    EXPECT_EQ(x.values[2], -1.5);
    const auto& dob = std::get<cpplink::DateColumn>(store.column(6));
    EXPECT_EQ(dob.values, (std::vector<int32_t>{19000, cpplink::kNullDate, 19001}));
    const auto& tags = std::get<cpplink::StringListColumn>(store.column(7));
    EXPECT_EQ(tags.offsets, (std::vector<uint64_t>{0, 2, 2, 2}));
    EXPECT_EQ(tags.dict.Size(), 2u);
}

}  // namespace

TEST(ArrowExport, ArrowValidatesTheLayout) {
    cpplink::BatchBuilder builder = EveryType();
    FillRows(&builder);
    ArrowSchema schema;
    ArrowArray array;
    std::string error;
    builder.ExportSchema(&schema);
    ASSERT_TRUE(builder.ExportBatch(&array, &error)) << error;
    EXPECT_EQ(builder.Rows(), 0) << "the rows moved out";

    auto imported = arrow::ImportRecordBatch(&array, &schema);
    ASSERT_TRUE(imported.ok()) << imported.status().message();
    const auto& batch = *imported;
    ASSERT_TRUE(batch->ValidateFull().ok()) << batch->ValidateFull().message();
    EXPECT_EQ(batch->num_rows(), 3);
    EXPECT_EQ(batch->schema()->field(0)->type()->ToString(), "string");
    EXPECT_EQ(batch->schema()->field(2)->type()->ToString(), "bool");
    EXPECT_EQ(batch->schema()->field(3)->type()->ToString(), "uint8");
    EXPECT_EQ(batch->schema()->field(4)->type()->ToString(), "uint32");
    EXPECT_EQ(batch->schema()->field(5)->type()->ToString(), "uint64");
    EXPECT_EQ(batch->schema()->field(6)->type()->ToString(), "double");
    EXPECT_EQ(batch->schema()->field(7)->type()->ToString(), "date32[day]");
    EXPECT_EQ(batch->schema()->field(8)->type()->ToString(), "list<item: string>");
    EXPECT_EQ(batch->column(1)->null_count(), 1);
    EXPECT_EQ(batch->column(0)->null_count(), 0);
    EXPECT_EQ(batch->column(0)->data()->buffers[0], nullptr)
        << "a column with no null carries no validity bitmap";
    EXPECT_EQ(batch->column(8)->null_count(), 1);
    const auto& tags = static_cast<const arrow::ListArray&>(*batch->column(8));
    EXPECT_EQ(tags.value_length(0), 3);
    EXPECT_EQ(tags.value_length(1), 0);
    EXPECT_TRUE(tags.IsNull(2));
    // Releasing the imported batch released the export; the structs say so.
    EXPECT_EQ(array.release, nullptr);
    EXPECT_EQ(schema.release, nullptr);
}

TEST(ArrowExport, TheConsumerReadsWhatTheProducerWrote) {
    cpplink::BatchBuilder builder = EveryType();
    FillRows(&builder);
    ArrowSchema schema;
    ArrowArray array;
    std::string error;
    builder.ExportSchema(&schema);
    ASSERT_TRUE(builder.ExportBatch(&array, &error)) << error;

    const cpplink::Schema spec = Parse(kSchema);
    cpplink::RecordStore store(spec);
    ASSERT_TRUE(cpplink::AppendRecordBatch(schema, array, spec, &store, &error)) << error;
    array.release(&array);
    schema.release(&schema);
    store.set_num_records(3);
    store.Finalize();
    ExpectRows(store);
}

TEST(ArrowExport, RefusesRaggedColumns) {
    cpplink::BatchBuilder builder;
    builder.AddColumn("a", cpplink::ExportType::kString);
    builder.AddColumn("b", cpplink::ExportType::kDouble);
    builder.AppendString(0, "x");
    builder.AppendString(0, "y");
    builder.AppendDouble(1, 1.0);
    ArrowArray array;
    std::string error;
    EXPECT_FALSE(builder.ExportBatch(&array, &error));
    EXPECT_NE(error.find("\"b\" has 1 rows where \"a\" has 2"), std::string::npos)
        << error;
    EXPECT_EQ(builder.Rows(), 2) << "a refused export leaves the rows in place";
}

TEST(ArrowExport, ABatchCanBeReusedAfterExport) {
    cpplink::BatchBuilder builder;
    builder.AddColumn("name", cpplink::ExportType::kString);
    builder.AddColumn("tags", cpplink::ExportType::kStringList);
    builder.AppendNull(0);
    builder.AppendStringList(1, {"a"});
    ArrowArray first;
    std::string error;
    ASSERT_TRUE(builder.ExportBatch(&first, &error)) << error;
    first.release(&first);
    // The second batch starts from zero: fresh offsets, no validity bitmap.
    builder.AppendString(0, "b");
    builder.AppendStringList(1, {"c", "d"});
    ArrowSchema schema;
    ArrowArray second;
    builder.ExportSchema(&schema);
    ASSERT_TRUE(builder.ExportBatch(&second, &error)) << error;
    auto imported = arrow::ImportRecordBatch(&second, &schema);
    ASSERT_TRUE(imported.ok()) << imported.status().message();
    ASSERT_TRUE((*imported)->ValidateFull().ok());
    EXPECT_EQ((*imported)->column(0)->null_count(), 0);
    const auto& names = static_cast<const arrow::StringArray&>(*(*imported)->column(0));
    EXPECT_EQ(names.GetString(0), "b");
    const auto& tags = static_cast<const arrow::ListArray&>(*(*imported)->column(1));
    EXPECT_EQ(tags.value_offset(0), 0);
    EXPECT_EQ(tags.value_length(0), 2);
}

// The io seam end to end: batches written as row groups and read back as the
// same batches, through nothing but the C structs on either side.
TEST(ArrowExport, ParquetRoundTrip) {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("cpplink_arrow_export_" + cpplink_test::ProcessId());
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "rows.parquet").string();

    ASSERT_TRUE(cpplink::ParquetSupported());
    cpplink::BatchBuilder builder = EveryType();
    cpplink::ParquetWriter writer;
    ArrowSchema schema;
    std::string error;
    builder.ExportSchema(&schema);
    ASSERT_TRUE(writer.Open(path, schema, &error)) << error;
    schema.release(&schema);
    for (int batch = 0; batch < 2; ++batch) {
        FillRows(&builder);
        ArrowArray array;
        ASSERT_TRUE(builder.ExportBatch(&array, &error)) << error;
        ASSERT_TRUE(writer.Write(&array, &error)) << error;
    }
    ASSERT_TRUE(writer.Close(&error)) << error;

    // Read only some columns, in an order of our choosing.
    ArrowSchema file;
    ASSERT_TRUE(cpplink::ReadParquetSchema(path, &file, &error)) << error;
    EXPECT_EQ(file.n_children, 9);
    EXPECT_EQ(cpplink::ArrowTypeName(*file.children[8]), "list<element: string>")
        << "parquet names the child element, and the name follows the file";
    file.release(&file);

    ArrowArrayStream stream;
    ASSERT_TRUE(cpplink::OpenParquetStream(path, {"tags", "id", "x"}, &stream, &error))
        << error;
    const cpplink::Schema spec = Parse(R"({"unique_id":"id","columns":[
        {"name":"x","type":"double"},
        {"name":"tags","type":"string_list"}]})");
    cpplink::RecordStore store(spec);
    uint64_t rows = 0;
    int batches = 0;
    ASSERT_TRUE(
        cpplink::AppendArrayStream(&stream, spec, &store, &rows, &batches, &error))
        << error;
    EXPECT_EQ(rows, 6u);
    EXPECT_EQ(batches, 2) << "one row group per batch written";
    store.set_num_records(6);
    store.Finalize();
    EXPECT_EQ(store.ids().Get(5), "3");
    const auto& tags = std::get<cpplink::StringListColumn>(store.column(1));
    EXPECT_EQ(tags.offsets, (std::vector<uint64_t>{0, 2, 2, 2, 4, 4, 4}));

    ArrowArrayStream missing;
    EXPECT_FALSE(cpplink::OpenParquetStream(path, {"gone"}, &missing, &error));
    EXPECT_NE(error.find("\"gone\" is not in"), std::string::npos) << error;
    cpplink_test::RemoveAll(dir);
}
