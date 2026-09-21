// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// The batch loader reads Arrow buffers where a producer left them, through the C
// Data Interface, and it is the one decoder both the parquet path and a data frame
// go through. The parquet tests already cover the types a file holds; these cover
// what only a frame produces -- a sliced array, a sliced batch, a category
// column, string views, an all-null column, a stream of batches -- and check that
// a table decoded from its buffers is the store the parquet path builds from the
// same table, member for member.

#include "cpplink/batch_loader.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <unistd.h>

#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace {

template <typename Builder, typename T>
std::shared_ptr<arrow::Array> Column(const std::vector<T>& values,
                                     const std::vector<bool>& valid = {}) {
    Builder builder;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!valid.empty() && !valid[i]) {
            EXPECT_TRUE(builder.AppendNull().ok());
        } else {
            EXPECT_TRUE(builder.Append(values[i]).ok());
        }
    }
    std::shared_ptr<arrow::Array> array;
    EXPECT_TRUE(builder.Finish(&array).ok());
    return array;
}

std::string Text(const cpplink::StringColumn& column, uint64_t row) {
    return std::string(column.dict.Value(column.ids[row]));
}

cpplink::Schema Parse(const std::string& json) {
    cpplink::Schema schema;
    std::string error;
    EXPECT_TRUE(cpplink::ParseSchema(json, &schema, &error)) << error;
    return schema;
}

// Exports the batch and decodes it, releasing the structs afterwards as a
// consumer must.
bool Append(const arrow::RecordBatch& batch, const cpplink::Schema& schema,
            cpplink::RecordStore* store, std::string* error) {
    ArrowArray array;
    ArrowSchema c_schema;
    EXPECT_TRUE(arrow::ExportRecordBatch(batch, &array, &c_schema).ok());
    const bool ok = cpplink::AppendRecordBatch(c_schema, array, schema, store, error);
    array.release(&array);
    c_schema.release(&c_schema);
    return ok;
}

std::shared_ptr<arrow::RecordBatch> Batch(
    const std::vector<std::shared_ptr<arrow::Field>>& fields,
    const std::vector<std::shared_ptr<arrow::Array>>& arrays) {
    return arrow::RecordBatch::Make(arrow::schema(fields), arrays.front()->length(),
                                    arrays);
}

// Every member of every column, so two stores built two ways can be compared.
void ExpectSameStore(const cpplink::RecordStore& a, const cpplink::RecordStore& b) {
    ASSERT_EQ(a.NumRecords(), b.NumRecords());
    ASSERT_EQ(a.NumColumns(), b.NumColumns());
    EXPECT_EQ(a.ids().text, b.ids().text);
    EXPECT_EQ(a.ids().offsets, b.ids().offsets);
    for (size_t c = 0; c < a.NumColumns(); ++c) {
        ASSERT_EQ(a.column(c).index(), b.column(c).index());
        std::visit(
            [&](const auto& x) {
                using T = std::decay_t<decltype(x)>;
                const auto& y = std::get<T>(b.column(c));
                if constexpr (std::is_same_v<T, cpplink::StringColumn>) {
                    EXPECT_EQ(x.ids, y.ids);
                    EXPECT_EQ(x.tf, y.tf);
                    ASSERT_EQ(x.dict.Size(), y.dict.Size());
                    for (uint32_t v = 0; v < x.dict.Size(); ++v) {
                        EXPECT_EQ(x.dict.Value(v), y.dict.Value(v));
                    }
                } else if constexpr (std::is_same_v<T, cpplink::StringListColumn>) {
                    EXPECT_EQ(x.ids, y.ids);
                    EXPECT_EQ(x.offsets, y.offsets);
                    EXPECT_EQ(x.tf, y.tf);
                    ASSERT_EQ(x.dict.Size(), y.dict.Size());
                    for (uint32_t v = 0; v < x.dict.Size(); ++v) {
                        EXPECT_EQ(x.dict.Value(v), y.dict.Value(v));
                    }
                } else if constexpr (std::is_same_v<T, cpplink::DateColumn>) {
                    EXPECT_EQ(x.values, y.values);
                    EXPECT_EQ(x.tf, y.tf);
                    EXPECT_EQ(x.tf_origin, y.tf_origin);
                } else if constexpr (std::is_same_v<T, cpplink::BooleanColumn>) {
                    EXPECT_EQ(x.values, y.values);
                    EXPECT_EQ(x.tf, y.tf);
                } else {
                    ASSERT_EQ(x.values.size(), y.values.size());
                    for (size_t i = 0; i < x.values.size(); ++i) {
                        if (std::isnan(x.values[i])) {
                            EXPECT_TRUE(std::isnan(y.values[i]));
                        } else {
                            EXPECT_EQ(x.values[i], y.values[i]);
                        }
                    }
                }
            },
            a.column(c));
    }
}

}  // namespace

// A slice of a frame is the common way a wrapper bounds the transient copy of
// an object column: it exports arrays whose `offset` is not zero, and the values
// before the slice must not be read, let alone interned.
TEST(BatchLoader, ReadsSlicedArraysAtTheirOffset) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"name","type":"string"},
        {"name":"tags","type":"string_list"},
        {"name":"dob","type":"date"},
        {"name":"x","type":"double"},
        {"name":"flag","type":"boolean"}]})");

    arrow::ListBuilder tags(arrow::default_memory_pool(),
                            std::make_shared<arrow::StringBuilder>());
    auto* tag = static_cast<arrow::StringBuilder*>(tags.value_builder());
    for (const char* row : {"before", "b", "c", "after"}) {
        ASSERT_TRUE(tags.Append().ok());
        ASSERT_TRUE(tag->AppendValues({row, "shared"}).ok());
    }
    std::shared_ptr<arrow::Array> tags_array;
    ASSERT_TRUE(tags.Finish(&tags_array).ok());

    const auto whole =
        Batch({arrow::field("id", arrow::utf8()), arrow::field("name", arrow::utf8()),
               arrow::field("tags", arrow::list(arrow::utf8())),
               arrow::field("dob", arrow::date32()), arrow::field("x", arrow::float64()),
               arrow::field("flag", arrow::boolean())},
              {Column<arrow::StringBuilder>(std::vector<std::string>{"0", "1", "2", "3"}),
               Column<arrow::StringBuilder>(
                   std::vector<std::string>{"zero", "one", "", "three"},
                   {true, true, false, true}),
               tags_array, Column<arrow::Date32Builder>(std::vector<int32_t>{1, 2, 3, 4}),
               Column<arrow::DoubleBuilder>(std::vector<double>{0.5, 1.5, 2.5, 3.5}),
               Column<arrow::BooleanBuilder>(std::vector<bool>{false, true, false, true},
                                             {true, true, false, true})});

    // Each column sliced on its own, so every array carries its own offset.
    std::vector<std::shared_ptr<arrow::Array>> sliced;
    for (int c = 0; c < whole->num_columns(); ++c) {
        sliced.push_back(whole->column(c)->Slice(1, 2));
    }
    const auto batch = arrow::RecordBatch::Make(whole->schema(), 2, sliced);

    cpplink::RecordStore store(schema);
    std::string error;
    ASSERT_TRUE(Append(*batch, schema, &store, &error)) << error;
    store.set_num_records(2);
    store.Finalize();

    EXPECT_EQ(store.ids().Get(0), "1");
    EXPECT_EQ(store.ids().Get(1), "2");
    const auto& name = std::get<cpplink::StringColumn>(store.column(0));
    EXPECT_EQ(Text(name, 0), "one");
    EXPECT_EQ(name.ids[1], cpplink::kNullId);
    EXPECT_EQ(name.dict.Size(), 1u) << "values outside the slice are not interned";

    const auto& tag_column = std::get<cpplink::StringListColumn>(store.column(1));
    const std::vector<uint64_t> offsets = {0, 2, 4};
    EXPECT_EQ(tag_column.offsets, offsets);
    EXPECT_EQ(tag_column.dict.Size(), 3u) << "b, shared and c; not before or after";
    EXPECT_EQ(tag_column.tf[tag_column.ids[1]], 2u) << "shared is on both rows";

    const auto& dob = std::get<cpplink::DateColumn>(store.column(2));
    EXPECT_EQ(dob.values, (std::vector<int32_t>{2, 3}));
    const auto& x = std::get<cpplink::DoubleColumn>(store.column(3));
    EXPECT_EQ(x.values, (std::vector<double>{1.5, 2.5}));
    const auto& flag = std::get<cpplink::BooleanColumn>(store.column(4));
    EXPECT_EQ(flag.values, (std::vector<int8_t>{1, cpplink::kNullBoolean}));
}

// A batch sliced as a whole is a struct with an offset, which applies to every
// child: the children keep their full length and the batch says where to read.
TEST(BatchLoader, AppliesTheBatchOffsetToEveryColumn) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"name","type":"string"}]})");
    const auto whole = Batch(
        {arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8())},
        {Column<arrow::Int64Builder>(std::vector<int64_t>{10, 11, 12, 13}),
         Column<arrow::StringBuilder>(std::vector<std::string>{"a", "b", "c", "d"})});
    const auto batch = whole->Slice(2, 2);
    cpplink::RecordStore store(schema);
    std::string error;
    ASSERT_TRUE(Append(*batch, schema, &store, &error)) << error;
    ASSERT_EQ(store.ids().offsets.size(), 3u);
    EXPECT_EQ(store.ids().Get(0), "12");
    EXPECT_EQ(store.ids().Get(1), "13");
    const auto& name = std::get<cpplink::StringColumn>(store.column(0));
    EXPECT_EQ(Text(name, 0), "c");
    EXPECT_EQ(Text(name, 1), "d");
}

// A pandas `category` arrives dictionary-encoded, which is the store's own shape:
// each distinct value is interned once and the rows are a remap of the indices.
// A category no row uses is not interned, or it would count as a distinct value
// and sit in every dictionary self-join with a frequency of zero.
TEST(BatchLoader, DictionaryColumnsInternEachValueOnce) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"city","type":"string"},
        {"name":"code","type":"string"},
        {"name":"tags","type":"string_list"}]})");

    const auto cities =
        Column<arrow::StringBuilder>(std::vector<std::string>{"paris", "rome", "oslo"});
    const auto city_indices = Column<arrow::Int8Builder>(std::vector<int8_t>{1, 0, 1, 0},
                                                         {true, true, false, true});
    const auto city = std::make_shared<arrow::DictionaryArray>(
        arrow::dictionary(arrow::int8(), arrow::utf8()), city_indices, cities);

    // An integer dictionary spells its digits once.
    const auto codes = Column<arrow::Int32Builder>(std::vector<int32_t>{2000, 90210});
    const auto code_indices =
        Column<arrow::Int32Builder>(std::vector<int32_t>{1, 1, 0, 0});
    const auto code = std::make_shared<arrow::DictionaryArray>(
        arrow::dictionary(arrow::int32(), arrow::int32()), code_indices, codes);

    // Dictionary-encoded list elements go through the same text visitor.
    const auto elements =
        Column<arrow::StringBuilder>(std::vector<std::string>{"x", "y"});
    const auto element_indices =
        Column<arrow::Int16Builder>(std::vector<int16_t>{1, 0, 0, 1, 1});
    const auto element = std::make_shared<arrow::DictionaryArray>(
        arrow::dictionary(arrow::int16(), arrow::utf8()), element_indices, elements);
    const auto tags = std::make_shared<arrow::ListArray>(
        arrow::list(element->type()), 4,
        Column<arrow::Int32Builder>(std::vector<int32_t>{0, 2, 2, 3, 5})
            ->data()
            ->buffers[1],
        element);

    const auto batch =
        Batch({arrow::field("id", arrow::utf8()), arrow::field("city", city->type()),
               arrow::field("code", code->type()), arrow::field("tags", tags->type())},
              {Column<arrow::StringBuilder>(std::vector<std::string>{"a", "b", "c", "d"}),
               city, code, tags});

    ArrowSchema c_schema;
    ASSERT_TRUE(arrow::ExportSchema(*batch->schema(), &c_schema).ok());
    cpplink::ColumnType type;
    EXPECT_TRUE(cpplink::ColumnTypeOf(*c_schema.children[1], &type));
    EXPECT_EQ(type, cpplink::ColumnType::kString);
    EXPECT_TRUE(cpplink::ColumnTypeOf(*c_schema.children[2], &type));
    EXPECT_EQ(type, cpplink::ColumnType::kString) << "an integer dictionary is text";
    EXPECT_EQ(cpplink::ArrowTypeName(*c_schema.children[1]),
              "dictionary<values=string, indices=int8, ordered=0>");
    c_schema.release(&c_schema);

    cpplink::RecordStore store(schema);
    std::string error;
    ASSERT_TRUE(Append(*batch, schema, &store, &error)) << error;
    store.set_num_records(4);
    store.Finalize();

    const auto& city_column = std::get<cpplink::StringColumn>(store.column(0));
    EXPECT_EQ(city_column.dict.Size(), 2u) << "oslo is a category no row holds";
    EXPECT_EQ(Text(city_column, 0), "rome");
    EXPECT_EQ(Text(city_column, 1), "paris");
    EXPECT_EQ(city_column.ids[2], cpplink::kNullId);
    EXPECT_EQ(city_column.ids[1], city_column.ids[3]);
    EXPECT_EQ(city_column.tf[city_column.ids[0]], 1u);

    const auto& code_column = std::get<cpplink::StringColumn>(store.column(1));
    EXPECT_EQ(Text(code_column, 0), "90210");
    EXPECT_EQ(Text(code_column, 3), "2000");
    EXPECT_EQ(code_column.dict.Size(), 2u);

    const auto& tag_column = std::get<cpplink::StringListColumn>(store.column(2));
    const std::vector<uint64_t> offsets = {0, 2, 2, 3, 4};
    EXPECT_EQ(tag_column.offsets, offsets) << "the last cell's y, y is one y";
    EXPECT_EQ(tag_column.dict.Value(tag_column.ids[3]), "y");
}

// polars hands text over as string views: short values inline, long ones in a
// data buffer the view points into.
TEST(BatchLoader, ReadsStringViews) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"name","type":"string"}]})");
    const std::string long_value(40, 'q');
    const auto batch =
        Batch({arrow::field("id", arrow::utf8_view()),
               arrow::field("name", arrow::utf8_view())},
              {Column<arrow::StringViewBuilder>(std::vector<std::string>{"1", "2", "3"}),
               Column<arrow::StringViewBuilder>(
                   std::vector<std::string>{"short", long_value, "short"},
                   {true, true, true})});
    ArrowSchema c_schema;
    ASSERT_TRUE(arrow::ExportSchema(*batch->schema(), &c_schema).ok());
    EXPECT_EQ(cpplink::ArrowTypeName(*c_schema.children[1]), "string_view");
    cpplink::ColumnType type;
    EXPECT_TRUE(cpplink::ColumnTypeOf(*c_schema.children[1], &type));
    EXPECT_EQ(type, cpplink::ColumnType::kString);
    c_schema.release(&c_schema);

    cpplink::RecordStore store(schema);
    std::string error;
    ASSERT_TRUE(Append(*batch, schema, &store, &error)) << error;
    EXPECT_EQ(store.ids().Get(2), "3");
    const auto& name = std::get<cpplink::StringColumn>(store.column(0));
    EXPECT_EQ(Text(name, 0), "short");
    EXPECT_EQ(Text(name, 1), long_value);
    EXPECT_EQ(name.ids[0], name.ids[2]);
}

// A column of nothing but None exports as the null type. It is missing on every
// row for whatever type the schema declares, not an error.
TEST(BatchLoader, NullTypedColumnsAreMissingEverywhere) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"name","type":"string"},
        {"name":"tags","type":"string_list"},
        {"name":"dob","type":"date"},
        {"name":"x","type":"double"},
        {"name":"flag","type":"boolean"}]})");
    const auto nulls = std::make_shared<arrow::NullArray>(2);
    const auto batch =
        Batch({arrow::field("id", arrow::utf8()), arrow::field("name", arrow::null()),
               arrow::field("tags", arrow::null()), arrow::field("dob", arrow::null()),
               arrow::field("x", arrow::null()), arrow::field("flag", arrow::null())},
              {Column<arrow::StringBuilder>(std::vector<std::string>{"1", "2"}), nulls,
               nulls, nulls, nulls, nulls});
    cpplink::RecordStore store(schema);
    std::string error;
    ASSERT_TRUE(Append(*batch, schema, &store, &error)) << error;
    store.set_num_records(2);
    store.Finalize();
    for (size_t c = 0; c < 5; ++c) EXPECT_EQ(store.NullCount(c), 2u) << c;

    // The type is not one a schema can take from the input, though.
    ArrowSchema c_schema;
    ASSERT_TRUE(arrow::ExportSchema(*batch->schema(), &c_schema).ok());
    cpplink::ColumnType type;
    EXPECT_FALSE(cpplink::ColumnTypeOf(*c_schema.children[1], &type));
    EXPECT_EQ(cpplink::ArrowTypeName(*c_schema.children[1]), "null");
    c_schema.release(&c_schema);
}

// Timestamps of every unit are days, as the parquet path has always read them,
// and the type is named the way Arrow prints it.
TEST(BatchLoader, TimestampsAreDays) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"seen","type":"date"}]})");
    arrow::TimestampBuilder builder(arrow::timestamp(arrow::TimeUnit::MICRO, "UTC"),
                                    arrow::default_memory_pool());
    ASSERT_TRUE(builder.Append(3 * 86400000000LL + 5).ok());
    ASSERT_TRUE(builder.AppendNull().ok());
    std::shared_ptr<arrow::Array> seen;
    ASSERT_TRUE(builder.Finish(&seen).ok());
    const auto batch =
        Batch({arrow::field("id", arrow::utf8()), arrow::field("seen", seen->type())},
              {Column<arrow::StringBuilder>(std::vector<std::string>{"1", "2"}), seen});
    ArrowSchema c_schema;
    ASSERT_TRUE(arrow::ExportSchema(*batch->schema(), &c_schema).ok());
    EXPECT_EQ(cpplink::ArrowTypeName(*c_schema.children[1]), "timestamp[us, tz=UTC]");
    EXPECT_EQ(cpplink::ArrowTypeName(*c_schema.children[1]),
              batch->schema()->field(1)->type()->ToString());
    c_schema.release(&c_schema);

    cpplink::RecordStore store(schema);
    std::string error;
    ASSERT_TRUE(Append(*batch, schema, &store, &error)) << error;
    const auto& column = std::get<cpplink::DateColumn>(store.column(0));
    EXPECT_EQ(column.values, (std::vector<int32_t>{3, cpplink::kNullDate}));
}

TEST(BatchLoader, RefusesWhatTheSchemaCannotRead) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"x","type":"string"}]})");
    const auto batch =
        Batch({arrow::field("id", arrow::utf8()), arrow::field("x", arrow::float64())},
              {Column<arrow::StringBuilder>(std::vector<std::string>{"1"}),
               Column<arrow::DoubleBuilder>(std::vector<double>{1.0})});
    cpplink::RecordStore store(schema);
    std::string error;
    EXPECT_FALSE(Append(*batch, schema, &store, &error));
    EXPECT_NE(error.find("x: expected a string or integer column, found double"),
              std::string::npos)
        << error;

    const auto missing = Parse(R"({"unique_id":"id","columns":[
        {"name":"gone","type":"string"}]})");
    cpplink::RecordStore other(missing);
    EXPECT_FALSE(Append(*batch, missing, &other, &error));
    EXPECT_NE(error.find("\"gone\" is not in the input"), std::string::npos) << error;
}

// A stream is drained batch by batch, each released before the next is fetched.
TEST(BatchLoader, DrainsAStream) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"name","type":"string"}]})");
    const auto fields = arrow::schema(
        {arrow::field("id", arrow::int64()), arrow::field("name", arrow::large_utf8())});
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {
        arrow::RecordBatch::Make(
            fields, 2,
            {Column<arrow::Int64Builder>(std::vector<int64_t>{1, 2}),
             Column<arrow::LargeStringBuilder>(std::vector<std::string>{"a", "b"})}),
        arrow::RecordBatch::Make(
            fields, 1,
            {Column<arrow::Int64Builder>(std::vector<int64_t>{3}),
             Column<arrow::LargeStringBuilder>(std::vector<std::string>{"a"})})};
    auto reader = arrow::RecordBatchReader::Make(batches, fields);
    ASSERT_TRUE(reader.ok());
    ArrowArrayStream stream;
    ASSERT_TRUE(arrow::ExportRecordBatchReader(*reader, &stream).ok());

    cpplink::RecordStore store(schema);
    uint64_t rows = 0;
    int count = 0;
    std::string error;
    ASSERT_TRUE(
        cpplink::AppendArrayStream(&stream, schema, &store, &rows, &count, &error))
        << error;
    EXPECT_EQ(stream.release, nullptr) << "the stream is released on return";
    EXPECT_EQ(rows, 3u);
    EXPECT_EQ(count, 2);
    store.set_num_records(3);
    store.Finalize();
    EXPECT_EQ(store.ids().Get(2), "3");
    const auto& name = std::get<cpplink::StringColumn>(store.column(0));
    EXPECT_EQ(name.ids[0], name.ids[2]);
    EXPECT_EQ(name.tf[name.ids[0]], 2u);
}

// The equivalence the design rests on: a table read from its buffers is the
// store the parquet path builds from the same table written to disk, member for
// member, across every column type.
TEST(BatchLoader, MatchesTheParquetPathMemberForMember) {
    const auto schema = Parse(R"({"unique_id":"id","columns":[
        {"name":"name","type":"string"},
        {"name":"postcode","type":"string"},
        {"name":"tags","type":"string_list"},
        {"name":"dob","type":"date"},
        {"name":"x","type":"double"},
        {"name":"flag","type":"boolean"}]})");

    arrow::LargeListBuilder tags(arrow::default_memory_pool(),
                                 std::make_shared<arrow::LargeStringBuilder>());
    auto* tag = static_cast<arrow::LargeStringBuilder*>(tags.value_builder());
    ASSERT_TRUE(tags.Append().ok());
    ASSERT_TRUE(tag->AppendValues({"y", "x", "y"}).ok());
    ASSERT_TRUE(tags.AppendNull().ok());
    ASSERT_TRUE(tags.Append().ok());
    ASSERT_TRUE(tags.Append().ok());
    ASSERT_TRUE(tag->AppendValues({"x"}).ok());
    std::shared_ptr<arrow::Array> tags_array;
    ASSERT_TRUE(tags.Finish(&tags_array).ok());

    arrow::TimestampBuilder dob_builder(arrow::timestamp(arrow::TimeUnit::NANO),
                                        arrow::default_memory_pool());
    ASSERT_TRUE(dob_builder
                    .AppendValues({86400000000000LL * 10, 0, 86400000000000LL * 10,
                                   86400000000000LL * 11},
                                  std::vector<bool>{true, false, true, true})
                    .ok());
    std::shared_ptr<arrow::Array> dob;
    ASSERT_TRUE(dob_builder.Finish(&dob).ok());

    const auto table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::int64()),
                       arrow::field("name", arrow::large_utf8()),
                       arrow::field("postcode", arrow::int32()),
                       arrow::field("tags", arrow::large_list(arrow::large_utf8())),
                       arrow::field("dob", dob->type()),
                       arrow::field("x", arrow::float32()),
                       arrow::field("flag", arrow::boolean())}),
        {Column<arrow::Int64Builder>(std::vector<int64_t>{1, 2, 3, 4}),
         Column<arrow::LargeStringBuilder>(
             std::vector<std::string>{"ann", "bob", "", "ann"},
             {true, true, false, true}),
         Column<arrow::Int32Builder>(std::vector<int32_t>{2000, 0, 2000, 3},
                                     {true, false, true, true}),
         tags_array, dob,
         Column<arrow::FloatBuilder>(std::vector<float>{0.5f, 0.f, 1.5f, 2.5f},
                                     {true, false, true, true}),
         Column<arrow::BooleanBuilder>(std::vector<bool>{true, false, false, true},
                                       {true, true, false, true})});

    const auto dir = std::filesystem::temp_directory_path() /
                     ("cpplink_batch_loader_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "table.parquet").string();
    auto sink = arrow::io::FileOutputStream::Open(path);
    ASSERT_TRUE(sink.ok());
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink,
                                           /*chunk_size=*/3)
                    .ok());

    cpplink::RecordStore from_file(schema);
    std::string error;
    ASSERT_TRUE(cpplink::LoadParquet(path, schema, &from_file, nullptr, &error)) << error;
    std::filesystem::remove_all(dir);

    // The same table, decoded from its buffers in batches of two.
    cpplink::RecordStore from_buffers(schema);
    arrow::TableBatchReader reader(*table);
    reader.set_chunksize(2);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (reader.ReadNext(&batch).ok() && batch != nullptr) {
        ASSERT_TRUE(Append(*batch, schema, &from_buffers, &error)) << error;
    }
    from_buffers.set_num_records(4);
    from_buffers.Finalize();

    ExpectSameStore(from_file, from_buffers);
    EXPECT_EQ(from_buffers.ids().Get(3), "4");
    const auto& dob_column = std::get<cpplink::DateColumn>(from_buffers.column(3));
    EXPECT_EQ(dob_column.values[0], 10);
}
