// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/record_store.hpp"

#include <cmath>
#include <limits>
#include <string>

#include <gtest/gtest.h>

namespace {

cpplink::Schema MakeSchema() {
    cpplink::Schema schema;
    schema.unique_id = "id";
    schema.columns = {
        {"surname", cpplink::ColumnType::kString},
        {"dob", cpplink::ColumnType::kDate},
        {"latitude", cpplink::ColumnType::kDouble},
        {"tokens", cpplink::ColumnType::kStringList},
    };
    return schema;
}

TEST(RecordStoreTest, CountsTermFrequenciesForCountableColumns) {
    cpplink::RecordStore store(MakeSchema());
    auto& surname = std::get<cpplink::StringColumn>(store.mutable_column(0));
    const uint32_t smith = surname.dict.Intern("smith");
    const uint32_t jones = surname.dict.Intern("jones");
    surname.ids = {smith, smith, jones, cpplink::kNullId, smith};
    store.set_num_records(5);
    store.Finalize();

    ASSERT_EQ(surname.tf.size(), 2u);
    EXPECT_EQ(surname.tf[smith], 3u);
    EXPECT_EQ(surname.tf[jones], 1u);
    EXPECT_EQ(store.DistinctValues(0), 2u);
    EXPECT_EQ(store.NullCount(0), 1u);
}

TEST(RecordStoreTest, DateFrequenciesAreDenseOverTheObservedRange) {
    cpplink::RecordStore store(MakeSchema());
    auto& dob = std::get<cpplink::DateColumn>(store.mutable_column(1));
    dob.values = {100, 102, 100, cpplink::kNullDate};
    store.set_num_records(4);
    store.Finalize();

    EXPECT_EQ(dob.tf_origin, 100);
    ASSERT_EQ(dob.tf.size(), 3u);  // 100, 101, 102
    EXPECT_EQ(dob.tf[0], 2u);
    EXPECT_EQ(dob.tf[1], 0u);
    EXPECT_EQ(dob.tf[2], 1u);
    EXPECT_EQ(store.DistinctValues(1), 2u);  // 101 was never seen
    EXPECT_EQ(store.NullCount(1), 1u);
}

// Exact agreement between two doubles is not a discrete event worth counting, so
// kDouble carries no term frequencies and cannot drive rare-value blocking.
TEST(RecordStoreTest, DoublesCarryNoTermFrequencies) {
    cpplink::RecordStore store(MakeSchema());
    auto& latitude = std::get<cpplink::DoubleColumn>(store.mutable_column(2));
    latitude.values = {-33.8, -33.8, std::numeric_limits<double>::quiet_NaN()};
    store.set_num_records(3);
    store.Finalize();

    EXPECT_EQ(store.DistinctValues(2), 0u);
    EXPECT_EQ(store.NullCount(2), 1u);
}

TEST(RecordStoreTest, ListFrequenciesCountRowsAndEmptyRowsAreNull) {
    cpplink::RecordStore store(MakeSchema());
    auto& tokens = std::get<cpplink::StringListColumn>(store.mutable_column(3));
    const uint32_t king = tokens.dict.Intern("king");
    const uint32_t street = tokens.dict.Intern("street");
    tokens.offsets = {0, 2, 3, 3};  // rows: {king,street}, {street}, {}
    tokens.ids = {king, street, street};
    store.set_num_records(3);
    store.Finalize();

    ASSERT_EQ(tokens.tf.size(), 2u);
    EXPECT_EQ(tokens.tf[king], 1u);
    EXPECT_EQ(tokens.tf[street], 2u);
    EXPECT_EQ(store.NullCount(3), 1u);
}

TEST(IdColumnTest, StoresIdentifiersWithoutAnIndex) {
    cpplink::IdColumn ids;
    ids.Append("r0");
    ids.Append("r1234");
    ids.Append("");
    EXPECT_EQ(ids.Get(0), "r0");
    EXPECT_EQ(ids.Get(1), "r1234");
    EXPECT_EQ(ids.Get(2), "");
    EXPECT_GT(ids.BytesUsed(), 0u);
}

TEST(RecordStoreTest, MemoryReportCoversEveryColumn) {
    cpplink::RecordStore store(MakeSchema());
    auto& surname = std::get<cpplink::StringColumn>(store.mutable_column(0));
    surname.ids.assign(10, surname.dict.Intern("smith"));
    std::get<cpplink::DateColumn>(store.mutable_column(1)).values.assign(10, 0);
    std::get<cpplink::DoubleColumn>(store.mutable_column(2)).values.assign(10, 0.0);
    store.set_num_records(10);
    store.Finalize();

    const cpplink::MemoryReport report = store.Memory();
    EXPECT_GT(report.Total(), 0u);
    bool saw_latitude = false;
    for (const cpplink::MemoryLine& line : report.lines) {
        if (line.structure.find("latitude") != std::string::npos) saw_latitude = true;
    }
    EXPECT_TRUE(saw_latitude);
}

}  // namespace
