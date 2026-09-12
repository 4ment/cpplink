// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/record_store.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "cpplink/derive.hpp"

namespace cpplink {
namespace {

Column MakeColumn(ColumnType type) {
    switch (type) {
        case ColumnType::kString:
            return StringColumn{};
        case ColumnType::kStringList:
            return StringListColumn{};
        case ColumnType::kDate:
            return DateColumn{};
        case ColumnType::kDouble:
            return DoubleColumn{};
        case ColumnType::kBoolean:
            return BooleanColumn{};
    }
    return StringColumn{};
}

}  // namespace

void IdColumn::Append(std::string_view value) {
    if (offsets.empty()) offsets.push_back(0);
    text.insert(text.end(), value.begin(), value.end());
    offsets.push_back(text.size());
}

std::string_view IdColumn::Get(uint64_t row) const {
    const uint64_t begin = offsets[row];
    const uint64_t end = offsets[row + 1];
    return std::string_view(text.data() + begin, end - begin);
}

uint64_t IdColumn::BytesUsed() const {
    return static_cast<uint64_t>(text.capacity()) +
           static_cast<uint64_t>(offsets.capacity()) * sizeof(uint64_t);
}

uint64_t MemoryReport::Total() const {
    uint64_t total = 0;
    for (const MemoryLine& line : lines) total += line.bytes;
    return total;
}

const char* PairModeName(PairMode mode) {
    switch (mode) {
        case PairMode::kAll:
            return "all pairs";
        case PairMode::kCrossDataset:
            return "cross-dataset pairs";
    }
    return "unknown";
}

RecordStore::RecordStore(Schema schema) : schema_(std::move(schema)) {
    columns_.reserve(schema_.columns.size());
    for (const ColumnSpec& spec : schema_.columns) {
        columns_.push_back(MakeColumn(spec.type));
    }
}

void RecordStore::Finalize() {
    // Derived columns are filled in before anything is counted, so a phonetic key
    // or a normalised name carries term frequencies, blocks and compares exactly
    // as a column read from the file does. Doing it here rather than in the loader
    // is what makes that true of every path that builds a store.
    BuildDerivedColumns(this);

    for (size_t i = 0; i < columns_.size(); ++i) {
        if (!HasTermFrequencies(schema_.columns[i].type)) continue;

        if (auto* col = std::get_if<StringColumn>(&columns_[i])) {
            col->tf.assign(col->dict.Size(), 0);
            for (uint32_t id : col->ids) {
                if (id != kNullId) ++col->tf[id];
            }
            col->dict.ReleaseIndex();
        } else if (auto* col = std::get_if<StringListColumn>(&columns_[i])) {
            col->tf.assign(col->dict.Size(), 0);
            for (uint32_t id : col->ids) ++col->tf[id];
            col->dict.ReleaseIndex();
        } else if (auto* col = std::get_if<DateColumn>(&columns_[i])) {
            int32_t low = 0;
            int32_t high = 0;
            bool seen = false;
            for (int32_t value : col->values) {
                if (value == kNullDate) continue;
                if (!seen) {
                    low = high = value;
                    seen = true;
                } else {
                    low = std::min(low, value);
                    high = std::max(high, value);
                }
            }
            if (!seen) {
                col->tf_origin = 0;
                col->tf.clear();
                continue;
            }
            col->tf_origin = low;
            col->tf.assign(static_cast<size_t>(high - low) + 1, 0);
            for (int32_t value : col->values) {
                if (value != kNullDate) ++col->tf[static_cast<size_t>(value - low)];
            }
        } else if (auto* col = std::get_if<BooleanColumn>(&columns_[i])) {
            col->tf.assign(2, 0);
            for (int8_t value : col->values) {
                if (value != kNullBoolean) ++col->tf[static_cast<size_t>(value)];
            }
        }
    }
}

uint32_t RecordStore::DistinctValues(size_t index) const {
    const Column& column = columns_[index];
    if (const auto* col = std::get_if<StringColumn>(&column)) return col->dict.Size();
    if (const auto* col = std::get_if<StringListColumn>(&column)) return col->dict.Size();
    // A dense table holds every value in its range, seen or not, so distinct is
    // the entries that were.
    const std::vector<uint32_t>* dense = nullptr;
    if (const auto* col = std::get_if<DateColumn>(&column)) dense = &col->tf;
    if (const auto* col = std::get_if<BooleanColumn>(&column)) dense = &col->tf;
    if (dense != nullptr) {
        uint32_t distinct = 0;
        for (uint32_t count : *dense) {
            if (count > 0) ++distinct;
        }
        return distinct;
    }
    return 0;  // kDouble is not counted.
}

uint64_t RecordStore::NullCount(size_t index) const {
    const Column& column = columns_[index];
    uint64_t nulls = 0;
    if (const auto* col = std::get_if<StringColumn>(&column)) {
        for (uint32_t id : col->ids) {
            if (id == kNullId) ++nulls;
        }
    } else if (const auto* col = std::get_if<StringListColumn>(&column)) {
        for (uint64_t row = 0; row + 1 < col->offsets.size(); ++row) {
            if (col->offsets[row + 1] == col->offsets[row]) ++nulls;
        }
    } else if (const auto* col = std::get_if<DateColumn>(&column)) {
        for (int32_t value : col->values) {
            if (value == kNullDate) ++nulls;
        }
    } else if (const auto* col = std::get_if<DoubleColumn>(&column)) {
        for (double value : col->values) {
            if (std::isnan(value)) ++nulls;
        }
    } else if (const auto* col = std::get_if<BooleanColumn>(&column)) {
        for (int8_t value : col->values) {
            if (value == kNullBoolean) ++nulls;
        }
    }
    return nulls;
}

MemoryReport RecordStore::Memory() const {
    MemoryReport report;
    if (!ids_.offsets.empty()) {
        report.lines.push_back(
            {schema_.unique_id + " (ids)", ids_.BytesUsed(), "arena, no index"});
    }
    for (size_t i = 0; i < columns_.size(); ++i) {
        const std::string& name = schema_.columns[i].name;
        const Column& column = columns_[i];
        if (const auto* col = std::get_if<StringColumn>(&column)) {
            report.lines.push_back({name + " dictionary", col->dict.BytesUsed(),
                                    std::to_string(col->dict.Size()) + " values"});
            report.lines.push_back({name + " ids",
                                    static_cast<uint64_t>(col->ids.capacity()) * 4,
                                    "u32 per record"});
            report.lines.push_back({name + " tf",
                                    static_cast<uint64_t>(col->tf.capacity()) * 4,
                                    "u32 per value"});
        } else if (const auto* col = std::get_if<StringListColumn>(&column)) {
            report.lines.push_back({name + " dictionary", col->dict.BytesUsed(),
                                    std::to_string(col->dict.Size()) + " values"});
            report.lines.push_back({name + " ids",
                                    static_cast<uint64_t>(col->ids.capacity()) * 4,
                                    std::to_string(col->ids.size()) + " entries, CSR"});
            report.lines.push_back({name + " offsets",
                                    static_cast<uint64_t>(col->offsets.capacity()) * 8,
                                    "u64 per record"});
            report.lines.push_back({name + " tf",
                                    static_cast<uint64_t>(col->tf.capacity()) * 4,
                                    "u32 per value"});
        } else if (const auto* col = std::get_if<DateColumn>(&column)) {
            report.lines.push_back({name + " values",
                                    static_cast<uint64_t>(col->values.capacity()) * 4,
                                    "i32 days per record"});
            report.lines.push_back({name + " tf",
                                    static_cast<uint64_t>(col->tf.capacity()) * 4,
                                    "dense over the observed range"});
        } else if (const auto* col = std::get_if<DoubleColumn>(&column)) {
            report.lines.push_back({name + " values",
                                    static_cast<uint64_t>(col->values.capacity()) * 8,
                                    "f64 per record, no tf"});
        } else if (const auto* col = std::get_if<BooleanColumn>(&column)) {
            report.lines.push_back({name + " values",
                                    static_cast<uint64_t>(col->values.capacity()),
                                    "i8 per record, tf is two counts"});
        }
    }
    return report;
}

void RecordStore::set_datasets(std::vector<uint64_t> starts) {
    dataset_starts_ = std::move(starts);
}

double RecordStore::PairSpace(PairMode mode) const {
    const double records = static_cast<double>(num_records_);
    const double all = records * (records - 1.0) / 2.0;
    if (mode == PairMode::kAll) return all;
    // Cross-dataset pairs are what is left of the triangle once each input's own
    // triangle is taken out of it.
    double within = 0.0;
    for (size_t d = 0; d < NumDatasets(); ++d) {
        const double size = static_cast<double>(DatasetEnd(d) - DatasetStart(d));
        within += size * (size - 1.0) / 2.0;
    }
    return all - within;
}

}  // namespace cpplink
