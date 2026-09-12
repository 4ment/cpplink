// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/inspect.hpp"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace cpplink {
namespace {

std::string WithThousands(uint64_t value) {
    std::string digits = std::to_string(value);
    std::string out;
    int count = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (count > 0 && count % 3 == 0) out.push_back(',');
        out.push_back(*it);
        ++count;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

std::string HumanBytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), value < 10.0 ? "%.2f %s" : "%.1f %s", value,
                  units[unit]);
    return buffer;
}

std::string Percent(uint64_t part, uint64_t whole) {
    if (whole == 0) return "-";
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%.2f%%",
                  100.0 * static_cast<double>(part) / static_cast<double>(whole));
    return buffer;
}

// The largest term frequency, as a share of rows. This is what decides whether a
// column can drive rare-value blocking: a column whose top value covers a large
// share of the data has a group that blows up quadratically.
std::string TopValueShare(const RecordStore& store, size_t index) {
    const Column& column = store.column(index);
    const std::vector<uint32_t>* tf = nullptr;
    if (const auto* col = std::get_if<StringColumn>(&column)) tf = &col->tf;
    if (const auto* col = std::get_if<StringListColumn>(&column)) tf = &col->tf;
    if (const auto* col = std::get_if<DateColumn>(&column)) tf = &col->tf;
    if (const auto* col = std::get_if<BooleanColumn>(&column)) tf = &col->tf;
    if (tf == nullptr || tf->empty()) return "-";
    const uint32_t top = *std::max_element(tf->begin(), tf->end());
    return Percent(top, store.NumRecords());
}

// Pairs a rare-value source would emit from this column at the given cap. This is
// the stage 2 cost formula, computable from the term frequencies alone.
std::string RareValuePairs(const RecordStore& store, size_t index, uint32_t cap) {
    const Column& column = store.column(index);
    const std::vector<uint32_t>* tf = nullptr;
    if (const auto* col = std::get_if<StringColumn>(&column)) tf = &col->tf;
    if (const auto* col = std::get_if<StringListColumn>(&column)) tf = &col->tf;
    if (const auto* col = std::get_if<DateColumn>(&column)) tf = &col->tf;
    if (const auto* col = std::get_if<BooleanColumn>(&column)) tf = &col->tf;
    if (tf == nullptr || tf->empty()) return "-";
    uint64_t pairs = 0;
    for (uint32_t count : *tf) {
        if (count >= 2 && count <= cap) {
            pairs += static_cast<uint64_t>(count) * (count - 1) / 2;
        }
    }
    return WithThousands(pairs);
}

std::string MeanListLength(const RecordStore& store, size_t index) {
    const auto* col = std::get_if<StringListColumn>(&store.column(index));
    if (col == nullptr || store.NumRecords() == 0) return "-";
    char buffer[16];
    std::snprintf(
        buffer, sizeof(buffer), "%.2f",
        static_cast<double>(col->ids.size()) / static_cast<double>(store.NumRecords()));
    return buffer;
}

}  // namespace

void PrintInspection(const RecordStore& store, const LoadStats& stats,
                     std::ostream& out) {
    constexpr uint32_t kRareCap = 100;

    out << "Records      " << WithThousands(store.NumRecords()) << "\n"
        << "Row groups   " << stats.row_groups << "\n"
        << "Load time    " << std::fixed << std::setprecision(1) << stats.seconds << " s";
    if (stats.seconds > 0.0) {
        out << "  ("
            << WithThousands(
                   static_cast<uint64_t>(static_cast<double>(stats.rows) / stats.seconds))
            << " rows/s)";
    }
    out << "\n\n";

    out << std::left << std::setw(18) << "Column" << std::setw(14) << "Type" << std::right
        << std::setw(14) << "Distinct" << std::setw(10) << "Null" << std::setw(10)
        << "Top val" << std::setw(9) << "Mean len" << std::setw(18) << "Rare pairs"
        << "\n";
    out << std::string(93, '-') << "\n";
    for (size_t i = 0; i < store.NumColumns(); ++i) {
        const ColumnSpec& spec = store.schema().columns[i];
        const uint32_t distinct = store.DistinctValues(i);
        out << std::left << std::setw(18) << spec.name << std::setw(14)
            << ColumnTypeName(spec.type) << std::right << std::setw(14)
            << (HasTermFrequencies(spec.type) ? WithThousands(distinct) : "-")
            << std::setw(10) << Percent(store.NullCount(i), store.NumRecords())
            << std::setw(10) << TopValueShare(store, i) << std::setw(9)
            << MeanListLength(store, i) << std::setw(18)
            << RareValuePairs(store, i, kRareCap) << "\n";
    }
    out << "\n\"Rare pairs\" is the candidate count a rare-value source would emit "
           "from that\ncolumn alone at a frequency cap of "
        << kRareCap << ", from the term frequencies only.\n\n";

    const MemoryReport report = store.Memory();
    out << std::left << std::setw(34) << "Structure" << std::right << std::setw(12)
        << "Bytes" << "   " << std::left << "Basis" << "\n";
    out << std::string(93, '-') << "\n";
    for (const MemoryLine& line : report.lines) {
        if (line.bytes == 0) continue;
        out << std::left << std::setw(34) << line.structure << std::right << std::setw(12)
            << HumanBytes(line.bytes) << "   " << std::left << line.basis << "\n";
    }
    out << std::string(93, '-') << "\n";
    out << std::left << std::setw(34) << "Resident total" << std::right << std::setw(12)
        << HumanBytes(report.Total());
    if (store.NumRecords() > 0) {
        out << "   " << std::left
            << (std::to_string(report.Total() / store.NumRecords()) +
                " bytes per record");
    }
    out << "\n";
}

}  // namespace cpplink
