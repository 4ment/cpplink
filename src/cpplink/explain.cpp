// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/explain.hpp"

#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace cpplink {
namespace {

std::string Bits(uint32_t value, uint8_t width) {
    if (width == 0) return "0";
    std::string out;
    for (int bit = width - 1; bit >= 0; --bit) {
        out.push_back(((value >> bit) & 1u) != 0 ? '1' : '0');
        if (bit % 4 == 0 && bit != 0) out.push_back(' ');
    }
    return out;
}

std::string Hex(uint32_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0') << std::setw(8) << value;
    return out.str();
}

// The stored value of one comparison's columns for one row, as text.
std::string ValueOf(const BoundComparison& comparison, uint64_t row) {
    if (comparison.strings != nullptr) {
        const uint32_t id = comparison.strings->ids[row];
        if (id == kNullId) return "<null>";
        return std::string(comparison.strings->dict.Value(id));
    }
    if (comparison.dates != nullptr) {
        const int32_t value = comparison.dates->values[row];
        if (value == kNullDate) return "<null>";
        return std::to_string(value) + "d";
    }
    if (comparison.numbers != nullptr) {
        const double value = comparison.numbers->values[row];
        if (std::isnan(value)) return "<null>";
        std::ostringstream out;
        out << std::fixed << std::setprecision(4) << value;
        if (comparison.numbers2 != nullptr) {
            out << ", " << comparison.numbers2->values[row];
        }
        return out.str();
    }
    if (comparison.lists != nullptr) {
        const uint64_t begin = comparison.lists->offsets[row];
        const uint64_t end = comparison.lists->offsets[row + 1];
        if (begin == end) return "<empty>";
        std::string out = "{";
        for (uint64_t i = begin; i < end; ++i) {
            if (i > begin) out += " ";
            out += std::string(comparison.lists->dict.Value(comparison.lists->ids[i]));
        }
        return out + "}";
    }
    return "?";
}

// Marked with an ASCII dot rather than an ellipsis: std::setw pads by bytes, and a
// multi-byte marker silently costs the column its alignment.
std::string Truncate(std::string text, size_t width) {
    if (text.size() <= width) return text;
    return text.substr(0, width - 1) + ".";
}

}  // namespace

bool FindRowById(const RecordStore& store, const std::string& id, uint64_t* row) {
    const IdColumn& ids = store.ids();
    if (ids.offsets.empty()) return false;
    for (uint64_t i = 0; i < store.NumRecords(); ++i) {
        if (ids.Get(i) == id) {
            *row = i;
            return true;
        }
    }
    return false;
}

void PrintGammaLayout(const ComparisonSet& comparisons, std::ostream& out) {
    out << std::left << std::setw(20) << "Comparison" << std::right << std::setw(8)
        << "Levels" << std::setw(7) << "Bits" << std::setw(9) << "Shift" << "   "
        << std::left << "Columns" << "\n";
    out << std::string(78, '-') << "\n";
    for (size_t i = 0; i < comparisons.Size(); ++i) {
        const BoundComparison& bound = comparisons.at(i);
        std::string columns;
        for (const std::string& name : bound.spec->columns) {
            if (!columns.empty()) columns += ", ";
            columns += name;
        }
        out << std::left << std::setw(20) << Truncate(bound.spec->name, 19) << std::right
            << std::setw(8) << bound.spec->levels.size() << std::setw(7)
            << static_cast<int>(bound.bits) << std::setw(9)
            << static_cast<int>(bound.shift) << "   " << std::left << columns << "\n";
    }
    out << std::string(78, '-') << "\n";
    out << "Packed width " << static_cast<int>(comparisons.Width()) << " bits of 32.\n";
}

void PrintPairExplanation(const RecordStore& store, const ComparisonSet& comparisons,
                          uint64_t a, uint64_t b, std::ostream& out) {
    const IdColumn& ids = store.ids();
    const bool has_ids = !ids.offsets.empty();
    out << "Pair  " << (has_ids ? std::string(ids.Get(a)) : "row " + std::to_string(a))
        << "  /  " << (has_ids ? std::string(ids.Get(b)) : "row " + std::to_string(b))
        << "\n\n";

    out << std::left << std::setw(18) << "Comparison" << std::setw(24) << "Level"
        << std::right << std::setw(7) << "Index" << "   " << std::left << "Values"
        << "\n";
    out << std::string(100, '-') << "\n";
    for (size_t i = 0; i < comparisons.Size(); ++i) {
        const BoundComparison& bound = comparisons.at(i);
        const uint8_t level = comparisons.EvaluateOne(i, a, b);
        out << std::left << std::setw(18) << Truncate(bound.spec->name, 17)
            << std::setw(24) << Truncate(bound.spec->levels[level].Describe(), 23)
            << std::right << std::setw(7) << static_cast<int>(level) << "   " << std::left
            << Truncate(ValueOf(bound, a), 28) << "  |  "
            << Truncate(ValueOf(bound, b), 28) << "\n";
    }

    const uint32_t gamma = comparisons.Evaluate(a, b);
    out << "\ngamma  " << Hex(gamma) << "   " << Bits(gamma, comparisons.Width())
        << "   (" << static_cast<int>(comparisons.Width()) << " bits)\n";
}

namespace {

// Right-aligned bits with an explicit sign, so a column of contributions reads as
// a ledger rather than as a list of numbers.
std::string Signed(double bits, int precision = 2) {
    std::ostringstream out;
    out << std::showpos << std::fixed << std::setprecision(precision) << bits;
    return out.str();
}

}  // namespace

void PrintPairWaterfall(const RecordStore& store, const ComparisonSet& comparisons,
                        const Scorer& scorer, uint64_t a, uint64_t b, std::ostream& out) {
    const uint32_t gamma = comparisons.Evaluate(a, b);
    const double prior = scorer.PriorWeight();

    out << "\n"
        << std::left << std::setw(18) << "Comparison" << std::setw(22) << "Level"
        << std::right << std::setw(10) << "bits" << std::setw(10) << "tf" << std::setw(12)
        << "running" << "\n";
    out << std::string(72, '-') << "\n";

    double running = prior;
    out << std::left << std::setw(18) << "(prior)" << std::setw(22) << "lambda"
        << std::right << std::setw(10) << Signed(prior) << std::setw(10) << ""
        << std::setw(12) << Signed(running) << "\n";

    for (size_t i = 0; i < comparisons.Size(); ++i) {
        const BoundComparison& bound = comparisons.at(i);
        const uint8_t level = comparisons.LevelOf(gamma, i);
        const double bits = scorer.LevelWeight(i, level);
        const double move = scorer.AdjustmentFor(i, gamma, a);
        running += bits + move;
        out << std::left << std::setw(18) << Truncate(bound.spec->name, 17)
            << std::setw(22) << Truncate(bound.spec->levels[level].Describe(), 21)
            << std::right << std::setw(10) << Signed(bits) << std::setw(10)
            << (move != 0.0 ? Signed(move) : std::string("")) << std::setw(12)
            << Signed(running) << "\n";
    }
    out << std::string(72, '-') << "\n";

    const double weight = scorer.Weight(gamma, a);
    out << "Match weight   " << std::fixed << std::setprecision(3) << weight
        << " bits    posterior " << std::setprecision(9) << ProbabilityForWeight(weight)
        << "\n";

    // Where the term-frequency moves came from. Without the counts the adjustment
    // is an unexplained number, and this is the report whose job is to explain it.
    bool any = false;
    for (size_t i = 0; i < comparisons.Size(); ++i) {
        if (!scorer.HasAdjustment(i)) continue;
        const double move = scorer.AdjustmentFor(i, gamma, a);
        if (move == 0.0) continue;
        if (!any) {
            out << "\nTerm frequency, for the comparisons that agreed exactly:\n";
            any = true;
        }
        const uint32_t frequency = scorer.FrequencyFor(i, a);
        const double share =
            store.NumRecords() > 0
                ? static_cast<double>(frequency) / static_cast<double>(store.NumRecords())
                : 0.0;
        out << "  " << std::left << std::setw(18)
            << Truncate(comparisons.at(i).spec->name, 17) << std::setw(24)
            << Truncate(ValueOf(comparisons.at(i), a), 23) << std::right << std::setw(12)
            << frequency << " rows" << std::setw(12) << std::scientific
            << std::setprecision(2) << share << std::setw(10) << std::defaultfloat
            << Signed(move) << " bits\n";
    }

    // The same three-way decision `predict` makes, so a pair can be traced from
    // here to whether it would have been emitted.
    const Zone zone = scorer.Classify(gamma);
    out << "\nPattern bracket  " << std::fixed << std::setprecision(3)
        << scorer.BaseWeight(gamma) + scorer.DeltaMin(gamma) << " to "
        << scorer.BaseWeight(gamma) + scorer.DeltaMax(gamma) << " bits, against a "
        << "threshold of " << scorer.threshold() << "\n";
    out << "Zone             " << ZoneName(zone) << " -- ";
    switch (zone) {
        case Zone::kDrop:
            out << "no pair with this pattern can clear the threshold, whatever "
                   "values it carries\n";
            break;
        case Zone::kEmit:
            out << "every pair with this pattern clears it, so no term-frequency "
                   "table is consulted\n";
            break;
        case Zone::kCheck:
            out << "the bracket straddles the threshold, so this pattern is scored "
                   "exactly\n";
            break;
        case Zone::kUnreachable:
            out << "no evaluation can produce this pattern\n";
            break;
    }
    out << (weight >= scorer.threshold() ? "This pair would be emitted.\n"
                                         : "This pair would not be emitted.\n");
}

}  // namespace cpplink
