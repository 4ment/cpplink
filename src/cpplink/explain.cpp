// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/explain.hpp"

#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

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
    if (comparison.strings != nullptr && comparison.lists != nullptr) {
        // A list_contains comparison, whose evidence is one column against the
        // other: printing either alone would leave the reader unable to check the
        // level the waterfall then charges for.
        BoundComparison scalar = comparison;
        scalar.lists = nullptr;
        BoundComparison list = comparison;
        list.strings = nullptr;
        return ValueOf(scalar, row) + " in " + ValueOf(list, row);
    }
    if (comparison.strings != nullptr) {
        // Every string column the comparison names, so an address and its
        // username both show and the level charged can be checked against
        // whichever it read.
        std::string out;
        for (const BoundComparison::StringSlot& slot : comparison.slots) {
            if (!out.empty()) out += " / ";
            const uint32_t id = slot.strings->ids[row];
            out += id == kNullId ? "<null>" : std::string(slot.strings->dict.Value(id));
        }
        return out;
    }
    if (comparison.dates != nullptr) {
        const int32_t value = comparison.dates->values[row];
        if (value == kNullDate) return "<null>";
        return std::to_string(value) + "d";
    }
    if (comparison.booleans != nullptr) {
        const int8_t value = comparison.booleans->values[row];
        if (value == kNullBoolean) return "<null>";
        return value != 0 ? "true" : "false";
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

PairWaterfall BuildPairWaterfall(const RecordStore& store,
                                 const ComparisonSet& comparisons, const Scorer& scorer,
                                 uint64_t a, uint64_t b, const Model* model) {
    PairWaterfall w;
    w.row_a = a;
    w.row_b = b;
    const IdColumn& ids = store.ids();
    if (!ids.offsets.empty()) {
        w.id_a = std::string(ids.Get(a));
        w.id_b = std::string(ids.Get(b));
    }
    w.records = store.NumRecords();
    w.gamma = comparisons.Evaluate(a, b);
    w.prior = scorer.PriorWeight();

    double running = w.prior;
    for (size_t i = 0; i < comparisons.Size(); ++i) {
        const BoundComparison& bound = comparisons.at(i);
        WaterfallStep step;
        step.name = bound.spec->name;
        step.level = comparisons.LevelOf(w.gamma, i);
        step.label = bound.spec->levels[step.level].Describe();
        step.value_a = ValueOf(bound, a);
        step.value_b = ValueOf(bound, b);
        step.m = std::nan("");
        step.u = std::nan("");
        if (model != nullptr && i < model->comparisons.size() &&
            step.level < model->comparisons[i].levels.size()) {
            step.m = model->comparisons[i].levels[step.level].m;
            step.u = model->comparisons[i].levels[step.level].u;
        }
        step.bits = scorer.LevelWeight(i, step.level);
        step.tf = scorer.AdjustmentFor(i, w.gamma, a, b);
        if (step.tf != 0.0) step.frequency = scorer.FrequencyFor(i, w.gamma, a);
        running += step.bits + step.tf;
        step.running = running;
        w.steps.push_back(std::move(step));
    }
    // The two-way corrections, where the model carries any. They are part of the
    // sum the scorer computes, so they have to be part of the ledger that explains
    // it: a waterfall missing them would total something the run never produced.
    for (size_t i = 0; i < scorer.InteractionCount(); ++i) {
        WaterfallInteraction term;
        term.name = scorer.InteractionName(i);
        term.bits = scorer.InteractionBits(i, w.gamma);
        running += term.bits;
        term.running = running;
        w.interactions.push_back(std::move(term));
    }

    w.weight = scorer.Weight(w.gamma, a, b);
    w.probability = ProbabilityForWeight(w.weight);
    w.bracket_low = scorer.BaseWeight(w.gamma) + scorer.DeltaMin(w.gamma);
    w.bracket_high = scorer.BaseWeight(w.gamma) + scorer.DeltaMax(w.gamma);
    w.threshold = scorer.threshold();
    w.zone = scorer.Classify(w.gamma);
    w.emitted = w.weight >= w.threshold;
    return w;
}

void PrintPairWaterfall(const PairWaterfall& w, std::ostream& out) {
    out << "\n"
        << std::left << std::setw(18) << "Comparison" << std::setw(22) << "Level"
        << std::right << std::setw(10) << "bits" << std::setw(10) << "tf" << std::setw(12)
        << "running" << "\n";
    out << std::string(72, '-') << "\n";

    out << std::left << std::setw(18) << "(prior)" << std::setw(22) << "lambda"
        << std::right << std::setw(10) << Signed(w.prior) << std::setw(10) << ""
        << std::setw(12) << Signed(w.prior) << "\n";
    for (const WaterfallStep& step : w.steps) {
        out << std::left << std::setw(18) << Truncate(step.name, 17) << std::setw(22)
            << Truncate(step.label, 21) << std::right << std::setw(10)
            << Signed(step.bits) << std::setw(10)
            << (step.tf != 0.0 ? Signed(step.tf) : std::string("")) << std::setw(12)
            << Signed(step.running) << "\n";
    }
    for (const WaterfallInteraction& term : w.interactions) {
        out << std::left << std::setw(18) << "(interaction)" << std::setw(22)
            << Truncate(term.name, 21) << std::right << std::setw(10) << Signed(term.bits)
            << std::setw(10) << "" << std::setw(12) << Signed(term.running) << "\n";
    }
    out << std::string(72, '-') << "\n";

    out << "Match weight   " << std::fixed << std::setprecision(3) << w.weight
        << " bits    posterior " << std::setprecision(9) << w.probability << "\n";

    // Where the term-frequency moves came from. Without the counts the adjustment
    // is an unexplained number, and this is the report whose job is to explain it.
    bool any = false;
    for (const WaterfallStep& step : w.steps) {
        if (step.tf == 0.0) continue;
        if (!any) {
            out << "\nTerm frequency, for the comparisons that moved the weight:\n";
            any = true;
        }
        const double share = w.records > 0 ? static_cast<double>(step.frequency) /
                                                 static_cast<double>(w.records)
                                           : 0.0;
        out << "  " << std::left << std::setw(18) << Truncate(step.name, 17)
            << std::setw(24) << Truncate(step.value_a, 23) << std::right << std::setw(12)
            << step.frequency << " rows" << std::setw(12) << std::scientific
            << std::setprecision(2) << share << std::setw(10) << std::defaultfloat
            << Signed(step.tf) << " bits\n";
    }

    // The same three-way decision `predict` makes, so a pair can be traced from
    // here to whether it would have been emitted.
    out << "\nPattern bracket  " << std::fixed << std::setprecision(3) << w.bracket_low
        << " to " << w.bracket_high << " bits, against a threshold of " << w.threshold
        << "\n";
    out << "Zone             " << ZoneName(w.zone) << " -- ";
    switch (w.zone) {
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
    out << (w.emitted ? "This pair would be emitted.\n"
                      : "This pair would not be emitted.\n");
}

void PrintPairWaterfall(const RecordStore& store, const ComparisonSet& comparisons,
                        const Scorer& scorer, uint64_t a, uint64_t b, std::ostream& out) {
    PrintPairWaterfall(BuildPairWaterfall(store, comparisons, scorer, a, b), out);
}

namespace {

// JSON has no NaN, and a rate the report was not given is absent rather than null.
void PutRate(nlohmann::json* item, const char* key, double value) {
    if (!std::isnan(value)) (*item)[key] = value;
}

}  // namespace

std::string PairWaterfallJson(const PairWaterfall& w) {
    nlohmann::json root;
    root["row_a"] = w.row_a;
    root["row_b"] = w.row_b;
    if (!w.id_a.empty() || !w.id_b.empty()) {
        root["id_a"] = w.id_a;
        root["id_b"] = w.id_b;
    }
    root["records"] = w.records;
    root["gamma"] = w.gamma;
    root["prior"] = w.prior;
    root["steps"] = nlohmann::json::array();
    for (const WaterfallStep& step : w.steps) {
        nlohmann::json item;
        item["name"] = step.name;
        item["level"] = step.level;
        item["label"] = step.label;
        item["values"] = {step.value_a, step.value_b};
        PutRate(&item, "m", step.m);
        PutRate(&item, "u", step.u);
        item["bits"] = step.bits;
        item["tf"] = step.tf;
        if (step.tf != 0.0) item["frequency"] = step.frequency;
        item["running"] = step.running;
        root["steps"].push_back(std::move(item));
    }
    root["interactions"] = nlohmann::json::array();
    for (const WaterfallInteraction& term : w.interactions) {
        nlohmann::json item;
        item["name"] = term.name;
        item["bits"] = term.bits;
        item["running"] = term.running;
        root["interactions"].push_back(std::move(item));
    }
    root["weight"] = w.weight;
    root["probability"] = w.probability;
    root["bracket"] = {w.bracket_low, w.bracket_high};
    root["threshold"] = w.threshold;
    root["zone"] = ZoneName(w.zone);
    root["emitted"] = w.emitted;
    return root.dump();
}

}  // namespace cpplink
