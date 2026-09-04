// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/recall.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cpplink {
namespace {

std::string Percent(uint64_t part, uint64_t whole) {
    if (whole == 0) return "-";
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%.2f%%",
                  100.0 * static_cast<double>(part) / static_cast<double>(whole));
    return buffer;
}

// Marked with an ASCII dot rather than an ellipsis: std::setw pads by bytes, and a
// multi-byte marker silently costs the column its alignment.
std::string Truncate(std::string text, size_t width) {
    if (text.size() <= width) return text;
    return text.substr(0, width - 1) + ".";
}

}  // namespace

bool LoadTruthPairs(const std::string& path, const RecordStore& store, TruthPairs* truth,
                    std::string* error) {
    std::ifstream in(path);
    if (!in) {
        *error = "cannot open truth file: " + path;
        return false;
    }

    std::vector<std::pair<std::string, std::string>> named;
    std::unordered_map<std::string, uint32_t> wanted;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            if (line.rfind("id_a", 0) == 0) continue;  // header
        }
        const size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        std::string a = line.substr(0, comma);
        std::string b = line.substr(comma + 1);
        while (!b.empty() && (b.back() == '\r' || b.back() == '\n')) b.pop_back();
        wanted.emplace(a, kNoRank);
        wanted.emplace(b, kNoRank);
        named.emplace_back(std::move(a), std::move(b));
    }
    truth->lines = named.size();

    const IdColumn& ids = store.ids();
    if (ids.offsets.empty()) {
        *error = "the schema has no unique_id, so truth pairs cannot be resolved";
        return false;
    }
    for (uint64_t row = 0; row < store.NumRecords(); ++row) {
        auto it = wanted.find(std::string(ids.Get(row)));
        if (it != wanted.end() && it->second == kNoRank) {
            it->second = static_cast<uint32_t>(row);
        }
    }

    truth->rows.reserve(named.size());
    for (const auto& pair : named) {
        const uint32_t a = wanted[pair.first];
        const uint32_t b = wanted[pair.second];
        if (a == kNoRank || b == kNoRank || a == b) {
            ++truth->unresolved;
            continue;
        }
        truth->rows.emplace_back(a, b);
    }
    return true;
}

void PrintRecallReport(const BlockingPlan& plan, const TruthPairs& truth,
                       std::ostream& out) {
    const uint64_t total = truth.rows.size();
    out << "Known pairs  " << total << " resolved";
    if (truth.unresolved > 0) {
        out << ", " << truth.unresolved << " unresolved";
    }
    out << "\n\n";

    std::vector<uint64_t> found(plan.Size(), 0);
    uint64_t union_found = 0;
    // Attribution to the first source that produces a pair, which is also the
    // source that would emit it: later sources are only credited with what they
    // add, not with what they duplicate.
    std::vector<uint64_t> unique_credit(plan.Size(), 0);
    for (const auto& pair : truth.rows) {
        bool any = false;
        for (size_t s = 0; s < plan.Size(); ++s) {
            if (!plan.Produces(s, pair.first, pair.second)) continue;
            ++found[s];
            if (!any) ++unique_credit[s];
            any = true;
        }
        if (any) ++union_found;
    }

    out << std::left << std::setw(30) << "Source" << std::right << std::setw(12)
        << "Found" << std::setw(10) << "Recall" << std::setw(12) << "First to"
        << "\n";
    out << std::string(64, '-') << "\n";
    for (size_t s = 0; s < plan.Size(); ++s) {
        out << std::left << std::setw(30) << Truncate(plan.at(s).name, 29) << std::right
            << std::setw(12) << found[s] << std::setw(10) << Percent(found[s], total)
            << std::setw(12) << unique_credit[s] << "\n";
    }
    out << std::string(64, '-') << "\n";
    out << std::left << std::setw(30) << "Union" << std::right << std::setw(12)
        << union_found << std::setw(10) << Percent(union_found, total) << "\n";
    if (union_found < total) {
        out << "\n"
            << (total - union_found)
            << " known pairs are reachable by no source. No amount of scoring "
               "recovers\nthem: they are never generated as candidates.\n";
    }
}

namespace {

// The key one value source would give a row, plus how many records share it.
// Returns false when the column is null on that row, which is the case that has
// to be told apart from "the values differ".
bool KeyAndFrequency(const BoundSource& source, uint64_t row, uint64_t* key,
                     uint32_t* frequency) {
    if (source.strings != nullptr) {
        const uint32_t id = source.strings->ids[row];
        if (id == kNullId) return false;
        *key = source.kind == SourceKind::kMinHash ? source.band_key[id] : id;
        *frequency = source.strings->tf[id];
        return true;
    }
    if (source.dates != nullptr) {
        const int32_t value = source.dates->values[row];
        if (value == kNullDate) return false;
        // Equality is all this needs, so the raw value serves as the key.
        *key = static_cast<uint64_t>(static_cast<int64_t>(value));
        *frequency =
            source.dates->tf[static_cast<size_t>(value - source.dates->tf_origin)];
        return true;
    }
    return false;
}

MissReason Classify(const BlockingPlan& plan, size_t index, uint64_t a, uint64_t b,
                    uint64_t* gap_needed) {
    const BoundSource& source = plan.at(index);
    if (plan.Produces(index, a, b)) return MissReason::kProduced;

    if (source.kind == SourceKind::kSortedNeighbourhood) {
        const uint32_t rank_a = source.rank[a];
        const uint32_t rank_b = source.rank[b];
        if (rank_a == kNoRank || rank_b == kNoRank) return MissReason::kNull;
        *gap_needed = rank_a > rank_b ? rank_a - rank_b : rank_b - rank_a;
        return MissReason::kOutsideWindow;
    }

    uint64_t key_a = 0;
    uint64_t key_b = 0;
    uint32_t frequency_a = 0;
    uint32_t frequency_b = 0;
    if (!KeyAndFrequency(source, a, &key_a, &frequency_a) ||
        !KeyAndFrequency(source, b, &key_b, &frequency_b)) {
        return MissReason::kNull;
    }
    if (key_a != key_b) return MissReason::kDisagreed;
    // The values match, so the only thing left that can have excluded them is the
    // rare-value cap.
    return MissReason::kTooCommon;
}

}  // namespace

const char* MissReasonName(MissReason reason) {
    switch (reason) {
        case MissReason::kProduced:
            return "produced";
        case MissReason::kNull:
            return "null";
        case MissReason::kDisagreed:
            return "disagreed";
        case MissReason::kTooCommon:
            return "too common";
        case MissReason::kOutsideWindow:
            return "outside window";
    }
    return "?";
}

void DiagnoseMisses(const BlockingPlan& plan, const ComparisonSet& comparisons,
                    const TruthPairs& truth, MissReport* report) {
    constexpr size_t kReasons = 5;
    report->truth_pairs = truth.rows.size();
    report->reasons.assign(plan.Size(), std::vector<uint64_t>(kReasons, 0));

    report->agreement.clear();
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        ColumnAgreement entry;
        entry.comparison = comparisons.at(c).spec->name;
        for (size_t f = 0; f < plan.Size() && !entry.blocked; ++f) {
            for (const std::string& column : comparisons.at(c).spec->columns) {
                if (plan.at(f).column == column) {
                    entry.blocked = true;
                    break;
                }
            }
        }
        report->agreement.push_back(std::move(entry));
    }

    std::vector<std::vector<uint64_t>> gaps(plan.Size());
    for (const auto& pair : truth.rows) {
        const uint32_t a = pair.first;
        const uint32_t b = pair.second;
        if (plan.ProducedByAny(a, b)) continue;
        ++report->missed;

        bool cap = false;
        bool window = false;
        for (size_t index = 0; index < plan.Size(); ++index) {
            uint64_t gap = 0;
            const MissReason reason = Classify(plan, index, a, b, &gap);
            ++report->reasons[index][static_cast<size_t>(reason)];
            if (reason == MissReason::kTooCommon) cap = true;
            if (reason == MissReason::kOutsideWindow) {
                gaps[index].push_back(gap);
                report->widest_window_needed =
                    std::max(report->widest_window_needed, gap);
                // Any window can be widened until it covers the file, so this only
                // counts as a fix while the widening stays within reason.
                if (gap <= plan.at(index).window * kPlausibleWindowFactor) window = true;
            }
        }
        // Cheapest fix first, and the three are exclusive so they sum to `missed`.
        if (cap) {
            ++report->fixable_by_cap;
            continue;
        }
        if (window) {
            ++report->fixable_by_window;
            continue;
        }
        ++report->unreachable;

        // Only the genuinely unreachable pairs are worth asking "what is left".
        bool anything = false;
        for (size_t c = 0; c < comparisons.Size(); ++c) {
            const uint8_t level = comparisons.EvaluateOne(c, a, b);
            const LevelType type = comparisons.at(c).spec->levels[level].type;
            if (type == LevelType::kExact) {
                ++report->agreement[c].exact;
                anything = true;
            } else if (type != LevelType::kNull && type != LevelType::kElse) {
                ++report->agreement[c].fuzzy;
                anything = true;
            }
        }
        if (!anything) {
            ++report->agreed_on_nothing;
            if (report->examples.size() < report->example_limit) {
                report->examples.emplace_back(a, b);
            }
        }
    }

    // What each plausible widening would actually buy, priced in candidate pairs.
    for (size_t index = 0; index < plan.Size(); ++index) {
        if (plan.at(index).kind != SourceKind::kSortedNeighbourhood) continue;
        const uint64_t rows = plan.at(index).order.size();
        const uint64_t current = plan.at(index).window;
        if (current == 0 || rows == 0) continue;
        for (const uint64_t factor :
             {uint64_t{2}, uint64_t{5}, uint64_t{10}, uint64_t{100}, uint64_t{1000}}) {
            const uint64_t window = std::min(current * factor, rows - 1);
            WindowGain gain;
            gain.source = plan.at(index).name;
            gain.window = window;
            for (const uint64_t gap : gaps[index]) {
                if (gap <= window) ++gain.recovered;
            }
            // Sum over start positions of min(window, rows - 1 - i).
            gain.candidate_pairs = rows > window
                                       ? rows * window - window * (window + 1) / 2
                                       : rows * (rows - 1) / 2;
            report->window_gains.push_back(std::move(gain));
            if (window >= rows - 1) break;
        }
    }
}

void PrintMissReport(const BlockingPlan& plan, const MissReport& report,
                     std::ostream& out) {
    out << "\nMissed " << report.missed << " of " << report.truth_pairs
        << " known pairs (" << Percent(report.missed, report.truth_pairs) << ")\n";
    if (report.missed == 0) return;

    out << "\nWhy each source missed them\n";
    out << std::left << std::setw(30) << "Source" << std::right << std::setw(12) << "null"
        << std::setw(12) << "disagreed" << std::setw(12) << "too common" << std::setw(16)
        << "outside window" << "\n";
    out << std::string(82, '-') << "\n";
    for (size_t index = 0; index < plan.Size(); ++index) {
        const auto& row = report.reasons[index];
        const bool window = plan.at(index).kind == SourceKind::kSortedNeighbourhood;
        out << std::left << std::setw(30) << Truncate(plan.at(index).name, 29)
            << std::right << std::setw(12) << row[static_cast<size_t>(MissReason::kNull)]
            << std::setw(12)
            << (window ? std::string("-")
                       : std::to_string(row[static_cast<size_t>(MissReason::kDisagreed)]))
            << std::setw(12)
            << (window ? std::string("-")
                       : std::to_string(row[static_cast<size_t>(MissReason::kTooCommon)]))
            << std::setw(16)
            << (window
                    ? std::to_string(row[static_cast<size_t>(MissReason::kOutsideWindow)])
                    : std::string("-"))
            << "\n";
    }
    out << std::string(82, '-') << "\n";

    out << "\nWhat would reach them\n";
    out << "  raising a frequency cap   " << std::setw(10) << report.fixable_by_cap
        << "  (" << Percent(report.fixable_by_cap, report.missed) << ")\n";
    out << "  widening a window         " << std::setw(10) << report.fixable_by_window
        << "  (" << Percent(report.fixable_by_window, report.missed) << ")  within "
        << kPlausibleWindowFactor << "x the current window\n";
    out << "  neither                   " << std::setw(10) << report.unreachable << "  ("
        << Percent(report.unreachable, report.missed)
        << ")  <- needs a signal the plan does not carry\n";

    if (!report.window_gains.empty()) {
        out << "\nWhat widening a window would buy, and cost\n";
        out << std::left << std::setw(30) << "Source" << std::right << std::setw(12)
            << "window" << std::setw(12) << "recovers" << std::setw(20)
            << "candidate pairs" << "\n";
        out << std::string(74, '-') << "\n";
        for (const WindowGain& gain : report.window_gains) {
            out << std::left << std::setw(30) << Truncate(gain.source, 29) << std::right
                << std::setw(12) << gain.window << std::setw(12) << gain.recovered
                << std::setw(20) << gain.candidate_pairs << "\n";
        }
        out << std::string(74, '-') << "\n";
        out << "  the widest gap any missed pair needs is " << report.widest_window_needed
            << "\n";
    }

    if (report.unreachable == 0) return;
    out << "\nOf those " << report.unreachable
        << ", what still agrees (a column agreeing here and not blocked on is "
           "free recall)\n";
    out << std::left << std::setw(22) << "Comparison" << std::right << std::setw(12)
        << "exact" << std::setw(10) << "share" << std::setw(12) << "fuzzy"
        << std::setw(10) << "share" << std::setw(12) << "blocked?" << "\n";
    out << std::string(78, '-') << "\n";
    for (const ColumnAgreement& entry : report.agreement) {
        if (entry.exact == 0 && entry.fuzzy == 0) continue;
        out << std::left << std::setw(22) << Truncate(entry.comparison, 21) << std::right
            << std::setw(12) << entry.exact << std::setw(10)
            << Percent(entry.exact, report.unreachable) << std::setw(12) << entry.fuzzy
            << std::setw(10) << Percent(entry.fuzzy, report.unreachable) << std::setw(12)
            << (entry.blocked ? "yes" : "no") << "\n";
    }
    out << std::string(78, '-') << "\n";
    out << "  agree on nothing at all   " << report.agreed_on_nothing << "  ("
        << Percent(report.agreed_on_nothing, report.unreachable)
        << ") -- no column-wise signal exists for these\n";
    if (!report.examples.empty()) {
        out << "\nSome of them, to check with `explain`:\n";
        for (const auto& pair : report.examples) {
            out << "  rows " << pair.first << "," << pair.second << "\n";
        }
    }
}

}  // namespace cpplink
