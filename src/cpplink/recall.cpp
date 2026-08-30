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

}  // namespace cpplink
