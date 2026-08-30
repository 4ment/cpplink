// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/explain_blocking.hpp"

#include <algorithm>
#include <cstdio>
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

// Marked with an ASCII dot rather than an ellipsis: std::setw pads by bytes, and a
// multi-byte marker silently costs the column its alignment.
std::string Truncate(std::string text, size_t width) {
    if (text.size() <= width) return text;
    return text.substr(0, width - 1) + ".";
}

}  // namespace

void PrintBlockingReport(const BlockingPlan& plan, const RecordStore& store,
                         bool count_union, std::ostream& out) {
    const uint64_t rows = store.NumRecords();
    const uint64_t all_pairs = rows * (rows - 1) / 2;
    out << "Records          " << WithThousands(rows) << "\n"
        << "Pairs unblocked  " << WithThousands(all_pairs) << "\n\n";

    out << std::left << std::setw(30) << "Source" << std::setw(11) << "EM-safe"
        << std::right << std::setw(20) << "Candidate pairs" << std::setw(16)
        << "Largest group" << "\n";
    out << std::string(77, '-') << "\n";

    uint64_t sum = 0;
    for (size_t s = 0; s < plan.Size(); ++s) {
        const uint64_t pairs = plan.CountPairs(s);
        sum += pairs;
        out << std::left << std::setw(30) << Truncate(plan.at(s).name, 29)
            << std::setw(11) << (plan.at(s).em_safe ? "yes" : "no") << std::right
            << std::setw(20) << WithThousands(pairs) << std::setw(16)
            << WithThousands(plan.LargestGroup(s)) << "\n";
    }
    out << std::string(77, '-') << "\n";
    out << std::left << std::setw(41) << "Sum over sources" << std::right << std::setw(20)
        << WithThousands(sum) << "\n";

    if (count_union) {
        const uint64_t united = plan.CountUnion();
        out << std::left << std::setw(41) << "Union, deduplicated" << std::right
            << std::setw(20) << WithThousands(united) << "\n";
        if (sum > 0) {
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "%.1f%% of the sum",
                          100.0 * static_cast<double>(united) / static_cast<double>(sum));
            out << std::left << std::setw(41) << "" << std::right << std::setw(20)
                << buffer << "\n";
        }
    }

    const double reduction =
        all_pairs > 0 ? static_cast<double>(sum) / static_cast<double>(all_pairs) : 0.0;
    out << "\nCounts are exact and come from the term frequencies, with no pairs "
           "enumerated.\nThe sum bounds the deduplicated union from above";
    if (!count_union) out << "; pass --count for the union";
    out << ".\nBlocking keeps " << std::scientific << std::setprecision(2) << reduction
        << " of all possible pairs.\n";
}

}  // namespace cpplink
