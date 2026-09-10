// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/explain_blocking.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

#include "cpplink/format.hpp"

namespace cpplink {
namespace {}  // namespace

void PrintBlockingReport(const BlockingPlan& plan, const RecordStore& store,
                         bool count_union, std::ostream& out) {
    const uint64_t rows = store.NumRecords();
    // The unblocked pair count is what the mode admits, not the whole triangle: in
    // link mode the reduction is measured against the cross-product, so the number
    // blocking is compared to is the number it is actually competing with.
    const uint64_t all_pairs = static_cast<uint64_t>(store.PairSpace(plan.mode()));
    out << "Records          " << WithThousands(rows) << "\n";
    if (store.NumDatasets() > 1) {
        out << "Inputs           " << store.NumDatasets() << "  (";
        for (size_t d = 0; d < store.NumDatasets(); ++d) {
            if (d > 0) out << " + ";
            out << WithThousands(store.DatasetEnd(d) - store.DatasetStart(d));
        }
        out << " rows)\n"
            << "Mode             " << PairModeName(plan.mode()) << "\n";
    }
    out << "Pairs unblocked  " << WithThousands(all_pairs) << "\n\n";

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
    // An unblocked source has reduced nothing, so the footer says that rather than
    // leaving a ratio of one to be read as a bug -- and its count comes from the
    // record counts, not from any term frequency.
    size_t unblocked = 0;
    for (size_t s = 0; s < plan.Size(); ++s) {
        if (plan.at(s).kind == SourceKind::kAllPairs) ++unblocked;
    }
    if (unblocked > 0) {
        out << "\nThis plan does no blocking: it produces every pair the mode "
               "admits, and\nthat count is closed form in the record counts. Any "
               "other source in it is\nredundant, since every pair it produces was "
               "produced already.\n";
        return;
    }

    out << "\nCounts are exact";
    if (plan.mode() == PairMode::kCrossDataset) {
        out << " and come from the grouped rows -- the term frequencies\n"
               "pool the inputs, so a cross-dataset count needs the split -- with no "
               "pairs\nenumerated";
    } else {
        out << " and come from the term frequencies, with no pairs "
               "enumerated";
    }
    out << ".\nThe sum bounds the deduplicated union from above";
    if (!count_union) out << "; pass --count for the union";
    out << ".\nBlocking keeps " << std::scientific << std::setprecision(2) << reduction
        << " of all possible pairs.\n";
}

}  // namespace cpplink
