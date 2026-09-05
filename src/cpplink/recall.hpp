// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

struct TruthPairs {
    std::vector<std::pair<uint32_t, uint32_t>> rows;
    uint64_t lines = 0;
    uint64_t unresolved = 0;  // ids in the file that are not in the data
};

// Reads an "id_a,id_b" CSV and resolves both ids to rows. Resolution is one pass
// over the id column against a set of only the wanted ids, rather than a full
// id index, which is not otherwise needed.
bool LoadTruthPairs(const std::string& path, const RecordStore& store, TruthPairs* truth,
                    std::string* error);

// What one source reaches, what it costs, and what it alone reaches. Recall on
// its own ranks nothing: every source can buy recall with candidates, so the
// number that decides whether a source earns its place is the known pairs it is
// the first to reach divided by the pairs it makes the pipeline evaluate.
struct SourceRecall {
    std::string name;
    uint64_t found = 0;            // known pairs this source produces
    uint64_t first_to = 0;         // known pairs no earlier source produces
    uint64_t candidate_pairs = 0;  // exact, from the term frequencies
};

// The blocking literature's three numbers -- pair completeness, pair quality and
// reduction ratio -- over one plan, plus the marginal split that says which
// source is paying for itself.
struct RecallMetrics {
    uint64_t truth_pairs = 0;   // resolved, and in scope for the mode
    uint64_t unresolved = 0;    // ids in the truth file the data does not hold
    uint64_t out_of_scope = 0;  // link mode: known pairs inside a single input
    uint64_t union_found = 0;
    uint64_t candidate_sum = 0;    // over sources; bounds the union from above
    uint64_t candidate_union = 0;  // deduplicated; only when `count_union`
    bool counted_union = false;
    double pair_space = 0.0;  // pairs the mode admits, before blocking
    std::vector<SourceRecall> sources;

    // What to price the union by: the deduplicated count when it was paid for,
    // the sum otherwise. The sum is an upper bound, so pair quality and reduction
    // ratio taken from it are lower bounds -- they understate the plan, which is
    // the safe direction for a number used to justify keeping a source.
    uint64_t UnionCandidates() const {
        return counted_union ? candidate_union : candidate_sum;
    }
};

// With hand-written rules you can reason about what they miss. With automatic
// sources you cannot, so recall must be measured or the approach is unfalsifiable.
// Asking `Produces` of each known pair costs O(pairs x sources) and enumerates
// nothing; `count_union` additionally enumerates, and is the only part of this
// that is not free.
RecallMetrics MeasureRecall(const BlockingPlan& plan, const RecordStore& store,
                            const TruthPairs& truth, bool count_union);

void PrintRecallReport(const RecallMetrics& metrics, std::ostream& out);

// The same numbers for a harness that sweeps a source's knob and plots the
// frontier, so the sweep reads a parser-stable document rather than scraping a
// fixed-width table.
void WriteRecallJson(const RecallMetrics& metrics, std::ostream& out);

// Why one source failed to produce a pair. A recall number says how many matches
// blocking loses; it does not say whether the loss is a parameter that is set too
// tight or a signal the plan does not carry, and those have completely different
// fixes.
enum class MissReason : uint8_t {
    kProduced = 0,   // this source did reach the pair
    kNull,           // the source's column is missing on one side or both
    kDisagreed,      // both sides have a value and the keys differ
    kTooCommon,      // the keys agree but the frequency cap excluded them
    kOutsideWindow,  // sorted neighbourhood, but further apart than the window
};

// How far a sorted-neighbourhood window may be stretched before "widen it" stops
// being a fix and becomes "enumerate the whole file". Ten times the configured
// window is already a tenfold candidate bill.
inline constexpr uint64_t kPlausibleWindowFactor = 10;

const char* MissReasonName(MissReason reason);

// What agrees on a missed pair, per comparison, split by how strong the agreement
// is. `exact` counts pairs landing on an exact level; `fuzzy` counts any other
// non-null, non-else level. A column agreeing exactly on missed pairs and not
// blocked on is free recall; one that only ever agrees fuzzily is the case a
// similarity-based source exists for.
struct ColumnAgreement {
    std::string comparison;
    uint64_t exact = 0;
    uint64_t fuzzy = 0;
    bool blocked = false;  // some source already blocks on this column
};

// What widening one sorted-neighbourhood window would buy and what it would cost.
// A window can always be widened until it reaches the whole file, so the only
// honest way to report it is against the candidate pairs it would generate.
struct WindowGain {
    std::string source;
    uint64_t window = 0;
    uint64_t recovered = 0;        // missed pairs this window would reach
    uint64_t candidate_pairs = 0;  // what it would cost to enumerate
};

struct MissReport {
    uint64_t truth_pairs = 0;
    uint64_t missed = 0;
    // reasons[source][reason]
    std::vector<std::vector<uint64_t>> reasons;
    // Missed pairs a looser parameter on an existing source would have reached.
    uint64_t fixable_by_cap = 0;
    uint64_t fixable_by_window = 0;
    uint64_t unreachable = 0;  // no plausible parameter change reaches them
    // The widest gap a sorted-neighbourhood source would have needed.
    uint64_t widest_window_needed = 0;
    std::vector<WindowGain> window_gains;
    // Over the unreachable pairs only: what still agrees.
    std::vector<ColumnAgreement> agreement;
    uint64_t agreed_on_nothing = 0;
    // A few missed pairs by row, so a claim about them can be checked against
    // `explain` rather than believed.
    std::vector<std::pair<uint32_t, uint32_t>> examples;
    size_t example_limit = 0;
};

// Classifies every truth pair the plan misses. Costs O(pairs x sources) and
// enumerates nothing, like the recall report it extends.
void DiagnoseMisses(const BlockingPlan& plan, const ComparisonSet& comparisons,
                    const TruthPairs& truth, MissReport* report);

void PrintMissReport(const BlockingPlan& plan, const MissReport& report,
                     std::ostream& out);

}  // namespace cpplink
