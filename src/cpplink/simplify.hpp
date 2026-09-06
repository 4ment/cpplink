// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// Fit fine, then simplify.
//
// `cpplink levels` places cut points but will not choose how many there are,
// because merging bins can only lose bits and no criterion tried there would stop
// at a sensible count. This is the other half of that: a comparison is fitted
// deliberately fine, and then adjacent levels the data cannot tell apart are made
// one.
//
// The test is the one a nested pair of models asks for. Two adjacent levels differ
// only in what they are worth, which is log2(m/u), so restrict the pairs landing on
// either of them to a 2x2 table of level against class:
//
//                    match      non-match
//     upper level     M_i          U_i
//     lower level     M_j          U_j
//
// and the hypothesis that the two levels carry the same weight is exactly the
// hypothesis that the table's rows and columns are independent. The likelihood
// ratio statistic is the usual G^2 = 2 sum O log(O/E), which is chi-square on one
// degree of freedom, and for one degree of freedom the tail is erfc(sqrt(G^2/2))
// with no table to look up. Merging is proposed where the test cannot reject.
//
// The rates being compared are the model's own m and u, because log2(m/u) is what
// they are made of. What the histogram supplies is the *scale*: how many pairs
// this run scores, split into the two classes by the responsibility the model
// gives each pattern. So the question the test answers is whether a run of this
// size can tell the two levels' weights apart.
//
// The histogram supplies one more thing, as a diagnostic and not as a criterion.
// A candidate stream is not a random sample of pairs, because blocking chose it,
// so the rate at which *scored* pairs land on a level is not u, and the weight
// implied by the stream's own counts is not log2(m/u). Printing the two beside
// each other says how much of a column's evidence blocking has already spent:
// where a plan blocks on exact agreement, the stream weight of the exact level
// collapses towards zero while the model's stays where it was.
//
// And the headline finding, which the report states rather than hides. A failure
// to reject is not a finding of equality, and the test's power grows with the run.
// On any file this project is aimed at the run is billions of pairs, every gap
// however small is significant, and **the test merges nothing**. That is not a
// defect in the arithmetic: it is the same shape as `levels` finding that BIC
// always takes the widest partition, and it means significance cannot choose a
// level count here any more than it could choose a cut point. What can is the
// effect size, so the gap in bits is printed beside every p-value and `min_gap`
// decides on it: a level whose weight sits a tenth of a bit from its neighbour's
// is not earning a gamma code, however many pairs prove the tenth of a bit real.
//
// And a merge has to be expressible. Levels are evaluated top-down, first hit
// wins, so making two of them one means deleting the stronger and letting the
// weaker absorb its pairs -- which is only the same partition if the stronger
// level's predicate implies the weaker one's. Descending Jaro thresholds imply
// each other and so do ascending edit distances; `levenshtein <= 1` and
// `jaro_winkler >= 0.88` do not, in either direction, and a comparison mixing
// metrics is refused rather than silently rewritten into a different model. The
// null level is never merged, because missing is not a degree of agreement.

struct SimplifyOptions {
    // The level at which the likelihood-ratio test rejects. Merging is proposed
    // where it cannot: p >= alpha.
    double alpha = 0.01;
    // Merge regardless where the two levels are worth less than this many bits
    // apart. Zero leaves the decision entirely to the test.
    double min_gap = 0.0;
    // Candidate pairs folded into the histogram. Zero folds every one; a cap makes
    // it a Bernoulli sample, which costs the test power in exactly the way the
    // header above describes.
    uint64_t pair_cap = 0;
    unsigned threads = 0;
    uint64_t seed = 1;
};

// One level of one comparison, as the candidate stream saw it.
struct LevelCounts {
    std::string label;
    LevelType type = LevelType::kElse;
    double threshold = 0.0;
    uint64_t pairs = 0;       // candidate pairs landing here
    double matches = 0.0;     // of those, the expected matching ones
    double nonmatches = 0.0;  // and the rest
    double m = 0.0;           // the model's, which is not the stream's
    double u = 0.0;
    double weight = 0.0;  // log2(m/u)
    // The same weight read off the stream's own counts, which is what the test
    // actually compares: log2((M/sum M) / (U/sum U)).
    double stream_weight = 0.0;
};

// Two adjacent levels, and whether the stream can tell them apart.
struct LevelMerge {
    size_t upper = 0;  // index into ComparisonSimplify::levels, the stronger one
    size_t lower = 0;
    double g2 = 0.0;
    double p = 1.0;
    double gap = 0.0;  // |difference in log2(m/u)|: the effect size
    // The same gap read off the stream's own counts, which on a column blocking
    // conditions on is a very different number and worth seeing.
    double stream_gap = 0.0;
    bool expressible = true;  // the upper level's predicate implies the lower's
    bool merged = false;
    // Whether the level this merge deletes is the exact one, on a comparison that
    // asks for term-frequency adjustment. The comparison keeps its adjustment,
    // because a fuzzy level gets one from the neighbourhood mass, but without
    // --fuzzy-tf there is no mass table and the adjustment goes with the level.
    bool drops_exact = false;
    std::string refusal;  // why not, when not
};

struct ComparisonSimplify {
    std::string name;
    size_t comparison = 0;
    std::vector<LevelCounts> levels;
    std::vector<LevelMerge> tests;  // every adjacent pair, upper level first
    // The levels that survive, by index into `levels`, in order. A merged run
    // keeps its *last* level, which is the loosest predicate and the one the
    // others imply.
    std::vector<size_t> kept;
    uint8_t bits = 0;
    uint8_t proposed_bits = 0;
    bool changed = false;
};

struct SimplifyReport {
    uint64_t records = 0;
    PairMode mode = PairMode::kAll;
    uint64_t enumerated = 0;
    uint64_t folded = 0;
    double rate = 1.0;
    uint64_t distinct = 0;
    double expected_matches = 0.0;  // what the model makes of the folded pairs

    std::vector<ComparisonSimplify> comparisons;
    uint8_t width = 0;  // packed gamma bits now
    uint8_t proposed_width = 0;
    size_t merged_levels = 0;

    unsigned threads = 0;
    double seconds = 0.0;
};

// Whether the model was fitted to this schema. A level index means nothing under
// a model whose comparisons carry different levels, which is the same rule a
// spill is refused by.
bool ModelMatches(const Model& model, const ComparisonSet& comparisons,
                  std::string* error);

SimplifyReport BuildSimplify(const RecordStore& store, const ComparisonSet& comparisons,
                             const BlockingPlan& plan, const Model& model,
                             const SimplifyOptions& options);

void PrintSimplifyReport(const SimplifyReport& report, std::ostream& out);
void WriteSimplifyJson(const SimplifyReport& report, std::ostream& out);

// The schema with the merged levels deleted, so what comes out is a file that
// runs. Takes and returns the file's own text and touches nothing but the `levels`
// arrays of the comparisons that changed, so everything else survives verbatim.
// The model that produced the report does not survive it: gamma is packed
// differently, so the schema has to be re-estimated.
bool RewriteSchema(const std::string& text, const SimplifyReport& report,
                   std::string* rewritten, std::string* error);

}  // namespace cpplink
