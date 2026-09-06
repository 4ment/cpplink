// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/neighbourhood.hpp"
#include "cpplink/profile.hpp"
#include "cpplink/recall.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// Levels that define themselves.
//
// Thresholds in a schema are guesses. `jaro_winkler >= 0.9` and `levenshtein <= 2`
// are written by hand, copied from the last schema, and never checked against the
// column they run on, even though level order *is* the model: the first level that
// fires wins, so where the cut points sit decides what every pair is worth. This
// checks them and proposes better ones, from two curves, neither of which needs a
// model, a blocking plan or a truth file.
//
// The u curve is exact and this pipeline already pays for it. Because values are
// interned, sweeping a similarity threshold over the dictionary self-join gives
//
//     u(t) = P(sim(a, b) >= t) for two random records carrying both values
//
// for the whole column in one scan over the distinct values rather than the rows.
// Almost no linkage system can afford that curve; the same self-join `BallMassTable`
// runs for the fuzzy term-frequency adjustment computes it, one bin instead of one
// level.
//
// The m curve comes from anchor pairs, which the profile's M side already
// manufactures: pairs agreeing exactly on a column set strong enough that agreement
// alone makes them matches. Bin the similarity of the held-out column over those
// pairs and that is m(t). It carries the anchor's upward bias, and that matters far
// less here than it does for m itself: the bias lifts the whole curve, and a cut
// point is chosen from the curve's shape.
//
// Given both curves the levels are a partition of the similarity axis, and the
// right partition is the one that throws away the least evidence. A comparison is
// worth
//
//     KL(m || u) = sum over levels of m_l * log2(m_l / u_l)
//
// bits to a matching pair, which is the mutual information between the discretised
// similarity and the latent match indicator, divided by lambda, in the limit lambda
// is actually in. Merging two bins can only lower it, so the criterion cannot pick
// the number of levels on its own and BIC does, over the anchor pairs the m side
// rests on. The partition itself is a dynamic program over the bins, which is
// Fayyad-Irani minimal-entropy partitioning against a latent target rather than an
// observed one.
//
// What it will not do is decide. It prints the current levels and the proposed ones
// measured the same way and in the same units, and writes the proposal out as a
// schema on request, but which of the two to keep is a judgement about the data
// that the numbers inform rather than settle. It also proposes nothing about the
// null level or the term-frequency flag, neither of which is a threshold question.

struct LevelsOptions {
    // The similarity grid. Below the floor the curve is one tail bin, because a
    // Jaro of 0.6 between two surnames is not evidence of anything and pricing the
    // self-join down there costs more than the answer is worth. The floor is
    // lowered automatically where the schema already declares a looser level, so
    // the current levels are always measured exactly rather than off the grid.
    double jaro_floor = 0.75;
    double jaro_step = 0.01;
    uint32_t edit_max = 4;
    // Levels one comparison may be given, including the exact level and the else.
    size_t max_levels = 8;
    // Levels to propose. Zero means the count the schema already uses, which is the
    // only setting under which the current and proposed partitions are comparable:
    // it holds gamma's width fixed and asks the one question the curves can answer
    // well, which is where the cuts go. Choosing the count is a different question
    // and the sweep below is what to answer it with.
    size_t levels = 0;
    // Anchor pairs a comparison needs before a curve is read off them.
    uint64_t min_match_pairs = 500;

    ProfileOptions profile;
    BallOptions ball;
    unsigned threads = 0;
};

// One bin of the similarity axis, in level order: most similar first.
struct SimilarityBin {
    double threshold = 0.0;  // the bin's lower edge on the grid
    bool exact = false;      // identical values, which is always a level of its own
    bool tail = false;       // everything below the floor, which the else absorbs
    double u = 0.0;          // P(two random present rows land here), exact
    double m = 0.0;          // P(an anchor pair lands here)
    uint64_t match_pairs = 0;
};

struct ProposedLevel {
    LevelType type = LevelType::kElse;
    double threshold = 0.0;
    std::string label;
    double m = 0.0;
    double u = 0.0;
    double weight = 0.0;   // log2(m/u): what landing here is worth
    double bits = 0.0;     // m * weight: what it contributes to the comparison
    bool floored = false;  // m or u hit the half-a-pair floor rather than measured
    size_t bin_begin = 0;  // the bins this level owns, for a second m curve
    size_t bin_end = 0;
    // The same three numbers read off known pairs rather than anchor pairs, where a
    // truth file was given. The partition is chosen without ever seeing these, so
    // they are what says whether the cut points generalise or fit the anchor.
    double truth_m = 0.0;
    double truth_weight = 0.0;
    double truth_bits = 0.0;
};

// One comparison's two curves, its current levels re-measured on them, and the
// partition that keeps the most evidence.
struct ComparisonLevels {
    std::string name;
    size_t comparison = 0;
    bool proposed = false;
    std::string refusal;

    LevelType metric = LevelType::kJaroWinkler;
    double floor = 0.0;  // the grid's lowest edge, after the schema widened it
    std::vector<SimilarityBin> bins;

    std::vector<ProposedLevel> current;
    std::vector<ProposedLevel> best;
    double current_bits = 0.0;
    double best_bits = 0.0;
    size_t best_count = 0;    // levels proposed
    size_t schema_count = 0;  // levels the schema declares, the null one aside
    // What BIC would take. Reported and not obeyed: see the note in the report.
    size_t bic_count = 0;
    // What each level count was worth, what BIC made of it, and what it costs gamma,
    // from two levels up to `max_levels`. Printed because the count is an
    // engineering choice these numbers inform rather than settle.
    std::vector<double> bits_by_count;
    std::vector<double> score_by_count;
    std::vector<size_t> width_by_count;
    // Whether the lowest cut sits on the grid's floor, which means the partition
    // wanted to go lower and --jaro-floor stopped it.
    bool floor_binds = false;

    bool truthed = false;
    uint64_t truth_pairs = 0;  // known pairs carrying the column on both rows
    double current_truth_bits = 0.0;
    double best_truth_bits = 0.0;
    std::vector<double> truth_m;  // the truth curve, by bin

    std::vector<std::string> anchor;  // the anchor the m curve was read off
    uint64_t match_pairs = 0;
    uint64_t values = 0;
    uint64_t value_pairs = 0;
    uint64_t compared = 0;  // value pairs that survived the signature bound
    double seconds = 0.0;
};

struct LevelsReport {
    uint64_t records = 0;
    PairMode mode = PairMode::kAll;
    bool truthed = false;
    uint64_t truth_pairs = 0;
    uint64_t anchor_rows = 0;
    std::string anchor_refusal;  // set when the profile could not build an anchor
    std::vector<ComparisonLevels> comparisons;
    unsigned threads = 0;
    double seconds = 0.0;
};

// `truth` is optional and changes nothing about the proposal: it is scored beside
// it, never fitted to, which is the only way the second curve says anything.
LevelsReport BuildLevels(const RecordStore& store, const ComparisonSet& comparisons,
                         PairMode mode, const LevelsOptions& options,
                         const TruthPairs* truth = nullptr);

void PrintLevelsReport(const LevelsReport& report, std::ostream& out);
// Both curves, both partitions and the level-count sweep, for a caller that wants
// to plot them rather than read them.
void WriteLevelsJson(const LevelsReport& report, std::ostream& out);

// The schema with the proposal substituted for every comparison that got one, so
// what comes out is a file that runs rather than a fragment to merge by hand. Takes
// and returns the file's own text and touches nothing but the `levels` arrays, so
// everything else it carries survives verbatim.
bool RewriteSchema(const std::string& text, const LevelsReport& report,
                   std::string* rewritten, std::string* error);

}  // namespace cpplink
