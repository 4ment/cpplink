// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"

namespace cpplink {

struct InteractionOptions {
    bool enabled = false;
    // How many terms may enter the model. Small on purpose: the point is to repair
    // the two or three places conditional independence actually fails, not to fit
    // every pair and lose the property that makes gamma sufficient.
    size_t max_terms = 2;
    // The effect-size floor, in bits: a term must move an average matching pair by
    // at least this much to be worth a table. Significance is printed beside it and
    // deliberately does not decide -- see the note on the report.
    double min_bits = 0.25;
    // No single cell may move the weight further than this. A cell holding two
    // pairs would otherwise put twenty bits on the pattern it names.
    double clamp_bits = 6.0;
    // Pairs a cell needs before its correction is taken at face value: below it the
    // correction is scaled by n / (n + this), toward zero. A cell the data barely
    // reaches is a ratio of two floors -- a column duplicated under two names can
    // never land on "one agrees and the other does not", and the empty cell naming
    // that was reading +3.7 bits before this. Five is the contingency table's own
    // bar for an expected count, and it is set there rather than at whatever
    // maximised F1 on a benchmark.
    double cell_support = 5.0;
    // Cells added to both sides before the fit, so an empty cell is a small number
    // rather than an infinite one.
    double prior_count = 0.5;
    // The share of the u-side joint a pair may owe to the file's own duplicates
    // before the term is refused. A uniformly random pair is a match with
    // probability lambda, so the sampled joint measures (1 - lambda) u + lambda m,
    // and for two high-cardinality columns the second term is the whole of it: on
    // `febrl3` the exact-agreement cell of two independent columns sits 1,565x
    // above the independent rate, all of it duplicates. The rate is subtracted,
    // and where the subtraction leaves nothing the term is refused rather than
    // guessed.
    double max_contamination = 0.5;
    // Matching pairs a session must carry per free parameter of the term. A two-way
    // table over five levels each has sixteen of them, and fitting sixteen numbers
    // off a few hundred matches produces association that is not there: on
    // `fake_1000` every one of the ten candidate pairs reads a correction between
    // -1.3 and -1.9 bits, which is not ten dependencies but one small sample. The
    // bar is the contingency table's own rule of thumb.
    double min_pairs_per_parameter = 0.0;
    // Sessions that must see the term independently. Two is the minimum that can
    // disagree, and disagreement is the only evidence available here that a fitted
    // association is noise: there is no held-out set, and the pooled fit will always
    // find something.
    size_t min_sessions = 2;
};

// Level-pair counts for every unordered pair of comparisons. One instance holds
// the M side of one estimation session or the U side of the random-pair sample;
// nothing here holds a row per pair, and the tables are 45 by 8 by 8 doubles for a
// ten-comparison schema whatever the data size.
class JointTables {
   public:
    JointTables() = default;
    explicit JointTables(const std::vector<size_t>& levels);

    // Fold in one pattern's already-unpacked levels, weighted by `mass`.
    void Add(const uint8_t* levels, double mass);
    void Merge(const JointTables& other);

    const std::vector<double>& Table(size_t left, size_t right) const;
    double Total() const { return total_; }
    size_t Levels(size_t comparison) const { return levels_[comparison]; }
    size_t Size() const { return levels_.size(); }

   private:
    size_t Index(size_t left, size_t right) const;

    std::vector<size_t> levels_;
    std::vector<std::vector<double>> tables_;
    double total_ = 0.0;
};

// What one estimation session contributed. A session that held a comparison out
// says nothing about any interaction touching it, which is the interaction form of
// the held-out discipline the per-column sessions already run.
struct SessionJoint {
    JointTables tables;
    std::vector<bool> excluded;
    bool usable = false;  // the session's m was merged into the model
};

// One candidate pair of comparisons, whether it was admitted and why not.
struct InteractionCandidate {
    std::string left;
    std::string right;
    double match_bits = 0.0;  // signed: what the term moves an average match by
    double effect = 0.0;      // mass-weighted mean |correction| under M
    double widest = 0.0;      // the largest single cell, in bits
    double g2 = 0.0;          // deviance against independence, M side plus U side
    uint64_t degrees = 0;
    double p_value = 1.0;
    size_t sessions = 0;
    double match_pairs = 0.0;    // matching mass a typical contributing session had
    double per_parameter = 0.0;  // that mass, per free parameter of the term
    // The least any single session says the term moves a match by, and zero where
    // two sessions disagree about the sign. A term no session can see on its own is
    // an artefact of pooling them.
    double weakest = 0.0;
    // The share of the u-side joint that was the file's own duplicates, weighted
    // by where a matching pair actually lands. Above max_contamination the term is
    // refused: what is left after the subtraction is noise.
    double contamination = 0.0;
    bool admitted = false;
    std::string reason;
};

struct InteractionReport {
    bool ran = false;
    // What was subtracted off the u side, and whether it can be trusted. lambda
    // from a session is a lower bound -- it counts only the matches blocking
    // reached -- so the subtraction removes less than it should and every
    // correction reads low.
    double lambda = 0.0;
    bool lambda_is_a_bound = false;
    std::string refusal;  // set when nothing could be fitted at all
    std::vector<InteractionCandidate> candidates;  // ranked by effect, all of them
    size_t admitted = 0;
    size_t clamped = 0;  // cells that hit clamp_bits
    double seconds = 0.0;
};

// Fits the two-way corrections and writes the admitted ones into `model`.
//
// The M side comes from the session histograms, weighted by the responsibility EM
// already computed, and is averaged over the sessions that left both comparisons
// free. The U side comes from the uniformly random pairs `u` is drawn from and
// never from candidates: a blocking source selecting on a column induces agreement
// correlation involving that column, so measuring U-side dependence on the
// candidate stream measures the plan.
//
// Both sides are then fitted to the model's own margins, which leaves the
// association untouched and makes the term a pure correction: adding it cannot
// move m or u by a millionth.
// `lambda` is the rate at which a uniformly random pair is a match, which is what
// contaminates the u side and what is subtracted from it.
void FitInteractions(const ComparisonSet& comparisons,
                     const std::vector<SessionJoint>& sessions,
                     const JointTables& random_pairs, double lambda,
                     const InteractionOptions& options, Model* model,
                     InteractionReport* report);

void PrintInteractionReport(const InteractionReport& report, std::ostream& out);

// Iterative proportional fitting of a two-way table to given margins. Exposed for
// the test that asserts it moves the margins and not the odds ratios.
void FitToMargins(const std::vector<double>& rows, const std::vector<double>& columns,
                  std::vector<double>* table);

}  // namespace cpplink
