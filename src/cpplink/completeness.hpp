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

namespace cpplink {

// Blocking recall without ground truth.
//
// `recall` answers "what fraction of the known pairs does this plan reach", and
// needs known pairs. On real data there are none, which is what makes automatic
// blocking unfalsifiable in production: a source that silently misses a third of
// the matches looks exactly like one that works.
//
// Pair completeness is a sum over agreement patterns, and every term of it is
// already in this pipeline:
//
//     PC = sum over reachable gamma of  m(gamma) * pi_fire(gamma)
//
// m(gamma) is the product of the learned m parameters, which is defined for every
// pattern -- including the dark ones, where no source fires and no pair was ever
// enumerated. That is the whole trick: EM extrapolates into patterns blocking
// never shows you, so the mass that blocking loses can be summed without ever
// seeing one of the pairs it lost.
//
// pi_fire is the probability that at least one source produces a matching pair
// with that pattern. For an exact-value source it is 1 or 0 and gamma says which.
// For a rare-value source it is the chance the shared value is under the cap,
// which the term-frequency table answers exactly. Those are the analytic sources.
// Sorted neighbourhood and MinHash are not analytic in gamma; leaving them out can
// only move mass from "reached" to "dark", so doing that yields a conservative
// bound, and estimating them from the observed capture overlap yields the estimate.
//
// That sum is the product estimator, and on real data it is not good enough:
// m(gamma) is a product over comparisons, and agreement among matches is
// correlated across columns, so the product overstates the chance that at least
// one blocked column agrees. Measured on `historical_50k` it reads 95.3% where the
// truth is 85.0%.
//
// The table estimator replaces the product where the data can. See
// `CompletenessReport` below: the cells an exact-value source covers are observed
// whole and are counted rather than modelled, and a log-linear model is asked
// only for the dark ones -- Fienberg's incomplete contingency table, which is what
// capture-recapture with dependent lists has always been. Independence and all
// two-way interactions are both fitted and BIC picks between them.
//
// The assumptions that remain, all of which the report states rather than hides:
//   1. m is not itself biased by blocking. This is what the per-column EM sessions
//      with a held-out column are for. When every source blocks one column EM can
//      learn nothing about it, and the report refuses rather than reporting.
//   2. A matching pair's shared value is drawn like a record's value, not like a
//      random agreeing pair's. This is what makes the rare-value cap answerable
//      from the term frequencies; `--value-weighting pairs` runs the other one.
//   3. Rows carrying one value are in no particular order, which is what a
//      sorted-neighbourhood window's hit rate is computed from. The report checks
//      that one against the data rather than assuming it.

// Which mechanism supplies a source's firing probability.
enum class CaptureClass : uint8_t {
    kAnalytic = 0,  // exact value, rare value: a closed form given gamma
    kObserved = 1,  // sorted neighbourhood, MinHash: estimated from the overlap
};

// One source's contribution to pi_fire.
struct SourceCapture {
    std::string name;
    std::string column;
    SourceKind kind = SourceKind::kExactValue;
    CaptureClass capture = CaptureClass::kAnalytic;
    bool bound_to_comparison = false;
    size_t comparison = 0;
    uint8_t exact_level = 0;
    // P(this source fires | the pair agrees exactly on its column, and matches).
    double fires_given_exact = 0.0;
    // Sorted neighbourhood only, and only as a cross-check: the same probability
    // from the term frequencies, against what the overlap actually shows.
    double window_analytic = 0.0;
    double window_observed = 0.0;
    bool has_window_check = false;
};

// How far two blocked comparisons are from independent among matches, over the
// cells that were actually observed. This is the bivariate residual the latent
// class literature uses as a dependence diagnostic: it names the pair, and BIC
// decides what to do about it.
struct Dependence {
    std::string left;
    std::string right;
    double g2 = 0.0;
    uint64_t df = 0;
};

struct CompletenessOptions {
    unsigned threads = 0;
    // Bernoulli-sample the capture fold. Every quantity read off it is a ratio of
    // counts, so a sample needs no reweighting.
    double sample = 1.0;
    uint64_t seed = 20260905;
    // Weight a matched pair's shared value by how many records carry it (the
    // default, and what a duplicate of a random record does) or by how many pairs
    // it forms (what a random agreeing pair does).
    bool pair_weighting = false;
    // How many observed pairs a pattern needs before its own observed capture rate
    // is trusted over the pooled one.
    uint64_t min_observed = 64;
    bool skip_observed = false;  // bound only: never walk the pair stream
};

struct CompletenessReport {
    uint64_t records = 0;
    std::vector<SourceCapture> sources;
    // pi_fire reads only the blocked comparisons, and every other comparison's m
    // sums to one over its levels, so those marginalise out: the estimate depends
    // on the m of the blocked columns and on nothing else in the model. Which is
    // also where it can fail, because EM cannot learn m for a column every source
    // conditions on. These are the sources whose m was never learned; while the
    // list is not empty, the numbers below are not evidence.
    std::vector<std::string> unlearned;
    bool trusted = true;
    // Whether some analytic source blocks a column no observed-class source
    // blocks. Without one there is nothing to condition the overlap on that is
    // not already correlated with the thing it is trying to measure, and the
    // correction is left out rather than made up.
    bool correction_available = false;
    // Read straight off the product model: the strongest analytic source per
    // pattern, and then the analytic sources combined across columns. They differ
    // only where a capped source is the only one firing, because an exact-value
    // source fires with probability one and "the best of them" is then the same
    // event as "any of them". Both inherit the conditional independence the
    // product is built on, which on real data is where they go wrong.
    double pc_bound = 0.0;
    double pc_analytic = 0.0;
    // The product model plus the observed-class correction.
    double pc_product = 0.0;

    // The table estimator, which does not take the joint over the blocked columns
    // from the product model but reads it off the data.
    //
    // A cell of that table is one combination of levels on the blocked
    // comparisons. Where an exact-value source fires, the candidate stream holds
    // *every* pair of the file in that cell, so the cell is completely observed
    // and its matching pairs can simply be counted (in expectation, through the
    // model's posterior). Where only a rare-value source fires, the cell is
    // observed with a known probability and is inflated by it. Where nothing
    // fires, the cell is dark, and only there does a model have to say anything.
    //
    // Filling the dark cells from an incomplete table is the classical
    // multiple-systems problem, and the classical answer applies: fit a log-linear
    // model to the observed cells and let it predict the missing one. Independence
    // needs two blocked columns, two-way interactions need three, exactly as
    // capture-recapture needs three lists before it can model dependence between
    // them.
    // A model can only fill the dark cells if the cells that were seen leave it
    // something to be fitted against: more observed cells than free parameters.
    // With two blocked columns and one-way margins the count is exactly equal --
    // the fit is saturated, the dark cell is whatever the model says, and there is
    // no evidence in it at all. That is the two-list capture-recapture problem,
    // and the answer here is the same as there: refuse rather than report.
    bool table_available = false;
    bool interaction_available = false;
    uint64_t cells = 0;
    uint64_t dark_cells = 0;
    uint64_t observed_cells = 0;
    double observed_matches = 0.0;  // expected matches in the completely seen cells
    double pc_independence = 0.0;
    double pc_interaction = 0.0;
    // Which extrapolation the observed cells actually support. Neither model wins
    // everywhere: independence is right on data generated under it and reads high
    // on data where agreement is correlated, and the interaction model is the
    // other way round. So the choice is made by fit rather than by preference --
    // the deviance over the completely observed cells, penalised by BIC.
    double deviance_independence = 0.0;
    double deviance_interaction = 0.0;
    double bic_independence = 0.0;
    double bic_interaction = 0.0;
    uint64_t parameters_independence = 0;
    uint64_t parameters_interaction = 0;
    std::vector<Dependence> dependence;  // worst pair first
    // The headline: the interaction fit where there are enough blocked columns for
    // one, the independence fit where there are not, and the product model where
    // there is no table at all.
    double pc_estimate = 0.0;
    const char* pc_basis = "product";
    double mass_total = 0.0;     // sum of m(gamma); a check that it is 1
    uint64_t patterns = 0;       // reachable patterns summed over
    uint64_t dark_patterns = 0;  // patterns no analytic source can reach
    double dark_mass = 0.0;      // the m mass those patterns carry

    // The capture fold.
    bool walked = false;
    uint64_t observed_pairs = 0;
    uint64_t analytic_captured = 0;
    uint64_t both_captured = 0;
    double pooled_rate = 0.0;
    uint64_t patterns_with_own_rate = 0;
    double seconds = 0.0;
    unsigned threads = 0;

    // Filled in only when a truth file was given, so the estimator can be scored.
    bool measured = false;
    double pc_measured = 0.0;
    uint64_t truth_pairs = 0;
};

bool EstimateCompleteness(const RecordStore& store, const ComparisonSet& comparisons,
                          const BlockingPlan& plan, const Model& model,
                          const CompletenessOptions& options, CompletenessReport* report,
                          std::string* error);

void PrintCompletenessReport(const CompletenessReport& report, std::ostream& out);
void WriteCompletenessJson(const CompletenessReport& report, std::ostream& out);

}  // namespace cpplink
