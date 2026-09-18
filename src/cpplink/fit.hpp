// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/histogram.hpp"

namespace cpplink {

// How well a session's fitted mixture explains the histogram it was fitted to.
//
// EM chooses lambda and m to maximise the likelihood of count[gamma] under a
// two-class model whose comparisons are independent within each class, and the
// likelihood it reaches says nothing about whether that model is the right shape.
// This is the check the fit itself cannot make. The fitted model predicts every
// pattern's count,
//
//     E[gamma] = N ( lambda prod_c m_c[gamma_c] + (1 - lambda) prod_c u_c[gamma_c] )
//
// over the comparisons the session left free, and the deviance
//
//     G^2 = 2 sum_gamma O[gamma] ln (O[gamma] / E[gamma])
//
// is the likelihood-ratio statistic of the saturated model against the fitted
// one. Divided by 2 N ln 2 it is KL(observed || fitted) in bits: the average
// number of bits per pair the model is wrong by, which is the scale every other
// number in this project is on, and the one that decides anything here. G^2
// itself grows with N at a fixed misfit, so its p-value is printed as the
// diagnostic and never used as the decision; the same rule `levels`, `simplify`
// and the interaction fit each had to learn.
//
// The same deviance marginalised to one pair of comparisons is the bivariate
// residual of the latent class literature, and it names which pair of
// comparisons the independence assumption fails on. A pair reading high here
// with no term fitted is a candidate for `--interactions`; the whole-histogram
// deviance after the admitted terms are in says whether they were enough.
//
// Nothing here touches a pair: the histogram is the input, and the cost is the
// number of distinct patterns, which is 1e2 to 1e5 whatever the run size.

// One pair of free comparisons, and how far the two-way table the session's
// pairs filled sits from the one the fitted mixture predicts for it.
struct PairResidual {
    std::string left;
    std::string right;
    double g2 = 0.0;       // 2 sum O ln(O/E) over the pair's table
    uint64_t degrees = 0;  // (L_left - 1)(L_right - 1) over the reachable levels
    double p_value = 1.0;
    double bits = 0.0;  // g2 / (2 N ln 2): the misfit per pair, in bits
    // The same residual once the admitted interaction terms are in the model.
    // A term on this pair fits its two-way table by construction, so what is left
    // is the disagreement between this session and the pooled table; a term on
    // another pair leaves this one exactly where it was.
    bool corrected = false;
    double corrected_g2 = 0.0;
    double corrected_bits = 0.0;
};

struct SessionFit {
    bool measured = false;
    std::string refusal;
    uint64_t pairs = 0;     // pairs folded into the histogram
    uint64_t patterns = 0;  // distinct patterns over the free comparisons
    double deviance = 0.0;  // 2 sum O ln(O/E) over every observed pattern
    // Cells the free comparisons can reach, less one, less the parameters the
    // session fitted (lambda and every free m); u is fixed and not counted.
    uint64_t degrees = 0;
    double p_value = 1.0;
    double bits = 0.0;  // deviance / (2 N ln 2)
    // What a model that was exactly right would still read, because N pairs
    // spread over K patterns are a sample: about (K - 1) / (2 N ln 2) bits, the
    // chi-square expectation of G^2 on the same scale. Misfit is what sits above
    // it, and on a small run most of `bits` can be this.
    double floor_bits = 0.0;
    // The same deviance under the model the admitted terms make. Absent unless a
    // term touches two comparisons this session left free.
    bool corrected = false;
    std::string corrected_refusal;
    double corrected_deviance = 0.0;
    double corrected_bits = 0.0;
    std::vector<PairResidual> residuals;  // worst first
};

// One admitted interaction as the log-linear fit produced it: the joint over the
// product of its own margins, under M and under U, before the shrinkage and the
// clamp the scorer applies. The residual is measured against the association the
// data showed rather than the version the scorer trusts, because the question is
// what is left in the histogram and not what the weight does about it.
struct InteractionRatios {
    size_t left = 0;             // comparison index
    size_t right = 0;            // comparison index, greater than left
    std::vector<double> match;   // left levels x right levels, row-major
    std::vector<double> random;  // same shape
};

struct FitOptions {
    // Cells the corrected model may enumerate to normalise the terms that share a
    // comparison. Past this the corrected reading is refused, not approximated.
    uint64_t cell_budget = 1u << 22;
};

// `m` and `u` are per comparison per level; `excluded` marks the comparisons the
// session held out, whose levels the histogram still carries and the model
// ignores. `lambda` is the session's own match rate, because the fit being
// judged is the session's.
SessionFit MeasureFit(const ComparisonSet& comparisons,
                      const std::vector<PatternCount>& entries,
                      const std::vector<std::vector<double>>& m,
                      const std::vector<std::vector<double>>& u, double lambda,
                      const std::vector<bool>& excluded,
                      const std::vector<InteractionRatios>& terms,
                      const FitOptions& options);

}  // namespace cpplink
