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
#include "cpplink/neighbourhood.hpp"
#include "cpplink/profile.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

struct EstimateOptions {
    uint64_t u_sample = 1000000;        // random pairs drawn to estimate fuzzy-level u
    uint64_t session_pairs = 10000000;  // per-session cap on pairs actually compared
    unsigned threads = 0;               // 0 asks the hardware
    int max_iterations = 500;
    double tolerance = 1e-6;
    double lambda_init = 0.01;
    double lambda = 0.0;  // > 0 overrides the derived value
    uint64_t seed = 20260903;
    // Compute u for the fuzzy levels exactly, by self-joining the column's
    // dictionary, instead of sampling random pairs and rescaling. Off by default
    // because it costs a quadratic scan over the distinct values, and it is
    // refused on a dictionary too large for `ball.budget`.
    bool fuzzy_u = false;
    BallOptions ball;

    // A session blocking on a column conditions on everything that column decides,
    // not only on the column itself. A comparison reading a column tied to it then
    // sees, among the session's *non*-matching pairs, an agreement rate that is
    // nothing like the u the model holds, and EM has no way to read that as
    // anything but evidence of matching. So those comparisons are held out of the
    // session too, which is the same held-out discipline one step wider.
    //
    // Measured on `historical_50k`: the session blocking on `first_name` implies a
    // match rate of 0.9996 where the truth is 0.0080, because `first_and_surname`
    // contains `first_name` and `first_name` all but determines `gender`. Holding
    // both out brings it to 0.018.
    bool exclude_tied = true;
    // Bits of u-side overlap past which two columns count as tied. Far below the
    // bit a redundancy has to be worth acting on in a report, because the question
    // here is not what to spend evidence on but whether a session can be read at
    // all.
    double tied_bits = 0.25;
    // Rows the tie pass reads. It is the profile's pairwise pass and nothing else:
    // no model, no candidate pair, and a fraction of a second on anything small.
    uint64_t tie_sample_rows = 500000;
};

// One EM session: the sources conditioning on a single column, unioned among
// themselves, with every comparison reading that column held out. Its m estimates
// are unbiased for the comparisons it did not hold out, and meaningless for the
// one it did -- which is why there has to be more than one session.
struct SessionReport {
    std::string column;
    std::vector<std::string> sources;
    std::vector<std::string> excluded;
    // Held out not because they read this column but because they read one tied to
    // it. Reported separately, because a session losing half its comparisons this
    // way is a fact about the schema.
    std::vector<std::string> tied;
    uint64_t candidates = 0;  // the exact bound from term frequencies
    uint64_t enumerated = 0;
    uint64_t folded = 0;
    double rate = 1.0;
    uint64_t distinct = 0;
    double seconds = 0.0;
    int iterations = 0;
    double change = 0.0;
    bool converged = false;
    double lambda = 0.0;           // the match rate among this session's pairs
    double implied_matches = 0.0;  // lambda x enumerated
    bool merged = false;           // whether its m estimates went into the model
    std::vector<std::string> notes;
    std::vector<std::string> warnings;
};

// One column's dictionary self-join: what it cost and what it bought, or why it
// was not run. Which columns got an exact fuzzy u is part of what a run reports,
// because the alternative is a sampled number wearing the same label.
struct BallReport {
    std::string comparison;
    bool built = false;
    std::string reason;
    uint64_t values = 0;
    uint64_t value_pairs = 0;
    uint64_t levels = 0;
    double seconds = 0.0;
};

struct EstimateReport {
    uint64_t u_pairs = 0;
    double u_seconds = 0.0;
    uint64_t u_exact_levels = 0;
    uint64_t u_ball_levels = 0;
    double ball_seconds = 0.0;
    double tie_seconds = 0.0;  // the pairwise pass that says which columns are tied
    std::vector<BallReport> balls;
    std::vector<SessionReport> sessions;
    std::vector<std::string> warnings;
};

// Learns u, m and lambda. u never sees a candidate pair: exact levels come from
// the term frequencies in closed form and the rest from uniformly random pairs.
// m comes from EM over per-column sessions. lambda is put back on the full-data
// pair count, because a blocked session's match rate is biased upward by exactly
// the thing blocking is for.
bool Estimate(const RecordStore& store, const ComparisonSet& comparisons,
              const BlockingPlan& plan, const EstimateOptions& options, Model* model,
              EstimateReport* report, std::string* error);

void PrintEstimateReport(const EstimateReport& report, std::ostream& out);

}  // namespace cpplink
