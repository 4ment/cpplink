// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// What the data can be worth, before a model exists to be wrong about it.
//
// Every stage after this one costs an hour or a day, and every one of them
// assumes things about the columns that nobody has checked. This one reads the
// term-frequency tables and the rows, and answers two questions with no model, no
// blocking plan, no known pairs and no pair enumerated.
//
// One: can these columns separate the classes at all? A comparison is worth
// log2(m/u) bits when it agrees; u for exact agreement is sum_v p_v^2, which the
// TF table already holds, and m <= 1, so the most a column can ever be worth is
// the collision entropy -log2(sum_v p_v^2). Its inverse is the effective
// cardinality: the number of equally likely values the column behaves as if it
// had, which is what `inspect`'s distinct count is a bad proxy for. Set the sum of
// those against the prior odds the pair space imposes and the answer is a ledger
// that ends in a margin. A negative margin means no model and no plan will ever
// separate these classes, which is worth knowing in the first minute.
//
// Two: how much of that evidence is the same evidence twice? Fellegi-Sunter
// assumes agreement is independent conditional on match status. The U half of that
// is a property of the value distributions alone, so it needs no matches and no
// sampling of pairs: the pairwise version of the same closed form is
//
//     u_cd = sum over (v,w) of p_vw^2      phi_cd = u_cd / (u_c * u_d)
//
// over the joint value table of two interned columns, and log2(phi_cd) is the
// number of bits the score double-counts. It is reported in bits rather than as a
// correlation because that is the correction term that would appear in the weight,
// which makes it actionable. Two structural checks ride the same pass: how nearly
// one column determines the other, and substring containment, which is the
// detector for a column that literally contains another.
//
// Both of those are counting measures rather than entropy measures, and that is
// deliberate. Mutual information between two high-cardinality columns is not
// estimable from the rows: when every joint cell holds one row the plug-in
// estimator reads almost the whole marginal entropy as shared, and on
// `historical_50k` it calls two independent columns 92% determined. Sample size
// cannot fix it, because both marginals and the joint all saturate at log2(n)
// together. The determination share degrades far more gracefully, and its one
// failure mode is a near-unique determinant, which is checked and refused rather
// than reported.
//
// The pairwise half carries a precondition, found by measuring rather than by
// reasoning. u is the rate two *non-matching* rows collide, and a deduplication
// file is full of matching rows that collide on everything at once, so what is
// actually measured is (1 - lambda) * u + lambda * m. For one column that is a
// small inflation. For the joint of two high-cardinality columns the independent
// rate is orders of magnitude below lambda and the measurement is nothing but the
// duplicates: on `historical_50k` the joint of `first_and_surname` and
// `postcode_fake` holds 85,584 collisions where independence predicts 52, and
// every one of the extras is a duplicate row rather than a dependence. So the
// number is reported only where the independent rate stands clear of the match
// rate, and refused where it does not. The row-level measures below are immune to
// this, because a duplicate row is one row and not one pair.
//
// The M half needs matching pairs, and gets them without a model or a truth file
// from *anchor pairs*: two rows agreeing exactly on a subset of columns whose joint
// collision rate is far enough below the match rate that agreement alone puts the
// posterior near one. Hold that subset out and every other column's agreement rate
// over those pairs is m, measured rather than assumed. The same pairs give the
// M-side half of the dependence map, which is the larger of the two correlations
// and the one the U-side table above cannot see.
//
// The bias is real and is stated rather than buried: anchors are the matches that
// happened to agree on a clean identifier, and agreement is positively correlated
// among matches, so every m here reads high. The spread across sessions built on
// different anchors is what that bias looks like from the inside, so it is
// reported next to the mean rather than averaged away.
//
// Two disciplines make the estimate mean what it says, and both are the ones the
// per-column EM sessions already run. A column is never estimated from a session
// whose anchor contains it, and never from one whose anchor contains a column that
// determines it or occurs inside it, because either way the anchor has already
// forced the agreement being measured. That check is what the pairwise pass above
// is for, which is why the M side is refused outright when it has not run.

struct ProfileOptions {
    // Rows the pairwise pass reads, selected by a hash of the row index so the
    // choice is deterministic, independent of order and spread across the inputs.
    // Every quantity printed is a ratio of counts, so a sample needs no
    // reweighting. Zero reads every row.
    uint64_t sample_rows = 2'000'000;
    bool pairs = true;
    // Matching pairs assumed present, which is what the prior odds are made of.
    // Zero means one duplicate per record, which is what the pair space of a
    // deduplication is usually asked to hold.
    uint64_t expected_matches = 0;
    unsigned threads = 0;
    uint64_t seed = 20260906;

    // The M side. Anchor pairs are rare by construction, and a row sample keeps
    // only the square of its own fraction of them, so this pass gets its own and
    // much larger budget rather than sharing the pairwise one. Zero reads every row.
    bool anchors = true;
    uint64_t anchor_rows = 4'000'000;
    // Posterior bits an anchor has to clear before the pairs it selects are read as
    // matches: the joint exact-agreement bits of the anchor columns, less the
    // pairwise overlap between them, set against the prior odds. Six is odds of
    // about 64 to 1 under m <= 1, and every anchor it admits on the three benchmark
    // datasets holds 99.6% true pairs or better. Raising it buys no precision that
    // was there to buy and costs accuracy: a stronger anchor selects a cleaner
    // subset of the matches, and every m read off it moves further up.
    double anchor_margin = 6.0;
    // Anchor pairs one session walks before it stops. m to a hundredth needs about
    // ten thousand, so this is loose enough not to bind on any real file.
    uint64_t anchor_pairs = 1'000'000;
};

// One column, from its term frequencies alone.
struct ColumnProfile {
    std::string name;
    ColumnType type = ColumnType::kString;
    // False for a column with no term frequencies, and for a list column, whose
    // exact agreement is not a single-value event and whose collision entropy
    // would therefore mean something other than what the table's heading says.
    bool scored = false;
    uint32_t distinct = 0;
    uint64_t nulls = 0;
    uint64_t present = 0;
    double coverage = 0.0;
    double top_share = 0.0;
    double collision = 0.0;         // the rate two distinct present rows collide
    double effective_values = 0.0;  // 1 / collision
    double bits = 0.0;              // -log2(collision): the exact-match ceiling
    double covered_bits = 0.0;      // coverage^2 * bits
    // Whether the collision rate is close enough to the match rate that the
    // duplicates in the file dominate it. Bits is then a floor and not an
    // estimate, which is the safe direction for a margin but is worth saying.
    bool bits_floored = false;
};

// Two columns, from one pass over the sampled rows.
struct ColumnPairProfile {
    size_t left = 0;  // index into ProfileReport::columns
    size_t right = 0;
    std::string left_name;
    std::string right_name;
    uint64_t rows = 0;  // sampled rows carrying both values
    // Share of rows kept by mapping each value of one column to its commonest
    // partner in the other: 1 exactly when the first determines the second. This
    // is the complement of the g3 error the functional-dependence literature
    // measures an approximate dependency by.
    double determines_right = 0.0;
    double determines_left = 0.0;
    // A near-unique column determines everything and says nothing by doing so, so
    // the direction it determines is refused rather than reported.
    double left_distinct_share = 0.0;
    double right_distinct_share = 0.0;
    // What the same share reads by ignoring the determinant and taking the target
    // column's own commonest value. A column with one dominant value is
    // determined by everything, so a determination only counts when it beats this.
    double baseline_right = 0.0;
    double baseline_left = 0.0;
    bool left_informative = false;   // whether determines_right means anything
    bool right_informative = false;  // whether determines_left means anything
    // Share of those rows on which one value occurs inside the other, and which
    // way round. String columns only.
    double containment = 0.0;
    bool left_inside_right = false;
    double u_left = 0.0;
    double u_right = 0.0;
    double u_joint = 0.0;
    // log2(u_joint / (u_left * u_right)): bits of agreement counted twice. Every
    // u here is the probability two *distinct* rows collide, estimated without
    // replacement, which is the only form that reads zero rather than 1/rows when
    // the joint table is all singletons.
    double redundant_bits = 0.0;
    double joint_collisions = 0.0;
    // Collisions independence predicts, which is what the observed count has to
    // stand clear of, and which itself has to stand clear of the match rate.
    double expected_collisions = 0.0;
    // Whether the joint is readable at all: enough collisions to be more than
    // noise, and an independent rate far enough above the match rate that the
    // file's own duplicates are not what is being measured.
    bool resolved = false;

    // The same question under M, from the anchor pairs: how often two matching
    // rows agree on both columns against how often independence says they would.
    // This is the correlation the U side above cannot see, and normally the larger
    // of the two.
    uint64_t m_pairs = 0;  // anchor pairs carrying both columns on both rows
    double m_left = 0.0;   // P(left agrees | match), over those pairs
    double m_right = 0.0;
    double m_joint = 0.0;
    // log2(m_joint / (m_left * m_right)): bits the score double-counts under M.
    double m_redundant_bits = 0.0;
    bool m_resolved = false;

    // What the weight actually double-counts when both columns agree, which is the
    // M-side overlap less the U-side one: the score adds log2(m/u) twice, and only
    // the difference of the two corrections is wrong. Zero unless both sides read.
    double NetRedundantBits() const {
        return m_resolved && resolved ? m_redundant_bits - redundant_bits : 0.0;
    }

    // How much the determinant buys over ignoring it, which is what ranks a pair:
    // a determination that only matches the target's own commonest value is not
    // one. Zero where the direction is not readable at all.
    double LeftLift() const {
        return left_informative ? determines_right - baseline_right : 0.0;
    }
    double RightLift() const {
        return right_informative ? determines_left - baseline_left : 0.0;
    }

    // Whether this pair is worth a line in the suspects table, and why.
    bool Suspect() const;
    std::string Verdict() const;
};

// One anchor: a column subset whose exact agreement makes a pair a match, and the
// pairs it selected. A session learns every column the anchor neither contains nor
// determines.
struct AnchorSession {
    std::vector<size_t> anchor;  // column indices, strongest first
    std::vector<std::string> anchor_names;
    std::vector<size_t> learns;   // columns this session may be read for
    double bits = 0.0;            // joint exact-agreement bits, pairwise overlap removed
    double posterior_bits = 0.0;  // bits + the prior: the odds a pair here is a match
    uint64_t groups = 0;          // key groups holding two rows or more
    uint64_t pairs = 0;           // anchor pairs walked
    uint64_t oversized = 0;       // groups too large to be a match cluster, skipped
    bool capped = false;          // the pair budget stopped the walk
    bool used = false;            // whether it produced enough pairs to be read
};

// One column's m, averaged over the sessions that could hold it out.
struct ColumnMatchProfile {
    size_t column = 0;
    std::string name;
    unsigned sessions = 0;
    uint64_t pairs = 0;     // anchor pairs carrying the column on both rows
    double coverage = 0.0;  // share of anchor pairs that do
    double m = 0.0;         // mean over sessions
    double m_low = 0.0;     // the spread across anchors, which is the bias showing
    double m_high = 0.0;
    double weight = 0.0;         // log2(m / u): what an agreement is worth
    double expected_bits = 0.0;  // what this column contributes to a match, on average
    // Whether m hit the continuity floor rather than being measured there: a
    // session where every anchor pair agreed, or none did.
    bool floored = false;
    bool estimated = false;
};

struct ProfileReport {
    uint64_t records = 0;
    PairMode mode = PairMode::kAll;
    double pair_space = 0.0;
    uint64_t expected_matches = 0;
    double match_rate = 0.0;      // lambda: matching pairs over the pair space
    double space_bits = 0.0;      // log2(pair space)
    double prior_bits = 0.0;      // -log2(pair space / matches), always negative
    double available_bits = 0.0;  // sum of covered_bits over the scored columns
    double redundant_bits = 0.0;  // pairwise U-side overlap, as a negative number
    double margin_bits = 0.0;

    std::vector<ColumnProfile> columns;
    std::vector<ColumnPairProfile> pairs;  // worst redundancy first
    uint64_t unresolved_pairs = 0;         // pairs the match rate swamps

    std::vector<AnchorSession> sessions;
    std::vector<ColumnMatchProfile> matches;
    bool anchored = false;       // whether the M side produced anything
    std::string anchor_refusal;  // why it did not, when it did not
    uint64_t anchor_rows = 0;
    uint64_t anchor_pairs = 0;
    // The ledger's second half: what a matching pair is expected to score, rather
    // than the most it could. Estimated over the columns m was learned for, so an
    // unestimated column contributes nothing and the margin is a floor.
    double expected_bits = 0.0;
    double double_counted_bits = 0.0;
    double estimated_margin_bits = 0.0;
    size_t estimated_columns = 0;
    size_t scored_columns = 0;

    bool walked = false;  // whether the pairwise pass ran
    uint64_t sampled_rows = 0;
    bool sampled = false;
    unsigned threads = 0;
    double seconds = 0.0;
};

ProfileReport BuildProfile(const RecordStore& store, PairMode mode,
                           const ProfileOptions& options);

// What one anchor's walk cost, for a caller that wants the pairs rather than the
// agreement counts the profile folds out of them.
struct AnchorWalk {
    uint64_t groups = 0;     // key groups holding two rows or more
    uint64_t pairs = 0;      // pairs visited
    uint64_t oversized = 0;  // groups too large to be a match cluster, skipped
    bool capped = false;     // the budget stopped the walk
};

// The rows an anchor pass reads: empty when it reads every row, and otherwise the
// hash-selected sample `options.anchor_rows` asks for.
std::vector<uint64_t> AnchorRows(uint64_t records, const ProfileOptions& options);

// Walks the pairs of one anchor, two distinct rows agreeing exactly on every column
// of `anchor` in the mode's pair space, and calls `visit` on each.
//
// This is the only source of matching pairs that exists before a model does, which
// is why it is public rather than private to the report above: the M side folds
// agreement counts through it, and the level proposal folds a similarity histogram.
AnchorWalk WalkAnchorPairs(const RecordStore& store, PairMode mode,
                           const std::vector<size_t>& anchor,
                           const std::vector<uint64_t>& rows, uint64_t budget,
                           const std::function<void(uint64_t, uint64_t)>& visit);

void PrintProfileReport(const ProfileReport& report, std::ostream& out);
void WriteProfileJson(const ProfileReport& report, std::ostream& out);

}  // namespace cpplink
