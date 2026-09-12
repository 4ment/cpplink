// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/neighbourhood.hpp"
#include "cpplink/record_store.hpp"

namespace cpplink {

// What a pattern's bracket says about the threshold, before any term-frequency
// table is touched. Drop and Emit are decided from gamma alone; only Check has to
// look a value up.
enum class Zone : uint8_t {
    kDrop = 0,
    kCheck = 1,
    kEmit = 2,
    // A pattern no evaluation can produce. A comparison with five levels occupies
    // three bits, so three of the eight codes in that field name no level; the
    // packed space is larger than the reachable space and the difference has to be
    // excluded rather than indexed.
    kUnreachable = 3,
};

const char* ZoneName(Zone zone);

// One comparison level's term-frequency adjustment, resolved against the store.
//
// TF makes the weight depend on the value and not just the level -- a shared
// "Zolnerowich" is not a shared "Smith" -- which is exactly what breaks gamma as a
// sufficient statistic. It is therefore kept out of EM and applied here.
//
// On an exact level the value-specific u is p_v: given one side is v, that is the
// chance the other side is v too. On a fuzzy level the same question has the same
// shape and a different answer -- given one side is v, the chance the other lands
// anywhere in v's ball -- which is the neighbourhood mass. Two sides give two
// masses, and their geometric mean is the symmetric reading that collapses back to
// p_v when the values are equal. So a fuzzy level gets an adjustment on exactly
// the same footing as an exact one, which is what splink cannot do: there an
// exact-match level is required before any adjustment is possible at all.
struct TermFrequencyAdjustment {
    bool active = false;
    size_t comparison = 0;
    uint8_t level = 0;
    bool fuzzy = false;  // the mass comes from the ball table, not from tf
    double damping = 1.0;
    double log_u_times_records = 0.0;  // log2(u[level] * records)
    double delta_max = 0.0;            // the rarest neighbourhood in the column
    double delta_min = 0.0;            // the commonest

    const StringColumn* strings = nullptr;
    const DateColumn* dates = nullptr;
    const BooleanColumn* booleans = nullptr;
    const BallMassTable* ball = nullptr;

    // log2(u / p) for the pair, damped: p is the shared value's frequency on an
    // exact level and the geometric mean of the two neighbourhood masses on a
    // fuzzy one.
    double Delta(uint64_t a, uint64_t b) const;
    // How many records carry that value. Zero where the value is null or the
    // column keeps no term frequencies; this is what Delta is computed from on an
    // exact level, and reporting it is what makes a term-frequency move
    // explicable rather than magic.
    uint32_t Frequency(uint64_t row) const;
    // The neighbourhood mass this row's value carries at this level, for the same
    // reason.
    double Mass(uint64_t row) const;
};

struct ScoreOptions {
    double threshold = 0.0;  // bits of match weight
    double tf_damping = 1.0;
    bool use_bounds = true;  // false scores every pair exactly, for verification
    // Bound the whole pair before comparing any of it. Admissible, so false
    // changes nothing but the work done -- which is what a test compares.
    bool use_ceiling = true;
    // Apply the model's two-way corrections, where it carries any. False scores
    // the plain conditionally-independent model out of the same file, which is how
    // the two are measured against each other.
    bool use_interactions = true;
};

// The scoring model bound to a store: base weight per pattern, the term-frequency
// brackets, and the zone each pattern falls in.
class Scorer {
   public:
    // `balls` is optional: without it only exact levels are term-frequency
    // adjusted, which is where this started and where splink stops.
    bool Bind(const Model& model, const ComparisonSet& comparisons,
              const RecordStore& store, const ScoreOptions& options, std::string* error,
              const BallTables* balls = nullptr);

    // prior + sum of level weights. Everything TF-independent.
    double BaseWeight(uint32_t gamma) const;
    // The widest and narrowest the TF adjustment can be for this pattern.
    double DeltaMax(uint32_t gamma) const;
    double DeltaMin(uint32_t gamma) const;
    Zone Classify(uint32_t gamma) const;

    // An upper bound on the match weight this pair can reach, from the cheap
    // level bounds alone: no string metric is evaluated and no pattern is
    // produced. Every comparison contributes the best it could still score, so
    // the true weight can never exceed this.
    double Ceiling(uint64_t a, uint64_t b) const;
    // Whether the pair can clear the threshold at all. False means it cannot,
    // whatever the metrics would have said, so the comparison never has to run.
    bool CanReach(uint64_t a, uint64_t b) const;
    // The exact TF-adjusted weight. On an exact level both rows carry the same
    // value and only `a` is read; a fuzzy level needs both.
    double Weight(uint32_t gamma, uint64_t a, uint64_t b) const;

    // The pieces the weight is made of, for explaining one pair rather than
    // scoring billions.
    double PriorWeight() const { return prior_; }
    double LevelWeight(size_t comparison, uint8_t level) const;
    bool HasAdjustment(size_t comparison) const;
    // The term-frequency move this pair gets for one comparison: zero unless the
    // pattern puts it on a level that has one.
    double AdjustmentFor(size_t comparison, uint32_t gamma, uint64_t a, uint64_t b) const;
    // The term frequency of the row's value on the exact level the pattern puts
    // this pair on, which is the one the move above was taken from.
    uint32_t FrequencyFor(size_t comparison, uint32_t gamma, uint64_t row) const;
    // Levels of this comparison that carry an adjustment, for the report that has
    // to say which ones did.
    bool AdjustsFuzzyLevels() const;

    // The fitted two-way corrections in force, for the waterfall that has to show
    // them: without these rows it would explain a different sum from the one that
    // runs.
    size_t InteractionCount() const { return interactions_.size(); }
    const std::string& InteractionName(size_t index) const;
    double InteractionBits(size_t index, uint32_t gamma) const;

    double threshold() const { return options_.threshold; }
    bool Dense() const { return dense_; }
    // Distinct patterns an evaluation can actually produce: the product of the
    // level counts, not two to the packed width.
    uint64_t PatternSpace() const { return reachable_; }
    // How many distinct patterns fall in each zone. Dense binding only.
    uint64_t PatternsIn(Zone zone) const;
    const std::vector<TermFrequencyAdjustment>& adjustments() const {
        return adjustments_;
    }

   private:
    double ComputeBase(uint32_t gamma) const;
    bool Reachable(uint32_t gamma) const;

    // One comparison's levels in decreasing order of what they could contribute:
    // the level weight plus, on the term-frequency level, the largest adjustment
    // the column's rarest value could earn. The first level in this order that
    // the cheap bounds admit is that comparison's share of the ceiling.
    struct OptimisticLevel {
        uint8_t level = 0;
        double value = 0.0;
    };

    // One model interaction resolved against the schema: two comparison indices
    // and a table indexed by the levels they land on.
    struct BoundInteraction {
        std::string name;
        size_t left = 0;
        size_t right = 0;
        uint8_t right_levels = 0;
        std::vector<double> bits;
    };

    const ComparisonSet* comparisons_ = nullptr;
    ScoreOptions options_;
    double prior_ = 0.0;
    std::vector<std::vector<double>> weight_;  // by comparison, by level
    std::vector<TermFrequencyAdjustment> adjustments_;

    std::vector<BoundInteraction> interactions_;
    // The most the corrections can add to any pair, summed over the terms. The
    // weight stops being separable once a term is in, so the ceiling adds each
    // term's largest cell rather than the cell the pair will land on: still an
    // upper bound, and a few bits looser.
    double interaction_ceiling_ = 0.0;

    std::vector<std::vector<OptimisticLevel>> optimistic_;
    bool dense_ = false;
    uint64_t reachable_ = 0;
    std::vector<double> base_;
    std::vector<double> delta_max_;
    std::vector<double> delta_min_;
    std::vector<uint8_t> zone_;
};

// Turns a posterior probability into the match weight that reaches it.
double WeightForProbability(double probability);
// And back: 1 / (1 + 2^-w).
double ProbabilityForWeight(double weight);

}  // namespace cpplink
