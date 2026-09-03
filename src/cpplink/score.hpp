// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
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

// One comparison's term-frequency adjustment, resolved against the store.
//
// TF makes the weight depend on the value and not just the level -- a shared
// "Zolnerowich" is not a shared "Smith" -- which is exactly what breaks gamma as a
// sufficient statistic. It is therefore kept out of EM and applied here.
struct TermFrequencyAdjustment {
    bool active = false;
    size_t comparison = 0;
    uint8_t exact_level = 0;
    double damping = 1.0;
    double log_u_times_records = 0.0;  // log2(u[exact] * records)
    double delta_max = 0.0;            // rarest value in the column
    double delta_min = 0.0;            // most common value in the column

    const StringColumn* strings = nullptr;
    const DateColumn* dates = nullptr;

    // log2(u / p_v) for the value row `row` carries, damped.
    double Delta(uint64_t row) const;
    // How many records carry that value. Zero where the value is null or the
    // column keeps no term frequencies; this is what Delta is computed from, and
    // reporting it is what makes a term-frequency move explicable rather than
    // magic.
    uint32_t Frequency(uint64_t row) const;
};

struct ScoreOptions {
    double threshold = 0.0;  // bits of match weight
    double tf_damping = 1.0;
    bool use_bounds = true;  // false scores every pair exactly, for verification
};

// The scoring model bound to a store: base weight per pattern, the term-frequency
// brackets, and the zone each pattern falls in.
class Scorer {
   public:
    bool Bind(const Model& model, const ComparisonSet& comparisons,
              const RecordStore& store, const ScoreOptions& options, std::string* error);

    // prior + sum of level weights. Everything TF-independent.
    double BaseWeight(uint32_t gamma) const;
    // The widest and narrowest the TF adjustment can be for this pattern.
    double DeltaMax(uint32_t gamma) const;
    double DeltaMin(uint32_t gamma) const;
    Zone Classify(uint32_t gamma) const;
    // The exact TF-adjusted weight. Both rows agree on every TF level by
    // construction, so the value is read from `a`.
    double Weight(uint32_t gamma, uint64_t a) const;

    // The pieces the weight is made of, for explaining one pair rather than
    // scoring billions.
    double PriorWeight() const { return prior_; }
    double LevelWeight(size_t comparison, uint8_t level) const;
    bool HasAdjustment(size_t comparison) const;
    // The term-frequency move this pair gets for one comparison: zero unless the
    // pattern puts it on that comparison's exact level.
    double AdjustmentFor(size_t comparison, uint32_t gamma, uint64_t row) const;
    uint32_t FrequencyFor(size_t comparison, uint64_t row) const;

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

    const ComparisonSet* comparisons_ = nullptr;
    ScoreOptions options_;
    double prior_ = 0.0;
    std::vector<std::vector<double>> weight_;  // by comparison, by level
    std::vector<TermFrequencyAdjustment> adjustments_;

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
