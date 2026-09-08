// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/score.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cpplink {
namespace {

// Above this packed width the pattern space is too large to tabulate, so the
// base weight and the bracket are recomputed per pair. It costs one add per
// comparison, which is nothing next to the comparison itself.
constexpr uint8_t kDenseWidth = 22;

}  // namespace

const char* ZoneName(Zone zone) {
    switch (zone) {
        case Zone::kDrop:
            return "drop";
        case Zone::kCheck:
            return "check";
        case Zone::kEmit:
            return "emit";
        case Zone::kUnreachable:
            return "unreachable";
    }
    return "?";
}

uint32_t TermFrequencyAdjustment::Frequency(uint64_t row) const {
    if (strings != nullptr) {
        const uint32_t id = strings->ids[row];
        if (id == kNullId) return 0;
        return strings->tf[id];
    }
    if (dates != nullptr) {
        const int32_t value = dates->values[row];
        if (value == kNullDate) return 0;
        return dates->tf[static_cast<size_t>(value - dates->tf_origin)];
    }
    return 0;
}

double TermFrequencyAdjustment::Mass(uint64_t row) const {
    if (ball == nullptr || strings == nullptr) return 0.0;
    const uint32_t id = strings->ids[row];
    if (id == kNullId) return 0.0;
    return ball->Mass(level, id);
}

double TermFrequencyAdjustment::Delta(uint64_t a, uint64_t b) const {
    if (!fuzzy) {
        const uint32_t frequency = Frequency(a);
        if (frequency == 0) return 0.0;
        // log2(u / p_v) with p_v = tf / records, folded so only one log is taken.
        return damping *
               (log_u_times_records - std::log2(static_cast<double>(frequency)));
    }
    // The two sides sit in different neighbourhoods, and the geometric mean is
    // the symmetric reading: it is p_v again when the values are equal, and it
    // moves half as far as either side would on its own.
    const double mass_a = Mass(a);
    const double mass_b = Mass(b);
    if (mass_a <= 0.0 || mass_b <= 0.0) return 0.0;
    const double log_records = log_u_times_records;
    return damping * (log_records - 0.5 * (std::log2(mass_a) + std::log2(mass_b)));
}

double WeightForProbability(double probability) {
    if (probability <= 0.0) return -std::numeric_limits<double>::infinity();
    if (probability >= 1.0) return std::numeric_limits<double>::infinity();
    return std::log2(probability / (1.0 - probability));
}

double ProbabilityForWeight(double weight) { return 1.0 / (1.0 + std::exp2(-weight)); }

bool Scorer::Bind(const Model& model, const ComparisonSet& comparisons,
                  const RecordStore& store, const ScoreOptions& options,
                  std::string* error, const BallTables* balls) {
    comparisons_ = &comparisons;
    options_ = options;
    prior_ = model.PriorWeight();
    weight_.assign(comparisons.Size(), {});
    adjustments_.clear();

    if (model.comparisons.size() != comparisons.Size()) {
        *error = "the model describes " + std::to_string(model.comparisons.size()) +
                 " comparisons but the schema declares " +
                 std::to_string(comparisons.Size());
        return false;
    }

    const double records = static_cast<double>(store.NumRecords());
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        const ComparisonSpec& spec = *comparisons.at(c).spec;
        const ModelComparison& learned = model.comparisons[c];
        if (learned.name != spec.name) {
            *error = "the model's comparison " + std::to_string(c) + " is \"" +
                     learned.name + "\" but the schema's is \"" + spec.name + "\"";
            return false;
        }
        if (learned.levels.size() != spec.levels.size()) {
            *error = "comparison \"" + spec.name + "\" has " +
                     std::to_string(spec.levels.size()) + " levels but the model has " +
                     std::to_string(learned.levels.size());
            return false;
        }
        weight_[c].resize(learned.levels.size());
        for (size_t l = 0; l < learned.levels.size(); ++l) {
            weight_[c][l] = learned.levels[l].Weight();
        }

        if (!spec.term_frequency) continue;
        // The fuzzy levels first, where a neighbourhood mass exists for them.
        // Their u is log2(u) alone rather than log2(u * records), because the mass
        // is already a probability where a term frequency is a count.
        if (balls != nullptr && balls->Has(c)) {
            const BallMassTable& table = balls->tables[c];
            for (size_t l = 0; l < spec.levels.size(); ++l) {
                const LevelType type = spec.levels[l].type;
                if (type != LevelType::kLevenshtein && type != LevelType::kJaroWinkler) {
                    continue;
                }
                if (!table.Covers(l)) continue;
                const double min_mass = table.MinMass(l);
                const double max_mass = table.MaxMass(l);
                if (min_mass <= 0.0 || max_mass <= 0.0) continue;
                if (learned.levels[l].u <= 0.0) continue;
                TermFrequencyAdjustment fuzzy;
                fuzzy.comparison = c;
                fuzzy.level = static_cast<uint8_t>(l);
                fuzzy.fuzzy = true;
                fuzzy.damping = options.tf_damping;
                fuzzy.log_u_times_records = std::log2(learned.levels[l].u);
                fuzzy.strings = comparisons.at(c).strings;
                fuzzy.ball = &table;
                fuzzy.delta_max =
                    fuzzy.damping * (fuzzy.log_u_times_records - std::log2(min_mass));
                fuzzy.delta_min =
                    fuzzy.damping * (fuzzy.log_u_times_records - std::log2(max_mass));
                fuzzy.active = true;
                adjustments_.push_back(fuzzy);
            }
        }

        TermFrequencyAdjustment adjustment;
        adjustment.comparison = c;
        adjustment.damping = options.tf_damping;
        bool has_exact = false;
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            if (spec.levels[l].type == LevelType::kExact) {
                adjustment.level = static_cast<uint8_t>(l);
                adjustment.log_u_times_records = std::log2(learned.levels[l].u * records);
                has_exact = true;
                break;
            }
        }
        // A list column's term frequencies count values, not sets, so an exact
        // level over a whole set has no frequency to look up. Doubles have no
        // frequencies at all. Both simply get no adjustment.
        const BoundComparison& bound = comparisons.at(c);
        if (bound.strings != nullptr && bound.lists == nullptr) {
            adjustment.strings = bound.strings;
        } else if (bound.dates != nullptr && bound.lists == nullptr) {
            adjustment.dates = bound.dates;
        }
        if (!has_exact ||
            (adjustment.strings == nullptr && adjustment.dates == nullptr)) {
            continue;
        }

        // The bracket. delta_max comes from the rarest value the column holds and
        // delta_min from the most common, so it brackets every pair that can
        // occur. Both are exact, which is what makes the drop and emit decisions
        // give bit-identical output to scoring everything.
        uint32_t rarest = std::numeric_limits<uint32_t>::max();
        uint32_t commonest = 0;
        const std::vector<uint32_t>& tf =
            adjustment.strings != nullptr ? adjustment.strings->tf : adjustment.dates->tf;
        for (const uint32_t frequency : tf) {
            if (frequency == 0) continue;
            rarest = std::min(rarest, frequency);
            commonest = std::max(commonest, frequency);
        }
        if (commonest == 0) continue;  // the column is entirely null
        adjustment.delta_max =
            adjustment.damping *
            (adjustment.log_u_times_records - std::log2(static_cast<double>(rarest)));
        adjustment.delta_min =
            adjustment.damping *
            (adjustment.log_u_times_records - std::log2(static_cast<double>(commonest)));
        adjustment.active = true;
        adjustments_.push_back(adjustment);
    }

    // The two-way corrections, resolved to comparison indices once. A term names
    // its comparisons rather than their positions, so a schema whose comparisons
    // have been reordered still gets the term it was fitted with -- and one whose
    // levels have changed is refused, because the table is indexed by level.
    interactions_.clear();
    interaction_ceiling_ = 0.0;
    if (options.use_interactions) {
        for (const ModelInteraction& term : model.interactions) {
            BoundInteraction bound;
            bound.name = term.left + " x " + term.right;
            bool found_left = false;
            bool found_right = false;
            for (size_t c = 0; c < comparisons.Size(); ++c) {
                if (comparisons.at(c).spec->name == term.left) {
                    bound.left = c;
                    found_left = true;
                }
                if (comparisons.at(c).spec->name == term.right) {
                    bound.right = c;
                    found_right = true;
                }
            }
            if (!found_left || !found_right) {
                *error = "the model has an interaction on \"" + bound.name +
                         "\" but the schema declares no such comparison";
                return false;
            }
            const size_t height = comparisons.at(bound.left).spec->levels.size();
            const size_t width = comparisons.at(bound.right).spec->levels.size();
            if (term.left_levels != height || term.right_levels != width) {
                *error = "the interaction on \"" + bound.name + "\" was fitted over " +
                         std::to_string(term.left_levels) + " by " +
                         std::to_string(term.right_levels) +
                         " levels but the schema declares " + std::to_string(height) +
                         " by " + std::to_string(width);
                return false;
            }
            bound.right_levels = static_cast<uint8_t>(width);
            bound.bits = term.bits;
            double widest = 0.0;
            for (const double value : bound.bits) widest = std::max(widest, value);
            interaction_ceiling_ += widest;
            interactions_.push_back(std::move(bound));
        }
    }

    // The ceiling's table. A level is worth its own weight plus, where it is the
    // term-frequency level, the most any value in that column could add; sorting
    // by that lets the ceiling stop at the first level the cheap bounds admit.
    optimistic_.assign(comparisons.Size(), {});
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        const size_t levels = comparisons.at(c).spec->levels.size();
        optimistic_[c].reserve(levels);
        for (size_t l = 0; l < levels; ++l) {
            OptimisticLevel entry;
            entry.level = static_cast<uint8_t>(l);
            entry.value = weight_[c][l];
            for (const TermFrequencyAdjustment& adjustment : adjustments_) {
                if (adjustment.comparison == c && adjustment.level == l) {
                    entry.value += adjustment.delta_max;
                }
            }
            optimistic_[c].push_back(entry);
        }
        std::sort(optimistic_[c].begin(), optimistic_[c].end(),
                  [](const OptimisticLevel& x, const OptimisticLevel& y) {
                      return x.value > y.value;
                  });
    }

    reachable_ = 1;
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        reachable_ *= comparisons.at(c).spec->levels.size();
    }
    dense_ = comparisons.Width() <= kDenseWidth;
    base_.clear();
    delta_max_.clear();
    delta_min_.clear();
    zone_.clear();
    if (!dense_) return true;

    const size_t span = size_t{1} << comparisons.Width();
    base_.resize(span);
    delta_max_.resize(span);
    delta_min_.resize(span);
    zone_.resize(span);
    for (size_t gamma = 0; gamma < span; ++gamma) {
        const uint32_t packed = static_cast<uint32_t>(gamma);
        if (!Reachable(packed)) {
            zone_[gamma] = static_cast<uint8_t>(Zone::kUnreachable);
            continue;
        }
        base_[gamma] = ComputeBase(packed);
        double high = 0.0;
        double low = 0.0;
        for (const TermFrequencyAdjustment& adjustment : adjustments_) {
            if (comparisons.LevelOf(packed, adjustment.comparison) != adjustment.level) {
                continue;
            }
            high += adjustment.delta_max;
            low += adjustment.delta_min;
        }
        delta_max_[gamma] = high;
        delta_min_[gamma] = low;
        Zone zone = Zone::kCheck;
        if (base_[gamma] + high < options_.threshold) {
            zone = Zone::kDrop;
        } else if (base_[gamma] + low >= options_.threshold) {
            zone = Zone::kEmit;
        }
        zone_[gamma] = static_cast<uint8_t>(zone);
    }
    return true;
}

bool Scorer::Reachable(uint32_t gamma) const {
    for (size_t c = 0; c < weight_.size(); ++c) {
        if (comparisons_->LevelOf(gamma, c) >= weight_[c].size()) return false;
    }
    return true;
}

double Scorer::ComputeBase(uint32_t gamma) const {
    double total = prior_;
    for (size_t c = 0; c < weight_.size(); ++c) {
        total += weight_[c][comparisons_->LevelOf(gamma, c)];
    }
    // The corrections are a function of gamma like everything else here, which is
    // why the tabulated path below 22 bits does not pay for them at all: they are
    // folded into base_ once at bind time and the hot loop never sees them.
    for (const BoundInteraction& interaction : interactions_) {
        total += interaction.bits[comparisons_->LevelOf(gamma, interaction.left) *
                                      interaction.right_levels +
                                  comparisons_->LevelOf(gamma, interaction.right)];
    }
    return total;
}

const std::string& Scorer::InteractionName(size_t index) const {
    return interactions_[index].name;
}

double Scorer::InteractionBits(size_t index, uint32_t gamma) const {
    const BoundInteraction& interaction = interactions_[index];
    return interaction
        .bits[comparisons_->LevelOf(gamma, interaction.left) * interaction.right_levels +
              comparisons_->LevelOf(gamma, interaction.right)];
}

double Scorer::BaseWeight(uint32_t gamma) const {
    return dense_ ? base_[gamma] : ComputeBase(gamma);
}

double Scorer::DeltaMax(uint32_t gamma) const {
    if (dense_) return delta_max_[gamma];
    double high = 0.0;
    for (const TermFrequencyAdjustment& adjustment : adjustments_) {
        if (comparisons_->LevelOf(gamma, adjustment.comparison) == adjustment.level) {
            high += adjustment.delta_max;
        }
    }
    return high;
}

double Scorer::DeltaMin(uint32_t gamma) const {
    if (dense_) return delta_min_[gamma];
    double low = 0.0;
    for (const TermFrequencyAdjustment& adjustment : adjustments_) {
        if (comparisons_->LevelOf(gamma, adjustment.comparison) == adjustment.level) {
            low += adjustment.delta_min;
        }
    }
    return low;
}

Zone Scorer::Classify(uint32_t gamma) const {
    if (!options_.use_bounds) return Zone::kCheck;
    if (dense_) {
        const Zone zone = static_cast<Zone>(zone_[gamma]);
        // Evaluate never produces an unreachable pattern, so reaching one would be
        // a packing bug; dropping it is the safe reading either way.
        return zone == Zone::kUnreachable ? Zone::kDrop : zone;
    }
    const double base = ComputeBase(gamma);
    if (base + DeltaMax(gamma) < options_.threshold) return Zone::kDrop;
    if (base + DeltaMin(gamma) >= options_.threshold) return Zone::kEmit;
    return Zone::kCheck;
}

double Scorer::Ceiling(uint64_t a, uint64_t b) const {
    // A correction is not separable across comparisons, so the pair's own cell is
    // not known until the levels are. Adding each term's largest cell keeps the
    // bound admissible at the cost of loosening it by that much.
    double total = prior_ + interaction_ceiling_;
    for (size_t c = 0; c < optimistic_.size(); ++c) {
        // Levels are in decreasing order of what they are worth, so the first one
        // the bounds admit is the most this comparison can contribute. The last
        // level of every comparison is "else", which is always possible, so this
        // loop always finds one.
        for (const OptimisticLevel& entry : optimistic_[c]) {
            if (comparisons_->LevelPossible(c, entry.level, a, b)) {
                total += entry.value;
                break;
            }
        }
    }
    return total;
}

bool Scorer::CanReach(uint64_t a, uint64_t b) const {
    if (!options_.use_ceiling) return true;
    return Ceiling(a, b) >= options_.threshold;
}

double Scorer::Weight(uint32_t gamma, uint64_t a, uint64_t b) const {
    double total = BaseWeight(gamma);
    for (const TermFrequencyAdjustment& adjustment : adjustments_) {
        if (comparisons_->LevelOf(gamma, adjustment.comparison) == adjustment.level) {
            total += adjustment.Delta(a, b);
        }
    }
    return total;
}

double Scorer::LevelWeight(size_t comparison, uint8_t level) const {
    if (comparison >= weight_.size()) return 0.0;
    if (level >= weight_[comparison].size()) return 0.0;
    return weight_[comparison][level];
}

bool Scorer::HasAdjustment(size_t comparison) const {
    for (const TermFrequencyAdjustment& adjustment : adjustments_) {
        if (adjustment.comparison == comparison) return true;
    }
    return false;
}

double Scorer::AdjustmentFor(size_t comparison, uint32_t gamma, uint64_t a,
                             uint64_t b) const {
    for (const TermFrequencyAdjustment& adjustment : adjustments_) {
        if (adjustment.comparison != comparison) continue;
        if (comparisons_->LevelOf(gamma, comparison) != adjustment.level) continue;
        return adjustment.Delta(a, b);
    }
    return 0.0;
}

bool Scorer::AdjustsFuzzyLevels() const {
    for (const TermFrequencyAdjustment& adjustment : adjustments_) {
        if (adjustment.fuzzy) return true;
    }
    return false;
}

uint32_t Scorer::FrequencyFor(size_t comparison, uint64_t row) const {
    for (const TermFrequencyAdjustment& adjustment : adjustments_) {
        if (adjustment.comparison == comparison && !adjustment.fuzzy) {
            return adjustment.Frequency(row);
        }
    }
    return 0;
}

uint64_t Scorer::PatternsIn(Zone zone) const {
    uint64_t count = 0;
    for (const uint8_t value : zone_) {
        if (static_cast<Zone>(value) == zone) ++count;
    }
    return count;
}

}  // namespace cpplink
