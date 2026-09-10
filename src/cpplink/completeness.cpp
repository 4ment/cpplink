// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/completeness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpplink/format.hpp"
#include "cpplink/pair_stream.hpp"

namespace cpplink {
namespace {

constexpr uint8_t kDenseWidth = 22;

uint64_t Mix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

double PairsIn(double n) { return n * (n - 1.0) / 2.0; }

std::string Fixed(double value, int places) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", places, value);
    return buffer;
}

// The chance two rows sharing a value land within `window` of each other in the
// sorted order. Rows carrying one value are contiguous and ordered by row, so the
// two positions are a pair drawn from n, and the pairs within the window are
// counted rather than assumed.
double WindowHitRate(double n, double window) {
    if (n < 2.0) return 0.0;
    const double k = std::min(window, n - 1.0);
    if (k <= 0.0) return 0.0;
    const double within = k * n - k * (k + 1.0) / 2.0;
    return within / PairsIn(n);
}

// Per-pattern capture counts. Dense below the same width the histogram uses, and
// a map above it, because the pattern space is what it is and neither container
// is interesting enough to argue about.
class CaptureCounts {
   public:
    explicit CaptureCounts(uint8_t width) : dense_(width <= kDenseWidth) {
        if (dense_) counts_.assign(size_t{2} << width, 0);
    }

    void Add(uint32_t gamma, bool both) {
        if (dense_) {
            counts_[size_t{gamma} * 2 + (both ? 1 : 0)] += 1;
            return;
        }
        std::pair<uint64_t, uint64_t>& entry = sparse_[gamma];
        entry.first += 1;
        if (both) entry.second += 1;
    }

    void Merge(const CaptureCounts& other) {
        if (dense_) {
            for (size_t i = 0; i < counts_.size(); ++i) counts_[i] += other.counts_[i];
            return;
        }
        for (const auto& entry : other.sparse_) {
            std::pair<uint64_t, uint64_t>& mine = sparse_[entry.first];
            mine.first += entry.second.first;
            mine.second += entry.second.second;
        }
    }

    // Pairs at this pattern that an analytic source captured, and how many of
    // those an observed-class source also captured.
    std::pair<uint64_t, uint64_t> At(uint32_t gamma) const {
        if (dense_) {
            const uint64_t analytic = counts_[size_t{gamma} * 2];
            return {analytic, counts_[size_t{gamma} * 2 + 1]};
        }
        const auto it = sparse_.find(gamma);
        if (it == sparse_.end()) return {0, 0};
        return it->second;
    }

   private:
    bool dense_ = false;
    std::vector<uint64_t> counts_;
    std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> sparse_;
};

// Per-thread, one cache line each: touched once per candidate pair.
struct alignas(64) FoldTally {
    uint64_t seen = 0;
    uint64_t analytic = 0;
    uint64_t both = 0;
    uint64_t window_denominator = 0;
    uint64_t window_numerator = 0;
    double window_expected = 0.0;
    uint64_t random = 0;
};

// How many records carry the value a row holds, on one source's column.
uint32_t ValueFrequency(const BoundSource& source, uint64_t row) {
    if (source.strings != nullptr) {
        const uint32_t id = source.strings->ids[row];
        return id == kNullId ? 0 : source.strings->tf[id];
    }
    if (source.dates != nullptr) {
        const int32_t value = source.dates->values[row];
        if (value == kNullDate) return 0;
        return source.dates->tf[static_cast<size_t>(value - source.dates->tf_origin)];
    }
    return 0;
}

const std::vector<uint32_t>* FrequencyTable(const BoundSource& source) {
    if (source.strings != nullptr) return &source.strings->tf;
    if (source.dates != nullptr) return &source.dates->tf;
    if (source.lists != nullptr) return &source.lists->tf;
    return nullptr;
}

// P(the shared value of a matching pair is under the cap). The weight is what a
// matching pair's value is drawn like: by record, because a duplicate copies some
// record's value, or by pair, which is what a random agreeing pair does.
double MassUnderCap(const std::vector<uint32_t>& tf, uint32_t cap, bool pair_weighting) {
    double under = 0.0;
    double total = 0.0;
    for (const uint32_t frequency : tf) {
        if (frequency < 2) continue;  // a value one record holds forms no pair
        const double n = static_cast<double>(frequency);
        const double weight = pair_weighting ? PairsIn(n) : n;
        total += weight;
        if (cap == 0 || frequency <= cap) under += weight;
    }
    if (total <= 0.0) return 0.0;
    return under / total;
}

// The model's posterior for a pattern, without the term-frequency move: the
// average pair at this pattern, which is what a cell of the table wants.
class Posterior {
   public:
    Posterior(const Model& model, const ComparisonSet& comparisons) {
        prior_ = model.PriorWeight();
        weight_.resize(comparisons.Size());
        for (size_t c = 0; c < comparisons.Size(); ++c) {
            const ModelComparison& learned = model.comparisons[c];
            weight_[c].resize(learned.levels.size());
            for (size_t l = 0; l < learned.levels.size(); ++l) {
                weight_[c][l] = learned.levels[l].Weight();
            }
        }
        comparisons_ = &comparisons;
    }

    double At(uint32_t gamma) const {
        double total = prior_;
        for (size_t c = 0; c < weight_.size(); ++c) {
            const uint8_t level = comparisons_->LevelOf(gamma, c);
            if (level < weight_[c].size()) total += weight_[c][level];
        }
        return 1.0 / (1.0 + std::exp2(-total));
    }

   private:
    const ComparisonSet* comparisons_ = nullptr;
    double prior_ = 0.0;
    std::vector<std::vector<double>> weight_;
};

// One blocked comparison: a dimension of the capture table.
struct BlockedColumn {
    size_t comparison = 0;
    uint8_t levels = 0;
    uint8_t exact_level = 0;
    double fires = 0.0;   // P(some analytic source on this column fires | exact)
    uint64_t stride = 0;  // mixed-radix stride into the table
};

// Iterative proportional fitting to a set of margins. Each margin is a subset of
// the table's dimensions; fitting to every one-way margin is the independence
// model, and adding every two-way margin is the log-linear model with all
// pairwise interactions and nothing higher.
//
// The table is incomplete -- the dark cells were never observed -- so IPF sits
// inside an EM loop that fills them from the current fit. This is Fienberg's
// treatment of an incomplete contingency table, and it is what a capture-recapture
// estimate with dependent lists has always been.
void FitLogLinear(const std::vector<BlockedColumn>& columns,
                  const std::vector<std::vector<size_t>>& margins,
                  const std::vector<double>& observed, const std::vector<bool>& dark,
                  int outer, int inner, std::vector<double>* fitted) {
    const size_t cells = observed.size();
    fitted->assign(cells, 1.0);
    std::vector<double> complete(cells, 0.0);
    std::vector<double> margin_observed;
    std::vector<double> margin_fitted;
    for (int step = 0; step < outer; ++step) {
        for (size_t cell = 0; cell < cells; ++cell) {
            complete[cell] = dark[cell] ? (*fitted)[cell] : observed[cell];
        }
        std::vector<double>& x = *fitted;
        x.assign(cells, 1.0);
        for (int pass = 0; pass < inner; ++pass) {
            for (const std::vector<size_t>& margin : margins) {
                size_t span = 1;
                for (const size_t d : margin) span *= columns[d].levels;
                margin_observed.assign(span, 0.0);
                margin_fitted.assign(span, 0.0);
                for (size_t cell = 0; cell < cells; ++cell) {
                    size_t index = 0;
                    for (const size_t d : margin) {
                        const size_t level =
                            (cell / columns[d].stride) % columns[d].levels;
                        index = index * columns[d].levels + level;
                    }
                    margin_observed[index] += complete[cell];
                    margin_fitted[index] += x[cell];
                }
                for (size_t cell = 0; cell < cells; ++cell) {
                    size_t index = 0;
                    for (const size_t d : margin) {
                        const size_t level =
                            (cell / columns[d].stride) % columns[d].levels;
                        index = index * columns[d].levels + level;
                    }
                    if (margin_fitted[index] <= 0.0) {
                        x[cell] = 0.0;
                    } else {
                        x[cell] *= margin_observed[index] / margin_fitted[index];
                    }
                }
            }
        }
    }
}

// The likelihood-ratio deviance of a fit against the cells that were actually
// seen. The dark cells are excluded: a model is judged on what it explains, not
// on what it was asked to invent.
double Deviance(const std::vector<double>& observed, const std::vector<double>& fitted,
                const std::vector<bool>& dark) {
    double g2 = 0.0;
    for (size_t cell = 0; cell < observed.size(); ++cell) {
        if (dark[cell]) continue;
        const double o = observed[cell];
        const double e = fitted[cell];
        if (o <= 0.0 || e <= 0.0) continue;
        g2 += 2.0 * o * std::log(o / e);
    }
    return g2;
}

// Free parameters of a log-linear model with the given margins, over a table with
// these dimensions: one for the total, the main effects, and one interaction per
// pair of levels for every two-way margin.
uint64_t ParameterCount(const std::vector<BlockedColumn>& columns,
                        const std::vector<std::vector<size_t>>& margins) {
    uint64_t count = 1;
    for (const std::vector<size_t>& margin : margins) {
        uint64_t term = 1;
        for (const size_t d : margin) term *= columns[d].levels - 1;
        count += term;
    }
    return count;
}

// The comparison that reads exactly this one column and has an exact level. A
// source blocks on a column; gamma speaks about comparisons; this is the join,
// and where it fails the source simply contributes nothing, which keeps the
// bound on the safe side.
bool FindExactComparison(const ComparisonSet& comparisons, const std::string& column,
                         size_t* index, uint8_t* level) {
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        const ComparisonSpec& spec = *comparisons.at(c).spec;
        if (spec.columns.size() != 1 || spec.columns[0] != column) continue;
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            if (spec.levels[l].type != LevelType::kExact) continue;
            *index = c;
            *level = static_cast<uint8_t>(l);
            return true;
        }
    }
    return false;
}

}  // namespace

bool EstimateCompleteness(const RecordStore& store, const ComparisonSet& comparisons,
                          const BlockingPlan& plan, const Model& model,
                          const CompletenessOptions& options, CompletenessReport* report,
                          std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    report->records = store.NumRecords();
    if (model.comparisons.size() != comparisons.Size()) {
        *error = "the model describes " + std::to_string(model.comparisons.size()) +
                 " comparisons but the schema declares " +
                 std::to_string(comparisons.Size());
        return false;
    }
    if (plan.Size() == 0) {
        *error = "the plan has no blocking sources, so there is nothing to score";
        return false;
    }
    // An unblocked source produces every pair the mode admits, so there is no
    // dark cell to fit and no firing probability to model: pair completeness is
    // one exactly, and saying so is more honest than estimating it.
    for (size_t s = 0; s < plan.Size(); ++s) {
        if (plan.at(s).kind != SourceKind::kAllPairs) continue;
        report->unblocked = true;
        report->pc_bound = 1.0;
        report->pc_analytic = 1.0;
        report->pc_product = 1.0;
        report->pc_estimate = 1.0;
        report->pc_basis = "unblocked";
        report->seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
        return true;
    }

    // Every source's firing probability, and which class supplies it.
    std::vector<size_t> analytic_sources;
    std::vector<size_t> observed_sources;
    size_t window_source = plan.Size();
    for (size_t s = 0; s < plan.Size(); ++s) {
        const BoundSource& source = plan.at(s);
        SourceCapture capture;
        capture.name = source.name;
        capture.column = source.column;
        capture.kind = source.kind;
        capture.bound_to_comparison = FindExactComparison(
            comparisons, source.column, &capture.comparison, &capture.exact_level);
        const std::vector<uint32_t>* tf = FrequencyTable(source);
        switch (source.kind) {
            case SourceKind::kExactValue:
            case SourceKind::kRareValue:
                capture.capture = CaptureClass::kAnalytic;
                if (capture.bound_to_comparison && tf != nullptr) {
                    capture.fires_given_exact =
                        MassUnderCap(*tf, source.max_frequency, options.pair_weighting);
                }
                break;
            case SourceKind::kAllPairs:  // handled above; the plan reaches every pair
            case SourceKind::kSortedNeighbourhood:
            case SourceKind::kMinHash:
                capture.capture = CaptureClass::kObserved;
                if (source.kind == SourceKind::kSortedNeighbourhood &&
                    capture.bound_to_comparison && tf != nullptr) {
                    capture.has_window_check = true;
                    if (window_source == plan.Size()) window_source = s;
                }
                break;
        }
        if (capture.capture == CaptureClass::kAnalytic &&
            capture.fires_given_exact > 0.0) {
            analytic_sources.push_back(s);
        } else if (capture.capture == CaptureClass::kObserved) {
            observed_sources.push_back(s);
        }
        report->sources.push_back(capture);
    }
    // EM holds out every comparison reading the column a session conditions on,
    // so a column that every source blocks on has no session left to learn its m.
    // The estimate is a sum weighted by exactly those m, so this is not a caveat
    // to mention in passing: it is the condition under which the answer means
    // anything, and it is readable off the model.
    for (const size_t s : analytic_sources) {
        const SourceCapture& capture = report->sources[s];
        if (!capture.bound_to_comparison) continue;
        if (model.comparisons[capture.comparison].sessions > 0) continue;
        report->unlearned.push_back(model.comparisons[capture.comparison].name);
        report->trusted = false;
    }
    std::sort(report->unlearned.begin(), report->unlearned.end());
    report->unlearned.erase(
        std::unique(report->unlearned.begin(), report->unlearned.end()),
        report->unlearned.end());

    if (analytic_sources.empty()) {
        *error =
            "no source has a firing probability this can compute: an exact-value or "
            "rare-value source on a column with an exact comparison level is what "
            "makes the bound possible";
        return false;
    }

    // Analytic sources sharing a column are combined by taking the strongest,
    // because a rare-value source produces a subset of what an exact-value source
    // on the same column produces. Only across columns is anything assumed.
    struct ColumnCapture {
        size_t comparison = 0;
        uint8_t exact_level = 0;
        double fires = 0.0;
    };
    std::vector<ColumnCapture> columns;
    for (const size_t s : analytic_sources) {
        const SourceCapture& capture = report->sources[s];
        bool merged = false;
        for (ColumnCapture& column : columns) {
            if (column.comparison != capture.comparison) continue;
            column.fires = std::max(column.fires, capture.fires_given_exact);
            merged = true;
            break;
        }
        if (merged) continue;
        ColumnCapture column;
        column.comparison = capture.comparison;
        column.exact_level = capture.exact_level;
        column.fires = capture.fires_given_exact;
        columns.push_back(column);
    }

    // What the overlap may be conditioned on. An observed-class source on the
    // same column as the analytic source doing the conditioning is not a second
    // look at the pair: both fire because that column agrees on a rare value, and
    // reading the overlap that way says the window catches everything the cap
    // caught. Only analytic sources on other columns can ask the question.
    std::vector<size_t> conditioning;
    for (const size_t s : analytic_sources) {
        bool shares_column = false;
        for (const size_t o : observed_sources) {
            if (plan.at(o).column == plan.at(s).column) shares_column = true;
        }
        if (!shares_column) conditioning.push_back(s);
    }
    report->correction_available = !conditioning.empty() && !observed_sources.empty();

    // The table's dimensions: one per comparison some analytic source blocks on.
    std::vector<BlockedColumn> blocked;
    for (const ColumnCapture& column : columns) {
        BlockedColumn dimension;
        dimension.comparison = column.comparison;
        dimension.levels =
            static_cast<uint8_t>(comparisons.at(column.comparison).spec->levels.size());
        dimension.exact_level = column.exact_level;
        dimension.fires = column.fires;
        blocked.push_back(dimension);
    }
    uint64_t table_cells = 1;
    for (BlockedColumn& dimension : blocked) {
        dimension.stride = table_cells;
        table_cells *= dimension.levels;
    }
    report->cells = table_cells;

    // The capture fold: which class produced each candidate, by pattern. Nothing
    // here holds a row per pair either.
    const unsigned threads = ResolveThreads(options.threads);
    CaptureCounts counts(comparisons.Width());
    std::vector<FoldTally> tally(threads);
    const bool sampling = options.sample < 1.0;
    const uint64_t sample_cut =
        sampling ? static_cast<uint64_t>(std::max(0.0, options.sample) *
                                         18446744073709549568.0)
                 : 0;
    const bool window_check = window_source < plan.Size();
    const size_t window_comparison =
        window_check ? report->sources[window_source].comparison : 0;
    const uint8_t window_level =
        window_check ? report->sources[window_source].exact_level : 0;

    Posterior posterior(model, comparisons);
    std::vector<double> cell_matches(table_cells, 0.0);
    if (!options.skip_observed) {
        std::vector<CaptureCounts> local(threads, CaptureCounts(comparisons.Width()));
        std::vector<std::vector<double>> cells(threads,
                                               std::vector<double>(table_cells, 0.0));
        for (unsigned t = 0; t < threads; ++t) {
            tally[t].random = Mix64(options.seed + t * 0x9E3779B97F4A7C15ull);
        }
        ForEachPairParallel(plan, plan.AllSources(), threads, [&](unsigned t) {
            FoldTally* fold = &tally[t];
            CaptureCounts* mine = &local[t];
            std::vector<double>* table = &cells[t];
            return [&, fold, mine, table](uint32_t a, uint32_t b) {
                if (sampling) {
                    fold->random = Mix64(fold->random);
                    if (fold->random >= sample_cut) return;
                }
                ++fold->seen;
                bool captured = false;
                for (const size_t s : analytic_sources) {
                    if (plan.Produces(s, a, b)) {
                        captured = true;
                        break;
                    }
                }
                // The table counts only analytic captures, because those are the
                // ones whose cell completeness is known. A pair only a window
                // reached tells you nothing about how full its cell is.
                if (!captured) return;
                const uint32_t gamma = comparisons.Evaluate(a, b);
                uint64_t cell = 0;
                for (const BlockedColumn& dimension : blocked) {
                    cell += static_cast<uint64_t>(
                                comparisons.LevelOf(gamma, dimension.comparison)) *
                            dimension.stride;
                }
                (*table)[cell] += posterior.At(gamma);

                if (!report->correction_available) return;
                bool analytic = false;
                for (const size_t s : conditioning) {
                    if (plan.Produces(s, a, b)) {
                        analytic = true;
                        break;
                    }
                }
                // Only a pair the conditioning set reached can say anything about
                // the observed class: it is the one part of the table where both
                // captures are visible and uncorrelated.
                if (!analytic) return;
                bool observed = false;
                for (const size_t s : observed_sources) {
                    if (plan.Produces(s, a, b)) {
                        observed = true;
                        break;
                    }
                }
                mine->Add(gamma, observed);
                ++fold->analytic;
                if (observed) ++fold->both;
                if (window_check &&
                    comparisons.LevelOf(gamma, window_comparison) == window_level) {
                    ++fold->window_denominator;
                    if (plan.Produces(window_source, a, b)) ++fold->window_numerator;
                    // The prediction is accumulated over the same pairs the
                    // observation is, because those pairs are not a random sample
                    // of agreeing pairs: they reached here through the analytic
                    // sources, which favour rare values, and comparing against an
                    // average over all values would be comparing two different
                    // questions.
                    fold->window_expected += WindowHitRate(
                        static_cast<double>(ValueFrequency(plan.at(window_source), a)),
                        static_cast<double>(plan.at(window_source).window));
                }
            };
        });
        for (unsigned t = 0; t < threads; ++t) {
            counts.Merge(local[t]);
            for (uint64_t cell = 0; cell < table_cells; ++cell) {
                cell_matches[cell] += cells[t][cell];
            }
        }
        report->walked = true;
    }
    for (unsigned t = 0; t < threads; ++t) {
        report->observed_pairs += tally[t].seen;
        report->analytic_captured += tally[t].analytic;
        report->both_captured += tally[t].both;
    }
    if (report->analytic_captured > 0) {
        report->pooled_rate = static_cast<double>(report->both_captured) /
                              static_cast<double>(report->analytic_captured);
    }
    if (window_check) {
        uint64_t denominator = 0;
        uint64_t numerator = 0;
        double expected = 0.0;
        for (unsigned t = 0; t < threads; ++t) {
            denominator += tally[t].window_denominator;
            numerator += tally[t].window_numerator;
            expected += tally[t].window_expected;
        }
        if (denominator > 0) {
            report->sources[window_source].window_observed =
                static_cast<double>(numerator) / static_cast<double>(denominator);
            report->sources[window_source].window_analytic =
                expected / static_cast<double>(denominator);
        }
    }

    // The sum. An odometer over the level tuples walks the reachable patterns and
    // nothing else -- the packed space is larger, and its extra codes name no
    // level and carry no mass.
    const size_t n = comparisons.Size();
    std::vector<uint8_t> level(n, 0);
    std::vector<uint8_t> levels(n, 0);
    for (size_t c = 0; c < n; ++c) {
        levels[c] = static_cast<uint8_t>(comparisons.at(c).spec->levels.size());
    }
    double mass_total = 0.0;
    double bound = 0.0;
    double analytic_total = 0.0;
    double estimate = 0.0;
    double dark_mass = 0.0;
    uint64_t dark_patterns = 0;
    uint64_t patterns = 0;
    uint64_t own_rate = 0;
    for (;;) {
        double mass = 1.0;
        uint32_t gamma = 0;
        for (size_t c = 0; c < n; ++c) {
            mass *= model.comparisons[c].levels[level[c]].m;
            gamma |= static_cast<uint32_t>(level[c]) << comparisons.at(c).shift;
        }
        ++patterns;
        mass_total += mass;

        double best = 0.0;
        double none = 1.0;
        for (const ColumnCapture& column : columns) {
            const double fires =
                level[column.comparison] == column.exact_level ? column.fires : 0.0;
            best = std::max(best, fires);
            none *= 1.0 - fires;
        }
        bound += mass * best;
        analytic_total += mass * (1.0 - none);
        if (best <= 0.0) {
            ++dark_patterns;
            dark_mass += mass;
        }

        double rate = report->pooled_rate;
        if (report->correction_available) {
            const std::pair<uint64_t, uint64_t> observed = counts.At(gamma);
            if (observed.first >= options.min_observed) {
                rate = static_cast<double>(observed.second) /
                       static_cast<double>(observed.first);
                if (mass > 0.0) ++own_rate;
            }
        } else {
            rate = 0.0;
        }
        estimate += mass * (1.0 - none * (1.0 - rate));

        size_t c = 0;
        for (; c < n; ++c) {
            if (++level[c] < levels[c]) break;
            level[c] = 0;
        }
        if (c == n) break;
    }

    report->patterns = patterns;
    report->dark_patterns = dark_patterns;
    report->mass_total = mass_total;
    // m sums to one over each comparison's levels, so the product sums to one over
    // the patterns. Dividing by the total keeps a model that was floored or
    // rounded from moving the answer.
    if (mass_total > 0.0) {
        report->pc_bound = bound / mass_total;
        report->pc_analytic = analytic_total / mass_total;
        report->pc_product = estimate / mass_total;
        report->dark_mass = dark_mass / mass_total;
    }
    report->patterns_with_own_rate = own_rate;

    // The table estimator. The cells an analytic source completely covers are
    // counted rather than modelled; only the dark ones are fitted, and how they
    // are fitted is the whole question.
    if (report->walked && blocked.size() >= 2) {
        std::vector<double> observed_hat(table_cells, 0.0);
        std::vector<bool> dark(table_cells, false);
        double captured = 0.0;
        double seen_population = 0.0;
        uint64_t dark_cells = 0;
        for (uint64_t cell = 0; cell < table_cells; ++cell) {
            double none = 1.0;
            for (const BlockedColumn& dimension : blocked) {
                const uint64_t level = (cell / dimension.stride) % dimension.levels;
                if (level == dimension.exact_level) none *= 1.0 - dimension.fires;
            }
            const double fires = 1.0 - none;
            if (fires <= 0.0) {
                dark[cell] = true;
                ++dark_cells;
                continue;
            }
            captured += cell_matches[cell];
            // A cell an exact-value source covers is seen whole, so this divides
            // by one. A cell only a capped source reaches is seen with a known
            // probability, and is inflated by it.
            observed_hat[cell] = cell_matches[cell] / fires;
            seen_population += observed_hat[cell];
        }
        report->dark_cells = dark_cells;
        report->observed_cells = table_cells - dark_cells;
        report->observed_matches = captured;

        std::vector<std::vector<size_t>> one_way;
        std::vector<std::vector<size_t>> two_way;
        for (size_t d = 0; d < blocked.size(); ++d) {
            one_way.push_back({d});
            two_way.push_back({d});
        }
        for (size_t d = 0; d < blocked.size(); ++d) {
            for (size_t e = d + 1; e < blocked.size(); ++e) two_way.push_back({d, e});
        }
        const double correction =
            report->correction_available ? report->pooled_rate : 0.0;
        report->parameters_independence = ParameterCount(blocked, one_way);
        report->parameters_interaction = ParameterCount(blocked, two_way);
        // Residual degrees of freedom on the cells that were seen. Zero means the
        // fit reproduces them exactly whatever the truth is, so the dark cells it
        // predicts carry no information.
        report->table_available =
            report->observed_cells > report->parameters_independence;
        report->interaction_available =
            blocked.size() >= 3 &&
            report->observed_cells > report->parameters_interaction;
        const double penalty = seen_population > 1.0 ? std::log(seen_population) : 0.0;
        std::vector<double> fitted;
        if (report->table_available) {
            FitLogLinear(blocked, one_way, observed_hat, dark, 60, 20, &fitted);
            double dark_total = 0.0;
            for (uint64_t cell = 0; cell < table_cells; ++cell) {
                if (dark[cell]) dark_total += fitted[cell];
            }
            if (seen_population + dark_total > 0.0) {
                report->pc_independence =
                    (captured + dark_total * correction) / (seen_population + dark_total);
            }
            // Which pairs of blocked comparisons the independence fit gets wrong,
            // and by how much. The two-way fit acts on all of them at once; this
            // says which ones it is acting on.
            for (size_t d = 0; d < blocked.size(); ++d) {
                for (size_t e = d + 1; e < blocked.size(); ++e) {
                    const size_t span =
                        static_cast<size_t>(blocked[d].levels) * blocked[e].levels;
                    std::vector<double> observed_margin(span, 0.0);
                    std::vector<double> fitted_margin(span, 0.0);
                    for (uint64_t cell = 0; cell < table_cells; ++cell) {
                        if (dark[cell]) continue;
                        const size_t left =
                            (cell / blocked[d].stride) % blocked[d].levels;
                        const size_t right =
                            (cell / blocked[e].stride) % blocked[e].levels;
                        const size_t index = left * blocked[e].levels + right;
                        observed_margin[index] += observed_hat[cell];
                        fitted_margin[index] += fitted[cell];
                    }
                    Dependence pair;
                    pair.left = model.comparisons[blocked[d].comparison].name;
                    pair.right = model.comparisons[blocked[e].comparison].name;
                    pair.df = static_cast<uint64_t>(blocked[d].levels - 1) *
                              (blocked[e].levels - 1);
                    for (size_t i = 0; i < span; ++i) {
                        if (observed_margin[i] <= 0.0 || fitted_margin[i] <= 0.0) {
                            continue;
                        }
                        pair.g2 += 2.0 * observed_margin[i] *
                                   std::log(observed_margin[i] / fitted_margin[i]);
                    }
                    report->dependence.push_back(pair);
                }
            }
            std::sort(
                report->dependence.begin(), report->dependence.end(),
                [](const Dependence& x, const Dependence& y) { return x.g2 > y.g2; });

            report->deviance_independence = Deviance(observed_hat, fitted, dark);
            report->bic_independence =
                report->deviance_independence +
                static_cast<double>(report->parameters_independence) * penalty;
            if (report->interaction_available) {
                FitLogLinear(blocked, two_way, observed_hat, dark, 60, 20, &fitted);
                dark_total = 0.0;
                for (uint64_t cell = 0; cell < table_cells; ++cell) {
                    if (dark[cell]) dark_total += fitted[cell];
                }
                if (seen_population + dark_total > 0.0) {
                    report->pc_interaction = (captured + dark_total * correction) /
                                             (seen_population + dark_total);
                }
                report->deviance_interaction = Deviance(observed_hat, fitted, dark);
                report->bic_interaction =
                    report->deviance_interaction +
                    static_cast<double>(report->parameters_interaction) * penalty;
            }
        }
    }

    if (report->table_available && report->walked && report->interaction_available &&
        report->bic_interaction < report->bic_independence) {
        report->pc_estimate = report->pc_interaction;
        report->pc_basis = "interaction";
    } else if (report->table_available && report->walked) {
        report->pc_estimate = report->pc_independence;
        report->pc_basis = "independence";
    } else {
        report->pc_estimate = report->pc_product;
        report->pc_basis = "product";
    }
    report->threads = threads;
    report->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return true;
}

void PrintCompletenessReport(const CompletenessReport& report, std::ostream& out) {
    out << "Blocking recall, estimated without ground truth\n\n";
    if (report.unblocked) {
        out << "This plan does no blocking: it enumerates every pair the mode "
               "admits, so\npair completeness is 100% by construction and nothing "
               "here was estimated.\n";
        if (report.measured) {
            out << "\nMeasured against the truth file  "
                << Fixed(100.0 * report.pc_measured, 3) << "%  over "
                << WithThousands(report.truth_pairs) << " pairs\n";
        }
        return;
    }
    out << std::left << std::setw(22) << "Source" << std::setw(14) << "Column"
        << std::setw(22) << "Kind" << std::setw(12) << "Class" << std::right
        << std::setw(12) << "P(fire|=)" << "\n";
    out << std::string(82, '-') << "\n";
    for (const SourceCapture& source : report.sources) {
        out << std::left << std::setw(22) << source.name << std::setw(14) << source.column
            << std::setw(22) << SourceKindName(source.kind) << std::setw(12)
            << (source.capture == CaptureClass::kAnalytic ? "analytic" : "observed")
            << std::right << std::setw(12);
        if (source.capture == CaptureClass::kAnalytic && source.bound_to_comparison) {
            out << Fixed(source.fires_given_exact, 4);
        } else {
            out << "-";
        }
        out << "\n";
    }
    out << std::string(82, '-') << "\n\n";

    if (!report.trusted) {
        out << "REFUSE: EM never learned m for ";
        for (size_t i = 0; i < report.unlearned.size(); ++i) {
            out << (i == 0 ? "" : ", ") << "\"" << report.unlearned[i] << "\"";
        }
        out << ",\nwhich every source in this plan blocks on. The estimate is a sum\n"
               "weighted by exactly those parameters, so what follows is the model's\n"
               "default and not a measurement. Block on a second column, or estimate\n"
               "the model under a plan that does, and run this again.\n\n";
    }
    out << "Patterns summed        " << WithThousands(report.patterns) << "  ("
        << WithThousands(report.dark_patterns) << " reachable by no analytic source)\n"
        << "Match mass in the dark " << Fixed(100.0 * report.dark_mass, 4) << "%\n";
    if (!report.correction_available) {
        out << "Observed-class correction  not available: every analytic source "
               "blocks a\n                           column an observed-class source "
               "also blocks, so the\n                           overlap would only "
               "report its own conditioning.\n";
    }
    if (report.table_available) {
        out << "Capture table          " << WithThousands(report.cells) << " cells, "
            << WithThousands(report.dark_cells) << " dark, over "
            << (report.cells > 0 ? report.pc_basis : "") << " margins\n";
    } else {
        out << "Capture table          not available: fewer than two blocked "
               "columns, so a\n                       dark cell cannot be told "
               "from the marginals\n";
    }
    if (report.walked) {
        out << "Capture fold           " << WithThousands(report.observed_pairs)
            << " candidates, " << WithThousands(report.analytic_captured)
            << " analytic-captured, " << Fixed(100.0 * report.pooled_rate, 2)
            << "% of those also observed-class\n"
            << "Patterns with their own observed rate  "
            << WithThousands(report.patterns_with_own_rate) << "\n";
    }
    out << "\n";

    out << std::left << std::setw(42) << "Pair completeness" << std::right
        << std::setw(10) << "Estimate" << "\n";
    out << std::string(52, '-') << "\n";
    out << std::left << std::setw(42) << "  product model, best source per pattern"
        << std::right << std::setw(10) << Fixed(100.0 * report.pc_bound, 3) << "\n"
        << std::left << std::setw(42) << "  product model, sources combined" << std::right
        << std::setw(10) << Fixed(100.0 * report.pc_analytic, 3) << "\n"
        << std::left << std::setw(42) << "  product model, with the correction"
        << std::right << std::setw(10) << Fixed(100.0 * report.pc_product, 3) << "\n";
    if (report.table_available && report.walked) {
        out << std::left << std::setw(42) << "  table, independent dark cells"
            << std::right << std::setw(10) << Fixed(100.0 * report.pc_independence, 3)
            << "\n";
        if (report.interaction_available) {
            out << std::left << std::setw(42) << "  table, two-way interactions"
                << std::right << std::setw(10) << Fixed(100.0 * report.pc_interaction, 3)
                << "\n";
        }
    }
    out << std::string(52, '-') << "\n"
        << std::left << std::setw(42)
        << (std::string("  ESTIMATE (") + report.pc_basis + ")") << std::right
        << std::setw(10) << Fixed(100.0 * report.pc_estimate, 3) << "\n";
    if (report.measured) {
        out << std::left << std::setw(42) << "  measured against known pairs"
            << std::right << std::setw(10) << Fixed(100.0 * report.pc_measured, 3)
            << "\n";
    }
    out << std::string(52, '-') << "\n";
    if (report.table_available && report.walked && report.interaction_available) {
        out << "\nModel choice: independence fits the observed cells with deviance "
            << Fixed(report.deviance_independence, 1) << " on\n"
            << report.parameters_independence << " parameters, two-way interactions with "
            << Fixed(report.deviance_interaction, 1) << " on "
            << report.parameters_interaction << ". BIC "
            << Fixed(report.bic_independence, 1) << " against "
            << Fixed(report.bic_interaction, 1) << ", so "
            << (report.bic_interaction < report.bic_independence ? "the interactions"
                                                                 : "independence")
            << " wins.\n";
    }
    if (!report.dependence.empty()) {
        out << "\nDependence between the blocked columns, worst first. G2 is what "
               "the\nindependence fit fails to explain about that pair, over the "
               "cells it saw.\n\n";
        out << std::left << std::setw(24) << "Comparison" << std::setw(24) << "against"
            << std::right << std::setw(14) << "G2" << std::setw(6) << "df" << "\n";
        out << std::string(68, '-') << "\n";
        for (size_t i = 0; i < report.dependence.size() && i < 6; ++i) {
            const Dependence& pair = report.dependence[i];
            out << std::left << std::setw(24) << Truncate(pair.left, 23) << std::setw(24)
                << Truncate(pair.right, 23) << std::right << std::setw(14)
                << Fixed(pair.g2, 1) << std::setw(6) << pair.df << "\n";
        }
        out << std::string(68, '-') << "\n";
    }
    if (report.measured) {
        const double error = report.pc_estimate - report.pc_measured;
        out << "Estimator error " << Fixed(100.0 * error, 3) << " points against "
            << WithThousands(report.truth_pairs) << " known pairs.\n";
        if (report.pc_bound > report.pc_measured) {
            out << "NOTE: the product model reads above the measured recall, which "
                   "is what\nconditional dependence between the blocked columns "
                   "looks like: agreement\nis correlated among matches, so a "
                   "product overstates the chance that at\nleast one column agrees. "
                   "The table estimator exists for exactly this.\n";
        }
    }

    for (const SourceCapture& source : report.sources) {
        if (!source.has_window_check) continue;
        out << "\nWindow mechanism check for \"" << source.name
            << "\": over the agreeing pairs\nthe fold saw, the term frequencies "
               "predict "
            << Fixed(100.0 * source.window_analytic, 2)
            << "% fall inside the window and\n"
            << Fixed(100.0 * source.window_observed, 2)
            << "% of them do. The gap is how far the rows carrying one value are "
               "from\nbeing in random order, which is the assumption behind every "
               "window figure here.\n";
    }

    out << "\nWhat each row assumes. The product rows take the joint distribution of\n"
           "agreement among matches to be the product of the learned m, which is the\n"
           "conditional independence assumption and is where they fail on real data.\n"
           "The table rows take it from the cells blocking observed completely and\n"
           "model only the dark ones. All of them assume m was not itself biased by\n"
           "blocking -- see the REFUSE line, which is when it was -- and that a\n"
           "matching pair's shared value is drawn like a record's, which is what\n"
           "makes a frequency cap answerable at all.\n";
}

void WriteCompletenessJson(const CompletenessReport& report, std::ostream& out) {
    out << "{\n";
    out << "  \"pc_bound\": " << Fixed(report.pc_bound, 6) << ",\n";
    out << "  \"pc_analytic\": " << Fixed(report.pc_analytic, 6) << ",\n";
    out << "  \"pc_product\": " << Fixed(report.pc_product, 6) << ",\n";
    out << "  \"pc_independence\": " << Fixed(report.pc_independence, 6) << ",\n";
    out << "  \"pc_interaction\": " << Fixed(report.pc_interaction, 6) << ",\n";
    out << "  \"pc_basis\": \"" << report.pc_basis << "\",\n";
    out << "  \"bic_independence\": " << Fixed(report.bic_independence, 3) << ",\n";
    out << "  \"bic_interaction\": " << Fixed(report.bic_interaction, 3) << ",\n";
    out << "  \"table_available\": " << (report.table_available ? "true" : "false")
        << ",\n";
    out << "  \"dark_cells\": " << report.dark_cells << ",\n";
    out << "  \"pc_estimate\": " << Fixed(report.pc_estimate, 6) << ",\n";
    out << "  \"dark_mass\": " << Fixed(report.dark_mass, 6) << ",\n";
    out << "  \"records\": " << report.records << ",\n";
    out << "  \"trusted\": " << (report.trusted ? "true" : "false") << ",\n";
    out << "  \"unblocked\": " << (report.unblocked ? "true" : "false") << ",\n";
    out << "  \"correction_available\": "
        << (report.correction_available ? "true" : "false") << ",\n";
    out << "  \"patterns\": " << report.patterns << ",\n";
    out << "  \"dark_patterns\": " << report.dark_patterns << ",\n";
    out << "  \"observed_pairs\": " << report.observed_pairs << ",\n";
    out << "  \"analytic_captured\": " << report.analytic_captured << ",\n";
    out << "  \"pooled_rate\": " << Fixed(report.pooled_rate, 6) << ",\n";
    out << "  \"measured\": " << (report.measured ? "true" : "false") << ",\n";
    out << "  \"pc_measured\": " << Fixed(report.pc_measured, 6) << ",\n";
    out << "  \"truth_pairs\": " << report.truth_pairs << ",\n";
    out << "  \"sources\": [\n";
    for (size_t s = 0; s < report.sources.size(); ++s) {
        const SourceCapture& source = report.sources[s];
        out << "    {\"name\": \"" << source.name << "\", \"column\": \"" << source.column
            << "\", \"kind\": \"" << SourceKindName(source.kind) << "\", \"class\": \""
            << (source.capture == CaptureClass::kAnalytic ? "analytic" : "observed")
            << "\", \"fires_given_exact\": " << Fixed(source.fires_given_exact, 6) << "}"
            << (s + 1 == report.sources.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

}  // namespace cpplink
