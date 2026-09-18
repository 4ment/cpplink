// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/fit.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cpplink/interaction.hpp"

namespace cpplink {
namespace {

// A level whose u sits at the estimator's floor is one nothing reaches, and it is
// neither a cell nor a parameter. The same bar the identification check uses.
constexpr double kReachable = 1e-12;

// 2 sum O ln(O/E), over the cells the data filled. Where the model puts no mass
// on a filled cell the statistic is infinite in principle; it is left out here
// and the floor on m and u makes that a cell nothing can reach.
double Deviance(const std::vector<double>& observed,
                const std::vector<double>& expected) {
    double statistic = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] <= 0.0 || expected[i] <= 0.0) continue;
        statistic += 2.0 * observed[i] * std::log(observed[i] / expected[i]);
    }
    return std::max(statistic, 0.0);
}

double BitsPerPair(double deviance, double pairs) {
    return pairs > 0.0 ? deviance / (2.0 * pairs * std::log(2.0)) : 0.0;
}

// The product of the free comparisons' level probabilities for one pattern, with
// the terms' ratios multiplied in. `ratio` picks the M or U half of each term.
struct PatternModel {
    const std::vector<std::vector<double>>* margin = nullptr;
    const std::vector<InteractionRatios>* terms = nullptr;
    bool match = true;
    double normaliser = 1.0;

    double Probability(const std::vector<size_t>& free, const uint8_t* levels) const {
        double probability = 1.0;
        for (const size_t c : free) probability *= (*margin)[c][levels[c]];
        for (const InteractionRatios& term : *terms) {
            const size_t width = (*margin)[term.right].size();
            const std::vector<double>& table = match ? term.match : term.random;
            probability *= table[levels[term.left] * width + levels[term.right]];
        }
        return probability / normaliser;
    }
};

}  // namespace

SessionFit MeasureFit(const ComparisonSet& comparisons,
                      const std::vector<PatternCount>& entries,
                      const std::vector<std::vector<double>>& m,
                      const std::vector<std::vector<double>>& u, double lambda,
                      const std::vector<bool>& excluded,
                      const std::vector<InteractionRatios>& terms,
                      const FitOptions& options) {
    const size_t count = comparisons.Size();
    SessionFit fit;
    std::vector<size_t> free;
    for (size_t c = 0; c < count; ++c) {
        if (!excluded[c]) free.push_back(c);
    }
    if (free.empty()) {
        fit.refusal = "the session left no comparison free";
        return fit;
    }
    double pairs = 0.0;
    for (const PatternCount& entry : entries) pairs += static_cast<double>(entry.count);
    if (pairs <= 0.0) {
        fit.refusal = "the session folded no pair";
        return fit;
    }
    fit.measured = true;
    fit.pairs = static_cast<uint64_t>(pairs);

    // Levels a pair can land on, per comparison: the cells of the table the
    // mixture is fitted to, and what its parameter count is made of.
    std::vector<uint64_t> reachable(count, 0);
    double cells = 1.0;
    double parameters = 1.0;  // lambda
    for (const size_t c : free) {
        for (const double value : u[c]) {
            if (value > kReachable) ++reachable[c];
        }
        reachable[c] = std::max<uint64_t>(reachable[c], 1);
        cells *= static_cast<double>(reachable[c]);
        parameters += static_cast<double>(reachable[c] - 1);
    }
    fit.degrees = cells - 1.0 > parameters
                      ? static_cast<uint64_t>(std::min(cells - 1.0 - parameters, 1e18))
                      : 0;

    // The histogram carries the held-out comparisons' levels and the model does
    // not read them, so the patterns are collapsed onto the free comparisons
    // first: two patterns differing only on a held-out level are one cell of
    // the table being judged, not two cells each owed the whole marginal.
    std::map<std::vector<uint8_t>, double> collapsed;
    {
        std::vector<uint8_t> key(count, 0);
        for (const PatternCount& entry : entries) {
            for (size_t c = 0; c < count; ++c) {
                key[c] = excluded[c] ? 0 : comparisons.LevelOf(entry.gamma, c);
            }
            collapsed[key] += static_cast<double>(entry.count);
        }
    }
    fit.patterns = collapsed.size();
    std::vector<uint8_t> levels;
    std::vector<double> observed;
    levels.reserve(collapsed.size() * count);
    observed.reserve(collapsed.size());
    for (const auto& cell : collapsed) {
        levels.insert(levels.end(), cell.first.begin(), cell.first.end());
        observed.push_back(cell.second);
    }
    const size_t patterns = observed.size();

    // --- the plain model -----------------------------------------------------
    const std::vector<InteractionRatios> none;
    PatternModel plain_m{&m, &none, true, 1.0};
    PatternModel plain_u{&u, &none, false, 1.0};
    std::vector<double> expected(patterns);
    for (size_t e = 0; e < patterns; ++e) {
        const uint8_t* at = levels.data() + e * count;
        expected[e] = pairs * (lambda * plain_m.Probability(free, at) +
                               (1.0 - lambda) * plain_u.Probability(free, at));
    }
    fit.deviance = Deviance(observed, expected);
    fit.p_value = ChiSquareTail(fit.deviance, fit.degrees);
    fit.bits = BitsPerPair(fit.deviance, pairs);
    fit.floor_bits =
        patterns > 1 ? BitsPerPair(static_cast<double>(patterns - 1), pairs) : 0.0;

    // One two-way table per pair of free comparisons, observed and predicted.
    struct PairTable {
        size_t a = 0;
        size_t b = 0;
        std::vector<double> observed;
        std::vector<double> expected;
    };
    std::vector<PairTable> tables;
    for (size_t i = 0; i < free.size(); ++i) {
        for (size_t j = i + 1; j < free.size(); ++j) {
            PairTable table;
            table.a = free[i];
            table.b = free[j];
            const size_t height = m[table.a].size();
            const size_t width = m[table.b].size();
            table.observed.assign(height * width, 0.0);
            table.expected.assign(height * width, 0.0);
            for (size_t e = 0; e < patterns; ++e) {
                const uint8_t* at = levels.data() + e * count;
                table.observed[at[table.a] * width + at[table.b]] += observed[e];
            }
            for (size_t x = 0; x < height; ++x) {
                for (size_t y = 0; y < width; ++y) {
                    table.expected[x * width + y] =
                        pairs * (lambda * m[table.a][x] * m[table.b][y] +
                                 (1.0 - lambda) * u[table.a][x] * u[table.b][y]);
                }
            }
            PairResidual residual;
            residual.left = comparisons.at(table.a).spec->name;
            residual.right = comparisons.at(table.b).spec->name;
            residual.g2 = Deviance(table.observed, table.expected);
            residual.degrees = reachable[table.a] > 1 && reachable[table.b] > 1
                                   ? (reachable[table.a] - 1) * (reachable[table.b] - 1)
                                   : 0;
            residual.p_value = ChiSquareTail(residual.g2, residual.degrees);
            residual.bits = BitsPerPair(residual.g2, pairs);
            fit.residuals.push_back(std::move(residual));
            tables.push_back(std::move(table));
        }
    }

    // --- the model with the admitted terms in --------------------------------
    // Only a term whose two comparisons the session left free applies: one on a
    // held-out comparison marginalises to exactly the main effect, because the
    // fit put the joint back on the model's own margins.
    //
    // A term arrives as the joint over the pooled model's margins, and this
    // session's m is not the pooled one. What the term asserts is its odds
    // ratios, so the joint is refitted to the session's own margins by IPF, which
    // moves the margins and leaves every odds ratio alone; the ratio read off
    // that table is what multiplies into a pattern here.
    std::vector<InteractionRatios> applicable;
    for (const InteractionRatios& term : terms) {
        if (excluded[term.left] || excluded[term.right]) continue;
        InteractionRatios local = term;
        const auto refit = [&](const std::vector<double>& rows,
                               const std::vector<double>& columns,
                               std::vector<double>* ratio) {
            const size_t width = columns.size();
            std::vector<double> joint(ratio->size(), 0.0);
            for (size_t i = 0; i < rows.size(); ++i) {
                for (size_t j = 0; j < width; ++j) {
                    joint[i * width + j] = rows[i] * columns[j] * (*ratio)[i * width + j];
                }
            }
            FitToMargins(rows, columns, &joint);
            for (size_t i = 0; i < rows.size(); ++i) {
                for (size_t j = 0; j < width; ++j) {
                    const double independent = rows[i] * columns[j];
                    (*ratio)[i * width + j] =
                        independent > 0.0 ? joint[i * width + j] / independent : 1.0;
                }
            }
        };
        refit(m[term.left], m[term.right], &local.match);
        refit(u[term.left], u[term.right], &local.random);
        applicable.push_back(std::move(local));
    }
    if (!applicable.empty()) {
        // Terms sharing a comparison make a product that no longer sums to one, so
        // the touched comparisons are enumerated to normalise it, and their one-
        // and two-way margins are read off the same enumeration: everything the
        // terms leave alone factors out.
        std::vector<size_t> touched;
        for (const InteractionRatios& term : applicable) {
            for (const size_t c : {term.left, term.right}) {
                if (std::find(touched.begin(), touched.end(), c) == touched.end()) {
                    touched.push_back(c);
                }
            }
        }
        std::sort(touched.begin(), touched.end());
        double touched_cells = 1.0;
        for (const size_t c : touched) touched_cells *= static_cast<double>(m[c].size());
        if (touched_cells > static_cast<double>(options.cell_budget)) {
            fit.corrected_refusal = "the admitted terms touch " +
                                    std::to_string(touched.size()) +
                                    " comparisons whose levels make " +
                                    std::to_string(static_cast<uint64_t>(touched_cells)) +
                                    " cells to normalise, past the budget";
        } else {
            PatternModel corrected_m{&m, &applicable, true, 1.0};
            PatternModel corrected_u{&u, &applicable, false, 1.0};
            std::vector<uint8_t> cell(count, 0);
            std::vector<std::vector<double>> one_m(count);
            std::vector<std::vector<double>> one_u(count);
            for (const size_t c : touched) {
                one_m[c].assign(m[c].size(), 0.0);
                one_u[c].assign(u[c].size(), 0.0);
            }
            // Two-way margins over touched pairs, keyed by position in `tables`.
            std::vector<std::vector<double>> two_m(tables.size());
            std::vector<std::vector<double>> two_u(tables.size());
            std::vector<size_t> touched_tables;
            for (size_t t = 0; t < tables.size(); ++t) {
                const bool a_touched =
                    std::binary_search(touched.begin(), touched.end(), tables[t].a);
                const bool b_touched =
                    std::binary_search(touched.begin(), touched.end(), tables[t].b);
                if (a_touched && b_touched) {
                    two_m[t].assign(tables[t].observed.size(), 0.0);
                    two_u[t].assign(tables[t].observed.size(), 0.0);
                    touched_tables.push_back(t);
                }
            }
            double z_m = 0.0;
            double z_u = 0.0;
            bool done = false;
            while (!done) {
                const double pm = corrected_m.Probability(touched, cell.data());
                const double pu = corrected_u.Probability(touched, cell.data());
                z_m += pm;
                z_u += pu;
                for (const size_t c : touched) {
                    one_m[c][cell[c]] += pm;
                    one_u[c][cell[c]] += pu;
                }
                for (const size_t t : touched_tables) {
                    const size_t width = m[tables[t].b].size();
                    const size_t at = cell[tables[t].a] * width + cell[tables[t].b];
                    two_m[t][at] += pm;
                    two_u[t][at] += pu;
                }
                // Mixed-radix increment over the touched comparisons.
                done = true;
                for (const size_t c : touched) {
                    if (++cell[c] < m[c].size()) {
                        done = false;
                        break;
                    }
                    cell[c] = 0;
                }
            }
            corrected_m.normaliser = z_m;
            corrected_u.normaliser = z_u;

            for (size_t e = 0; e < patterns; ++e) {
                const uint8_t* at = levels.data() + e * count;
                expected[e] =
                    pairs * (lambda * corrected_m.Probability(free, at) +
                             (1.0 - lambda) * corrected_u.Probability(free, at));
            }
            fit.corrected = true;
            fit.corrected_deviance = Deviance(observed, expected);
            fit.corrected_bits = BitsPerPair(fit.corrected_deviance, pairs);

            // A pair the terms leave alone keeps its residual; a pair with one
            // touched comparison takes that comparison's normalised margin; a pair
            // with both takes the enumerated two-way margin.
            for (size_t t = 0; t < tables.size(); ++t) {
                const PairTable& table = tables[t];
                const bool a_touched =
                    std::binary_search(touched.begin(), touched.end(), table.a);
                const bool b_touched =
                    std::binary_search(touched.begin(), touched.end(), table.b);
                if (!a_touched && !b_touched) continue;
                const size_t height = m[table.a].size();
                const size_t width = m[table.b].size();
                std::vector<double> predicted(height * width, 0.0);
                for (size_t x = 0; x < height; ++x) {
                    for (size_t y = 0; y < width; ++y) {
                        double joint_m = 0.0;
                        double joint_u = 0.0;
                        if (a_touched && b_touched) {
                            joint_m = two_m[t][x * width + y] / z_m;
                            joint_u = two_u[t][x * width + y] / z_u;
                        } else if (a_touched) {
                            joint_m = one_m[table.a][x] / z_m * m[table.b][y];
                            joint_u = one_u[table.a][x] / z_u * u[table.b][y];
                        } else {
                            joint_m = m[table.a][x] * one_m[table.b][y] / z_m;
                            joint_u = u[table.a][x] * one_u[table.b][y] / z_u;
                        }
                        predicted[x * width + y] =
                            pairs * (lambda * joint_m + (1.0 - lambda) * joint_u);
                    }
                }
                PairResidual& residual = fit.residuals[t];
                residual.corrected = true;
                residual.corrected_g2 = Deviance(table.observed, predicted);
                residual.corrected_bits = BitsPerPair(residual.corrected_g2, pairs);
            }
        }
    }

    std::stable_sort(
        fit.residuals.begin(), fit.residuals.end(),
        [](const PairResidual& x, const PairResidual& y) { return x.bits > y.bits; });
    return fit;
}

}  // namespace cpplink
