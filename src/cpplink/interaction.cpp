// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/interaction.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cpplink {
namespace {

// A margin at or below this is a level nothing reaches, and a correction on it
// would be a ratio of two numbers that are both noise.
constexpr double kDeadMargin = 1e-9;

// IPF converges linearly on a two-way table and in practice in under twenty
// sweeps; the cap is there so a degenerate seed cannot spin.
constexpr int kMaxSweeps = 200;
constexpr double kSweepTolerance = 1e-13;

std::string Truncate(std::string text, size_t width) {
    if (text.size() <= width) return text;
    return text.substr(0, width - 1) + ".";
}

std::string Fixed(double value, int places) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(places) << value;
    return buffer.str();
}

// The regularized upper incomplete gamma Q(a, x), by the series below the
// crossover and the continued fraction above it. This is the chi-square tail and
// nothing more; it is printed beside every candidate and decides none of them.
double GammaQ(double a, double x) {
    if (x <= 0.0) return 1.0;
    const double log_gamma = std::lgamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a;
        double sum = term;
        for (int i = 1; i < 500; ++i) {
            term *= x / (a + i);
            sum += term;
            if (std::fabs(term) < std::fabs(sum) * 1e-15) break;
        }
        return 1.0 - sum * std::exp(-x + a * std::log(x) - log_gamma);
    }
    double b = x + 1.0 - a;
    double c = 1e300;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < 500; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c;
        if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) < 1e-15) break;
    }
    return std::exp(-x + a * std::log(x) - log_gamma) * h;
}

double ChiSquareTail(double statistic, uint64_t degrees) {
    if (degrees == 0 || statistic <= 0.0) return 1.0;
    return GammaQ(0.5 * static_cast<double>(degrees), 0.5 * statistic);
}

// The deviance of a table against the independence fit built from its own
// margins: 2 sum O ln(O/E), on (rows - 1)(columns - 1) degrees of freedom.
double Deviance(const std::vector<double>& table, size_t rows, size_t columns) {
    std::vector<double> row(rows, 0.0);
    std::vector<double> column(columns, 0.0);
    double total = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            const double value = table[i * columns + j];
            row[i] += value;
            column[j] += value;
            total += value;
        }
    }
    if (total <= 0.0) return 0.0;
    double statistic = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            const double observed = table[i * columns + j];
            if (observed <= 0.0) continue;
            const double expected = row[i] * column[j] / total;
            if (expected <= 0.0) continue;
            statistic += 2.0 * observed * std::log(observed / expected);
        }
    }
    return std::max(statistic, 0.0);
}

}  // namespace

JointTables::JointTables(const std::vector<size_t>& levels) : levels_(levels) {
    const size_t count = levels_.size();
    tables_.resize(count * (count - 1) / 2);
    for (size_t left = 0; left + 1 < count; ++left) {
        for (size_t right = left + 1; right < count; ++right) {
            tables_[Index(left, right)].assign(levels_[left] * levels_[right], 0.0);
        }
    }
}

size_t JointTables::Index(size_t left, size_t right) const {
    const size_t count = levels_.size();
    return left * (2 * count - left - 1) / 2 + (right - left - 1);
}

void JointTables::Add(const uint8_t* levels, double mass) {
    const size_t count = levels_.size();
    for (size_t left = 0; left + 1 < count; ++left) {
        for (size_t right = left + 1; right < count; ++right) {
            std::vector<double>& table = tables_[Index(left, right)];
            table[levels[left] * levels_[right] + levels[right]] += mass;
        }
    }
    total_ += mass;
}

void JointTables::Merge(const JointTables& other) {
    for (size_t t = 0; t < tables_.size(); ++t) {
        for (size_t i = 0; i < tables_[t].size(); ++i) {
            tables_[t][i] += other.tables_[t][i];
        }
    }
    total_ += other.total_;
}

const std::vector<double>& JointTables::Table(size_t left, size_t right) const {
    return tables_[Index(left, right)];
}

void FitToMargins(const std::vector<double>& rows, const std::vector<double>& columns,
                  std::vector<double>* table) {
    const size_t height = rows.size();
    const size_t width = columns.size();
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        double change = 0.0;
        for (size_t i = 0; i < height; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < width; ++j) sum += (*table)[i * width + j];
            if (sum <= 0.0) continue;
            const double scale = rows[i] / sum;
            change = std::max(change, std::fabs(scale - 1.0));
            for (size_t j = 0; j < width; ++j) (*table)[i * width + j] *= scale;
        }
        for (size_t j = 0; j < width; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < height; ++i) sum += (*table)[i * width + j];
            if (sum <= 0.0) continue;
            const double scale = columns[j] / sum;
            change = std::max(change, std::fabs(scale - 1.0));
            for (size_t i = 0; i < height; ++i) (*table)[i * width + j] *= scale;
        }
        if (change < kSweepTolerance) break;
    }
}

void FitInteractions(const ComparisonSet& comparisons,
                     const std::vector<SessionJoint>& sessions,
                     const JointTables& random_pairs, double lambda,
                     const InteractionOptions& options, Model* model,
                     InteractionReport* report) {
    const auto started = std::chrono::steady_clock::now();
    report->ran = true;
    report->lambda = lambda;
    report->lambda_is_a_bound = model->lambda_basis.compare(0, 11, "lower bound") == 0;
    const size_t count = comparisons.Size();
    if (count < 2) {
        report->refusal = "an interaction needs two comparisons";
        return;
    }
    if (random_pairs.Total() <= 0.0) {
        report->refusal = "no random pair was drawn, so the u side cannot be measured";
        return;
    }

    // A level the model marks unreachable carries no weight, and a correction on
    // it would be a ratio of two floors. Its cells are left at exactly zero.
    std::vector<std::vector<bool>> live(count);
    for (size_t c = 0; c < count; ++c) {
        const ModelComparison& learned = model->comparisons[c];
        live[c].assign(learned.levels.size(), true);
        for (size_t l = 0; l < learned.levels.size(); ++l) {
            const ModelLevel& level = learned.levels[l];
            live[c][l] = !(level.u_exact && level.m == level.u) && level.u > kDeadMargin;
        }
    }

    // Every candidate gets a slot here, refused ones included, so the ranking below
    // can index the two lists with the same number.
    std::vector<ModelInteraction> terms;
    for (size_t left = 0; left + 1 < count; ++left) {
        for (size_t right = left + 1; right < count; ++right) {
            InteractionCandidate candidate;
            candidate.left = model->comparisons[left].name;
            candidate.right = model->comparisons[right].name;
            const size_t height = model->comparisons[left].levels.size();
            const size_t width = model->comparisons[right].levels.size();

            // The m side, per session. A session that held either comparison out
            // has nothing to say about the pair, and one that did not gets its own
            // reading: a real dependence is visible to every session that can see
            // it, and noise is not. That agreement is what admits the term, and it
            // is the same discipline m itself is merged under.
            std::vector<std::vector<double>> per_session;
            std::vector<double> per_session_mass;
            std::vector<double> joint(height * width, 0.0);
            double mass = 0.0;
            for (const SessionJoint& session : sessions) {
                if (!session.usable) continue;
                if (session.excluded[left] || session.excluded[right]) continue;
                const std::vector<double>& table = session.tables.Table(left, right);
                double total = 0.0;
                for (const double value : table) total += value;
                if (total <= 0.0) continue;
                std::vector<double> share(joint.size(), 0.0);
                for (size_t i = 0; i < joint.size(); ++i) {
                    share[i] = table[i] / total;
                    joint[i] += share[i];
                }
                per_session.push_back(std::move(share));
                per_session_mass.push_back(total);
                mass += total;
            }
            const size_t contributing = per_session.size();
            candidate.sessions = contributing;
            if (contributing == 0) {
                candidate.reason =
                    "no session leaves both comparisons free, so the term is not "
                    "identified";
                report->candidates.push_back(std::move(candidate));
                terms.emplace_back();
                continue;
            }
            // Two readings of the pooled table: a probability, which the
            // contamination subtraction needs, and a count at the size of a typical
            // session, which the deviance needs.
            std::vector<double> match_share = joint;
            for (double& value : match_share) {
                value /= static_cast<double>(contributing);
            }
            mass /= static_cast<double>(contributing);
            candidate.match_pairs = mass;

            std::vector<double> m_margin(height, 0.0);
            std::vector<double> u_margin(width, 0.0);
            std::vector<double> m_right(width, 0.0);
            std::vector<double> u_left(height, 0.0);
            for (size_t i = 0; i < height; ++i) {
                m_margin[i] = live[left][i] ? model->comparisons[left].levels[i].m : 0.0;
                u_left[i] = live[left][i] ? model->comparisons[left].levels[i].u : 0.0;
            }
            for (size_t j = 0; j < width; ++j) {
                m_right[j] = live[right][j] ? model->comparisons[right].levels[j].m : 0.0;
                u_margin[j] =
                    live[right][j] ? model->comparisons[right].levels[j].u : 0.0;
            }

            // A level nothing reaches is not a free parameter, so the term's size is
            // counted over the live ones: half the degrees of freedom are the m side
            // and half the u side.
            size_t live_rows = 0;
            size_t live_columns = 0;
            for (size_t i = 0; i < height; ++i) live_rows += live[left][i] ? 1 : 0;
            for (size_t j = 0; j < width; ++j) live_columns += live[right][j] ? 1 : 0;
            candidate.degrees =
                live_rows > 1 && live_columns > 1
                    ? 2 * static_cast<uint64_t>((live_rows - 1) * (live_columns - 1))
                    : 0;
            candidate.per_parameter =
                candidate.degrees > 0
                    ? mass / (0.5 * static_cast<double>(candidate.degrees))
                    : mass;

            // The u side has this file's own duplicates in it, because a uniformly
            // random pair is a match with probability lambda and the joint of two
            // high-cardinality columns is nothing but those matches. Take the known
            // rate back out: a random draw shows (1 - lambda) u + lambda m, and both
            // lambda and m are already in hand.
            const std::vector<double>& sampled = random_pairs.Table(left, right);
            const double draws = random_pairs.Total();
            const double matches = lambda * draws;
            std::vector<double> u_fit(height * width, 0.0);
            double contaminated = 0.0;
            double weighed = 0.0;
            for (size_t i = 0; i < height; ++i) {
                for (size_t j = 0; j < width; ++j) {
                    if (!live[left][i] || !live[right][j]) continue;
                    const size_t at = i * width + j;
                    const double duplicates = matches * match_share[at];
                    const double observed = sampled[at];
                    if (observed > 0.0) {
                        contaminated +=
                            match_share[at] * std::min(1.0, duplicates / observed);
                    }
                    weighed += match_share[at];
                    u_fit[at] =
                        std::max(observed - duplicates, 0.0) + options.prior_count;
                }
            }
            candidate.contamination = weighed > 0.0 ? contaminated / weighed : 0.0;
            if (candidate.contamination > options.max_contamination) {
                candidate.reason = "the u side is " +
                                   Fixed(100.0 * candidate.contamination, 0) +
                                   "% this file's own duplicates";
                report->candidates.push_back(std::move(candidate));
                terms.emplace_back();
                continue;
            }
            if (candidate.per_parameter < options.min_pairs_per_parameter) {
                candidate.reason = "only " + Fixed(candidate.per_parameter, 0) +
                                   " matching pairs per free parameter";
                report->candidates.push_back(std::move(candidate));
                terms.emplace_back();
                continue;
            }

            std::vector<double> u_reference = u_fit;
            candidate.g2 = Deviance(u_reference, height, width);
            FitToMargins(u_left, u_margin, &u_fit);

            // One reading per session, then the pooled one. The correction is the
            // log ratio of the joint to the product of its own margins under M, less
            // the same under U. Both halves are needed: two columns agreeing
            // together among matches are double-counted only to the extent they do
            // not also agree together among non-matches, which the main-effect u
            // already prices.
            const auto correction = [&](const std::vector<double>& share, double scale,
                                        std::vector<double>* bits, double* match_bits,
                                        double* effect, double* widest) {
                std::vector<double> m_fit(height * width, 0.0);
                for (size_t i = 0; i < height; ++i) {
                    for (size_t j = 0; j < width; ++j) {
                        if (!live[left][i] || !live[right][j]) continue;
                        const size_t at = i * width + j;
                        m_fit[at] = share[at] * scale + options.prior_count;
                    }
                }
                const double deviance = Deviance(m_fit, height, width);
                FitToMargins(m_margin, m_right, &m_fit);
                bits->assign(height * width, 0.0);
                double signed_total = 0.0;
                double absolute_total = 0.0;
                double weight_total = 0.0;
                for (size_t i = 0; i < height; ++i) {
                    for (size_t j = 0; j < width; ++j) {
                        if (!live[left][i] || !live[right][j]) continue;
                        const size_t at = i * width + j;
                        const double independent_m = m_margin[i] * m_right[j];
                        const double independent_u = u_left[i] * u_margin[j];
                        if (independent_m <= 0.0 || independent_u <= 0.0) continue;
                        if (m_fit[at] <= 0.0 || u_fit[at] <= 0.0) continue;
                        double value = std::log2(m_fit[at] / independent_m) -
                                       std::log2(u_fit[at] / independent_u);
                        // Both sides have to have seen the cell. The binding one is
                        // the m side, which holds a session's matching pairs where
                        // the u side holds the whole random sample.
                        const double seen = std::min(share[at] * scale,
                                                     sampled[at] - matches * share[at]);
                        value *= std::max(seen, 0.0) /
                                 (std::max(seen, 0.0) + options.cell_support);
                        value = std::max(-options.clamp_bits,
                                         std::min(options.clamp_bits, value));
                        (*bits)[at] = value;
                        signed_total += m_fit[at] * value;
                        absolute_total += m_fit[at] * std::fabs(value);
                        weight_total += m_fit[at];
                        *widest = std::max(*widest, std::fabs(value));
                    }
                }
                if (weight_total > 0.0) {
                    *match_bits = signed_total / weight_total;
                    *effect = absolute_total / weight_total;
                }
                return deviance;
            };

            std::vector<double> bits;
            candidate.g2 += correction(match_share, mass, &bits, &candidate.match_bits,
                                       &candidate.effect, &candidate.widest);
            candidate.p_value = ChiSquareTail(candidate.g2, candidate.degrees);
            for (const double value : bits) {
                if (std::fabs(value) >= options.clamp_bits) ++report->clamped;
            }

            // What each session on its own says the term moves a match by. A real
            // dependence is seen by all of them; a small sample produces one number
            // per session and no two alike.
            candidate.weakest = candidate.match_bits;
            for (size_t s = 0; s < contributing; ++s) {
                std::vector<double> ignored_bits;
                double session_bits = 0.0;
                double ignored_effect = 0.0;
                double ignored_widest = 0.0;
                correction(per_session[s], per_session_mass[s], &ignored_bits,
                           &session_bits, &ignored_effect, &ignored_widest);
                if (session_bits * candidate.match_bits <= 0.0) {
                    candidate.weakest = 0.0;
                } else if (std::fabs(session_bits) < std::fabs(candidate.weakest)) {
                    candidate.weakest = session_bits;
                }
            }

            ModelInteraction term;
            term.left = candidate.left;
            term.right = candidate.right;
            term.left_levels = static_cast<uint8_t>(height);
            term.right_levels = static_cast<uint8_t>(width);
            term.bits = std::move(bits);
            term.match_bits = candidate.match_bits;
            term.effect = candidate.effect;
            term.sessions = contributing;
            report->candidates.push_back(std::move(candidate));
            terms.push_back(std::move(term));
        }
    }

    // Ranked by effect size, and admitted on effect size. This is the third place
    // in this project where a significance test could have chosen the model size
    // and the third where it must not: G-squared scales with the size of the run,
    // so at 18M candidates every pair of comparisons is significant and at 10k
    // none is. The p-value is printed because it is the diagnostic; the bits are
    // printed because they are the decision.
    std::vector<size_t> order(report->candidates.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        return report->candidates[x].effect > report->candidates[y].effect;
    });

    std::vector<ModelInteraction> kept;
    std::vector<InteractionCandidate> ranked;
    ranked.reserve(order.size());
    for (const size_t index : order) {
        InteractionCandidate candidate = report->candidates[index];
        if (candidate.reason.empty()) {
            if (candidate.sessions < options.min_sessions) {
                candidate.reason = "only " + std::to_string(candidate.sessions) +
                                   " session can see it, so nothing can disagree";
            } else if (std::fabs(candidate.weakest) < options.min_bits) {
                candidate.reason = candidate.weakest == 0.0
                                       ? "two sessions disagree about its sign"
                                       : "one session moves a match by only " +
                                             Fixed(candidate.weakest, 2) + " bits";
            } else if (kept.size() >= options.max_terms) {
                candidate.reason = "past --max-interactions";
            } else if (candidate.effect < options.min_bits) {
                candidate.reason = "moves a match by less than --interaction-bits";
            } else {
                candidate.admitted = true;
                kept.push_back(terms[index]);
            }
        }
        ranked.push_back(std::move(candidate));
    }
    model->interactions = std::move(kept);
    report->candidates = std::move(ranked);
    report->admitted = model->interactions.size();
    report->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

void PrintInteractionReport(const InteractionReport& report, std::ostream& out) {
    if (!report.ran) return;
    if (!report.refusal.empty()) {
        out << "\nInteractions: refused -- " << report.refusal << "\n";
        return;
    }
    out << "\nTwo-way interactions, ranked by what they move a match by. " << std::fixed
        << std::setprecision(1) << report.seconds << " s.\n";
    out << "The u side had " << std::scientific << std::setprecision(2) << report.lambda
        << " of its pairs subtracted as this file's own duplicates.\n\n";
    out << std::left << std::setw(20) << "Comparison" << std::setw(20) << "Comparison"
        << std::right << std::setw(10) << "effect" << std::setw(11) << "per match"
        << std::setw(9) << "weakest" << std::setw(9) << "dup u" << std::setw(11)
        << "pairs/par" << std::setw(12) << "G^2" << std::setw(10) << "p"
        << "  " << std::left << "verdict" << "\n";
    out << std::string(132, '-') << "\n";
    for (const InteractionCandidate& candidate : report.candidates) {
        std::ostringstream p;
        p << std::scientific << std::setprecision(1) << candidate.p_value;
        out << std::left << std::setw(20) << Truncate(candidate.left, 19) << std::setw(20)
            << Truncate(candidate.right, 19) << std::right << std::setw(10)
            << Fixed(candidate.effect, 2) << std::setw(11)
            << Fixed(candidate.match_bits, 2) << std::setw(9)
            << Fixed(candidate.weakest, 2) << std::setw(8)
            << Fixed(100.0 * candidate.contamination, 0) << "%" << std::setw(11)
            << Fixed(candidate.per_parameter, 0) << std::setw(12)
            << Fixed(candidate.g2, 1) << std::setw(10) << p.str() << "  " << std::left
            << (candidate.admitted ? std::string("fitted") : candidate.reason) << "\n";
    }
    out << std::string(132, '-') << "\n";
    out << report.admitted << " of " << report.candidates.size()
        << " candidate pairs entered the model";
    if (report.clamped > 0) {
        out << "; " << report.clamped << " cell(s) hit the clamp";
    }
    out << ".\n";
    out << "\"effect\" is the mean absolute correction an average matching pair gets,\n"
        << "in bits; \"weakest\" is the least any one session says on its own, and a\n"
        << "term has to clear --interaction-bits there. \"dup u\" is how much of the u\n"
        << "side was this file's own duplicates before they were subtracted back out.\n"
        << "G^2 is printed because it is the diagnostic, and ignored because its\n"
        << "power is the size of the run rather than the size of the effect.\n";
    if (report.lambda_is_a_bound && report.admitted > 0) {
        out << "\nwarning: lambda is a lower bound, so the u side keeps some of this\n"
            << "file's duplicates and every correction above reads more negative than\n"
            << "it is. Measured on a 1,000-record fixture whose lambda was low by\n"
            << "2.3x, that inverted the sign of the two largest terms. Pass --lambda\n"
            << "from a count you trust, or divide this one by the pair completeness\n"
            << "`cpplink completeness` estimates, before believing a term.\n";
    }
}

}  // namespace cpplink
