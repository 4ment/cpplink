// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/estimate.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cpplink/histogram.hpp"

namespace cpplink {
namespace {

// m and u are floored rather than allowed to reach zero: a zero makes the log
// weight infinite and takes the whole pattern out of the likelihood permanently.
// The floor is reported, never silently applied -- an unsupported level is a
// modelling bug, and hiding it as a number would bury it.
constexpr double kFloor = 1e-12;

// Neumaier compensated summation. Responsibilities span many orders of magnitude
// over 1e5 patterns, so the naive accumulation loses the tail.
class Neumaier {
   public:
    void Add(double value) {
        const double total = sum_ + value;
        if (std::fabs(sum_) >= std::fabs(value)) {
            correction_ += (sum_ - total) + value;
        } else {
            correction_ += (value - total) + sum_;
        }
        sum_ = total;
    }
    double Total() const { return sum_ + correction_; }

   private:
    double sum_ = 0.0;
    double correction_ = 0.0;
};

uint64_t Mix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

unsigned ThreadCount(unsigned requested) {
    if (requested > 0) return requested;
    const unsigned available = std::thread::hardware_concurrency();
    return available > 0 ? available : 1;
}

std::string Join(const std::vector<std::string>& parts) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += ", ";
        out += parts[i];
    }
    return out;
}

std::string WithThousands(uint64_t value) {
    std::string digits = std::to_string(value);
    std::string out;
    int count = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (count > 0 && count % 3 == 0) out.push_back(',');
        out.push_back(*it);
        ++count;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// Whether a level's u can be had exactly, without drawing a single pair.
//
// A null level placed first fires exactly when either side has no value, which is
// a count over the data. An exact level on one interned column fires exactly when
// two independent draws land on the same value, which is the term frequencies'
// second moment. Nothing else is closed form.
bool ExactU(const RecordStore& store, const ComparisonSet& comparisons, size_t index,
            size_t level, uint64_t nulls, double* value) {
    const BoundComparison& bound = comparisons.at(index);
    const std::vector<LevelSpec>& levels = bound.spec->levels;
    const double records = static_cast<double>(store.NumRecords());
    if (records < 2.0) return false;

    if (levels[level].type == LevelType::kNull) {
        if (level != 0) return false;  // an earlier level could pre-empt it
        const double present = (records - static_cast<double>(nulls)) / records;
        *value = 1.0 - present * present;
        return true;
    }
    if (levels[level].type != LevelType::kExact) return false;
    if (bound.spec->columns.size() != 1 || bound.lists != nullptr) return false;
    // Only a null level may sit above it, or equality is not the level's event.
    for (size_t l = 0; l < level; ++l) {
        if (levels[l].type != LevelType::kNull) return false;
    }

    const std::vector<uint32_t>* tf = nullptr;
    if (bound.strings != nullptr) {
        tf = &bound.strings->tf;
    } else if (bound.dates != nullptr) {
        tf = &bound.dates->tf;
    }
    if (tf == nullptr) return false;

    Neumaier collisions;
    for (const uint32_t frequency : *tf) {
        const double share = static_cast<double>(frequency) / records;
        collisions.Add(share * share);
    }
    *value = collisions.Total();
    return true;
}

// u is what a uniformly random pair does, so this draws uniformly random pairs.
// It is the one place estimation touches pairs that blocking never proposed, and
// it is deliberate: u must not know that blocking exists.
void SampleRandomPairs(const RecordStore& store, const ComparisonSet& comparisons,
                       const EstimateOptions& options,
                       std::vector<std::vector<uint64_t>>* counts, uint64_t* drawn) {
    const uint64_t records = store.NumRecords();
    const unsigned threads = ThreadCount(options.threads);
    std::vector<std::vector<std::vector<uint64_t>>> partials(threads, *counts);
    std::vector<uint64_t> per_thread(threads, 0);

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        const uint64_t share =
            options.u_sample / threads + (t < options.u_sample % threads ? 1 : 0);
        workers.emplace_back([&, t, share] {
            uint64_t state = Mix64(options.seed + 0x5DEECE66Dull * (t + 1));
            uint64_t taken = 0;
            for (uint64_t i = 0; i < share; ++i) {
                state = Mix64(state);
                const uint64_t a = state % records;
                state = Mix64(state);
                const uint64_t b = state % records;
                if (a == b) continue;  // a pair is two distinct records
                const uint32_t gamma = comparisons.Evaluate(a, b);
                for (size_t c = 0; c < comparisons.Size(); ++c) {
                    ++partials[t][c][comparisons.LevelOf(gamma, c)];
                }
                ++taken;
            }
            per_thread[t] = taken;
        });
    }
    for (std::thread& worker : workers) worker.join();

    *drawn = 0;
    for (unsigned t = 0; t < threads; ++t) {
        *drawn += per_thread[t];
        for (size_t c = 0; c < counts->size(); ++c) {
            for (size_t l = 0; l < (*counts)[c].size(); ++l) {
                (*counts)[c][l] += partials[t][c][l];
            }
        }
    }
}

struct EmResult {
    std::vector<std::vector<double>> m;
    std::vector<std::vector<double>> support;  // matching mass reaching each level
    double lambda = 0.0;
    int iterations = 0;
    double change = 0.0;
    bool converged = false;
};

// Starting m: a null level is as common among matches as in the data, and the
// remaining mass halves down the levels, because level order is the model and the
// strongest evidence is written first.
std::vector<std::vector<double>> InitialM(const ComparisonSet& comparisons,
                                          const std::vector<std::vector<double>>& u) {
    std::vector<std::vector<double>> m(comparisons.Size());
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        const std::vector<LevelSpec>& levels = comparisons.at(c).spec->levels;
        m[c].assign(levels.size(), 0.0);
        double null_mass = 0.0;
        double weight_total = 0.0;
        int rank = 0;
        for (size_t l = 0; l < levels.size(); ++l) {
            if (levels[l].type == LevelType::kNull) {
                null_mass += u[c][l];
                m[c][l] = u[c][l];
            } else {
                m[c][l] = std::pow(0.5, rank++);
                weight_total += m[c][l];
            }
        }
        const double spare = std::max(1.0 - null_mass, kFloor);
        for (size_t l = 0; l < levels.size(); ++l) {
            if (levels[l].type != LevelType::kNull && weight_total > 0.0) {
                m[c][l] = spare * m[c][l] / weight_total;
            }
        }
    }
    return m;
}

// EM over the histogram. The cost is patterns x iterations x comparisons and has
// nothing to do with how many pairs were folded to build it -- which is the whole
// reason the histogram is the interface.
EmResult RunEm(const ComparisonSet& comparisons, const std::vector<PatternCount>& entries,
               const std::vector<std::vector<double>>& u,
               const std::vector<bool>& excluded, const EstimateOptions& options) {
    const size_t count = comparisons.Size();
    EmResult result;
    result.m = InitialM(comparisons, u);
    result.support.resize(count);
    result.lambda = options.lambda_init;

    // The level each pattern assigns each comparison, unpacked once: the inner
    // loop runs iterations x patterns x comparisons times and should not re-shift.
    std::vector<uint8_t> levels(entries.size() * count);
    for (size_t e = 0; e < entries.size(); ++e) {
        for (size_t c = 0; c < count; ++c) {
            levels[e * count + c] = comparisons.LevelOf(entries[e].gamma, c);
        }
    }

    std::vector<std::vector<double>> weight(count);
    std::vector<std::vector<Neumaier>> numerator(count);
    for (size_t c = 0; c < count; ++c) {
        weight[c].assign(result.m[c].size(), 0.0);
        result.support[c].assign(result.m[c].size(), 0.0);
    }

    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        for (size_t c = 0; c < count; ++c) {
            if (excluded[c]) continue;
            for (size_t l = 0; l < weight[c].size(); ++l) {
                weight[c][l] = std::log2(std::max(result.m[c][l], kFloor) /
                                         std::max(u[c][l], kFloor));
            }
            numerator[c].assign(weight[c].size(), Neumaier());
        }
        const double prior = std::log2(result.lambda / (1.0 - result.lambda));
        Neumaier matched;
        Neumaier total;

        for (size_t e = 0; e < entries.size(); ++e) {
            double bits = prior;
            for (size_t c = 0; c < count; ++c) {
                if (excluded[c]) continue;
                bits += weight[c][levels[e * count + c]];
            }
            const double responsibility = 1.0 / (1.0 + std::exp2(-bits));
            const double mass = static_cast<double>(entries[e].count) * responsibility;
            for (size_t c = 0; c < count; ++c) {
                if (excluded[c]) continue;
                numerator[c][levels[e * count + c]].Add(mass);
            }
            matched.Add(mass);
            total.Add(static_cast<double>(entries[e].count));
        }

        const double matched_total = matched.Total();
        const double lambda = total.Total() > 0.0 ? matched_total / total.Total() : 0.0;
        double change = std::fabs(lambda - result.lambda);
        for (size_t c = 0; c < count; ++c) {
            if (excluded[c]) continue;
            for (size_t l = 0; l < result.m[c].size(); ++l) {
                const double value =
                    matched_total > 0.0 ? numerator[c][l].Total() / matched_total : 0.0;
                change = std::max(change, std::fabs(value - result.m[c][l]));
                result.m[c][l] = value;
                result.support[c][l] = numerator[c][l].Total();
            }
        }
        result.lambda = std::min(std::max(lambda, kFloor), 1.0 - kFloor);
        result.iterations = iteration;
        result.change = change;
        if (change < options.tolerance) {
            result.converged = true;
            break;
        }
    }
    return result;
}

// A session that has converged to the swapped labelling reports the non-match
// class as the match class, and every parameter in it is inverted. It cannot be
// caught from the likelihood, which is symmetric under the swap, so it is caught
// from the meaning: matches must not disagree more often than random pairs do.
// This is the one condition that disqualifies a session's m.
void CheckSession(const ComparisonSet& comparisons, const EmResult& em,
                  const std::vector<std::vector<double>>& u,
                  const std::vector<bool>& excluded, SessionReport* session) {
    size_t inverted = 0;
    size_t checked = 0;
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        if (excluded[c]) continue;
        const size_t last = em.m[c].size() - 1;
        ++checked;
        if (em.m[c][last] > u[c][last]) ++inverted;
    }
    if (checked > 0 && inverted * 2 > checked) {
        session->warnings.push_back(
            "matches disagree more often than random pairs in " +
            std::to_string(inverted) + " of " + std::to_string(checked) +
            " comparisons: EM has probably converged to the swapped labelling");
    }
    // A strong source really does block almost nothing but matches -- an exact
    // email block is close to a deterministic rule -- so a high match rate is a
    // fact about the source, not a fault, and it is reported rather than acted on.
    if (em.lambda > 0.5) {
        session->notes.push_back(
            "most of this session's pairs are matches, so its m is close to a "
            "direct count over the blocked pairs rather than a mixture split");
    }
    if (!em.converged) {
        session->notes.push_back("EM stopped at the iteration limit, last change " +
                                 std::to_string(em.change));
    }
}

}  // namespace

bool Estimate(const RecordStore& store, const ComparisonSet& comparisons,
              const BlockingPlan& plan, const EstimateOptions& options, Model* model,
              EstimateReport* report, std::string* error) {
    const size_t count = comparisons.Size();
    if (count == 0) {
        *error = "the schema declares no comparisons to estimate";
        return false;
    }
    if (store.NumRecords() < 2) {
        *error = "estimation needs at least two records";
        return false;
    }

    // --- u -------------------------------------------------------------------
    const auto u_started = std::chrono::steady_clock::now();
    std::vector<std::vector<uint64_t>> sampled(count);
    for (size_t c = 0; c < count; ++c) {
        sampled[c].assign(comparisons.at(c).spec->levels.size(), 0);
    }
    uint64_t drawn = 0;
    SampleRandomPairs(store, comparisons, options, &sampled, &drawn);

    std::vector<uint64_t> nulls(count, 0);
    for (size_t c = 0; c < count; ++c) {
        uint64_t missing = 0;
        for (uint64_t row = 0; row < store.NumRecords(); ++row) {
            if (comparisons.IsNullValue(c, row)) ++missing;
        }
        nulls[c] = missing;
    }

    std::vector<std::vector<double>> u(count);
    std::vector<std::vector<bool>> u_exact(count);
    for (size_t c = 0; c < count; ++c) {
        const size_t levels = comparisons.at(c).spec->levels.size();
        u[c].assign(levels, 0.0);
        u_exact[c].assign(levels, false);
        double exact_mass = 0.0;
        for (size_t l = 0; l < levels; ++l) {
            double value = 0.0;
            if (ExactU(store, comparisons, c, l, nulls[c], &value)) {
                u[c][l] = value;
                u_exact[c][l] = true;
                exact_mass += value;
                ++report->u_exact_levels;
            }
        }
        // The sampled levels take what the exact ones leave, so the two sources
        // of truth cannot disagree about the total. A level the sample never hit
        // gets half a count, which is a floor with a meaning rather than a guess.
        const double spare = std::max(1.0 - exact_mass, 0.0);
        double raw_total = 0.0;
        for (size_t l = 0; l < levels; ++l) {
            if (!u_exact[c][l]) {
                raw_total += std::max(static_cast<double>(sampled[c][l]), 0.5);
            }
        }
        for (size_t l = 0; l < levels; ++l) {
            if (u_exact[c][l]) continue;
            const double raw = std::max(static_cast<double>(sampled[c][l]), 0.5);
            u[c][l] = raw_total > 0.0 ? spare * raw / raw_total : 0.0;
        }
    }
    report->u_pairs = drawn;
    report->u_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - u_started)
            .count();

    // --- sessions ------------------------------------------------------------
    // One session per column any EM-safe source conditions on. A source whose
    // selection event does not factor on a column subset is skipped here and used
    // only at prediction: estimating from it would bias every m at once, with no
    // column to hold out to repair it.
    std::vector<std::string> columns;
    std::vector<std::vector<size_t>> by_column;
    for (size_t s = 0; s < plan.Size(); ++s) {
        if (!plan.at(s).em_safe) continue;
        const std::string& column = plan.at(s).column;
        size_t index = columns.size();
        for (size_t i = 0; i < columns.size(); ++i) {
            if (columns[i] == column) index = i;
        }
        if (index == columns.size()) {
            columns.push_back(column);
            by_column.emplace_back();
        }
        by_column[index].push_back(s);
    }
    if (columns.empty()) {
        *error = "no EM-safe blocking source, so m cannot be estimated";
        return false;
    }

    std::vector<std::vector<double>> m_total(count);
    std::vector<std::vector<double>> support_total(count);
    std::vector<size_t> sessions_for(count, 0);
    for (size_t c = 0; c < count; ++c) {
        m_total[c].assign(comparisons.at(c).spec->levels.size(), 0.0);
        support_total[c].assign(comparisons.at(c).spec->levels.size(), 0.0);
    }

    double best_matches = 0.0;
    std::string best_column;
    for (size_t i = 0; i < columns.size(); ++i) {
        SessionReport session;
        session.column = columns[i];
        std::vector<bool> excluded(count, false);
        size_t usable = 0;
        for (size_t c = 0; c < count; ++c) {
            const std::vector<std::string>& used = comparisons.at(c).spec->columns;
            if (std::find(used.begin(), used.end(), columns[i]) != used.end()) {
                excluded[c] = true;
                session.excluded.push_back(comparisons.at(c).spec->name);
            } else {
                ++usable;
            }
        }
        for (const size_t s : by_column[i]) {
            session.sources.push_back(plan.at(s).name);
            session.candidates += plan.CountPairs(s);
        }
        if (usable == 0) {
            session.warnings.push_back(
                "every comparison reads this column, so the session can learn "
                "nothing");
            report->sessions.push_back(std::move(session));
            continue;
        }
        // Two free comparisons give a 2x2 table with three degrees of freedom and
        // the mixture has three free parameters, so the fit is saturated: it has
        // two exact solutions and EM cannot tell them apart. Three is the floor
        // for identifiability, and falling below it is a schema problem the run
        // has to say out loud rather than quietly return the wrong root.
        if (usable < 3) {
            session.warnings.push_back(
                "only " + std::to_string(usable) +
                " comparison(s) are free in this session, which does not identify "
                "the mixture; give the schema more comparisons that do not read \"" +
                columns[i] + "\"");
            report->sessions.push_back(std::move(session));
            continue;
        }

        HistogramOptions histogram_options;
        histogram_options.threads = options.threads;
        histogram_options.pair_cap = options.session_pairs;
        histogram_options.seed = options.seed + i;
        PatternHistogram histogram(comparisons.Width());
        HistogramStats stats;
        BuildHistogram(plan, comparisons, by_column[i], histogram_options, &histogram,
                       &stats);
        session.enumerated = stats.enumerated;
        session.folded = stats.folded;
        session.rate = stats.rate;
        session.distinct = stats.distinct;
        session.seconds = stats.seconds;

        if (histogram.TotalPairs() == 0) {
            session.warnings.push_back("the session produced no candidate pair");
            report->sessions.push_back(std::move(session));
            continue;
        }

        const std::vector<PatternCount> entries = histogram.Entries();
        const EmResult em = RunEm(comparisons, entries, u, excluded, options);
        session.iterations = em.iterations;
        session.change = em.change;
        session.converged = em.converged;
        session.lambda = em.lambda;
        session.implied_matches = em.lambda * static_cast<double>(stats.enumerated);
        CheckSession(comparisons, em, u, excluded, &session);

        session.merged = session.warnings.empty();
        if (session.merged) {
            for (size_t c = 0; c < count; ++c) {
                if (excluded[c]) continue;
                ++sessions_for[c];
                for (size_t l = 0; l < m_total[c].size(); ++l) {
                    m_total[c][l] += em.m[c][l];
                    support_total[c][l] += em.support[c][l];
                }
            }
            if (session.implied_matches > best_matches) {
                best_matches = session.implied_matches;
                best_column = columns[i];
            }
        } else {
            session.warnings.push_back(
                "this session's m estimates were not merged into the model");
        }
        report->sessions.push_back(std::move(session));
    }

    // --- assemble ------------------------------------------------------------
    const std::vector<std::vector<double>> fallback = InitialM(comparisons, u);
    model->records = store.NumRecords();
    model->comparisons.clear();
    for (size_t c = 0; c < count; ++c) {
        const ComparisonSpec& spec = *comparisons.at(c).spec;
        ModelComparison entry;
        entry.name = spec.name;
        entry.columns = spec.columns;
        entry.term_frequency = spec.term_frequency;
        entry.sessions = sessions_for[c];
        // A level whose u is exactly zero cannot fire for any pair: nothing in the
        // data reaches it, so it is given no weight rather than a floored one. The
        // null level of a column with no missing value is the usual case, and a
        // floor there would make "both present" the strongest signal in the model.
        std::vector<bool> reachable(spec.levels.size(), true);
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            reachable[l] = !(u_exact[c][l] && u[c][l] <= 0.0);
        }
        // A level no matching pair reached is not impossible, it is below the
        // resolution of the evidence, and half a pair is that resolution. Flooring
        // at kFloor instead would put -40 bits on one comparison and let a single
        // disagreement veto every score the model ever produces.
        double mass = 0.0;
        for (const double support : support_total[c]) mass += support;
        const double m_floor = mass > 1.0 ? 0.5 / mass : kFloor;
        double m_total_mass = 0.0;
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            if (!reachable[l]) continue;
            const double raw = sessions_for[c] > 0
                                   ? m_total[c][l] / static_cast<double>(sessions_for[c])
                                   : fallback[c][l];
            m_total_mass += std::max(raw, m_floor);
        }
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            ModelLevel level;
            level.label = spec.levels[l].Describe();
            level.u = std::min(std::max(u[c][l], kFloor), 1.0 - kFloor);
            level.u_exact = u_exact[c][l];
            level.u_observed = sampled[c][l];
            level.m_estimated = sessions_for[c] > 0;
            level.m_support = support_total[c][l];
            if (!reachable[l]) {
                level.m = level.u;  // weight exactly zero
                entry.levels.push_back(std::move(level));
                continue;
            }
            const double raw = level.m_estimated
                                   ? m_total[c][l] / static_cast<double>(sessions_for[c])
                                   : fallback[c][l];
            level.m = std::max(raw, m_floor) / m_total_mass;
            level.m = std::min(std::max(level.m, kFloor), 1.0 - kFloor);
            entry.levels.push_back(std::move(level));
        }
        if (sessions_for[c] == 0) {
            report->warnings.push_back(
                "comparison \"" + spec.name +
                "\" was held out of every session, so its m is a starting value and "
                "not an estimate");
        }
        model->comparisons.push_back(std::move(entry));
    }

    // --- lambda --------------------------------------------------------------
    const double records = static_cast<double>(store.NumRecords());
    const double all_pairs = records * (records - 1.0) / 2.0;
    if (options.lambda > 0.0) {
        model->lambda = options.lambda;
        model->lambda_basis = "given on the command line";
    } else if (best_matches > 0.0) {
        model->lambda = best_matches / all_pairs;
        std::ostringstream basis;
        basis << "lower bound: session " << best_column << " implies "
              << WithThousands(static_cast<uint64_t>(best_matches)) << " matches over "
              << WithThousands(static_cast<uint64_t>(all_pairs)) << " pairs";
        model->lambda_basis = basis.str();
        report->warnings.push_back(
            "lambda is a lower bound: it counts only the matches a session's "
            "blocking reached, and blocking recall is below one. Pass --lambda to "
            "set it from a count you trust.");
    } else {
        model->lambda = 1.0 / all_pairs;
        model->lambda_basis = "no session produced an estimate; one match assumed";
        report->warnings.push_back(
            "no session produced a usable lambda, so the prior is a placeholder");
    }
    return true;
}

void PrintEstimateReport(const EstimateReport& report, std::ostream& out) {
    out << "u from " << WithThousands(report.u_pairs) << " random pairs in " << std::fixed
        << std::setprecision(1) << report.u_seconds << " s, plus "
        << report.u_exact_levels << " levels in closed form from the term "
        << "frequencies\n\n";

    out << std::left << std::setw(14) << "Session" << std::right << std::setw(18)
        << "Enumerated" << std::setw(14) << "Compared" << std::setw(10) << "Patterns"
        << std::setw(7) << "Iters" << std::setw(11) << "Match rate" << std::setw(9)
        << "Seconds" << "\n";
    out << std::string(83, '-') << "\n";
    for (const SessionReport& session : report.sessions) {
        out << std::left << std::setw(14) << session.column << std::right << std::setw(18)
            << WithThousands(session.enumerated) << std::setw(14)
            << WithThousands(session.folded) << std::setw(10)
            << WithThousands(session.distinct) << std::setw(7) << session.iterations
            << std::setw(11) << std::fixed << std::setprecision(6) << session.lambda
            << std::setw(9) << std::setprecision(1) << session.seconds << "\n";
    }
    out << std::string(83, '-') << "\n\n";

    for (const SessionReport& session : report.sessions) {
        out << "Session " << session.column << "\n"
            << "  sources    " << Join(session.sources) << "\n"
            << "  held out   " << Join(session.excluded) << "\n";
        if (session.rate < 1.0) {
            out << "  sampled    " << std::scientific << std::setprecision(2)
                << session.rate << " of " << WithThousands(session.enumerated)
                << " enumerated pairs\n";
        }
        if (session.iterations > 0) {
            out << "  em         " << session.iterations << " iterations, last change "
                << std::scientific << std::setprecision(1) << session.change << ", "
                << (session.converged ? "converged" : "NOT converged") << "\n"
                << "  implies    " << std::fixed << std::setprecision(0)
                << session.implied_matches << " matching pairs among the "
                << WithThousands(session.enumerated) << " it reached\n";
        }
        for (const std::string& note : session.notes) {
            out << "  note       " << note << "\n";
        }
        for (const std::string& warning : session.warnings) {
            out << "  warning    " << warning << "\n";
        }
        out << "\n";
    }
    for (const std::string& warning : report.warnings) {
        out << "warning: " << warning << "\n";
    }
    if (!report.warnings.empty()) out << "\n";
}

}  // namespace cpplink
