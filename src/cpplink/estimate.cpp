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

#include "cpplink/format.hpp"
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

// Whether a level's u can be had exactly, without drawing a single pair.
//
// A null level placed first fires exactly when either side has no value, which is
// a count over the data. An exact level on one interned column fires exactly when
// two independent draws land on the same value, which is the term frequencies'
// second moment. Nothing else is closed form.
// Sum over values and inputs of the squared per-input count.
//
// Term frequencies pool the inputs -- they say a value occurs n times, not how
// those n split between the files -- so this is the one quantity the closed form
// for a link run's u cannot read off them. It costs one pass per input over one
// column and one count array, which is nothing beside the sampling it replaces.
bool WithinDatasetCollisions(const RecordStore& store, const BoundComparison& bound,
                             double* total) {
    const size_t datasets = store.NumDatasets();
    size_t values = 0;
    if (bound.strings != nullptr) {
        values = bound.strings->tf.size();
    } else if (bound.dates != nullptr) {
        values = bound.dates->tf.size();
    } else {
        return false;
    }

    std::vector<uint32_t> counts;
    Neumaier sum;
    for (size_t d = 0; d < datasets; ++d) {
        counts.assign(values, 0);
        const uint64_t end = store.DatasetEnd(d);
        for (uint64_t row = store.DatasetStart(d); row < end; ++row) {
            if (bound.strings != nullptr) {
                const uint32_t id = bound.strings->ids[row];
                if (id != kNullId) ++counts[id];
            } else {
                const int32_t date = bound.dates->values[row];
                if (date != kNullDate) {
                    ++counts[static_cast<size_t>(date - bound.dates->tf_origin)];
                }
            }
        }
        for (const uint32_t count : counts) {
            const double n = static_cast<double>(count);
            sum.Add(n * n);
        }
    }
    *total = sum.Total();
    return true;
}

// The same second moment for an exact level of a comparison over several string
// columns, which cannot come off the term frequencies: the level reads one column
// but the comparison is null wherever *any* of them is, so the rows that can agree
// on the level's column are the rows where the whole comparison is present. One
// pass over the rows counts exactly those, per value and per input, and hands
// back the three sums the dedup and link forms need. It is the shape the email
// comparison takes, and it matters there more than anywhere: an address is the
// column whose u sits near 1/N, where a sampled u has nothing to see.
void RestrictedCollisions(const RecordStore& store, const ComparisonSet& comparisons,
                          size_t index, const StringColumn& column, double* total_sq,
                          double* present, double* within_sq) {
    const size_t datasets = store.NumDatasets();
    const size_t values = column.tf.size();
    std::vector<uint32_t> pooled(values, 0);
    std::vector<uint32_t> counts;
    Neumaier within;
    for (size_t d = 0; d < datasets; ++d) {
        counts.assign(values, 0);
        const uint64_t end = store.DatasetEnd(d);
        for (uint64_t row = store.DatasetStart(d); row < end; ++row) {
            if (comparisons.IsNullValue(index, row)) continue;
            const uint32_t id = column.ids[row];
            ++counts[id];
            ++pooled[id];
        }
        for (const uint32_t count : counts) {
            const double n = static_cast<double>(count);
            within.Add(n * n);
        }
    }
    Neumaier squares;
    Neumaier rows;
    for (const uint32_t count : pooled) {
        const double n = static_cast<double>(count);
        squares.Add(n * n);
        rows.Add(n);
    }
    *total_sq = squares.Total();
    *present = rows.Total();
    *within_sq = within.Total();
}

bool ExactU(const RecordStore& store, const ComparisonSet& comparisons, size_t index,
            size_t level, const std::vector<uint64_t>& nulls, PairMode mode,
            double* value) {
    const BoundComparison& bound = comparisons.at(index);
    const std::vector<LevelSpec>& levels = bound.spec->levels;
    const double records = static_cast<double>(store.NumRecords());
    if (records < 2.0) return false;
    const size_t datasets = store.NumDatasets();
    const bool cross = mode == PairMode::kCrossDataset;

    // The denominator both closed forms are taken over: every ordered pair of
    // distinct rows for a dedup run, and every ordered cross-input draw for a
    // link one.
    //
    // A row against itself is not a pair either run can see, and `SampleRandomPairs`
    // says so by skipping `a == b`; the closed form has to agree with it. For a
    // dedup that is `records` of the `records * records` ordered draws, which is
    // nothing while u is large and is the whole of u once u approaches 1/records --
    // exactly the regime the strongest columns sit in. A near-unique column read
    // over the wrong denominator loses several bits of weight. The link denominator
    // drops the diagonal already, because the within-input draws it subtracts
    // contain it.
    double space = records * records;
    if (cross) {
        double within = 0.0;
        for (size_t d = 0; d < datasets; ++d) {
            const double size =
                static_cast<double>(store.DatasetEnd(d) - store.DatasetStart(d));
            within += size * size;
        }
        space -= within;
        if (space <= 0.0) return false;
    } else {
        space -= records;
    }

    if (levels[level].type == LevelType::kNull) {
        if (level != 0) return false;  // an earlier level could pre-empt it
        if (!cross) {
            // Every input's nulls, not the first input's: a dedup run over more
            // than one file is `--mode link-and-dedup` and reads them all.
            double missing = 0.0;
            for (size_t d = 0; d < datasets; ++d) {
                missing += static_cast<double>(nulls[d]);
            }
            const double present = records - missing;
            *value = 1.0 - present * (present - 1.0) / space;
            return true;
        }
        // Neither side null, over cross-input draws: the present counts of two
        // different inputs multiplied, which is the square of their sum less the
        // sum of their squares.
        double present_total = 0.0;
        double present_squares = 0.0;
        for (size_t d = 0; d < datasets; ++d) {
            const double size =
                static_cast<double>(store.DatasetEnd(d) - store.DatasetStart(d));
            const double present = size - static_cast<double>(nulls[d]);
            present_total += present;
            present_squares += present * present;
        }
        *value = 1.0 - (present_total * present_total - present_squares) / space;
        return true;
    }
    if (levels[level].type != LevelType::kExact) return false;
    if (bound.lists != nullptr) return false;
    // Only a null level may sit above it, or equality is not the level's event.
    for (size_t l = 0; l < level; ++l) {
        if (levels[l].type != LevelType::kNull) return false;
    }

    if (bound.slots.size() > 1) {
        double total_sq = 0.0;
        double present = 0.0;
        double within_sq = 0.0;
        RestrictedCollisions(store, comparisons, index,
                             *bound.slots[levels[level].column].strings, &total_sq,
                             &present, &within_sq);
        *value = (total_sq - (cross ? within_sq : present)) / space;
        return true;
    }
    if (bound.spec->columns.size() != 1) return false;

    const std::vector<uint32_t>* tf = nullptr;
    if (bound.strings != nullptr) {
        tf = &bound.strings->tf;
    } else if (bound.dates != nullptr) {
        tf = &bound.dates->tf;
    }
    if (tf == nullptr) return false;

    Neumaier collisions;
    Neumaier present;
    for (const uint32_t frequency : *tf) {
        const double count = static_cast<double>(frequency);
        collisions.Add(count * count);
        present.Add(count);
    }
    double agreeing = collisions.Total();
    if (cross) {
        // Two draws from the same input are not pairs this run can see, so their
        // agreements come back out of the numerator exactly as they came out of
        // the denominator.
        double within = 0.0;
        if (!WithinDatasetCollisions(store, bound, &within)) return false;
        agreeing -= within;
    } else {
        // Sum of c^2 counts every present row agreeing with itself once. Sum of
        // c(c - 1) is what is left: the ordered pairs of distinct rows that agree,
        // over the denominator that now counts the same thing.
        agreeing -= present.Total();
    }
    *value = agreeing / space;
    return true;
}

// u is what a uniformly random pair does, so this draws uniformly random pairs.
// It is the one place estimation touches pairs that blocking never proposed, and
// it is deliberate: u must not know that blocking exists.
void SampleRandomPairs(const RecordStore& store, const ComparisonSet& comparisons,
                       const EstimateOptions& options, PairMode mode,
                       std::vector<std::vector<uint64_t>>* counts, uint64_t* drawn,
                       JointTables* joint) {
    const uint64_t records = store.NumRecords();
    const bool cross = mode == PairMode::kCrossDataset;
    const unsigned threads = ThreadCount(options.threads);
    std::vector<std::vector<std::vector<uint64_t>>> partials(threads, *counts);
    std::vector<uint64_t> per_thread(threads, 0);
    // The u side of every two-way interaction, from the same draws and at the cost
    // of one increment per comparison pair. It has to come from here rather than
    // from the candidate stream: a source selecting on a column induces agreement
    // correlation involving that column, so u-side dependence measured on
    // candidates measures the blocking plan.
    std::vector<JointTables> joints(joint != nullptr ? threads : 0,
                                    joint != nullptr ? *joint : JointTables());

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        const uint64_t share =
            options.u_sample / threads + (t < options.u_sample % threads ? 1 : 0);
        workers.emplace_back([&, t, share] {
            uint64_t state = Mix64(options.seed + 0x5DEECE66Dull * (t + 1));
            uint64_t taken = 0;
            std::vector<uint8_t> levels(comparisons.Size(), 0);
            for (uint64_t i = 0; i < share; ++i) {
                state = Mix64(state);
                const uint64_t a = state % records;
                state = Mix64(state);
                uint64_t b = state % records;
                if (cross) {
                    // Draw the partner from the rows outside a's own input, by
                    // picking a position in what is left once that input is taken
                    // out and stepping over the hole. With two inputs -- the case
                    // linking is about -- this is exactly uniform over cross pairs;
                    // with more it favours the smaller inputs slightly, which is
                    // why the number of inputs is reported beside u.
                    const uint64_t start = store.DatasetStart(store.DatasetOf(a));
                    const uint64_t end = store.DatasetEndFor(a);
                    const uint64_t outside = records - (end - start);
                    if (outside == 0) continue;
                    b = state % outside;
                    if (b >= start) b += end - start;
                }
                if (a == b) continue;  // a pair is two distinct records
                const uint32_t gamma = comparisons.Evaluate(a, b);
                for (size_t c = 0; c < comparisons.Size(); ++c) {
                    levels[c] = comparisons.LevelOf(gamma, c);
                    ++partials[t][c][levels[c]];
                }
                if (joint != nullptr) joints[t].Add(levels.data(), 1.0);
                ++taken;
            }
            per_thread[t] = taken;
        });
    }
    for (std::thread& worker : workers) worker.join();

    *drawn = 0;
    for (unsigned t = 0; t < threads; ++t) {
        *drawn += per_thread[t];
        if (joint != nullptr) joint->Merge(joints[t]);
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

// The m side of the interaction tables: the responsibility EM converged on,
// folded into a two-way count per comparison pair. It is one more pass over the
// patterns, which is 1e5 of them whatever the run size, and it reuses the fit
// rather than refitting anything.
JointTables SessionJointTables(const ComparisonSet& comparisons,
                               const std::vector<PatternCount>& entries,
                               const std::vector<std::vector<double>>& u,
                               const std::vector<bool>& excluded, const EmResult& em,
                               const std::vector<size_t>& level_counts) {
    const size_t count = comparisons.Size();
    JointTables tables(level_counts);
    std::vector<std::vector<double>> weight(count);
    for (size_t c = 0; c < count; ++c) {
        weight[c].assign(em.m[c].size(), 0.0);
        if (excluded[c]) continue;
        for (size_t l = 0; l < weight[c].size(); ++l) {
            weight[c][l] =
                std::log2(std::max(em.m[c][l], kFloor) / std::max(u[c][l], kFloor));
        }
    }
    const double prior = std::log2(em.lambda / (1.0 - em.lambda));
    std::vector<uint8_t> levels(count, 0);
    for (const PatternCount& entry : entries) {
        double bits = prior;
        for (size_t c = 0; c < count; ++c) {
            levels[c] = comparisons.LevelOf(entry.gamma, c);
            if (!excluded[c]) bits += weight[c][levels[c]];
        }
        const double responsibility = 1.0 / (1.0 + std::exp2(-bits));
        tables.Add(levels.data(), static_cast<double>(entry.count) * responsibility);
    }
    return tables;
}

// Whether blocking on one column conditions on the other. Containment and
// determination are the row-level shapes, and the u-side overlap is the same
// question in bits; any of the three means the second column's agreement is partly
// decided by the first, so a session blocking on one cannot be read for the other.
//
// The determination bar is 0.90 here rather than the 0.99 the profile's suspects
// table uses, because these are different questions. A report asks whether a column
// is redundant enough to drop; this asks whether a rate is readable at all, and
// `first_name` determining `gender` on 0.956 of rows is quite enough to make it not
// be.
bool ColumnsAreTied(const ColumnPairProfile& pair, double bits) {
    constexpr double kTiedShare = 0.90;
    if (pair.containment >= kTiedShare) return true;
    return pair.resolved && pair.redundant_bits >= bits;
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
    std::vector<size_t> level_counts(count);
    for (size_t c = 0; c < count; ++c) {
        level_counts[c] = comparisons.at(c).spec->levels.size();
    }
    JointTables random_joint(level_counts);
    SampleRandomPairs(store, comparisons, options, plan.mode(), &sampled, &drawn,
                      options.interactions.enabled ? &random_joint : nullptr);

    // Nulls are counted per input, because in link mode the two sides of a pair are
    // drawn from different ones and a column present in the first file and empty in
    // the second is not the same event as one half-empty across a single file.
    std::vector<std::vector<uint64_t>> nulls(count);
    for (size_t c = 0; c < count; ++c) {
        nulls[c].assign(store.NumDatasets(), 0);
        for (size_t d = 0; d < store.NumDatasets(); ++d) {
            uint64_t missing = 0;
            const uint64_t end = store.DatasetEnd(d);
            for (uint64_t row = store.DatasetStart(d); row < end; ++row) {
                if (comparisons.IsNullValue(c, row)) ++missing;
            }
            nulls[c][d] = missing;
        }
    }

    // The dictionary self-join, which turns the fuzzy levels' u from a sampled
    // number into an exact one. Term frequencies pool the inputs, so this is a
    // dedup-only closed form for the same reason the exact level's is.
    BallTables balls;
    if (options.fuzzy_u && plan.mode() != PairMode::kCrossDataset) {
        // The self-join is threaded by the same knob as everything else here:
        // --threads is the whole command's budget, not the pair walk's alone.
        BallOptions ball = options.ball;
        ball.threads = options.threads;
        balls.Build(comparisons, store.NumRecords(), ball);
        report->ball_seconds = balls.seconds;
        for (size_t c = 0; c < count; ++c) {
            BallReport entry;
            entry.comparison = comparisons.at(c).spec->name;
            entry.built = balls.Has(c);
            entry.reason = balls.reasons[c];
            if (entry.built) {
                entry.values = balls.tables[c].Values();
                entry.value_pairs = balls.tables[c].ValuePairs();
                entry.seconds = balls.tables[c].Seconds();
            }
            report->balls.push_back(entry);
        }
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
            if (ExactU(store, comparisons, c, l, nulls[c], plan.mode(), &value)) {
                u[c][l] = value;
                u_exact[c][l] = true;
                exact_mass += value;
                ++report->u_exact_levels;
                continue;
            }
            // A fuzzy level's u is what the self-join counted: the mass of value
            // pairs landing on it, over the same denominator.
            const LevelType type = comparisons.at(c).spec->levels[l].type;
            if (!balls.Has(c) || !balls.tables[c].Covers(l)) continue;
            if (type != LevelType::kLevenshtein && type != LevelType::kJaroWinkler) {
                continue;
            }
            u[c][l] = balls.tables[c].LevelU(l);
            u_exact[c][l] = true;
            exact_mass += u[c][l];
            ++report->u_ball_levels;
            ++report->balls[c].levels;
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

    // Which columns are tied to which, from the profile's pairwise pass over the
    // rows. No model, no plan and no candidate pair enters this.
    // An unblocked session holds nothing out, so a tie cannot widen a hold-out
    // there -- but it is exactly what makes such a session unreliable, and the
    // report has to be able to name the pairs. The rows are walked for either.
    ProfileReport profile;
    if (options.exclude_tied) {
        ProfileOptions profile_options;
        profile_options.anchors = false;  // the m side is not what this needs
        profile_options.sample_rows = options.tie_sample_rows;
        profile_options.threads = options.threads;
        profile_options.seed = options.seed;
        profile = BuildProfile(store, plan.mode(), profile_options);
        report->tie_seconds = profile.seconds;
    }
    const auto tied_to = [&](const std::string& column, const std::string& other) {
        // A declared derivation is a tie by construction and needs no rows to see.
        // The pairwise test below would have to rediscover it, and it can refuse
        // to: the joint of a column and a key derived from it is exactly the shape
        // a file's own duplicates swamp, and a near-unique source is a determinant
        // the profile declines to read.
        if (SameSource(store.schema(), column, other)) return true;
        for (const ColumnPairProfile& pair : profile.pairs) {
            const bool here = (pair.left_name == column && pair.right_name == other) ||
                              (pair.right_name == column && pair.left_name == other);
            if (here) return ColumnsAreTied(pair, options.tied_bits);
        }
        return false;
    };

    std::vector<SessionJoint> session_joints;
    double best_matches = 0.0;
    std::string best_column;
    // Whether the session behind lambda enumerated every admissible pair. Lambda
    // is a lower bound only because blocking misses matches; a session that does
    // no blocking misses none, and the warning would be false there.
    bool best_unblocked = false;
    for (size_t i = 0; i < columns.size(); ++i) {
        SessionReport session;
        // The unblocked source names no column, and holds none out: it conditions
        // on nothing, so every comparison is free in its session at once.
        const bool unblocked = columns[i].empty();
        session.column = unblocked ? "(all pairs)" : columns[i];
        std::vector<bool> excluded(count, false);
        size_t usable = 0;
        for (size_t c = 0; c < count; ++c) {
            const std::vector<std::string>& used = comparisons.at(c).spec->columns;
            if (std::find(used.begin(), used.end(), columns[i]) != used.end()) {
                excluded[c] = true;
                session.excluded.push_back(comparisons.at(c).spec->name);
                continue;
            }
            bool tied = false;
            for (const std::string& reads : used) {
                tied = tied || tied_to(columns[i], reads);
            }
            if (tied) {
                excluded[c] = true;
                session.tied.push_back(comparisons.at(c).spec->name);
            } else {
                ++usable;
            }
        }
        for (const size_t s : by_column[i]) {
            session.sources.push_back(plan.at(s).name);
            session.candidates += plan.CountPairs(s);
        }
        // A session that conditions on a column is protected from that column's
        // dependence by holding it out. An unblocked one holds nothing out, so a
        // dependence between two comparisons has nothing to repair it and goes
        // straight into m: measured on a file where one column contains two
        // others, this session's lambda reads 67x the truth and end-to-end F1
        // halves. Naming the pairs is the least the report can do.
        if (unblocked && options.exclude_tied) {
            for (size_t c = 0; c < count; ++c) {
                for (size_t d = c + 1; d < count; ++d) {
                    bool pair_tied = false;
                    for (const std::string& left : comparisons.at(c).spec->columns) {
                        for (const std::string& right : comparisons.at(d).spec->columns) {
                            pair_tied = pair_tied || tied_to(left, right);
                        }
                    }
                    if (!pair_tied) continue;
                    // A report-level warning rather than a session one: the tie
                    // test is deliberately loose (it is a hold-out widener, where
                    // a false positive costs only a session), and refusing the
                    // only session there is would leave the model at its defaults.
                    report->warnings.push_back(
                        "\"" + comparisons.at(c).spec->name + "\" and \"" +
                        comparisons.at(d).spec->name +
                        "\" read tied columns and no session holds either out, "
                        "because this plan does no blocking: their shared evidence "
                        "is counted twice and m is biased upwards. Block on a "
                        "column to get a hold-out, or drop one of the two.");
                }
            }
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
        if (options.interactions.enabled) {
            SessionJoint contribution;
            contribution.tables =
                SessionJointTables(comparisons, entries, u, excluded, em, level_counts);
            contribution.excluded = excluded;
            contribution.usable = session.merged;
            session_joints.push_back(std::move(contribution));
        }
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
                best_column = session.column;
                best_unblocked = unblocked;
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
    // The pair space lambda is put back on is the one the run is over: the whole
    // triangle for a dedup, the cross-product alone for a link.
    const double all_pairs = store.PairSpace(plan.mode());
    if (options.lambda > 0.0) {
        model->lambda = options.lambda;
        model->lambda_basis = "given on the command line";
    } else if (best_matches > 0.0) {
        model->lambda = best_matches / all_pairs;
        std::ostringstream basis;
        basis << (best_unblocked ? "session " : "lower bound: session ") << best_column
              << " implies " << WithThousands(static_cast<uint64_t>(best_matches))
              << " matches over " << WithThousands(static_cast<uint64_t>(all_pairs))
              << " pairs";
        model->lambda_basis = basis.str();
        if (!best_unblocked) {
            report->warnings.push_back(
                "lambda is a lower bound: it counts only the matches a session's "
                "blocking reached, and blocking recall is below one. Pass --lambda to "
                "set it from a count you trust.");
        }
    } else {
        model->lambda = 1.0 / all_pairs;
        model->lambda_basis = "no session produced an estimate; one match assumed";
        report->warnings.push_back(
            "no session produced a usable lambda, so the prior is a placeholder");
    }

    // --- interactions --------------------------------------------------------
    // Last, because a correction is measured against the margins the main effects
    // settled on and would be a different number against any others.
    if (options.interactions.enabled) {
        FitInteractions(comparisons, session_joints, random_joint, model->lambda,
                        options.interactions, model, &report->interactions);
    }
    return true;
}

void PrintEstimateReport(const EstimateReport& report, std::ostream& out) {
    out << "u from " << WithThousands(report.u_pairs) << " random pairs in " << std::fixed
        << std::setprecision(1) << report.u_seconds << " s, plus "
        << report.u_exact_levels << " levels in closed form from the term "
        << "frequencies\n";
    if (report.tie_seconds > 0.0) {
        out << "Column ties from one pairwise pass over the rows in "
            << std::setprecision(1) << report.tie_seconds << " s, no candidate pair\n";
    }
    if (!report.balls.empty()) {
        out << "Dictionary self-join gave " << report.u_ball_levels
            << " fuzzy levels an exact u in " << std::setprecision(1)
            << report.ball_seconds << " s\n";
        for (const BallReport& ball : report.balls) {
            out << "  " << std::left << std::setw(16) << Truncate(ball.comparison, 15);
            if (ball.built) {
                out << WithThousands(ball.values) << " values, "
                    << WithThousands(ball.value_pairs) << " value pairs, " << ball.levels
                    << " levels, " << std::setprecision(2) << ball.seconds << " s\n";
            } else {
                out << ball.reason << "\n";
            }
        }
    }
    out << "\n";

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
        if (!session.tied.empty()) {
            out << "  also tied  " << Join(session.tied)
                << "   (blocking on this column conditions on theirs)\n";
        }
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
    PrintInteractionReport(report.interactions, out);
}

}  // namespace cpplink
