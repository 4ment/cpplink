// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/levels.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "cpplink/format.hpp"
#include "cpplink/pair_stream.hpp"
#include "cpplink/string_metrics.hpp"

namespace cpplink {
namespace {

// Values of the outer loop handed out at a time. The dictionary is in no
// particular order, so a fixed stripe keeps the threads even with no planning.
constexpr uint32_t kStripe = 256;

double Log2(double value) { return std::log(value) / std::log(2.0); }

std::string Fixed(double value, int decimals) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return buffer;
}

std::string Signed(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%+.2f", value);
    return buffer;
}

// u runs from about 1 down to 1e-9 across the levels of one comparison, so no
// fixed number of decimals reads at both ends.
std::string Compact(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3g", value);
    return buffer;
}

// The similarity axis, from the top down.
//
// Bin 0 is identical values and is always a level of its own: the exact level is
// what term-frequency adjustment requires and what every schema writes. The last
// bin is the tail below the floor, which the else absorbs and which is counted by
// subtraction rather than enumerated, so no pair the signature bound rejects has to
// be looked at twice.
//
// The grid is the uniform one *plus every threshold the schema already declares*,
// which is what lets one histogram and one metric evaluation per value pair answer
// both questions: no bin then straddles a current level's boundary, so each bin
// belongs to exactly one of them.
struct Grid {
    LevelType metric = LevelType::kJaroWinkler;
    // Lower edge of each bin, descending. edges[0] is unused (bin 0 is equality)
    // and the tail has no edge, so this holds bins - 1 entries.
    std::vector<double> edges;
    size_t bins = 0;

    double Floor() const { return edges.empty() ? 0.0 : edges.back(); }

    // First hit wins, top down, which is the same rule the levels themselves run
    // under. Linear over a couple of dozen edges, against a Jaro that costs
    // hundreds of nanoseconds.
    size_t BinOfSimilarity(double similarity) const {
        if (similarity >= 1.0) return 0;
        for (size_t b = 1; b + 1 < bins; ++b) {
            if (similarity >= edges[b]) return b;
        }
        return bins - 1;
    }
    size_t BinOfDistance(int distance) const {
        if (distance <= 0) return 0;
        const size_t b = static_cast<size_t>(distance);
        return b + 1 < bins ? b : bins - 1;
    }
};

// Which of the comparison's own levels every bin lands on, and whether the
// comparison is one this can be asked about at all.
struct CurrentLevels {
    std::vector<size_t> of_bin;
    bool usable = false;
    std::string refusal;
};

// The metric a comparison's fuzzy levels are written in, or a reason there is not
// exactly one of them.
bool MetricOf(const ComparisonSpec& spec, LevelType* metric, std::string* refusal) {
    bool jaro = false;
    bool edit = false;
    for (const LevelSpec& level : spec.levels) {
        if (level.type == LevelType::kJaroWinkler) jaro = true;
        if (level.type == LevelType::kLevenshtein) edit = true;
    }
    if (jaro && edit) {
        *refusal =
            "its levels mix jaro_winkler and levenshtein, and one similarity axis "
            "cannot measure both";
        return false;
    }
    if (!jaro && !edit) {
        *refusal = "it has no fuzzy level, so there is no threshold to place";
        return false;
    }
    *metric = jaro ? LevelType::kJaroWinkler : LevelType::kLevenshtein;
    return true;
}

Grid BuildGrid(const ComparisonSpec& spec, LevelType metric,
               const LevelsOptions& options) {
    Grid grid;
    grid.metric = metric;
    if (metric == LevelType::kLevenshtein) {
        uint32_t top = options.edit_max;
        for (const LevelSpec& level : spec.levels) {
            if (level.type != LevelType::kLevenshtein) continue;
            top = std::max(top, static_cast<uint32_t>(level.threshold));
        }
        // One bin per distance, then the tail: bin d is exactly distance d.
        grid.edges.assign(top + 1, 0.0);
        for (uint32_t d = 0; d <= top; ++d) grid.edges[d] = static_cast<double>(d);
        grid.bins = grid.edges.size() + 1;
        return grid;
    }

    // The schema may already declare a level looser than the sweep's floor, and
    // measuring it off the grid would make the two halves of the report
    // incomparable, so the floor moves down to meet it.
    double floor = options.jaro_floor;
    for (const LevelSpec& level : spec.levels) {
        if (level.type != LevelType::kJaroWinkler) continue;
        floor = std::min(floor, level.threshold);
    }
    std::vector<double> edges;
    for (double edge = 1.0 - options.jaro_step; edge > floor - 1e-9;
         edge -= options.jaro_step) {
        edges.push_back(std::round(edge * 1e6) / 1e6);
    }
    for (const LevelSpec& level : spec.levels) {
        if (level.type != LevelType::kJaroWinkler) continue;
        edges.push_back(std::round(level.threshold * 1e6) / 1e6);
    }
    std::sort(edges.begin(), edges.end(), std::greater<double>());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    grid.edges.assign(1, 1.0);  // placeholder for the equality bin
    grid.edges.insert(grid.edges.end(), edges.begin(), edges.end());
    grid.bins = grid.edges.size() + 1;
    return grid;
}

// Every bin belongs to exactly one current level, because the grid carries the
// schema's own thresholds as edges.
CurrentLevels MapCurrentLevels(const ComparisonSpec& spec, const Grid& grid) {
    CurrentLevels map;
    map.of_bin.assign(grid.bins, spec.levels.size() - 1);
    size_t at_equality = spec.levels.size() - 1;
    for (size_t l = 0; l < spec.levels.size(); ++l) {
        const LevelType type = spec.levels[l].type;
        if (type == LevelType::kNull) continue;
        if (type == LevelType::kExact || type == grid.metric) {
            at_equality = l;
            break;
        }
        if (type == LevelType::kElse) break;
    }
    map.of_bin[0] = at_equality;
    for (size_t b = 1; b + 1 < grid.bins; ++b) {
        size_t chosen = spec.levels.size() - 1;
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            const LevelSpec& level = spec.levels[l];
            if (level.type == LevelType::kNull) continue;
            if (level.type == LevelType::kElse) break;
            if (level.type != grid.metric) continue;
            const bool fires = grid.metric == LevelType::kJaroWinkler
                                   ? grid.edges[b] >= level.threshold - 1e-9
                                   : grid.edges[b] <= level.threshold + 1e-9;
            if (fires) {
                chosen = l;
                break;
            }
        }
        map.of_bin[b] = chosen;
    }
    map.usable = true;
    return map;
}

// One dictionary self-join: the exact u curve, and the same pairs charged to the
// comparison's current levels.
struct Scan {
    std::vector<double> bin;    // ordered value-pair draws landing in each bin
    std::vector<double> level;  // the same, by current level
    uint64_t compared = 0;      // value pairs that survived the signature bound
};

Scan RunSelfJoin(const BoundComparison& bound, const Grid& grid, const CurrentLevels& map,
                 size_t levels, unsigned threads) {
    const std::vector<uint32_t>& tf = bound.strings->tf;
    const uint32_t values = static_cast<uint32_t>(tf.size());
    const SignatureTable& signatures = *bound.signatures;
    const Dictionary& dict = bound.strings->dict;
    const bool jaro = grid.metric == LevelType::kJaroWinkler;
    const double loosest = jaro ? grid.Floor() : 0.0;
    const int longest = jaro ? 0 : static_cast<int>(grid.edges.back());

    std::vector<Scan> local(threads);
    std::atomic<uint32_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            Scan& mine = local[t];
            mine.bin.assign(grid.bins, 0.0);
            mine.level.assign(levels, 0.0);
            for (;;) {
                const uint32_t begin = next.fetch_add(kStripe);
                if (begin >= values) break;
                const uint32_t end = std::min<uint32_t>(begin + kStripe, values);
                for (uint32_t i = begin; i < end; ++i) {
                    if (tf[i] == 0) continue;
                    const double count_i = static_cast<double>(tf[i]);
                    const uint64_t mask_i = signatures.Mask(i);
                    const uint32_t length_i = signatures.Length(i);
                    const std::string_view left = dict.Value(i);
                    for (uint32_t j = i + 1; j < values; ++j) {
                        if (tf[j] == 0) continue;
                        const uint64_t mask_j = signatures.Mask(j);
                        const uint32_t length_j = signatures.Length(j);
                        size_t bin = 0;
                        if (jaro) {
                            if (JaroWinklerUpperBound(mask_i, length_i, mask_j,
                                                      length_j) < loosest) {
                                continue;
                            }
                            bin = grid.BinOfSimilarity(JaroWinkler(left, dict.Value(j)));
                        } else {
                            if (LevenshteinLowerBound(mask_i, length_i, mask_j,
                                                      length_j) > longest) {
                                continue;
                            }
                            bin = grid.BinOfDistance(
                                BoundedLevenshtein(left, dict.Value(j), longest));
                        }
                        ++mine.compared;
                        if (bin + 1 >= grid.bins) continue;  // the tail, by subtraction
                        const double draws = 2.0 * count_i * static_cast<double>(tf[j]);
                        mine.bin[bin] += draws;
                        mine.level[map.of_bin[bin]] += draws;
                    }
                }
            }
        });
    }
    for (std::thread& worker : workers) worker.join();

    Scan total;
    total.bin.assign(grid.bins, 0.0);
    total.level.assign(levels, 0.0);
    for (unsigned t = 0; t < threads; ++t) {
        total.compared += local[t].compared;
        for (size_t b = 0; b < grid.bins; ++b) total.bin[b] += local[t].bin[b];
        for (size_t l = 0; l < levels; ++l) total.level[l] += local[t].level[l];
    }
    // A value against itself is an exact agreement, and there are c(c-1) ordered
    // draws of two *distinct* rows carrying it. Adding the diagonal here keeps the
    // inner loop to the pairs that need a metric, and keeps the denominator the
    // without-replacement one every other u in this pipeline uses.
    for (uint32_t v = 0; v < values; ++v) {
        const double count = static_cast<double>(tf[v]);
        if (count < 2.0) continue;
        total.bin[0] += count * (count - 1.0);
        total.level[map.of_bin[0]] += count * (count - 1.0);
    }
    return total;
}

// Which bin two rows land in on this comparison's axis, or `grid.bins` when either
// side is missing. The anchor curve and the truth curve both come through here, so
// they cannot drift apart.
size_t BinOfRows(const Grid& grid, const Dictionary& dict,
                 const std::vector<uint32_t>& ids, int longest, uint64_t a, uint64_t b) {
    const uint32_t left = ids[a];
    const uint32_t right = ids[b];
    if (left == kNullId || right == kNullId) return grid.bins;
    if (left == right) return 0;
    if (grid.metric == LevelType::kJaroWinkler) {
        return grid.BinOfSimilarity(JaroWinkler(dict.Value(left), dict.Value(right)));
    }
    return grid.BinOfDistance(
        BoundedLevenshtein(dict.Value(left), dict.Value(right), longest));
}

// The bits a contiguous run of bins is worth to a matching pair. Zero mass under M
// contributes nothing, which is what makes a level nobody lands on free to merge
// away rather than infinitely bad.
double GroupBits(const std::vector<SimilarityBin>& bins, size_t begin, size_t end,
                 double m_floor, double u_floor, double* mass, double* rate) {
    double m = 0.0;
    double u = 0.0;
    for (size_t b = begin; b < end; ++b) {
        m += bins[b].m;
        u += bins[b].u;
    }
    *mass = m;
    *rate = u;
    if (m <= 0.0) return 0.0;
    return std::max(m, m_floor) * Log2(std::max(m, m_floor) / std::max(u, u_floor));
}

}  // namespace

LevelsReport BuildLevels(const RecordStore& store, const ComparisonSet& comparisons,
                         PairMode mode, const LevelsOptions& options,
                         const TruthPairs* truth) {
    const auto started = std::chrono::steady_clock::now();
    LevelsReport report;
    report.records = store.NumRecords();
    report.mode = mode;
    report.truthed = truth != nullptr && !truth->rows.empty();
    report.truth_pairs = report.truthed ? truth->rows.size() : 0;
    report.threads = ResolveThreads(options.threads);

    // The m curve needs matching pairs, and anchors are where they come from. This
    // is the same estimate the profile's M side reports, read at a finer grain: one
    // histogram over a similarity axis rather than one agreement rate.
    const ProfileReport profile = BuildProfile(store, mode, options.profile);
    const std::vector<uint64_t> rows = AnchorRows(report.records, options.profile);
    report.anchor_rows = rows.empty() ? report.records : rows.size();
    if (!profile.anchored) {
        report.anchor_refusal = profile.anchor_refusal.empty()
                                    ? std::string("the profile found no anchor")
                                    : profile.anchor_refusal;
    }

    for (size_t c = 0; c < comparisons.Size(); ++c) {
        const auto comparison_started = std::chrono::steady_clock::now();
        const BoundComparison& bound = comparisons.at(c);
        const ComparisonSpec& spec = *bound.spec;
        ComparisonLevels out;
        out.comparison = c;
        out.name = spec.name;

        if (bound.strings == nullptr || bound.lists != nullptr ||
            spec.columns.size() != 1) {
            out.refusal = "it is not a single string column";
            report.comparisons.push_back(std::move(out));
            continue;
        }
        LevelType metric = LevelType::kJaroWinkler;
        if (!MetricOf(spec, &metric, &out.refusal)) {
            report.comparisons.push_back(std::move(out));
            continue;
        }
        if (bound.signatures == nullptr) {
            out.refusal = "it has no signature table, so the self-join has no bound";
            report.comparisons.push_back(std::move(out));
            continue;
        }
        out.metric = metric;

        // Which store column this reads, for its null count and for the anchor
        // sessions that are allowed to speak for it.
        size_t column = store.NumColumns();
        for (size_t i = 0; i < store.schema().columns.size(); ++i) {
            if (store.schema().columns[i].name == spec.columns[0]) column = i;
        }
        if (column >= store.NumColumns()) {
            out.refusal = "its column is not in the store";
            report.comparisons.push_back(std::move(out));
            continue;
        }
        if (!report.anchor_refusal.empty()) {
            out.refusal = "there are no anchor pairs: " + report.anchor_refusal;
            report.comparisons.push_back(std::move(out));
            continue;
        }

        // The anchor with the most pairs among those allowed to speak for this
        // column. One anchor rather than several pooled: two anchors both select
        // the pairs that agree on both, and counting those twice would weight the
        // curve towards the cleanest matches, which is the bias already in it.
        const AnchorSession* session = nullptr;
        for (const AnchorSession& candidate : profile.sessions) {
            if (!candidate.used) continue;
            if (std::find(candidate.learns.begin(), candidate.learns.end(), column) ==
                candidate.learns.end()) {
                continue;
            }
            if (session == nullptr || candidate.pairs > session->pairs) {
                session = &candidate;
            }
        }
        if (session == nullptr) {
            out.refusal =
                "every anchor either contains this column or has already spoken for "
                "it, so no session may be read for it";
            report.comparisons.push_back(std::move(out));
            continue;
        }
        out.anchor = session->anchor_names;

        const uint32_t values = static_cast<uint32_t>(bound.strings->tf.size());
        out.values = values;
        out.value_pairs = static_cast<uint64_t>(values) * (values - 1) / 2;
        if (out.value_pairs > options.ball.budget) {
            out.refusal = "a dictionary of " + WithThousands(values) + " values is " +
                          WithThousands(out.value_pairs / 1000000) +
                          "M value pairs, over --ball-budget";
            report.comparisons.push_back(std::move(out));
            continue;
        }

        const Grid grid = BuildGrid(spec, metric, options);
        out.floor = grid.Floor();
        const CurrentLevels map = MapCurrentLevels(spec, grid);
        const Scan scan =
            RunSelfJoin(bound, grid, map, spec.levels.size(), report.threads);
        out.compared = scan.compared;

        const double present =
            static_cast<double>(report.records - store.NullCount(column));
        const double draws = present * (present - 1.0);
        if (draws <= 0.0) {
            out.refusal = "fewer than two rows carry a value";
            report.comparisons.push_back(std::move(out));
            continue;
        }

        // The m curve, over the anchor's pairs, on the same bins.
        std::vector<uint64_t> match(grid.bins, 0);
        std::vector<uint64_t> match_level(spec.levels.size(), 0);
        uint64_t walked = 0;
        const Dictionary& dict = bound.strings->dict;
        const std::vector<uint32_t>& ids = bound.strings->ids;
        const int longest =
            metric == LevelType::kJaroWinkler ? 0 : static_cast<int>(grid.edges.back());
        WalkAnchorPairs(store, mode, session->anchor, rows, options.profile.anchor_pairs,
                        [&](uint64_t a, uint64_t b) {
                            const size_t bin = BinOfRows(grid, dict, ids, longest, a, b);
                            if (bin >= grid.bins) return;
                            ++walked;
                            ++match[bin];
                            ++match_level[map.of_bin[bin]];
                        });
        out.match_pairs = walked;
        if (walked < options.min_match_pairs) {
            out.refusal = "its anchor left only " + WithThousands(walked) +
                          " pairs carrying the column on both rows, and a curve needs " +
                          WithThousands(options.min_match_pairs);
            report.comparisons.push_back(std::move(out));
            continue;
        }

        const double inverse_matches = 1.0 / static_cast<double>(walked);
        const double m_floor = 0.5 * inverse_matches;
        const double u_floor = 0.5 / draws;
        out.bins.assign(grid.bins, SimilarityBin());
        double above = 0.0;
        uint64_t above_matches = 0;
        for (size_t b = 0; b + 1 < grid.bins; ++b) {
            out.bins[b].threshold = grid.edges[b];
            out.bins[b].exact = b == 0;
            out.bins[b].u = scan.bin[b] / draws;
            out.bins[b].m = static_cast<double>(match[b]) * inverse_matches;
            out.bins[b].match_pairs = match[b];
            above += out.bins[b].u;
            above_matches += match[b];
        }
        SimilarityBin& tail = out.bins.back();
        tail.tail = true;
        tail.threshold = grid.Floor();
        tail.u = std::max(0.0, 1.0 - above);
        tail.match_pairs = walked - above_matches;
        tail.m = static_cast<double>(tail.match_pairs) * inverse_matches;

        // The truth curve, on the same bins, if there is a truth file. It is scored
        // against and never fitted to: nothing below reads it.
        std::vector<uint64_t> known(grid.bins, 0);
        std::vector<uint64_t> known_level(spec.levels.size(), 0);
        if (report.truthed) {
            for (const std::pair<uint32_t, uint32_t>& pair : truth->rows) {
                const size_t bin =
                    BinOfRows(grid, dict, ids, longest, pair.first, pair.second);
                if (bin >= grid.bins) continue;
                ++out.truth_pairs;
                ++known[bin];
                ++known_level[map.of_bin[bin]];
            }
            out.truthed = out.truth_pairs > 0;
        }
        const double inverse_truth =
            out.truthed ? 1.0 / static_cast<double>(out.truth_pairs) : 0.0;
        const double truth_floor = out.truthed ? 0.5 * inverse_truth : 0.0;
        if (out.truthed) {
            out.truth_m.assign(grid.bins, 0.0);
            for (size_t b = 0; b < grid.bins; ++b) {
                out.truth_m[b] = static_cast<double>(known[b]) * inverse_truth;
            }
        }

        // The current levels, on the same two populations.
        double current_above = 0.0;
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            if (spec.levels[l].type == LevelType::kNull) continue;
            ProposedLevel level;
            level.type = spec.levels[l].type;
            level.threshold = spec.levels[l].threshold;
            level.label = spec.levels[l].Describe();
            level.m = static_cast<double>(match_level[l]) * inverse_matches;
            level.u = spec.levels[l].type == LevelType::kElse
                          ? std::max(0.0, 1.0 - current_above)
                          : scan.level[l] / draws;
            if (spec.levels[l].type != LevelType::kElse) current_above += level.u;
            level.floored = level.m > 0.0 && (level.m < m_floor || level.u < u_floor);
            if (level.m > 0.0) {
                const double m = std::max(level.m, m_floor);
                const double u = std::max(level.u, u_floor);
                level.weight = Log2(m / u);
                level.bits = m * level.weight;
                out.current_bits += level.bits;
            }
            if (out.truthed) {
                level.truth_m = static_cast<double>(known_level[l]) * inverse_truth;
                if (level.truth_m > 0.0) {
                    level.truth_weight = Log2(std::max(level.truth_m, truth_floor) /
                                              std::max(level.u, u_floor));
                    level.truth_bits =
                        std::max(level.truth_m, truth_floor) * level.truth_weight;
                    out.current_truth_bits += level.truth_bits;
                }
            }
            out.current.push_back(std::move(level));
        }

        // The partition. Bin 0 is its own level, so the dynamic program chooses
        // where to cut bins 1..B-1 into the remaining groups, the last of which
        // ends at the tail and becomes the else.
        const size_t bins = grid.bins;
        const size_t max_groups = std::min(options.max_levels, bins) - 1;
        const double negative = -std::numeric_limits<double>::infinity();
        std::vector<std::vector<double>> best(max_groups + 1,
                                              std::vector<double>(bins + 1, negative));
        std::vector<std::vector<size_t>> cut(max_groups + 1,
                                             std::vector<size_t>(bins + 1, 0));
        double mass = 0.0;
        double rate = 0.0;
        for (size_t b = 1; b < bins; ++b) {
            best[1][b] = GroupBits(out.bins, b, bins, m_floor, u_floor, &mass, &rate);
        }
        for (size_t k = 2; k <= max_groups; ++k) {
            for (size_t b = 1; b < bins; ++b) {
                for (size_t c2 = b + 1; c2 + k - 1 <= bins; ++c2) {
                    if (best[k - 1][c2] == negative) continue;
                    const double here =
                        GroupBits(out.bins, b, c2, m_floor, u_floor, &mass, &rate) +
                        best[k - 1][c2];
                    if (here > best[k][b]) {
                        best[k][b] = here;
                        cut[k][b] = c2;
                    }
                }
            }
        }

        const double exact_bits =
            GroupBits(out.bins, 0, 1, m_floor, u_floor, &mass, &rate);
        const double n = static_cast<double>(walked);
        double best_score = negative;
        size_t reachable = 0;
        for (size_t groups = 1; groups <= max_groups; ++groups) {
            if (best[groups][1] == negative) break;
            const size_t count = groups + 1;  // the exact level, plus the groups
            const double bits = exact_bits + best[groups][1];
            out.bits_by_count.push_back(bits);
            // BIC over the anchor pairs the m curve rests on: one free m per level
            // beyond the first, since u is exact and costs nothing to state.
            out.score_by_count.push_back(n * bits -
                                         0.5 * static_cast<double>(count - 1) * Log2(n));
            size_t width = 1;
            while ((static_cast<size_t>(1) << width) < count) ++width;
            out.width_by_count.push_back(width);
            reachable = count;
            if (out.score_by_count.back() > best_score) {
                best_score = out.score_by_count.back();
                out.bic_count = count;
            }
        }
        if (reachable < 2) {
            out.refusal = "no partition of the similarity axis carries any evidence";
            report.comparisons.push_back(std::move(out));
            continue;
        }

        // The proposal is made at the schema's own count unless one was asked for,
        // which is what makes the two partitions comparable: same gamma width, same
        // number of cuts, and the only difference is where they sit.
        for (const LevelSpec& level : spec.levels) {
            if (level.type != LevelType::kNull) ++out.schema_count;
        }
        size_t want = options.levels > 0 ? options.levels : out.schema_count;
        want = std::max<size_t>(2, std::min(want, reachable));
        out.best_count = want;
        out.best_bits = out.bits_by_count[want - 2];

        // Walk the chosen partition back out as levels.
        std::vector<std::pair<size_t, size_t>> groups;
        groups.emplace_back(0, 1);
        size_t at = 1;
        for (size_t k = out.best_count - 1; k >= 1; --k) {
            const size_t end = k == 1 ? bins : cut[k][at];
            groups.emplace_back(at, end);
            at = end;
            if (k == 1) break;
        }
        for (size_t g = 0; g < groups.size(); ++g) {
            ProposedLevel level;
            const bool last = g + 1 == groups.size();
            if (g == 0) {
                level.type = LevelType::kExact;
            } else if (last) {
                level.type = LevelType::kElse;
            } else {
                level.type = metric;
                level.threshold = grid.edges[groups[g].second - 1];
            }
            LevelSpec described;
            described.type = level.type;
            described.threshold = level.threshold;
            level.label = described.Describe();
            level.bits = GroupBits(out.bins, groups[g].first, groups[g].second, m_floor,
                                   u_floor, &mass, &rate);
            level.m = mass;
            level.u = rate;
            level.floored = mass > 0.0 && (mass < m_floor || rate < u_floor);
            if (mass > 0.0) {
                level.weight = Log2(std::max(mass, m_floor) / std::max(rate, u_floor));
            }
            if (level.type == metric && std::abs(level.threshold - grid.Floor()) < 1e-9) {
                out.floor_binds = true;
            }
            level.bin_begin = groups[g].first;
            level.bin_end = groups[g].second;
            if (out.truthed) {
                for (size_t b = level.bin_begin; b < level.bin_end; ++b) {
                    level.truth_m += out.truth_m[b];
                }
                if (level.truth_m > 0.0) {
                    level.truth_weight = Log2(std::max(level.truth_m, truth_floor) /
                                              std::max(level.u, u_floor));
                    level.truth_bits =
                        std::max(level.truth_m, truth_floor) * level.truth_weight;
                    out.best_truth_bits += level.truth_bits;
                }
            }
            out.best.push_back(std::move(level));
        }

        out.proposed = true;
        out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                    comparison_started)
                          .count();
        report.comparisons.push_back(std::move(out));
    }

    report.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return report;
}

namespace {

void PrintLevelTable(const std::vector<ProposedLevel>& levels, double total,
                     std::ostream& out) {
    for (const ProposedLevel& level : levels) {
        out << "    " << std::left << std::setw(28) << Truncate(level.label, 27)
            << std::right << std::setw(9) << Fixed(level.m, 4) << std::setw(12)
            << Compact(level.u) << std::setw(9)
            << (level.m > 0.0 ? Signed(level.weight) : "-") << std::setw(9)
            << (level.m > 0.0 ? Signed(level.bits) : "-")
            << (level.floored ? "  floored" : "") << "\n";
    }
    out << "    " << std::string(28, ' ') << std::right << std::setw(30) << ""
        << std::setw(9) << Fixed(total, 2) << " bits\n";
}

}  // namespace

void PrintLevelsReport(const LevelsReport& report, std::ostream& out) {
    out << "Records      " << WithThousands(report.records) << "\n"
        << "Mode         " << PairModeName(report.mode) << "\n"
        << "Anchor rows  " << WithThousands(report.anchor_rows) << "\n"
        << "Time         " << Fixed(report.seconds, 1) << " s  (" << report.threads
        << " threads)\n\n";

    size_t proposed = 0;
    for (const ComparisonLevels& item : report.comparisons) {
        out << item.name << "\n";
        if (!item.proposed) {
            out << "  No proposal: " << item.refusal << ".\n\n";
            continue;
        }
        ++proposed;
        out << "  " << LevelTypeName(item.metric) << " over "
            << WithThousands(item.values) << " values, " << WithThousands(item.compared)
            << " of " << WithThousands(item.value_pairs)
            << " value pairs past the bound\n"
            << "  m from " << WithThousands(item.match_pairs) << " anchor pairs on ";
        for (size_t i = 0; i < item.anchor.size(); ++i) {
            out << (i > 0 ? " + " : "") << item.anchor[i];
        }
        out << "\n\n";

        out << "  Current" << std::string(29, ' ') << std::right << std::setw(6) << "m"
            << std::setw(12) << "u" << std::setw(9) << "Weight" << std::setw(9) << "Bits"
            << "\n";
        PrintLevelTable(item.current, item.current_bits, out);

        out << "\n  Proposed, " << item.best_count << " levels\n";
        PrintLevelTable(item.best, item.best_bits, out);

        out << "\n  Levels ";
        for (size_t i = 0; i < item.bits_by_count.size(); ++i) {
            out << std::setw(8) << (i + 2);
        }
        out << "\n  Bits   ";
        for (double bits : item.bits_by_count) out << std::setw(8) << Fixed(bits, 2);
        out << "\n  y bits ";
        for (size_t width : item.width_by_count) out << std::setw(8) << width;
        out << "\n         ";
        for (size_t i = 0; i < item.bits_by_count.size(); ++i) {
            std::string mark;
            if (i + 2 == item.best_count) mark += "*";
            if (i + 2 == item.bic_count) mark += "bic";
            out << std::setw(8) << mark;
        }
        out << "\n";
        if (item.truthed) {
            out << "  Against " << WithThousands(item.truth_pairs)
                << " known pairs, which the proposal never saw: current "
                << Fixed(item.current_truth_bits, 2) << " bits,\n  proposed "
                << Fixed(item.best_truth_bits, 2) << " bits, "
                << Signed(item.best_truth_bits - item.current_truth_bits) << "\n";
        }
        if (item.floor_binds) {
            out << "  The lowest cut sits on the grid floor of " << Fixed(item.floor, 2)
                << ", so the partition wanted to go\n  lower and --jaro-floor stopped "
                   "it.\n";
        }
        out << "\n";
    }

    if (proposed == 0) {
        out << "Nothing was proposed. The report above says why for each comparison.\n";
        return;
    }

    out << std::left << std::setw(22) << "Comparison" << std::right << std::setw(8)
        << "Levels" << std::setw(10) << "Current" << std::setw(10) << "Proposed"
        << std::setw(9) << "Gain";
    if (report.truthed) {
        out << std::setw(10) << "Cur (T)" << std::setw(10) << "Prop (T)" << std::setw(9)
            << "Gain (T)";
    }
    out << "\n" << std::string(report.truthed ? 88 : 59, '-') << "\n";
    double current_total = 0.0;
    double best_total = 0.0;
    double current_truth = 0.0;
    double best_truth = 0.0;
    for (const ComparisonLevels& item : report.comparisons) {
        if (!item.proposed) continue;
        current_total += item.current_bits;
        best_total += item.best_bits;
        current_truth += item.current_truth_bits;
        best_truth += item.best_truth_bits;
        out << std::left << std::setw(22) << Truncate(item.name, 21) << std::right
            << std::setw(8) << item.best_count << std::setw(10)
            << Fixed(item.current_bits, 2) << std::setw(10) << Fixed(item.best_bits, 2)
            << std::setw(9) << Signed(item.best_bits - item.current_bits);
        if (report.truthed) {
            out << std::setw(10)
                << (item.truthed ? Fixed(item.current_truth_bits, 2) : "-")
                << std::setw(10) << (item.truthed ? Fixed(item.best_truth_bits, 2) : "-")
                << std::setw(9)
                << (item.truthed ? Signed(item.best_truth_bits - item.current_truth_bits)
                                 : "-");
        }
        out << "\n";
    }
    out << std::left << std::setw(22) << "total" << std::right << std::setw(8) << ""
        << std::setw(10) << Fixed(current_total, 2) << std::setw(10)
        << Fixed(best_total, 2) << std::setw(9) << Signed(best_total - current_total);
    if (report.truthed) {
        out << std::setw(10) << Fixed(current_truth, 2) << std::setw(10)
            << Fixed(best_truth, 2) << std::setw(9) << Signed(best_truth - current_truth);
    }
    out << "\n\n";

    out << "Bits is what a level contributes to a matching pair, m * log2(m/u), and "
           "the\n"
        << "column sums to what the whole comparison is worth. u is exact, from the "
           "value\n"
        << "self-join over the whole column. m is from anchor pairs and reads high, "
           "which\n"
        << "is why the current levels are re-measured on the same two curves rather "
           "than\n"
        << "read off a fitted model: both sides carry the same bias and the difference "
           "is\n"
        << "what the gain is.\n\n"
        << "The proposal is made at the count the schema already uses, so gamma is the "
           "same\n"
        << "width and the only thing that moved is where the cuts are. Merging two "
           "bins can\n"
        << "only lose bits, so the count cannot be read off Bits, and BIC does not "
           "settle\n"
        << "it either: over tens of thousands of anchor pairs a hundredth of a bit a "
           "pair\n"
        << "is hundreds of bits of likelihood, so bic marks the widest partition on "
           "offer\n"
        << "almost every time. The cost it cannot see is gamma's, which is a table "
           "size and\n"
        << "not a per-pair one: y bits is the field width each count needs, and it is "
           "the\n"
        << "step from one width to the next that is worth paying a real gain for. Use\n"
        << "--levels to see the partition at any other count.\n";
    if (report.truthed) {
        out << "\nThe (T) columns re-measure both partitions against the known pairs "
               "in the\ntruth file, which nothing above them ever read: the cuts are "
               "placed on the\nanchor curve alone. A gain that survives the move to "
               "the truth curve is a\nproperty of the column; one that does not was a "
               "property of the anchor.\n";
    }
}

void WriteLevelsJson(const LevelsReport& report, std::ostream& out) {
    nlohmann::json root;
    root["records"] = report.records;
    root["mode"] = PairModeName(report.mode);
    root["truthed"] = report.truthed;
    root["truth_pairs"] = report.truth_pairs;
    root["anchor_rows"] = report.anchor_rows;
    root["anchor_refusal"] = report.anchor_refusal;
    root["seconds"] = report.seconds;

    root["comparisons"] = nlohmann::json::array();
    for (const ComparisonLevels& item : report.comparisons) {
        nlohmann::json entry;
        entry["name"] = item.name;
        entry["proposed"] = item.proposed;
        if (!item.proposed) {
            entry["refusal"] = item.refusal;
            root["comparisons"].push_back(std::move(entry));
            continue;
        }
        entry["metric"] = LevelTypeName(item.metric);
        entry["floor"] = item.floor;
        entry["anchor"] = item.anchor;
        entry["match_pairs"] = item.match_pairs;
        entry["values"] = item.values;
        entry["value_pairs"] = item.value_pairs;
        entry["compared"] = item.compared;
        entry["current_bits"] = item.current_bits;
        entry["best_bits"] = item.best_bits;
        entry["truthed"] = item.truthed;
        entry["truth_pairs"] = item.truth_pairs;
        entry["current_truth_bits"] = item.current_truth_bits;
        entry["best_truth_bits"] = item.best_truth_bits;
        entry["best_count"] = item.best_count;
        entry["schema_count"] = item.schema_count;
        entry["bic_count"] = item.bic_count;
        entry["floor_binds"] = item.floor_binds;
        entry["bits_by_count"] = item.bits_by_count;
        entry["score_by_count"] = item.score_by_count;
        entry["width_by_count"] = item.width_by_count;
        entry["seconds"] = item.seconds;

        const auto levels = [](const std::vector<ProposedLevel>& source) {
            nlohmann::json array = nlohmann::json::array();
            for (const ProposedLevel& level : source) {
                nlohmann::json item;
                item["type"] = LevelTypeName(level.type);
                item["label"] = level.label;
                if (level.type == LevelType::kJaroWinkler ||
                    level.type == LevelType::kLevenshtein) {
                    item["threshold"] = level.threshold;
                }
                item["m"] = level.m;
                item["u"] = level.u;
                item["weight"] = level.weight;
                item["bits"] = level.bits;
                item["floored"] = level.floored;
                item["truth_m"] = level.truth_m;
                item["truth_weight"] = level.truth_weight;
                item["truth_bits"] = level.truth_bits;
                array.push_back(std::move(item));
            }
            return array;
        };
        entry["current"] = levels(item.current);
        entry["best"] = levels(item.best);

        entry["bins"] = nlohmann::json::array();
        for (const SimilarityBin& bin : item.bins) {
            nlohmann::json cell;
            cell["threshold"] = bin.threshold;
            cell["exact"] = bin.exact;
            cell["tail"] = bin.tail;
            cell["u"] = bin.u;
            cell["m"] = bin.m;
            cell["match_pairs"] = bin.match_pairs;
            if (item.truthed) cell["truth_m"] = item.truth_m[&bin - item.bins.data()];
            entry["bins"].push_back(std::move(cell));
        }
        root["comparisons"].push_back(std::move(entry));
    }
    out << root.dump(2) << "\n";
}

bool RewriteSchema(const std::string& text, const LevelsReport& report,
                   std::string* rewritten, std::string* error) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& failure) {
        *error = std::string("schema is not valid JSON: ") + failure.what();
        return false;
    }
    if (!root.contains("comparisons") || !root["comparisons"].is_array()) {
        *error = "schema declares no comparisons to rewrite";
        return false;
    }
    for (const ComparisonLevels& item : report.comparisons) {
        if (!item.proposed) continue;
        if (item.comparison >= root["comparisons"].size()) {
            *error = "schema and report disagree about how many comparisons there are";
            return false;
        }
        nlohmann::json& target = root["comparisons"][item.comparison];
        nlohmann::json levels = nlohmann::json::array();
        // The null level is not a threshold question, so it is carried over rather
        // than proposed.
        if (target.contains("levels") && target["levels"].is_array() &&
            !target["levels"].empty() &&
            target["levels"][0].value("type", "") == std::string("null")) {
            levels.push_back(target["levels"][0]);
        }
        for (const ProposedLevel& level : item.best) {
            nlohmann::json entry;
            entry["type"] = LevelTypeName(level.type);
            if (level.type == LevelType::kJaroWinkler ||
                level.type == LevelType::kLevenshtein) {
                entry["threshold"] = level.threshold;
            }
            levels.push_back(std::move(entry));
        }
        target["levels"] = std::move(levels);
    }
    *rewritten = root.dump(2) + "\n";
    return true;
}

}  // namespace cpplink
