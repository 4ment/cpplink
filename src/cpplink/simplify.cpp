// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/simplify.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "cpplink/format.hpp"
#include "cpplink/histogram.hpp"
#include "cpplink/score.hpp"

namespace cpplink {
namespace {

// Bits a level index needs, which is what merging a level away buys.
uint8_t BitsFor(size_t level_count) {
    uint8_t bits = 1;
    while ((1u << bits) < level_count) ++bits;
    return bits;
}

// Whether every pair the first level fires on also fires the second, which is
// what makes deleting the first and letting the second absorb its pairs the same
// partition rather than a different model.
//
// Only an ordered chain of one metric can say yes. `levenshtein <= 1` and
// `jaro_winkler >= 0.88` imply each other in neither direction, and nor does an
// exact match imply `list_overlap >= 2`, because two identical lists need not
// hold two elements. The pairwise levels are refused under an exact one for the
// same reason turned around: two identical lists may be empty, and then there is
// no element pair for a closest one to be.
bool Implies(const LevelSpec& upper, const LevelSpec& lower) {
    // Before anything else, including the else level, which absorbs whatever is
    // above it: a null level would be absorbed too, and missing is not a weak
    // agreement to be rounded down into one.
    if (upper.type == LevelType::kNull || lower.type == LevelType::kNull) return false;
    if (lower.type == LevelType::kElse) return true;
    if (upper.type == LevelType::kElse) return false;
    if (upper.type == LevelType::kExact) {
        switch (lower.type) {
            case LevelType::kLevenshtein:
            case LevelType::kDateWithin:
            case LevelType::kNumericWithin:
            case LevelType::kGeoWithin:
                return lower.threshold >= 0.0;
            case LevelType::kJaroWinkler:
                return lower.threshold <= 1.0;
            default:
                return false;
        }
    }
    // Exact membership is the value at distance zero from an element, so it
    // implies the fuzzy membership levels the way an exact match implies a fuzzy
    // one. It says nothing about the pairwise levels, which read the two lists
    // against each other rather than either value against a list.
    if (upper.type == LevelType::kListContains) {
        switch (lower.type) {
            case LevelType::kContainsLevenshtein:
                return lower.threshold >= 0.0;
            case LevelType::kContainsJaroWinkler:
                return lower.threshold <= 1.0;
            default:
                return false;
        }
    }
    if (upper.type != lower.type) return false;
    switch (upper.type) {
        case LevelType::kLevenshtein:
        case LevelType::kDateWithin:
        case LevelType::kNumericWithin:
        case LevelType::kGeoWithin:
        case LevelType::kListLevenshtein:
        case LevelType::kContainsLevenshtein:
            return upper.threshold <= lower.threshold;
        case LevelType::kJaroWinkler:
        case LevelType::kListJaccard:
        case LevelType::kListOverlap:
        case LevelType::kListJaroWinkler:
        case LevelType::kContainsJaroWinkler:
            return upper.threshold >= lower.threshold;
        default:
            return false;
    }
}

std::string WhyNot(const LevelSpec& upper, const LevelSpec& lower) {
    if (upper.type == LevelType::kNull || lower.type == LevelType::kNull) {
        return "missing is not a degree of agreement";
    }
    if (upper.type != lower.type && upper.type != LevelType::kExact &&
        upper.type != LevelType::kListContains) {
        return "different metrics: neither predicate implies the other";
    }
    return "the stronger predicate does not imply the weaker one";
}

double Cell(double observed, double expected) {
    if (observed <= 0.0 || expected <= 0.0) return 0.0;
    return 2.0 * observed * std::log(observed / expected);
}

// G^2 for the 2x2 table of level against class. Zero exactly when the two levels
// carry the same weight, which is the hypothesis being tested.
double LikelihoodRatio(double m1, double u1, double m2, double u2) {
    const double total = m1 + u1 + m2 + u2;
    if (total <= 0.0) return 0.0;
    const double top = m1 + u1;
    const double bottom = m2 + u2;
    const double matches = m1 + m2;
    const double nonmatches = u1 + u2;
    return Cell(m1, top * matches / total) + Cell(u1, top * nonmatches / total) +
           Cell(m2, bottom * matches / total) + Cell(u2, bottom * nonmatches / total);
}

// P(chi-square on one degree of freedom > x), which for one degree of freedom is
// closed form and needs no table.
double ChiSquareTail(double x) {
    if (!(x > 0.0)) return 1.0;
    return std::erfc(std::sqrt(x / 2.0));
}

double Log2(double value) { return value > 0.0 ? std::log2(value) : 0.0; }

// A run of consecutive levels that have been made one. `last` is the level that
// survives: the loosest predicate, which every other level in the run implies.
//
// A group carries the model's m and u, which add over a merged run because they
// are probabilities over the levels of one comparison, and the observed pairs,
// which are the histogram's.
struct Group {
    size_t first = 0;
    size_t last = 0;
    double m = 0.0;
    double u = 0.0;
    uint64_t pairs = 0;
};

double Weight(const Group& group) {
    if (group.m <= 0.0 || group.u <= 0.0) return 0.0;
    return std::log2(group.m / group.u);
}

double StreamWeight(double matches, double nonmatches, double total_matches,
                    double total_nonmatches) {
    if (total_matches <= 0.0 || total_nonmatches <= 0.0) return 0.0;
    const double m = matches / total_matches;
    const double u = nonmatches / total_nonmatches;
    if (m <= 0.0 || u <= 0.0) return 0.0;
    return Log2(m / u);
}

}  // namespace

bool ModelMatches(const Model& model, const ComparisonSet& comparisons,
                  std::string* error) {
    if (model.comparisons.size() != comparisons.Size()) {
        *error = "the model describes " + std::to_string(model.comparisons.size()) +
                 " comparisons but the schema declares " +
                 std::to_string(comparisons.Size());
        return false;
    }
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
    }
    return true;
}

SimplifyReport BuildSimplify(const RecordStore& store, const ComparisonSet& comparisons,
                             const BlockingPlan& plan, const Model& model,
                             const SimplifyOptions& options) {
    const auto started = std::chrono::steady_clock::now();
    SimplifyReport report;
    report.records = store.NumRecords();
    report.mode = plan.mode();
    report.width = comparisons.Width();

    HistogramOptions histogram_options;
    histogram_options.threads = options.threads;
    histogram_options.pair_cap = options.pair_cap;
    histogram_options.seed = options.seed;
    PatternHistogram histogram(comparisons.Width());
    HistogramStats stats;
    BuildHistogram(plan, comparisons, plan.AllSources(), histogram_options, &histogram,
                   &stats);
    report.enumerated = stats.enumerated;
    report.folded = stats.folded;
    report.rate = stats.rate;
    report.distinct = stats.distinct;
    report.threads = stats.threads;

    // What the stream put on each level, split into the two classes by the
    // responsibility the model gives each pattern. This is the only place the
    // model enters the test: the counts are the histogram's.
    const size_t count = comparisons.Size();
    std::vector<std::vector<uint64_t>> pairs(count);
    std::vector<std::vector<double>> matches(count);
    std::vector<std::vector<double>> nonmatches(count);
    for (size_t c = 0; c < count; ++c) {
        const size_t levels = comparisons.at(c).spec->levels.size();
        pairs[c].assign(levels, 0);
        matches[c].assign(levels, 0.0);
        nonmatches[c].assign(levels, 0.0);
    }

    for (const PatternCount& entry : histogram.Entries()) {
        double weight = model.PriorWeight();
        for (size_t c = 0; c < count; ++c) {
            const uint8_t level = comparisons.LevelOf(entry.gamma, c);
            const std::vector<ModelLevel>& learned = model.comparisons[c].levels;
            if (level < learned.size()) weight += learned[level].Weight();
        }
        const double posterior = ProbabilityForWeight(weight);
        const double folded = static_cast<double>(entry.count);
        report.expected_matches += folded * posterior;
        for (size_t c = 0; c < count; ++c) {
            const uint8_t level = comparisons.LevelOf(entry.gamma, c);
            if (level >= pairs[c].size()) continue;
            pairs[c][level] += entry.count;
            matches[c][level] += folded * posterior;
            nonmatches[c][level] += folded * (1.0 - posterior);
        }
    }

    for (size_t c = 0; c < count; ++c) {
        const ComparisonSpec& spec = *comparisons.at(c).spec;
        ComparisonSimplify item;
        item.name = spec.name;
        item.comparison = c;
        item.bits = spec.bits;

        double total_matches = 0.0;
        double total_nonmatches = 0.0;
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            total_matches += matches[c][l];
            total_nonmatches += nonmatches[c][l];
        }
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            LevelCounts level;
            level.label = spec.levels[l].Describe();
            level.type = spec.levels[l].type;
            level.threshold = spec.levels[l].threshold;
            level.pairs = pairs[c][l];
            level.matches = matches[c][l];
            level.nonmatches = nonmatches[c][l];
            level.m = model.comparisons[c].levels[l].m;
            level.u = model.comparisons[c].levels[l].u;
            level.weight = model.comparisons[c].levels[l].Weight();
            level.stream_weight = StreamWeight(level.matches, level.nonmatches,
                                               total_matches, total_nonmatches);
            item.levels.push_back(std::move(level));
        }

        // The scale the test is run at: the pairs this run actually scores, split
        // into the two classes by the responsibilities the model gives them. The
        // rates being compared are the model's own m and u, which is what
        // log2(m/u) is made of; the run supplies only how many pairs there are to
        // tell them apart with.
        const double scale_m = report.expected_matches;
        const double scale_u =
            static_cast<double>(histogram.TotalPairs()) - report.expected_matches;

        // Agglomerative, least distinguishable pair first, so a run of three
        // levels that cannot be separated becomes one rather than two.
        std::vector<Group> groups;
        for (size_t l = 0; l < spec.levels.size(); ++l) {
            groups.push_back(Group{l, l, model.comparisons[c].levels[l].m,
                                   model.comparisons[c].levels[l].u, pairs[c][l]});
        }
        std::vector<LevelMerge> merges;
        const auto test = [&](const Group& upper, const Group& lower, double* g2,
                              double* p, double* gap) {
            *g2 = LikelihoodRatio(scale_m * upper.m, scale_u * upper.u, scale_m * lower.m,
                                  scale_u * lower.u);
            *p = ChiSquareTail(*g2);
            *gap = std::abs(Weight(upper) - Weight(lower));
        };
        while (groups.size() > 1) {
            size_t best = groups.size();
            double best_p = 0.0;
            double best_g2 = 0.0;
            for (size_t g = 0; g + 1 < groups.size(); ++g) {
                if (!Implies(spec.levels[groups[g].last],
                             spec.levels[groups[g + 1].last])) {
                    continue;
                }
                double g2 = 0.0;
                double p = 1.0;
                double gap = 0.0;
                test(groups[g], groups[g + 1], &g2, &p, &gap);
                const bool mergeable = p >= options.alpha ||
                                       (options.min_gap > 0.0 && gap < options.min_gap);
                if (!mergeable) continue;
                if (best == groups.size() || p > best_p) {
                    best = g;
                    best_p = p;
                    best_g2 = g2;
                }
            }
            if (best == groups.size()) break;

            LevelMerge merge;
            merge.upper = groups[best].last;
            merge.lower = groups[best + 1].last;
            merge.g2 = best_g2;
            merge.p = best_p;
            merge.gap = std::abs(Weight(groups[best]) - Weight(groups[best + 1]));
            merge.stream_gap = std::abs(item.levels[merge.upper].stream_weight -
                                        item.levels[merge.lower].stream_weight);
            merge.merged = true;
            merge.drops_exact =
                spec.term_frequency && spec.levels[merge.upper].type == LevelType::kExact;
            merges.push_back(merge);

            groups[best].last = groups[best + 1].last;
            groups[best].m += groups[best + 1].m;
            groups[best].u += groups[best + 1].u;
            groups[best].pairs += groups[best + 1].pairs;
            groups.erase(groups.begin() + static_cast<std::ptrdiff_t>(best) + 1);
        }

        // What is left, and why it stayed: every adjacent pair of the final
        // grouping, so a comparison that merged nothing still says how far from
        // merging it was.
        for (size_t g = 0; g + 1 < groups.size(); ++g) {
            LevelMerge merge;
            merge.upper = groups[g].last;
            merge.lower = groups[g + 1].last;
            merge.stream_gap = std::abs(item.levels[merge.upper].stream_weight -
                                        item.levels[merge.lower].stream_weight);
            merge.gap = std::abs(Weight(groups[g]) - Weight(groups[g + 1]));
            if (!Implies(spec.levels[merge.upper], spec.levels[merge.lower])) {
                merge.expressible = false;
                merge.refusal =
                    WhyNot(spec.levels[merge.upper], spec.levels[merge.lower]);
            } else {
                double gap = 0.0;
                test(groups[g], groups[g + 1], &merge.g2, &merge.p, &gap);
            }
            merges.push_back(merge);
        }
        std::sort(
            merges.begin(), merges.end(),
            [](const LevelMerge& a, const LevelMerge& b) { return a.upper < b.upper; });
        item.tests = std::move(merges);

        for (const Group& group : groups) item.kept.push_back(group.last);
        item.proposed_bits = BitsFor(item.kept.size());
        item.changed = item.kept.size() < spec.levels.size();
        report.merged_levels += spec.levels.size() - item.kept.size();
        report.proposed_width =
            static_cast<uint8_t>(report.proposed_width + item.proposed_bits);
        report.comparisons.push_back(std::move(item));
    }

    report.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return report;
}

namespace {

std::string Fixed(double value, int decimals) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

std::string Probability(double p) {
    if (p < 0.0001) return "<0.0001";
    return Fixed(p, 4);
}

}  // namespace

void PrintSimplifyReport(const SimplifyReport& report, std::ostream& out) {
    out << "Records      " << std::setw(16) << WithThousands(report.records) << "\n";
    out << "Candidates   " << std::setw(16) << WithThousands(report.enumerated);
    if (report.folded != report.enumerated) {
        out << "  " << WithThousands(report.folded) << " folded at "
            << Fixed(report.rate, 4);
    }
    out << "\n";
    out << "Matches      " << std::setw(16)
        << WithThousands(static_cast<uint64_t>(report.expected_matches + 0.5))
        << "  expected, by the model\n";
    out << "Packed gamma " << std::setw(16) << static_cast<int>(report.width) << " bits";
    if (report.proposed_width != report.width) {
        out << " -> " << static_cast<int>(report.proposed_width);
    }
    out << "\n\n";

    for (const ComparisonSimplify& item : report.comparisons) {
        out << item.name << "  " << item.levels.size() << " levels";
        if (item.changed) out << " -> " << item.kept.size();
        out << ", " << static_cast<int>(item.bits) << " bits";
        if (item.proposed_bits != item.bits) {
            out << " -> " << static_cast<int>(item.proposed_bits);
        }
        out << "\n";
        out << "  " << std::left << std::setw(26) << "level" << std::right
            << std::setw(14) << "pairs" << std::setw(10) << "m" << std::setw(12) << "u"
            << std::setw(10) << "weight" << std::setw(10) << "stream" << "\n";
        for (size_t l = 0; l < item.levels.size(); ++l) {
            const LevelCounts& level = item.levels[l];
            const bool kept =
                std::find(item.kept.begin(), item.kept.end(), l) != item.kept.end();
            out << "  " << std::left << std::setw(26)
                << Truncate((kept ? "" : "- ") + level.label, 26) << std::right
                << std::setw(14) << WithThousands(level.pairs) << std::setw(10)
                << Fixed(level.m, 4) << std::setw(12) << Fixed(level.u, 6)
                << std::setw(10) << Fixed(level.weight, 2) << std::setw(10)
                << Fixed(level.stream_weight, 2) << "\n";
        }
        if (!item.tests.empty()) {
            out << "\n  " << std::left << std::setw(40) << "adjacent pair" << std::right
                << std::setw(14) << "G^2" << std::setw(10) << "p" << std::setw(8) << "gap"
                << "  verdict\n";
            for (const LevelMerge& test : item.tests) {
                const std::string names =
                    item.levels[test.upper].label + " / " + item.levels[test.lower].label;
                out << "  " << std::left << std::setw(40) << Truncate(names, 40)
                    << std::right;
                if (!test.expressible) {
                    out << std::setw(14) << "-" << std::setw(10) << "-" << std::setw(8)
                        << Fixed(test.gap, 2) << "  " << test.refusal << "\n";
                    continue;
                }
                out << std::setw(14) << Fixed(test.g2, 1) << std::setw(10)
                    << Probability(test.p) << std::setw(8) << Fixed(test.gap, 2) << "  "
                    << (test.merged ? (test.drops_exact ? "merged, drops the exact level"
                                                        : "merged")
                                    : "kept")
                    << "\n";
            }
        }
        out << "\n";
    }

    bool drops_exact = false;
    for (const ComparisonSimplify& item : report.comparisons) {
        for (const LevelMerge& test : item.tests) {
            drops_exact = drops_exact || test.drops_exact;
        }
    }
    if (report.merged_levels == 0) {
        out << "Nothing merged: every adjacent pair of levels is one a run this size\n"
               "can tell apart, or one whose predicates do not nest.\n";
    } else {
        out << WithThousands(report.merged_levels)
            << " level(s) merged. Re-run estimate against the rewritten schema: the\n"
               "merged parameters are not the pooled ones, and gamma packs "
               "differently.\n";
        if (drops_exact) {
            out << "\nOne merge deletes an exact level from a comparison that asks for\n"
                   "term-frequency adjustment. The adjustment moves to the fuzzy level\n"
                   "that absorbed it, which needs the neighbourhood masses that\n"
                   "--fuzzy-tf builds; without them that comparison loses its "
                   "adjustment.\n";
        }
    }
    out << "\nG^2 is the likelihood ratio for the 2x2 table of level against class,\n"
           "which is chi-square on one degree of freedom, and p is its tail. The\n"
           "rates compared are the model's own m and u, which is what log2(m/u) is\n"
           "made of; the run supplies the scale, so what the test asks is whether a\n"
           "run this size can tell the two levels' weights apart.\n\n"
           "Failing to reject is not equality, and that power grows with the run. On\n"
           "a file this project is aimed at nothing will merge and on a small one\n"
           "everything will, so `gap` -- the bits between the two weights -- is\n"
           "printed beside every p. That is the effect size, and --min-gap decides on\n"
           "it instead: a level a tenth of a bit from its neighbour is not earning a\n"
           "gamma code, however many pairs prove the tenth of a bit real.\n\n"
           "`stream` is the weight the run's own counts imply, which is not log2(m/u)\n"
           "because blocking chose the run. Where it has collapsed towards zero on a\n"
           "level, blocking has already spent that column's evidence.\n";
}

void WriteSimplifyJson(const SimplifyReport& report, std::ostream& out) {
    nlohmann::json root;
    root["records"] = report.records;
    root["mode"] = PairModeName(report.mode);
    root["enumerated"] = report.enumerated;
    root["folded"] = report.folded;
    root["rate"] = report.rate;
    root["distinct_patterns"] = report.distinct;
    root["expected_matches"] = report.expected_matches;
    root["gamma_bits"] = report.width;
    root["proposed_gamma_bits"] = report.proposed_width;
    root["merged_levels"] = report.merged_levels;
    root["seconds"] = report.seconds;
    root["comparisons"] = nlohmann::json::array();
    for (const ComparisonSimplify& item : report.comparisons) {
        nlohmann::json entry;
        entry["name"] = item.name;
        entry["bits"] = item.bits;
        entry["proposed_bits"] = item.proposed_bits;
        entry["changed"] = item.changed;
        entry["levels"] = nlohmann::json::array();
        for (size_t l = 0; l < item.levels.size(); ++l) {
            const LevelCounts& level = item.levels[l];
            nlohmann::json one;
            one["label"] = level.label;
            one["type"] = LevelTypeName(level.type);
            one["pairs"] = level.pairs;
            one["matches"] = level.matches;
            one["nonmatches"] = level.nonmatches;
            one["m"] = level.m;
            one["u"] = level.u;
            one["weight"] = level.weight;
            one["stream_weight"] = level.stream_weight;
            one["kept"] =
                std::find(item.kept.begin(), item.kept.end(), l) != item.kept.end();
            entry["levels"].push_back(std::move(one));
        }
        entry["tests"] = nlohmann::json::array();
        for (const LevelMerge& test : item.tests) {
            nlohmann::json one;
            one["upper"] = item.levels[test.upper].label;
            one["lower"] = item.levels[test.lower].label;
            one["expressible"] = test.expressible;
            one["g2"] = test.g2;
            one["p"] = test.p;
            one["gap"] = test.gap;
            one["stream_gap"] = test.stream_gap;
            one["merged"] = test.merged;
            one["drops_exact"] = test.drops_exact;
            if (!test.refusal.empty()) one["refusal"] = test.refusal;
            entry["tests"].push_back(std::move(one));
        }
        root["comparisons"].push_back(std::move(entry));
    }
    out << root.dump(2) << "\n";
}

bool RewriteSchema(const std::string& text, const SimplifyReport& report,
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
    for (const ComparisonSimplify& item : report.comparisons) {
        if (!item.changed) continue;
        if (item.comparison >= root["comparisons"].size()) {
            *error = "schema and report disagree about how many comparisons there are";
            return false;
        }
        nlohmann::json& target = root["comparisons"][item.comparison];
        if (!target.contains("levels") || !target["levels"].is_array() ||
            target["levels"].size() != item.levels.size()) {
            *error = "comparison \"" + item.name +
                     "\" has a different number of levels in the file than in the "
                     "report";
            return false;
        }
        nlohmann::json levels = nlohmann::json::array();
        // The surviving levels keep their own entries verbatim, so a label or
        // anything else the file carries on them survives the rewrite.
        for (const size_t kept : item.kept) levels.push_back(target["levels"][kept]);
        target["levels"] = std::move(levels);
    }
    *rewritten = root.dump(2) + "\n";
    return true;
}

}  // namespace cpplink
