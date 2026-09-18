// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/common.hpp"
#include "cpplink/completeness.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/explain_blocking.hpp"
#include "cpplink/id_index.hpp"
#include "cpplink/levels.hpp"
#include "cpplink/model.hpp"
#include "cpplink/neighbourhood.hpp"
#include "cpplink/pipeline.hpp"
#include "cpplink/profile.hpp"
#include "cpplink/recall.hpp"
#include "cpplink/score.hpp"
#include "cpplink/simplify.hpp"

namespace cpplink {
namespace python {

namespace {

TruthPairs LoadTruth(const RecordStore& store, const std::string& path) {
    TruthPairs truth;
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = LoadTruthPairs(path, store, &truth, &error);
    }
    Check(ok, error);
    return truth;
}

ProfileReport ProfileStage(Session& session, const ProfileOptions& options,
                           const std::string& truth_path) {
    TruthPairs truth;
    if (!truth_path.empty()) truth = LoadTruth(session.store(), truth_path);
    py::gil_scoped_release release;
    return BuildProfile(session.store(), session.mode(), options,
                        truth_path.empty() ? nullptr : &truth);
}

LevelsReport LevelsStage(Session& session, const LevelsOptions& options,
                         const std::string& truth_path) {
    Check(options.max_levels >= 2, "levels: max_levels must be at least 2");
    const ComparisonSet& comparisons = session.Comparisons();
    TruthPairs truth;
    if (!truth_path.empty()) truth = LoadTruth(session.store(), truth_path);
    py::gil_scoped_release release;
    return BuildLevels(session.store(), comparisons, session.mode(), options,
                       truth_path.empty() ? nullptr : &truth);
}

SimplifyReport SimplifyStage(Session& session, const Model& model,
                             const SimplifyOptions& options) {
    Check(options.alpha > 0.0 && options.alpha < 1.0,
          "simplify: alpha must be between 0 and 1");
    const ComparisonSet& comparisons = session.Comparisons();
    std::string error;
    Check(ModelMatches(model, comparisons, &error), "simplify: " + error);
    const BlockingPlan& plan = session.Plan(/*for_estimation=*/false);
    py::gil_scoped_release release;
    return BuildSimplify(session.store(), comparisons, plan, model, options);
}

// `levels --out` and `simplify --out`: the schema text with the proposal
// written into it.
template <typename Report>
std::string RewriteSchemaText(const std::string& text, const Report& report) {
    std::string rewritten;
    std::string error;
    Check(RewriteSchema(text, report, &rewritten, &error), error);
    return rewritten;
}

struct RecallOutcome {
    RecallMetrics metrics;
    std::string text;
    bool diagnosed = false;
    MissReport misses;
    std::string miss_text;
};

RecallOutcome RecallStage(Session& session, const std::string& truth_path, bool why,
                          bool count, size_t show_misses) {
    const BlockingPlan& plan = session.Plan(/*for_estimation=*/false);
    const TruthPairs truth = LoadTruth(session.store(), truth_path);
    RecallOutcome result;
    if (show_misses > 0) why = true;
    const ComparisonSet* comparisons = nullptr;
    if (why) {
        Check(!session.schema().comparisons.empty(),
              "recall: why needs the schema to declare \"comparisons\", to say what "
              "still agrees on a missed pair");
        comparisons = &session.Comparisons();
    }
    {
        py::gil_scoped_release release;
        result.metrics = MeasureRecall(plan, session.store(), truth, count);
        if (why) {
            result.diagnosed = true;
            result.misses.example_limit = show_misses;
            DiagnoseMisses(plan, *comparisons, truth, &result.misses);
        }
    }
    result.text =
        CaptureText([&](std::ostream& out) { PrintRecallReport(result.metrics, out); });
    if (why) {
        result.miss_text = CaptureText(
            [&](std::ostream& out) { PrintMissReport(plan, result.misses, out); });
    }
    return result;
}

CompletenessReport CompletenessStage(Session& session, const Model& model,
                                     const CompletenessOptions& options,
                                     const std::string& truth_path) {
    Check(options.sample > 0.0 && options.sample <= 1.0,
          "completeness: sample wants a rate in (0, 1]");
    const BlockingPlan& plan = session.Plan(/*for_estimation=*/false);
    const ComparisonSet& comparisons = session.Comparisons();
    CompletenessReport report;
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = EstimateCompleteness(session.store(), comparisons, plan, model, options,
                                  &report, &error);
    }
    Check(ok, error);
    // With a truth file the estimator is scored against the thing it exists to
    // replace, which is the only way to know whether to believe it without one.
    if (!truth_path.empty()) {
        const TruthPairs truth = LoadTruth(session.store(), truth_path);
        py::gil_scoped_release release;
        const RecallMetrics metrics = MeasureRecall(plan, session.store(), truth, false);
        if (metrics.truth_pairs > 0) {
            report.measured = true;
            report.truth_pairs = metrics.truth_pairs;
            report.pc_measured = static_cast<double>(metrics.union_found) /
                                 static_cast<double>(metrics.truth_pairs);
        }
    }
    return report;
}

// One pair explained: the levels it lands on, and with a model, the waterfall.
struct Explanation {
    uint64_t row_a = 0;
    uint64_t row_b = 0;
    std::string id_a;
    std::string id_b;
    uint32_t gamma = 0;
    std::vector<std::pair<std::string, std::string>> levels;  // comparison, label
    bool scored = false;
    PairWaterfall waterfall;
    std::string json;
    std::string text;
};

Explanation ExplainStage(Session& session, uint64_t row_a, uint64_t row_b,
                         const py::object& model_object, double threshold,
                         double tf_damping, bool fuzzy_tf, const BallOptions& ball,
                         bool interactions) {
    const RecordStore& store = session.store();
    Check(row_a < store.NumRecords() && row_b < store.NumRecords(),
          "row out of range; the file has " + std::to_string(store.NumRecords()) +
              " records");
    const ComparisonSet& comparisons = session.Comparisons();
    Explanation result;
    result.row_a = row_a;
    result.row_b = row_b;
    result.id_a = QualifiedId(store, row_a);
    result.id_b = QualifiedId(store, row_b);
    result.gamma = comparisons.Evaluate(row_a, row_b);
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        const BoundComparison& bound = comparisons.at(c);
        const uint8_t level = comparisons.LevelOf(result.gamma, c);
        result.levels.emplace_back(bound.spec->name,
                                   bound.spec->levels[level].Describe());
    }

    std::ostringstream text;
    PrintGammaLayout(comparisons, text);
    text << "\n";
    PrintPairExplanation(store, comparisons, row_a, row_b, text);
    if (!model_object.is_none()) {
        const Model& model = model_object.cast<const Model&>();
        ScoreOptions score;
        score.threshold = threshold;
        score.tf_damping = tf_damping;
        score.use_interactions = interactions;
        BallTables balls;
        if (fuzzy_tf) {
            py::gil_scoped_release release;
            BuildBallTables(comparisons, store, ball, &balls, text);
        }
        Scorer scorer;
        std::string error;
        Check(scorer.Bind(model, comparisons, store, score, &error,
                          fuzzy_tf ? &balls : nullptr),
              "explain: " + error);
        result.scored = true;
        result.waterfall =
            BuildPairWaterfall(store, comparisons, scorer, row_a, row_b, &model);
        result.json = PairWaterfallJson(result.waterfall);
        PrintPairWaterfall(result.waterfall, text);
    }
    result.text = text.str();
    return result;
}

// One source priced, as `explain-blocking` prints it.
struct SourceSummary {
    std::string name;
    std::string kind;
    std::string column;
    bool em_safe = true;
    uint64_t pairs = 0;
    uint64_t largest_group = 0;
};

struct BlockingReport {
    uint64_t records = 0;
    std::string mode;
    double pair_space = 0.0;
    std::vector<SourceSummary> sources;
    uint64_t candidate_sum = 0;
    bool counted_union = false;
    uint64_t candidate_union = 0;
    std::string text;
};

BlockingReport ExplainBlockingStage(Session& session, bool count) {
    const BlockingPlan& plan = session.Plan(/*for_estimation=*/false);
    const RecordStore& store = session.store();
    BlockingReport report;
    report.records = store.NumRecords();
    report.mode = PairModeName(plan.mode());
    report.pair_space = store.PairSpace(plan.mode());
    {
        py::gil_scoped_release release;
        for (size_t s = 0; s < plan.Size(); ++s) {
            const BoundSource& source = plan.at(s);
            SourceSummary summary;
            summary.name = source.name;
            summary.kind = SourceKindName(source.kind);
            summary.column = source.column;
            summary.em_safe = source.em_safe;
            summary.pairs = plan.CountPairs(s);
            summary.largest_group = plan.LargestGroup(s);
            report.candidate_sum += summary.pairs;
            report.sources.push_back(summary);
        }
        if (count) {
            report.counted_union = true;
            report.candidate_union = plan.CountUnion();
        }
    }
    report.text = CaptureText(
        [&](std::ostream& out) { PrintBlockingReport(plan, store, count, out); });
    return report;
}

template <typename Report, typename Printer, typename Writer>
void BindTextAndJson(py::class_<Report>* cls, Printer printer, Writer writer) {
    cls->def_property_readonly("text", [printer](const Report& r) {
        return CaptureText([&](std::ostream& out) { printer(r, out); });
    });
    cls->def("__repr__", [printer](const Report& r) {
        return CaptureText([&](std::ostream& out) { printer(r, out); });
    });
    cls->def(
        "json",
        [writer](const Report& r) {
            return CaptureText([&](std::ostream& out) { writer(r, out); });
        },
        "The report as its --json form, one document.");
}

void BindProfile(py::module_& m) {
    py::class_<ProfileOptions>(m, "ProfileOptions", "Every knob of `cpplink profile`.")
        .def(py::init<>())
        .def_readwrite("sample_rows", &ProfileOptions::sample_rows)
        .def_readwrite("pairs", &ProfileOptions::pairs)
        .def_readwrite("expected_matches", &ProfileOptions::expected_matches)
        .def_readwrite("threads", &ProfileOptions::threads)
        .def_readwrite("seed", &ProfileOptions::seed)
        .def_readwrite("anchors", &ProfileOptions::anchors)
        .def_readwrite("anchor_rows", &ProfileOptions::anchor_rows)
        .def_readwrite("anchor_margin", &ProfileOptions::anchor_margin)
        .def_readwrite("anchor_pairs", &ProfileOptions::anchor_pairs);

    py::class_<ColumnProfile>(m, "ColumnProfile")
        .def_readonly("name", &ColumnProfile::name)
        .def_property_readonly(
            "type", [](const ColumnProfile& c) { return ColumnTypeName(c.type); })
        .def_readonly("scored", &ColumnProfile::scored)
        .def_readonly("distinct", &ColumnProfile::distinct)
        .def_readonly("nulls", &ColumnProfile::nulls)
        .def_readonly("present", &ColumnProfile::present)
        .def_readonly("coverage", &ColumnProfile::coverage)
        .def_readonly("top_share", &ColumnProfile::top_share)
        .def_readonly("collision", &ColumnProfile::collision)
        .def_readonly("effective_values", &ColumnProfile::effective_values)
        .def_readonly("bits", &ColumnProfile::bits)
        .def_readonly("covered_bits", &ColumnProfile::covered_bits)
        .def_readonly("bits_floored", &ColumnProfile::bits_floored);

    py::class_<ColumnPairProfile>(m, "ColumnPairProfile")
        .def_readonly("left", &ColumnPairProfile::left)
        .def_readonly("right", &ColumnPairProfile::right)
        .def_readonly("left_name", &ColumnPairProfile::left_name)
        .def_readonly("right_name", &ColumnPairProfile::right_name)
        .def_readonly("rows", &ColumnPairProfile::rows)
        .def_readonly("determines_right", &ColumnPairProfile::determines_right)
        .def_readonly("determines_left", &ColumnPairProfile::determines_left)
        .def_readonly("left_distinct_share", &ColumnPairProfile::left_distinct_share)
        .def_readonly("right_distinct_share", &ColumnPairProfile::right_distinct_share)
        .def_readonly("baseline_right", &ColumnPairProfile::baseline_right)
        .def_readonly("baseline_left", &ColumnPairProfile::baseline_left)
        .def_readonly("left_informative", &ColumnPairProfile::left_informative)
        .def_readonly("right_informative", &ColumnPairProfile::right_informative)
        .def_readonly("containment", &ColumnPairProfile::containment)
        .def_readonly("left_inside_right", &ColumnPairProfile::left_inside_right)
        .def_readonly("u_left", &ColumnPairProfile::u_left)
        .def_readonly("u_right", &ColumnPairProfile::u_right)
        .def_readonly("u_joint", &ColumnPairProfile::u_joint)
        .def_readonly("redundant_bits", &ColumnPairProfile::redundant_bits)
        .def_readonly("joint_collisions", &ColumnPairProfile::joint_collisions)
        .def_readonly("expected_collisions", &ColumnPairProfile::expected_collisions)
        .def_readonly("resolved", &ColumnPairProfile::resolved)
        .def_readonly("m_pairs", &ColumnPairProfile::m_pairs)
        .def_readonly("m_left", &ColumnPairProfile::m_left)
        .def_readonly("m_right", &ColumnPairProfile::m_right)
        .def_readonly("m_joint", &ColumnPairProfile::m_joint)
        .def_readonly("m_redundant_bits", &ColumnPairProfile::m_redundant_bits)
        .def_readonly("m_resolved", &ColumnPairProfile::m_resolved)
        .def("net_redundant_bits", &ColumnPairProfile::NetRedundantBits)
        .def("suspect", &ColumnPairProfile::Suspect)
        .def("verdict", &ColumnPairProfile::Verdict);

    py::class_<AnchorSession>(m, "AnchorSession")
        .def_readonly("anchor", &AnchorSession::anchor)
        .def_readonly("anchor_names", &AnchorSession::anchor_names)
        .def_readonly("learns", &AnchorSession::learns)
        .def_readonly("bits", &AnchorSession::bits)
        .def_readonly("posterior_bits", &AnchorSession::posterior_bits)
        .def_readonly("groups", &AnchorSession::groups)
        .def_readonly("pairs", &AnchorSession::pairs)
        .def_readonly("oversized", &AnchorSession::oversized)
        .def_readonly("capped", &AnchorSession::capped)
        .def_readonly("used", &AnchorSession::used);

    py::class_<ColumnMatchProfile>(m, "ColumnMatchProfile")
        .def_readonly("column", &ColumnMatchProfile::column)
        .def_readonly("name", &ColumnMatchProfile::name)
        .def_readonly("sessions", &ColumnMatchProfile::sessions)
        .def_readonly("pairs", &ColumnMatchProfile::pairs)
        .def_readonly("coverage", &ColumnMatchProfile::coverage)
        .def_readonly("m", &ColumnMatchProfile::m)
        .def_readonly("m_low", &ColumnMatchProfile::m_low)
        .def_readonly("m_high", &ColumnMatchProfile::m_high)
        .def_readonly("weight", &ColumnMatchProfile::weight)
        .def_readonly("expected_bits", &ColumnMatchProfile::expected_bits)
        .def_readonly("floored", &ColumnMatchProfile::floored)
        .def_readonly("estimated", &ColumnMatchProfile::estimated)
        .def_readonly("truthed", &ColumnMatchProfile::truthed)
        .def_readonly("truth_pairs", &ColumnMatchProfile::truth_pairs)
        .def_readonly("truth_coverage", &ColumnMatchProfile::truth_coverage)
        .def_readonly("truth_m", &ColumnMatchProfile::truth_m)
        .def_readonly("truth_weight", &ColumnMatchProfile::truth_weight)
        .def_readonly("truth_expected_bits", &ColumnMatchProfile::truth_expected_bits);

    py::class_<ProfileReport> profile(
        m, "ProfileReport",
        "What the columns can be worth, what a match will\n"
        "score, and which pairs are the same evidence twice.");
    profile.def_readonly("records", &ProfileReport::records)
        .def_property_readonly(
            "mode", [](const ProfileReport& r) { return PairModeName(r.mode); })
        .def_readonly("pair_space", &ProfileReport::pair_space)
        .def_readonly("expected_matches", &ProfileReport::expected_matches)
        .def_readonly("match_rate", &ProfileReport::match_rate)
        .def_readonly("space_bits", &ProfileReport::space_bits)
        .def_readonly("prior_bits", &ProfileReport::prior_bits)
        .def_readonly("available_bits", &ProfileReport::available_bits)
        .def_readonly("redundant_bits", &ProfileReport::redundant_bits)
        .def_readonly("margin_bits", &ProfileReport::margin_bits)
        .def_readonly("columns", &ProfileReport::columns)
        .def_readonly("pairs", &ProfileReport::pairs)
        .def_readonly("unresolved_pairs", &ProfileReport::unresolved_pairs)
        .def_readonly("sessions", &ProfileReport::sessions)
        .def_readonly("matches", &ProfileReport::matches)
        .def_readonly("anchored", &ProfileReport::anchored)
        .def_readonly("anchor_refusal", &ProfileReport::anchor_refusal)
        .def_readonly("anchor_rows", &ProfileReport::anchor_rows)
        .def_readonly("anchor_pairs", &ProfileReport::anchor_pairs)
        .def_readonly("expected_bits", &ProfileReport::expected_bits)
        .def_readonly("double_counted_bits", &ProfileReport::double_counted_bits)
        .def_readonly("estimated_margin_bits", &ProfileReport::estimated_margin_bits)
        .def_readonly("estimated_columns", &ProfileReport::estimated_columns)
        .def_readonly("scored_columns", &ProfileReport::scored_columns)
        .def_readonly("truthed", &ProfileReport::truthed)
        .def_readonly("truth_pairs", &ProfileReport::truth_pairs)
        .def_readonly("truth_expected_bits", &ProfileReport::truth_expected_bits)
        .def_readonly("truth_margin_bits", &ProfileReport::truth_margin_bits)
        .def_readonly("truth_mean_error", &ProfileReport::truth_mean_error)
        .def_readonly("walked", &ProfileReport::walked)
        .def_readonly("sampled_rows", &ProfileReport::sampled_rows)
        .def_readonly("sampled", &ProfileReport::sampled)
        .def_readonly("threads", &ProfileReport::threads)
        .def_readonly("seconds", &ProfileReport::seconds);
    BindTextAndJson(&profile, &PrintProfileReport, &WriteProfileJson);
}

void BindLevels(py::module_& m) {
    py::class_<LevelsOptions>(m, "LevelsOptions", "Every knob of `cpplink levels`.")
        .def(py::init<>())
        .def_readwrite("jaro_floor", &LevelsOptions::jaro_floor)
        .def_readwrite("jaro_step", &LevelsOptions::jaro_step)
        .def_readwrite("edit_max", &LevelsOptions::edit_max)
        .def_readwrite("max_levels", &LevelsOptions::max_levels)
        .def_readwrite("levels", &LevelsOptions::levels)
        .def_readwrite("min_match_pairs", &LevelsOptions::min_match_pairs)
        .def_readwrite("profile", &LevelsOptions::profile)
        .def_readwrite("ball", &LevelsOptions::ball)
        .def_readwrite("threads", &LevelsOptions::threads);

    py::class_<SimilarityBin>(m, "SimilarityBin")
        .def_readonly("threshold", &SimilarityBin::threshold)
        .def_readonly("exact", &SimilarityBin::exact)
        .def_readonly("tail", &SimilarityBin::tail)
        .def_readonly("u", &SimilarityBin::u)
        .def_readonly("m", &SimilarityBin::m)
        .def_readonly("match_pairs", &SimilarityBin::match_pairs);

    py::class_<ProposedLevel>(m, "ProposedLevel")
        .def_property_readonly(
            "type", [](const ProposedLevel& l) { return LevelTypeName(l.type); })
        .def_readonly("threshold", &ProposedLevel::threshold)
        .def_readonly("label", &ProposedLevel::label)
        .def_readonly("m", &ProposedLevel::m)
        .def_readonly("u", &ProposedLevel::u)
        .def_readonly("weight", &ProposedLevel::weight)
        .def_readonly("bits", &ProposedLevel::bits)
        .def_readonly("floored", &ProposedLevel::floored)
        .def_readonly("bin_begin", &ProposedLevel::bin_begin)
        .def_readonly("bin_end", &ProposedLevel::bin_end)
        .def_readonly("truth_m", &ProposedLevel::truth_m)
        .def_readonly("truth_weight", &ProposedLevel::truth_weight)
        .def_readonly("truth_bits", &ProposedLevel::truth_bits);

    py::class_<ComparisonLevels>(m, "ComparisonLevels")
        .def_readonly("name", &ComparisonLevels::name)
        .def_readonly("comparison", &ComparisonLevels::comparison)
        .def_readonly("proposed", &ComparisonLevels::proposed)
        .def_readonly("refusal", &ComparisonLevels::refusal)
        .def_property_readonly(
            "metric", [](const ComparisonLevels& c) { return LevelTypeName(c.metric); })
        .def_readonly("floor", &ComparisonLevels::floor)
        .def_readonly("bins", &ComparisonLevels::bins)
        .def_readonly("current", &ComparisonLevels::current)
        .def_readonly("best", &ComparisonLevels::best)
        .def_readonly("current_bits", &ComparisonLevels::current_bits)
        .def_readonly("best_bits", &ComparisonLevels::best_bits)
        .def_readonly("best_count", &ComparisonLevels::best_count)
        .def_readonly("schema_count", &ComparisonLevels::schema_count)
        .def_readonly("bic_count", &ComparisonLevels::bic_count)
        .def_readonly("bits_by_count", &ComparisonLevels::bits_by_count)
        .def_readonly("score_by_count", &ComparisonLevels::score_by_count)
        .def_readonly("width_by_count", &ComparisonLevels::width_by_count)
        .def_readonly("floor_binds", &ComparisonLevels::floor_binds)
        .def_readonly("truthed", &ComparisonLevels::truthed)
        .def_readonly("truth_pairs", &ComparisonLevels::truth_pairs)
        .def_readonly("current_truth_bits", &ComparisonLevels::current_truth_bits)
        .def_readonly("best_truth_bits", &ComparisonLevels::best_truth_bits)
        .def_readonly("truth_m", &ComparisonLevels::truth_m)
        .def_readonly("anchor", &ComparisonLevels::anchor)
        .def_readonly("match_pairs", &ComparisonLevels::match_pairs)
        .def_readonly("values", &ComparisonLevels::values)
        .def_readonly("value_pairs", &ComparisonLevels::value_pairs)
        .def_readonly("compared", &ComparisonLevels::compared)
        .def_readonly("seconds", &ComparisonLevels::seconds);

    py::class_<LevelsReport> levels(m, "LevelsReport",
                                    "The fuzzy thresholds checked against the data.");
    levels.def_readonly("records", &LevelsReport::records)
        .def_property_readonly("mode",
                               [](const LevelsReport& r) { return PairModeName(r.mode); })
        .def_readonly("truthed", &LevelsReport::truthed)
        .def_readonly("truth_pairs", &LevelsReport::truth_pairs)
        .def_readonly("anchor_rows", &LevelsReport::anchor_rows)
        .def_readonly("anchor_refusal", &LevelsReport::anchor_refusal)
        .def_readonly("comparisons", &LevelsReport::comparisons)
        .def_readonly("threads", &LevelsReport::threads)
        .def_readonly("seconds", &LevelsReport::seconds)
        .def_property_readonly("proposed", [](const LevelsReport& r) {
            for (const ComparisonLevels& c : r.comparisons) {
                if (c.proposed) return true;
            }
            return false;
        });
    BindTextAndJson(&levels, &PrintLevelsReport, &WriteLevelsJson);
    m.def("_rewrite_schema", &RewriteSchemaText<LevelsReport>, py::arg("text"),
          py::arg("report"));
}

void BindSimplify(py::module_& m) {
    py::class_<SimplifyOptions>(m, "SimplifyOptions", "Every knob of `cpplink simplify`.")
        .def(py::init<>())
        .def_readwrite("alpha", &SimplifyOptions::alpha)
        .def_readwrite("min_gap", &SimplifyOptions::min_gap)
        .def_readwrite("pair_cap", &SimplifyOptions::pair_cap)
        .def_readwrite("threads", &SimplifyOptions::threads)
        .def_readwrite("seed", &SimplifyOptions::seed);

    py::class_<LevelCounts>(m, "LevelCounts")
        .def_readonly("label", &LevelCounts::label)
        .def_property_readonly("type",
                               [](const LevelCounts& l) { return LevelTypeName(l.type); })
        .def_readonly("threshold", &LevelCounts::threshold)
        .def_readonly("pairs", &LevelCounts::pairs)
        .def_readonly("matches", &LevelCounts::matches)
        .def_readonly("nonmatches", &LevelCounts::nonmatches)
        .def_readonly("m", &LevelCounts::m)
        .def_readonly("u", &LevelCounts::u)
        .def_readonly("weight", &LevelCounts::weight)
        .def_readonly("stream_weight", &LevelCounts::stream_weight);

    py::class_<LevelMerge>(m, "LevelMerge")
        .def_readonly("upper", &LevelMerge::upper)
        .def_readonly("lower", &LevelMerge::lower)
        .def_readonly("g2", &LevelMerge::g2)
        .def_readonly("p", &LevelMerge::p)
        .def_readonly("gap", &LevelMerge::gap)
        .def_readonly("stream_gap", &LevelMerge::stream_gap)
        .def_readonly("expressible", &LevelMerge::expressible)
        .def_readonly("merged", &LevelMerge::merged)
        .def_readonly("drops_exact", &LevelMerge::drops_exact)
        .def_readonly("refusal", &LevelMerge::refusal);

    py::class_<ComparisonSimplify>(m, "ComparisonSimplify")
        .def_readonly("name", &ComparisonSimplify::name)
        .def_readonly("comparison", &ComparisonSimplify::comparison)
        .def_readonly("levels", &ComparisonSimplify::levels)
        .def_readonly("tests", &ComparisonSimplify::tests)
        .def_readonly("kept", &ComparisonSimplify::kept)
        .def_readonly("bits", &ComparisonSimplify::bits)
        .def_readonly("proposed_bits", &ComparisonSimplify::proposed_bits)
        .def_readonly("changed", &ComparisonSimplify::changed);

    py::class_<SimplifyReport> simplify(m, "SimplifyReport",
                                        "The adjacent levels a run cannot tell apart.");
    simplify.def_readonly("records", &SimplifyReport::records)
        .def_property_readonly(
            "mode", [](const SimplifyReport& r) { return PairModeName(r.mode); })
        .def_readonly("enumerated", &SimplifyReport::enumerated)
        .def_readonly("folded", &SimplifyReport::folded)
        .def_readonly("rate", &SimplifyReport::rate)
        .def_readonly("distinct", &SimplifyReport::distinct)
        .def_readonly("expected_matches", &SimplifyReport::expected_matches)
        .def_readonly("comparisons", &SimplifyReport::comparisons)
        .def_readonly("width", &SimplifyReport::width)
        .def_readonly("proposed_width", &SimplifyReport::proposed_width)
        .def_readonly("merged_levels", &SimplifyReport::merged_levels)
        .def_readonly("threads", &SimplifyReport::threads)
        .def_readonly("seconds", &SimplifyReport::seconds);
    BindTextAndJson(&simplify, &PrintSimplifyReport, &WriteSimplifyJson);
    m.def("_rewrite_schema", &RewriteSchemaText<SimplifyReport>, py::arg("text"),
          py::arg("report"));
}

void BindRecall(py::module_& m) {
    py::class_<SourceRecall>(m, "SourceRecall")
        .def_readonly("name", &SourceRecall::name)
        .def_readonly("found", &SourceRecall::found)
        .def_readonly("first_to", &SourceRecall::first_to)
        .def_readonly("candidate_pairs", &SourceRecall::candidate_pairs);

    py::class_<RecallMetrics> recall(
        m, "RecallMetrics", "Pair completeness, pair quality and reduction ratio.");
    recall.def_readonly("truth_pairs", &RecallMetrics::truth_pairs)
        .def_readonly("unresolved", &RecallMetrics::unresolved)
        .def_readonly("ambiguous", &RecallMetrics::ambiguous)
        .def_readonly("out_of_scope", &RecallMetrics::out_of_scope)
        .def_readonly("union_found", &RecallMetrics::union_found)
        .def_readonly("candidate_sum", &RecallMetrics::candidate_sum)
        .def_readonly("candidate_union", &RecallMetrics::candidate_union)
        .def_readonly("counted_union", &RecallMetrics::counted_union)
        .def_readonly("pair_space", &RecallMetrics::pair_space)
        .def_readonly("sources", &RecallMetrics::sources)
        .def_property_readonly("union_candidates", &RecallMetrics::UnionCandidates)
        .def_property_readonly(
            "pair_completeness",
            [](const RecallMetrics& r) {
                return r.truth_pairs == 0 ? 0.0
                                          : static_cast<double>(r.union_found) /
                                                static_cast<double>(r.truth_pairs);
            },
            "Known pairs the union reaches, as a fraction.");
    BindTextAndJson(&recall, &PrintRecallReport, &WriteRecallJson);

    py::class_<ColumnAgreement>(m, "ColumnAgreement")
        .def_readonly("comparison", &ColumnAgreement::comparison)
        .def_readonly("exact", &ColumnAgreement::exact)
        .def_readonly("fuzzy", &ColumnAgreement::fuzzy)
        .def_readonly("blocked", &ColumnAgreement::blocked);

    py::class_<WindowGain>(m, "WindowGain")
        .def_readonly("source", &WindowGain::source)
        .def_readonly("window", &WindowGain::window)
        .def_readonly("recovered", &WindowGain::recovered)
        .def_readonly("candidate_pairs", &WindowGain::candidate_pairs);

    py::class_<MissReport>(m, "MissReport", "Why blocking missed each known pair it did.")
        .def_readonly("truth_pairs", &MissReport::truth_pairs)
        .def_readonly("missed", &MissReport::missed)
        .def_readonly("reasons", &MissReport::reasons,
                      "reasons[source][reason]; see `reason_names`.")
        .def_property_readonly_static(
            "reason_names",
            [](const py::object&) {
                std::vector<std::string> names;
                for (int r = 0; r <= static_cast<int>(MissReason::kOutsideWindow); ++r) {
                    names.push_back(MissReasonName(static_cast<MissReason>(r)));
                }
                return names;
            })
        .def_readonly("fixable_by_cap", &MissReport::fixable_by_cap)
        .def_readonly("fixable_by_window", &MissReport::fixable_by_window)
        .def_readonly("unreachable", &MissReport::unreachable)
        .def_readonly("widest_window_needed", &MissReport::widest_window_needed)
        .def_readonly("window_gains", &MissReport::window_gains)
        .def_readonly("agreement", &MissReport::agreement)
        .def_readonly("agreed_on_nothing", &MissReport::agreed_on_nothing)
        .def_readonly("examples", &MissReport::examples)
        .def_readonly("example_limit", &MissReport::example_limit);

    py::class_<RecallOutcome>(m, "RecallResult",
                              "The recall report, and the miss diagnostic when asked.")
        .def_readonly("metrics", &RecallOutcome::metrics)
        .def_readonly("text", &RecallOutcome::text)
        .def_property_readonly(
            "misses",
            [](const RecallOutcome& r) -> py::object {
                if (!r.diagnosed) return py::none();
                return py::cast(r.misses);
            },
            "The MissReport, or None where `why` was not asked.")
        .def_readonly("miss_text", &RecallOutcome::miss_text)
        .def("json",
             [](const RecallOutcome& r) {
                 return CaptureText(
                     [&](std::ostream& out) { WriteRecallJson(r.metrics, out); });
             })
        .def("__repr__", [](const RecallOutcome& r) { return r.text + r.miss_text; });
}

void BindCompleteness(py::module_& m) {
    py::class_<CompletenessOptions>(m, "CompletenessOptions",
                                    "Every knob of `cpplink completeness`.")
        .def(py::init<>())
        .def_readwrite("threads", &CompletenessOptions::threads)
        .def_readwrite("sample", &CompletenessOptions::sample)
        .def_readwrite("seed", &CompletenessOptions::seed)
        .def_readwrite("pair_weighting", &CompletenessOptions::pair_weighting)
        .def_readwrite("min_observed", &CompletenessOptions::min_observed)
        .def_readwrite("skip_observed", &CompletenessOptions::skip_observed);

    py::class_<SourceCapture>(m, "SourceCapture")
        .def_readonly("name", &SourceCapture::name)
        .def_readonly("column", &SourceCapture::column)
        .def_property_readonly(
            "kind", [](const SourceCapture& s) { return SourceKindName(s.kind); })
        .def_property_readonly(
            "capture",
            [](const SourceCapture& s) {
                return s.capture == CaptureClass::kAnalytic ? "analytic" : "observed";
            })
        .def_readonly("bound_to_comparison", &SourceCapture::bound_to_comparison)
        .def_readonly("comparison", &SourceCapture::comparison)
        .def_readonly("exact_level", &SourceCapture::exact_level)
        .def_readonly("fires_given_exact", &SourceCapture::fires_given_exact)
        .def_readonly("window_analytic", &SourceCapture::window_analytic)
        .def_readonly("window_observed", &SourceCapture::window_observed)
        .def_readonly("has_window_check", &SourceCapture::has_window_check);

    py::class_<Dependence>(m, "Dependence")
        .def_readonly("left", &Dependence::left)
        .def_readonly("right", &Dependence::right)
        .def_readonly("g2", &Dependence::g2)
        .def_readonly("df", &Dependence::df);

    py::class_<CompletenessReport> completeness(
        m, "CompletenessReport", "Blocking recall estimated with no known pairs.");
    completeness.def_readonly("records", &CompletenessReport::records)
        .def_readonly("unblocked", &CompletenessReport::unblocked)
        .def_readonly("sources", &CompletenessReport::sources)
        .def_readonly("unlearned", &CompletenessReport::unlearned)
        .def_readonly("trusted", &CompletenessReport::trusted)
        .def_readonly("correction_available", &CompletenessReport::correction_available)
        .def_readonly("pc_bound", &CompletenessReport::pc_bound)
        .def_readonly("pc_analytic", &CompletenessReport::pc_analytic)
        .def_readonly("pc_product", &CompletenessReport::pc_product)
        .def_readonly("table_available", &CompletenessReport::table_available)
        .def_readonly("interaction_available", &CompletenessReport::interaction_available)
        .def_readonly("cells", &CompletenessReport::cells)
        .def_readonly("dark_cells", &CompletenessReport::dark_cells)
        .def_readonly("observed_cells", &CompletenessReport::observed_cells)
        .def_readonly("observed_matches", &CompletenessReport::observed_matches)
        .def_readonly("pc_independence", &CompletenessReport::pc_independence)
        .def_readonly("pc_interaction", &CompletenessReport::pc_interaction)
        .def_readonly("deviance_independence", &CompletenessReport::deviance_independence)
        .def_readonly("deviance_interaction", &CompletenessReport::deviance_interaction)
        .def_readonly("bic_independence", &CompletenessReport::bic_independence)
        .def_readonly("bic_interaction", &CompletenessReport::bic_interaction)
        .def_readonly("parameters_independence",
                      &CompletenessReport::parameters_independence)
        .def_readonly("parameters_interaction",
                      &CompletenessReport::parameters_interaction)
        .def_readonly("dependence", &CompletenessReport::dependence)
        .def_readonly("pc_estimate", &CompletenessReport::pc_estimate)
        .def_property_readonly(
            "pc_basis",
            [](const CompletenessReport& r) { return std::string(r.pc_basis); })
        .def_readonly("mass_total", &CompletenessReport::mass_total)
        .def_readonly("patterns", &CompletenessReport::patterns)
        .def_readonly("dark_patterns", &CompletenessReport::dark_patterns)
        .def_readonly("dark_mass", &CompletenessReport::dark_mass)
        .def_readonly("walked", &CompletenessReport::walked)
        .def_readonly("observed_pairs", &CompletenessReport::observed_pairs)
        .def_readonly("analytic_captured", &CompletenessReport::analytic_captured)
        .def_readonly("both_captured", &CompletenessReport::both_captured)
        .def_readonly("pooled_rate", &CompletenessReport::pooled_rate)
        .def_readonly("patterns_with_own_rate",
                      &CompletenessReport::patterns_with_own_rate)
        .def_readonly("seconds", &CompletenessReport::seconds)
        .def_readonly("threads", &CompletenessReport::threads)
        .def_readonly("measured", &CompletenessReport::measured)
        .def_readonly("pc_measured", &CompletenessReport::pc_measured)
        .def_readonly("truth_pairs", &CompletenessReport::truth_pairs);
    BindTextAndJson(&completeness, &PrintCompletenessReport, &WriteCompletenessJson);
}

void BindExplain(py::module_& m) {
    py::class_<WaterfallStep>(m, "WaterfallStep", "One comparison's row of the ledger.")
        .def_readonly("name", &WaterfallStep::name)
        .def_readonly("level", &WaterfallStep::level)
        .def_readonly("label", &WaterfallStep::label)
        .def_readonly("value_a", &WaterfallStep::value_a)
        .def_readonly("value_b", &WaterfallStep::value_b)
        .def_readonly("m", &WaterfallStep::m)
        .def_readonly("u", &WaterfallStep::u)
        .def_readonly("bits", &WaterfallStep::bits)
        .def_readonly("tf", &WaterfallStep::tf)
        .def_readonly("frequency", &WaterfallStep::frequency)
        .def_readonly("running", &WaterfallStep::running);

    py::class_<WaterfallInteraction>(m, "WaterfallInteraction")
        .def_readonly("name", &WaterfallInteraction::name)
        .def_readonly("bits", &WaterfallInteraction::bits)
        .def_readonly("running", &WaterfallInteraction::running);

    py::class_<PairWaterfall>(m, "PairWaterfall",
                              "The match weight of one pair as a ledger ending at the\n"
                              "weight the scorer produces.")
        .def_readonly("row_a", &PairWaterfall::row_a)
        .def_readonly("row_b", &PairWaterfall::row_b)
        .def_readonly("id_a", &PairWaterfall::id_a)
        .def_readonly("id_b", &PairWaterfall::id_b)
        .def_readonly("records", &PairWaterfall::records)
        .def_readonly("gamma", &PairWaterfall::gamma)
        .def_readonly("prior", &PairWaterfall::prior)
        .def_readonly("steps", &PairWaterfall::steps)
        .def_readonly("interactions", &PairWaterfall::interactions)
        .def_readonly("weight", &PairWaterfall::weight)
        .def_readonly("probability", &PairWaterfall::probability)
        .def_readonly("bracket_low", &PairWaterfall::bracket_low)
        .def_readonly("bracket_high", &PairWaterfall::bracket_high)
        .def_readonly("threshold", &PairWaterfall::threshold)
        .def_property_readonly("zone",
                               [](const PairWaterfall& w) { return ZoneName(w.zone); })
        .def_readonly("emitted", &PairWaterfall::emitted)
        .def_property_readonly("text",
                               [](const PairWaterfall& w) {
                                   return CaptureText([&](std::ostream& out) {
                                       PrintPairWaterfall(w, out);
                                   });
                               })
        .def("json", &PairWaterfallJson)
        .def("__repr__", [](const PairWaterfall& w) {
            return CaptureText([&](std::ostream& out) { PrintPairWaterfall(w, out); });
        });

    py::class_<Explanation>(m, "Explanation",
                            "One pair: the level each comparison lands on, and with a\n"
                            "model, the waterfall of bits behind its score.")
        .def_readonly("row_a", &Explanation::row_a)
        .def_readonly("row_b", &Explanation::row_b)
        .def_readonly("id_a", &Explanation::id_a)
        .def_readonly("id_b", &Explanation::id_b)
        .def_readonly("gamma", &Explanation::gamma)
        .def_property_readonly(
            "levels",
            [](const Explanation& e) {
                py::dict levels;
                for (const auto& [name, label] : e.levels) levels[py::str(name)] = label;
                return levels;
            },
            "Comparison name to the label of the level the pair landed on.")
        .def_property_readonly(
            "waterfall",
            [](const Explanation& e) -> py::object {
                if (!e.scored) return py::none();
                return py::cast(e.waterfall);
            },
            "The PairWaterfall, or None without a model.")
        .def_property_readonly("weight",
                               [](const Explanation& e) -> py::object {
                                   if (!e.scored) return py::none();
                                   return py::float_(e.waterfall.weight);
                               })
        .def("json",
             [](const Explanation& e) {
                 Check(e.scored, "explain: json needs a model");
                 return e.json;
             })
        .def_readonly("text", &Explanation::text)
        .def("__repr__", [](const Explanation& e) { return e.text; });
}

void BindExplainBlocking(py::module_& m) {
    py::class_<SourceSummary>(m, "SourceSummary", "One blocking source, priced.")
        .def_readonly("name", &SourceSummary::name)
        .def_readonly("kind", &SourceSummary::kind)
        .def_readonly("column", &SourceSummary::column)
        .def_readonly("em_safe", &SourceSummary::em_safe)
        .def_readonly("pairs", &SourceSummary::pairs)
        .def_readonly("largest_group", &SourceSummary::largest_group);

    py::class_<BlockingReport>(m, "BlockingReport",
                               "Every source priced without enumerating a pair.")
        .def_readonly("records", &BlockingReport::records)
        .def_readonly("mode", &BlockingReport::mode)
        .def_readonly("pair_space", &BlockingReport::pair_space)
        .def_readonly("sources", &BlockingReport::sources)
        .def_readonly("candidate_sum", &BlockingReport::candidate_sum)
        .def_readonly("counted_union", &BlockingReport::counted_union)
        .def_readonly("candidate_union", &BlockingReport::candidate_union)
        .def_readonly("text", &BlockingReport::text)
        .def("__repr__", [](const BlockingReport& r) { return r.text; });
}

}  // namespace

void BindDiagnostics(py::module_& m, SessionClass* session) {
    BindProfile(m);
    BindLevels(m);
    BindSimplify(m);
    BindRecall(m);
    BindCompleteness(m);
    BindExplain(m);
    BindExplainBlocking(m);

    session->def("profile", &ProfileStage, py::arg("options"), py::arg("truth") = "")
        .def("levels", &LevelsStage, py::arg("options"), py::arg("truth") = "")
        .def("simplify", &SimplifyStage, py::arg("model"), py::arg("options"))
        .def("recall", &RecallStage, py::arg("truth"), py::arg("why") = false,
             py::arg("count") = false, py::arg("show_misses") = 0)
        .def("completeness", &CompletenessStage, py::arg("model"), py::arg("options"),
             py::arg("truth") = "")
        .def("explain_rows", &ExplainStage, py::arg("row_a"), py::arg("row_b"),
             py::arg("model") = py::none(), py::arg("threshold") = 0.0,
             py::arg("tf_damping") = 1.0, py::arg("fuzzy_tf") = false,
             py::arg("ball") = BallOptions{}, py::arg("interactions") = true)
        .def("explain_blocking", &ExplainBlockingStage, py::arg("count") = false);
}

}  // namespace python
}  // namespace cpplink
