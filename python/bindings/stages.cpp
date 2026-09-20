// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/common.hpp"
#include "cpplink/cluster.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/fit.hpp"
#include "cpplink/merge_edges.hpp"
#include "cpplink/model.hpp"
#include "cpplink/neighbourhood.hpp"
#include "cpplink/pipeline.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/recall.hpp"
#include "cpplink/rescore.hpp"
#include "cpplink/score.hpp"
#include "cpplink/spill.hpp"

namespace cpplink {
namespace python {

namespace {

// A report beside the text the command line prints for it, for the reports
// whose printer needs more than the report: the scorer, the plan, the store.
struct EstimateOutcome {
    Model model;
    EstimateReport report;
    std::string text;
};

struct PredictOutcome : PredictReport {
    std::string text;  // the ball tables, the plan when verbose, then the report
    std::shared_ptr<PredictionTable> table;  // every prediction, in memory
};

struct RescoreOutcome : RescoreReport {
    std::string text;
    std::shared_ptr<PredictionTable> table;  // every prediction, in memory
};

struct ClusterOutcome {
    ClusterAssignment assignment;
    ClusterReport report;
    bool measured = false;
    ClusterQuality quality;
    std::string text;
    std::shared_ptr<ClusterTable> table;  // the clusters, in memory; null from a file
};

struct MergeOutcome : MergeReport {
    std::string text;
};

EstimateOutcome EstimateStage(Session& session, const EstimateOptions& options,
                              const std::string& out) {
    const ComparisonSet& comparisons = session.Comparisons();
    const BlockingPlan& plan = session.Plan(/*for_estimation=*/true);
    EstimateOutcome result;
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = Estimate(session.store(), comparisons, plan, options, &result.model,
                      &result.report, &error);
    }
    Check(ok, error);
    result.text = CaptureText(
        [&](std::ostream& stream) { PrintEstimateReport(result.report, stream); });
    if (!out.empty()) Check(WriteModelJson(result.model, out, &error), error);
    return result;
}

// One `EdgeFormat` keyword: "bin", "csv" or None for the default, which also
// says whether a single-file `out` is allowed to carry one.
bool FormatFrom(const py::object& format, EdgeFormat* out, const char* command) {
    if (format.is_none()) return false;
    const std::string text = py::cast<std::string>(format);
    if (text == "bin") {
        *out = EdgeFormat::kBinary;
    } else if (text == "csv") {
        *out = EdgeFormat::kCsv;
    } else {
        throw Error(std::string(command) + ": format wants bin or csv");
    }
    return true;
}

PredictOutcome PredictStage(const py::object& self, const Model& model,
                            const std::string& out, const py::object& threshold,
                            const py::object& probability, const py::object& format,
                            unsigned threads, uint64_t limit, double tf_damping,
                            bool bounds, bool ceiling, bool fuzzy_tf,
                            const BallOptions& ball, const std::string& spill,
                            double spill_sample, bool signatures, bool ladders,
                            bool interactions, bool verbose) {
    Session& session = py::cast<Session&>(self);
    PredictOptions options;
    options.threads = threads;
    // Every prediction is kept in memory whether or not a file is written: the
    // table is what the front end returns, and what `cluster` reads next.
    auto edges = std::make_shared<EdgeTable>();
    options.table = edges.get();
    options.max_edges = limit;
    options.spill_dir = spill;
    if (spill_sample < 0.0 || spill_sample > 1.0) {
        throw Error("predict: spill_sample wants a rate in (0, 1]");
    }
    options.spill_sample = spill_sample;
    const bool format_given = FormatFrom(format, &options.format, "predict");
    std::string error;
    if (!out.empty()) {
        Check(ResolveEdgeOutput("predict", out, format_given, &options.out_dir,
                                &options.merge_path, &error),
              error);
    } else if (format_given) {
        throw Error("predict: format names the shard format, and no out was given");
    }
    ScoreOptions score;
    score.threshold = ThresholdFrom(threshold, probability, "predict");
    score.tf_damping = tf_damping;
    score.use_bounds = bounds;
    score.use_ceiling = ceiling;
    score.use_interactions = interactions;

    const BlockingPlan& plan = session.Plan(/*for_estimation=*/false);
    std::unique_ptr<ComparisonSet> own;
    const ComparisonSet* comparisons = &session.Comparisons();
    if (!signatures || !ladders) {
        own = session.BindComparisons(signatures, ladders);
        comparisons = own.get();
    }

    PredictOutcome result;
    std::ostringstream text;
    BallTables balls;
    Scorer scorer;
    {
        py::gil_scoped_release release;
        if (fuzzy_tf) {
            BallOptions with_threads = ball;
            with_threads.threads = threads;
            BuildBallTables(*comparisons, session.store(), with_threads, &balls, text);
        }
    }
    Check(scorer.Bind(model, *comparisons, session.store(), score, &error,
                      fuzzy_tf ? &balls : nullptr),
          "predict: " + error);
    if (verbose) {
        std::ostringstream plan_text;
        PrintPredictPlan(session.store(), plan, *comparisons, scorer, options,
                         session.stats().seconds, plan_text);
        text << plan_text.str();
        // Progress goes to the process's stderr as lines, which a terminal shows
        // as the run goes and a notebook does not capture.
        py::print(plan_text.str(), py::arg("end") = "");
        options.progress = &std::cerr;
        options.progress_columns = 0;
    }
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = Predict(session.store(), *comparisons, plan, scorer, options,
                     static_cast<PredictReport*>(&result), &error);
    }
    Check(ok, error);
    PrintPredictReport(result, scorer, text);
    result.text = text.str();
    session.set_last_edges(edges);
    result.table = std::make_shared<PredictionTable>(self, edges);
    return result;
}

RescoreOutcome RescoreStage(const py::object& self, const Model& model,
                            const std::string& spill, const std::string& out,
                            const py::object& threshold, const py::object& probability,
                            const py::object& format, unsigned threads, uint64_t limit,
                            double tf_damping, bool bounds) {
    Session& session = py::cast<Session&>(self);
    RescoreOptions options;
    options.spill_dir = spill;
    auto edges = std::make_shared<EdgeTable>();
    options.table = edges.get();
    options.threads = threads;
    options.max_edges = limit;
    const bool format_given = FormatFrom(format, &options.format, "rescore");
    std::string error;
    if (!out.empty()) {
        Check(ResolveEdgeOutput("rescore", out, format_given, &options.out_dir,
                                &options.merge_path, &error),
              error);
    } else if (format_given) {
        throw Error("rescore: format names the shard format, and no out was given");
    }
    ScoreOptions score;
    score.threshold = ThresholdFrom(threshold, probability, "rescore");
    score.tf_damping = tf_damping;
    score.use_bounds = bounds;

    // The plan is what the command line builds too, though the spill replaces
    // the enumeration: it is how the schema's blocking section is checked.
    session.Plan(/*for_estimation=*/false);
    const ComparisonSet& comparisons = session.Comparisons();
    Scorer scorer;
    Check(scorer.Bind(model, comparisons, session.store(), score, &error),
          "rescore: " + error);
    RescoreOutcome result;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = Rescore(session.store(), comparisons, scorer, options,
                     static_cast<RescoreReport*>(&result), &error);
    }
    Check(ok, error);
    result.text = CaptureText(
        [&](std::ostream& stream) { PrintRescoreReport(result, scorer, stream); });
    session.set_last_edges(edges);
    result.table = std::make_shared<PredictionTable>(self, edges);
    return result;
}

// Clustering over a store: the Linker's, or one loaded for its ids alone.
ClusterOutcome ClusterOver(const RecordStore& store, ClusterOptions options,
                           const py::object& threshold, const py::object& probability,
                           const std::string& out, uint64_t min_size,
                           const std::string& truth) {
    options.out_path = out;
    options.min_size = min_size;
    if (!threshold.is_none() || !probability.is_none()) {
        options.threshold = ThresholdFrom(threshold, probability, "cluster");
    }
    ClusterOutcome result;
    std::string error;
    bool ok = false;
    TruthPairs pairs;
    {
        py::gil_scoped_release release;
        ok = Cluster(store, options, &result.assignment, &result.report, &error) &&
             WriteClusters(result.assignment, store, options, &result.report.written,
                           &error);
        if (ok && !truth.empty()) {
            ok = LoadTruthPairs(truth, store, &pairs, &error);
            if (ok) {
                result.measured = true;
                result.quality = MeasureClusters(result.assignment, pairs);
            }
        }
    }
    Check(ok, error);
    result.text = CaptureText([&](std::ostream& stream) {
        PrintClusterReport(result.report, stream);
        if (result.measured) PrintClusterQuality(result.quality, stream);
    });
    return result;
}

// `predictions` is None for the last `predict`'s own rows, a path to a shard
// directory or a merged file, or a table -- a data frame, a pyarrow table --
// that speaks the Arrow C stream protocol and is read as a merged file is.
ClusterOutcome ClusterStage(const py::object& self, const py::object& predictions,
                            const py::object& threshold, const py::object& probability,
                            const std::string& out, uint64_t min_size,
                            const std::string& truth) {
    Session& session = py::cast<Session&>(self);
    ClusterOptions options;
    py::object capsule;  // keeps a stream's struct alive while it is read
    if (predictions.is_none()) {
        if (!session.last_edges()) {
            throw Error(
                "cluster: no predictions in memory; run predict first, or give "
                "a table or a path");
        }
        options.edges = session.last_edges().get();
    } else if (py::isinstance<py::str>(predictions)) {
        options.edge_path = py::cast<std::string>(predictions);
    } else if (py::hasattr(predictions, "__arrow_c_stream__")) {
        capsule = predictions.attr("__arrow_c_stream__")();
        void* pointer = PyCapsule_GetPointer(capsule.ptr(), "arrow_array_stream");
        if (pointer == nullptr) {
            PyErr_Clear();
            throw Error(
                "cluster: __arrow_c_stream__ did not return an arrow_array_stream");
        }
        options.stream = static_cast<ArrowArrayStream*>(pointer);
    } else {
        throw Error(
            "cluster: predictions wants None, a path, or a table with "
            "__arrow_c_stream__");
    }
    ClusterOutcome result = ClusterOver(session.store(), options, threshold, probability,
                                        out, min_size, truth);
    result.table = std::make_shared<ClusterTable>(self, result.assignment, min_size);
    return result;
}

// `cpplink cluster` as the command runs it: only the id column is loaded.
ClusterOutcome ClusterFile(const std::string& schema_path,
                           const std::vector<std::string>& files,
                           const std::string& predictions, const py::object& threshold,
                           const py::object& probability, const std::string& out,
                           uint64_t min_size, const std::string& truth) {
    std::unique_ptr<RecordStore> store;
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = LoadIdsOnly(schema_path, files, &store, &error);
    }
    Check(ok, error);
    ClusterOptions options;
    options.edge_path = predictions;
    return ClusterOver(*store, options, threshold, probability, out, min_size, truth);
}

MergeOutcome MergePredictions(const std::string& shards, const std::string& out,
                              const py::object& format, const std::string& source,
                              const py::object& threshold, const py::object& probability,
                              size_t batch_rows, const std::string& schema_path,
                              const std::vector<std::string>& files) {
    MergeOptions options;
    options.edge_dir = shards;
    options.out_path = out;
    options.batch_rows = batch_rows;
    if (format.is_none()) {
        const std::string suffix = std::filesystem::path(out).extension();
        if (suffix == ".parquet" || suffix == ".pq")
            options.format = MergeFormat::kParquet;
    } else {
        const std::string text = py::cast<std::string>(format);
        if (text == "csv") {
            options.format = MergeFormat::kCsv;
        } else if (text == "parquet") {
            options.format = MergeFormat::kParquet;
        } else {
            throw Error("merge_predictions: format wants csv or parquet");
        }
    }
    if (source == "bin") {
        options.source = MergeSource::kBinary;
    } else if (source == "csv") {
        options.source = MergeSource::kCsv;
    } else if (source == "auto") {
        options.source = MergeSource::kAuto;
    } else {
        throw Error("merge_predictions: source wants auto, bin or csv");
    }
    if (!threshold.is_none() || !probability.is_none()) {
        options.threshold = ThresholdFrom(threshold, probability, "merge_predictions");
    }
    if (schema_path.empty() != files.empty()) {
        throw Error("merge_predictions: naming the ids takes both schema and files");
    }
    std::unique_ptr<RecordStore> store;
    std::string error;
    bool ok = true;
    MergeOutcome result;
    {
        py::gil_scoped_release release;
        if (!schema_path.empty()) ok = LoadIdsOnly(schema_path, files, &store, &error);
        if (ok) {
            ok = MergeEdges(store.get(), options, static_cast<MergeReport*>(&result),
                            &error);
        }
    }
    Check(ok, error);
    result.text =
        CaptureText([&](std::ostream& stream) { PrintMergeReport(result, stream); });
    return result;
}

// A numpy view over one of the assignment's vectors, read-only and holding the
// assignment alive for as long as the view is.
py::array View(const py::object& owner, const std::vector<uint32_t>& values) {
    py::array_t<uint32_t> array({static_cast<py::ssize_t>(values.size())},
                                {static_cast<py::ssize_t>(sizeof(uint32_t))},
                                values.data(), owner);
    array.attr("setflags")(py::arg("write") = false);
    return array;
}

}  // namespace

void BindStages(py::module_& m, SessionClass* session) {
    py::class_<BallOptions>(m, "BallOptions",
                            "The dictionary self-join behind --fuzzy-tf and --fuzzy-u.")
        .def(py::init<>())
        .def_readwrite("budget", &BallOptions::budget)
        .def_readwrite("threads", &BallOptions::threads);

    py::class_<InteractionOptions>(
        m, "InteractionOptions",
        "The two-way corrections `estimate --interactions` fits.")
        .def(py::init<>())
        .def_readwrite("enabled", &InteractionOptions::enabled)
        .def_readwrite("max_terms", &InteractionOptions::max_terms)
        .def_readwrite("min_bits", &InteractionOptions::min_bits)
        .def_readwrite("clamp_bits", &InteractionOptions::clamp_bits)
        .def_readwrite("cell_support", &InteractionOptions::cell_support)
        .def_readwrite("prior_count", &InteractionOptions::prior_count)
        .def_readwrite("max_contamination", &InteractionOptions::max_contamination)
        .def_readwrite("min_pairs_per_parameter",
                       &InteractionOptions::min_pairs_per_parameter)
        .def_readwrite("min_sessions", &InteractionOptions::min_sessions);

    py::class_<EstimateOptions>(m, "EstimateOptions", "Every knob of `cpplink estimate`.")
        .def(py::init<>())
        .def_readwrite("u_sample", &EstimateOptions::u_sample)
        .def_readwrite("session_pairs", &EstimateOptions::session_pairs)
        .def_readwrite("threads", &EstimateOptions::threads)
        .def_readwrite("max_iterations", &EstimateOptions::max_iterations)
        .def_readwrite("tolerance", &EstimateOptions::tolerance)
        .def_readwrite("lambda_init", &EstimateOptions::lambda_init)
        .def_readwrite("lambda_", &EstimateOptions::lambda)
        .def_readwrite("seed", &EstimateOptions::seed)
        .def_readwrite("fuzzy_u", &EstimateOptions::fuzzy_u)
        .def_readwrite("ball", &EstimateOptions::ball)
        .def_readwrite("exclude_tied", &EstimateOptions::exclude_tied)
        .def_readwrite("tied_bits", &EstimateOptions::tied_bits)
        .def_readwrite("tie_sample_rows", &EstimateOptions::tie_sample_rows)
        .def_readwrite("interactions", &EstimateOptions::interactions);

    py::class_<PairResidual>(m, "PairResidual",
                             "One pair of free comparisons against the fitted mixture.")
        .def_readonly("left", &PairResidual::left)
        .def_readonly("right", &PairResidual::right)
        .def_readonly("g2", &PairResidual::g2)
        .def_readonly("degrees", &PairResidual::degrees)
        .def_readonly("p_value", &PairResidual::p_value)
        .def_readonly("bits", &PairResidual::bits)
        .def_readonly("corrected", &PairResidual::corrected)
        .def_readonly("corrected_g2", &PairResidual::corrected_g2)
        .def_readonly("corrected_bits", &PairResidual::corrected_bits);

    py::class_<SessionFit>(m, "SessionFit",
                           "A session's fitted mixture against its own histogram.")
        .def_readonly("measured", &SessionFit::measured)
        .def_readonly("refusal", &SessionFit::refusal)
        .def_readonly("pairs", &SessionFit::pairs)
        .def_readonly("patterns", &SessionFit::patterns)
        .def_readonly("deviance", &SessionFit::deviance)
        .def_readonly("degrees", &SessionFit::degrees)
        .def_readonly("p_value", &SessionFit::p_value)
        .def_readonly("bits", &SessionFit::bits)
        .def_readonly("floor_bits", &SessionFit::floor_bits)
        .def_readonly("corrected", &SessionFit::corrected)
        .def_readonly("corrected_refusal", &SessionFit::corrected_refusal)
        .def_readonly("corrected_deviance", &SessionFit::corrected_deviance)
        .def_readonly("corrected_bits", &SessionFit::corrected_bits)
        .def_readonly("residuals", &SessionFit::residuals);

    py::class_<SessionReport>(m, "SessionReport", "One EM session of `estimate`.")
        .def_readonly("column", &SessionReport::column)
        .def_readonly("sources", &SessionReport::sources)
        .def_readonly("excluded", &SessionReport::excluded)
        .def_readonly("tied", &SessionReport::tied)
        .def_readonly("candidates", &SessionReport::candidates)
        .def_readonly("enumerated", &SessionReport::enumerated)
        .def_readonly("folded", &SessionReport::folded)
        .def_readonly("rate", &SessionReport::rate)
        .def_readonly("distinct", &SessionReport::distinct)
        .def_readonly("seconds", &SessionReport::seconds)
        .def_readonly("iterations", &SessionReport::iterations)
        .def_readonly("change", &SessionReport::change)
        .def_readonly("converged", &SessionReport::converged)
        .def_readonly("lambda_", &SessionReport::lambda)
        .def_readonly("implied_matches", &SessionReport::implied_matches)
        .def_readonly("merged", &SessionReport::merged)
        .def_readonly("fit", &SessionReport::fit)
        .def_readonly("notes", &SessionReport::notes)
        .def_readonly("warnings", &SessionReport::warnings);

    py::class_<BallReport>(m, "BallReport")
        .def_readonly("comparison", &BallReport::comparison)
        .def_readonly("built", &BallReport::built)
        .def_readonly("reason", &BallReport::reason)
        .def_readonly("values", &BallReport::values)
        .def_readonly("value_pairs", &BallReport::value_pairs)
        .def_readonly("levels", &BallReport::levels)
        .def_readonly("seconds", &BallReport::seconds);

    py::class_<InteractionCandidate>(m, "InteractionCandidate")
        .def_readonly("left", &InteractionCandidate::left)
        .def_readonly("right", &InteractionCandidate::right)
        .def_readonly("match_bits", &InteractionCandidate::match_bits)
        .def_readonly("effect", &InteractionCandidate::effect)
        .def_readonly("widest", &InteractionCandidate::widest)
        .def_readonly("g2", &InteractionCandidate::g2)
        .def_readonly("degrees", &InteractionCandidate::degrees)
        .def_readonly("p_value", &InteractionCandidate::p_value)
        .def_readonly("sessions", &InteractionCandidate::sessions)
        .def_readonly("match_pairs", &InteractionCandidate::match_pairs)
        .def_readonly("per_parameter", &InteractionCandidate::per_parameter)
        .def_readonly("weakest", &InteractionCandidate::weakest)
        .def_readonly("contamination", &InteractionCandidate::contamination)
        .def_readonly("admitted", &InteractionCandidate::admitted)
        .def_readonly("reason", &InteractionCandidate::reason);

    py::class_<InteractionReport>(m, "InteractionReport")
        .def_readonly("ran", &InteractionReport::ran)
        .def_readonly("lambda_", &InteractionReport::lambda)
        .def_readonly("lambda_is_a_bound", &InteractionReport::lambda_is_a_bound)
        .def_readonly("refusal", &InteractionReport::refusal)
        .def_readonly("candidates", &InteractionReport::candidates)
        .def_readonly("admitted", &InteractionReport::admitted)
        .def_readonly("clamped", &InteractionReport::clamped)
        .def_readonly("seconds", &InteractionReport::seconds)
        .def_property_readonly("text", [](const InteractionReport& r) {
            return CaptureText(
                [&](std::ostream& out) { PrintInteractionReport(r, out); });
        });

    py::class_<EstimateReport>(m, "EstimateReport",
                               "What `estimate` reports about the fit.")
        .def_readonly("u_pairs", &EstimateReport::u_pairs)
        .def_readonly("u_inputs", &EstimateReport::u_inputs)
        .def_readonly("u_seconds", &EstimateReport::u_seconds)
        .def_readonly("u_exact_levels", &EstimateReport::u_exact_levels)
        .def_readonly("u_ball_levels", &EstimateReport::u_ball_levels)
        .def_readonly("ball_seconds", &EstimateReport::ball_seconds)
        .def_readonly("tie_seconds", &EstimateReport::tie_seconds)
        .def_readonly("balls", &EstimateReport::balls)
        .def_readonly("sessions", &EstimateReport::sessions)
        .def_readonly("interactions", &EstimateReport::interactions)
        .def_readonly("warnings", &EstimateReport::warnings)
        .def_property_readonly("text",
                               [](const EstimateReport& r) {
                                   return CaptureText([&](std::ostream& out) {
                                       PrintEstimateReport(r, out);
                                   });
                               })
        // The command line's --report <file>: every residual pair of every
        // session, where `text` names each session's worst pair only.
        .def_property_readonly("full_text",
                               [](const EstimateReport& r) {
                                   return CaptureText([&](std::ostream& out) {
                                       PrintEstimateReport(r, out, ReportDetail::kFull);
                                   });
                               })
        .def("__repr__", [](const EstimateReport& r) {
            return CaptureText([&](std::ostream& out) { PrintEstimateReport(r, out); });
        });

    py::class_<EstimateOutcome>(m, "_EstimateOutcome")
        .def_readonly("model", &EstimateOutcome::model)
        .def_readonly("report", &EstimateOutcome::report)
        .def_readonly("text", &EstimateOutcome::text);

    py::class_<PredictOutcome>(m, "PredictReport", "What one scoring run did.")
        .def_readonly("enumerated", &PredictReport::enumerated)
        .def_readonly("skipped", &PredictReport::skipped)
        .def_readonly("dropped", &PredictReport::dropped)
        .def_readonly("checked", &PredictReport::checked)
        .def_readonly("certain", &PredictReport::certain)
        .def_readonly("predictions", &PredictReport::edges)
        .def_readonly("tf_lookups", &PredictReport::tf_lookups)
        .def_readonly("patterns_drop", &PredictReport::patterns_drop)
        .def_readonly("patterns_check", &PredictReport::patterns_check)
        .def_readonly("patterns_certain", &PredictReport::patterns_certain)
        .def_readonly("pattern_space", &PredictReport::pattern_space)
        .def_readonly("threads", &PredictReport::threads)
        .def_readonly("seconds", &PredictReport::seconds)
        .def_readonly("truncated", &PredictReport::truncated)
        .def_readonly("spilled", &PredictReport::spilled)
        .def_property_readonly(
            "mode", [](const PredictOutcome& r) { return PairModeName(r.mode); })
        .def_readonly("datasets", &PredictReport::datasets)
        .def_readonly("shards", &PredictReport::shards)
        .def_readonly("merged_path", &PredictReport::merged_path)
        .def_readonly("merge_seconds", &PredictReport::merge_seconds)
        .def_readonly("text", &PredictOutcome::text)
        .def_readonly("table", &PredictOutcome::table,
                      "Every prediction of the run, as an `ArrowTable`.")
        .def("__repr__", [](const PredictOutcome& r) { return r.text; });

    py::class_<SpillManifest>(m, "SpillManifest")
        .def_readonly("threshold", &SpillManifest::threshold)
        .def_readonly("sample_rate", &SpillManifest::sample_rate)
        .def_readonly("records", &SpillManifest::records)
        .def_readonly("candidates", &SpillManifest::candidates)
        .def_readonly("spilled", &SpillManifest::spilled)
        .def_readonly("above_threshold", &SpillManifest::above_threshold)
        .def_readonly("gamma_width", &SpillManifest::gamma_width)
        .def_readonly("layout", &SpillManifest::layout);

    py::class_<RescoreOutcome>(m, "RescoreReport", "What replaying a spill did.")
        .def_readonly("pairs", &RescoreReport::pairs)
        .def_readonly("predictions", &RescoreReport::edges)
        .def_readonly("dropped", &RescoreReport::dropped)
        .def_readonly("tf_lookups", &RescoreReport::tf_lookups)
        .def_readonly("threads", &RescoreReport::threads)
        .def_readonly("seconds", &RescoreReport::seconds)
        .def_readonly("truncated", &RescoreReport::truncated)
        .def_readonly("below_spill_threshold", &RescoreReport::below_spill_threshold)
        .def_readonly("manifest", &RescoreReport::manifest)
        .def_readonly("shards", &RescoreReport::shards)
        .def_readonly("merged_path", &RescoreReport::merged_path)
        .def_readonly("merge_seconds", &RescoreReport::merge_seconds)
        .def_readonly("text", &RescoreOutcome::text)
        .def_readonly("table", &RescoreOutcome::table,
                      "Every prediction of the replay, as an `ArrowTable`.")
        .def("__repr__", [](const RescoreOutcome& r) { return r.text; });

    py::class_<ClusterAssignment>(
        m, "ClusterAssignment",
        "The partition: `root[row]` is the representative row,\n"
        "`size[root]` the cluster's size.")
        .def_property_readonly("root",
                               [](const py::object& self) {
                                   return View(
                                       self, self.cast<const ClusterAssignment&>().root);
                               })
        .def_property_readonly("size",
                               [](const py::object& self) {
                                   return View(
                                       self, self.cast<const ClusterAssignment&>().size);
                               })
        .def_readonly("clusters", &ClusterAssignment::clusters)
        .def_readonly("singletons", &ClusterAssignment::singletons)
        .def_readonly("clustered", &ClusterAssignment::clustered)
        .def_readonly("largest", &ClusterAssignment::largest)
        .def_readonly("implied_pairs", &ClusterAssignment::implied_pairs)
        .def("same_cluster", &ClusterAssignment::SameCluster, py::arg("a"), py::arg("b"))
        .def("__len__", [](const ClusterAssignment& a) { return a.root.size(); });

    py::class_<SizeBucket>(m, "SizeBucket")
        .def_readonly("label", &SizeBucket::label)
        .def_readonly("clusters", &SizeBucket::clusters)
        .def_readonly("records", &SizeBucket::records);

    py::class_<ClusterReport>(m, "ClusterReport", "What the union-find pass did.")
        .def_readonly("records", &ClusterReport::records)
        .def_readonly("predictions_read", &ClusterReport::edges_read)
        .def_readonly("predictions_used", &ClusterReport::edges_used)
        .def_readonly("merges", &ClusterReport::merges)
        .def_readonly("min_weight", &ClusterReport::min_weight)
        .def_readonly("max_weight", &ClusterReport::max_weight)
        .def_readonly("clusters", &ClusterReport::clusters)
        .def_readonly("singletons", &ClusterReport::singletons)
        .def_readonly("clustered", &ClusterReport::clustered)
        .def_readonly("largest", &ClusterReport::largest)
        .def_readonly("implied_pairs", &ClusterReport::implied_pairs)
        .def_readonly("written", &ClusterReport::written)
        .def_readonly("unresolved", &ClusterReport::unresolved)
        .def_readonly("ambiguous", &ClusterReport::ambiguous)
        .def_readonly("buckets", &ClusterReport::buckets)
        .def_readonly("shards", &ClusterReport::shards)
        .def_readonly("predictions_file", &ClusterReport::edge_file)
        .def_readonly("index_seconds", &ClusterReport::index_seconds)
        .def_readonly("seconds", &ClusterReport::seconds)
        .def_property_readonly("text",
                               [](const ClusterReport& r) {
                                   return CaptureText([&](std::ostream& out) {
                                       PrintClusterReport(r, out);
                                   });
                               })
        .def("__repr__", [](const ClusterReport& r) {
            return CaptureText([&](std::ostream& out) { PrintClusterReport(r, out); });
        });

    py::class_<ClusterQuality>(
        m, "ClusterQuality",
        "The partition against known pairs, both closed transitively.")
        .def_readonly("listed_pairs", &ClusterQuality::listed_pairs)
        .def_readonly("truth_pairs", &ClusterQuality::truth_pairs)
        .def_readonly("recovered", &ClusterQuality::recovered)
        .def_readonly("implied_pairs", &ClusterQuality::implied_pairs)
        .def_readonly("truth_clusters", &ClusterQuality::truth_clusters)
        .def_readonly("largest_truth_cluster", &ClusterQuality::largest_truth_cluster)
        .def_readonly("precision", &ClusterQuality::precision)
        .def_readonly("recall", &ClusterQuality::recall)
        .def_readonly("f1", &ClusterQuality::f1)
        .def_property_readonly("text",
                               [](const ClusterQuality& q) {
                                   return CaptureText([&](std::ostream& out) {
                                       PrintClusterQuality(q, out);
                                   });
                               })
        .def("__repr__", [](const ClusterQuality& q) {
            return CaptureText([&](std::ostream& out) { PrintClusterQuality(q, out); });
        });

    py::class_<ClusterOutcome>(m, "ClusterResult",
                               "The partition, its report and quality.")
        .def_readonly("assignment", &ClusterOutcome::assignment)
        .def_readonly("report", &ClusterOutcome::report)
        .def_property_readonly(
            "quality",
            [](const ClusterOutcome& c) -> py::object {
                if (!c.measured) return py::none();
                return py::cast(c.quality);
            },
            "Against the truth file, or None where none was given.")
        .def_readonly("text", &ClusterOutcome::text)
        .def_readonly("table", &ClusterOutcome::table,
                      "The clusters of `min_size` or more, as an `ArrowTable`; None\n"
                      "from `cluster_file`, which has no session to hold.")
        .def("__repr__", [](const ClusterOutcome& c) { return c.text; });

    py::class_<MergeOutcome>(m, "MergeReport", "What `merge_predictions` wrote.")
        .def_readonly("shards", &MergeReport::shards)
        .def_readonly("ignored", &MergeReport::ignored)
        .def_property_readonly("source",
                               [](const MergeOutcome& r) {
                                   return r.source == MergeSource::kCsv ? "csv" : "bin";
                               })
        .def_property_readonly("format",
                               [](const MergeOutcome& r) {
                                   return r.format == MergeFormat::kParquet ? "parquet"
                                                                            : "csv";
                               })
        .def_readonly("out_path", &MergeReport::out_path)
        .def_readonly("row_indices", &MergeReport::row_indices)
        .def_readonly("datasets", &MergeReport::datasets)
        .def_readonly("read", &MergeReport::read)
        .def_readonly("written", &MergeReport::written)
        .def_readonly("seconds", &MergeReport::seconds)
        .def_readonly("text", &MergeOutcome::text)
        .def("__repr__", [](const MergeOutcome& r) { return r.text; });

    session->def("estimate", &EstimateStage, py::arg("options"), py::arg("out") = "")
        .def("predict", &PredictStage, py::arg("model"), py::arg("out") = "",
             py::arg("threshold") = py::none(), py::arg("probability") = py::none(),
             py::arg("format") = py::none(), py::arg("threads") = 0, py::arg("limit") = 0,
             py::arg("tf_damping") = 1.0, py::arg("bounds") = true,
             py::arg("ceiling") = true, py::arg("fuzzy_tf") = false,
             py::arg("ball") = BallOptions{}, py::arg("spill") = "",
             py::arg("spill_sample") = 0.0, py::arg("signatures") = true,
             py::arg("ladders") = true, py::arg("interactions") = true,
             py::arg("verbose") = false)
        .def("rescore", &RescoreStage, py::arg("model"), py::arg("spill"),
             py::arg("out") = "", py::arg("threshold") = py::none(),
             py::arg("probability") = py::none(), py::arg("format") = py::none(),
             py::arg("threads") = 0, py::arg("limit") = 0, py::arg("tf_damping") = 1.0,
             py::arg("bounds") = true)
        .def("cluster", &ClusterStage, py::arg("predictions") = py::none(),
             py::arg("threshold") = py::none(), py::arg("probability") = py::none(),
             py::arg("out") = "", py::arg("min_size") = 2, py::arg("truth") = "");

    m.def("cluster_file", &ClusterFile, py::arg("schema"), py::arg("files"),
          py::arg("predictions"), py::arg("threshold") = py::none(),
          py::arg("probability") = py::none(), py::arg("out") = "",
          py::arg("min_size") = 2, py::arg("truth") = "",
          "`cpplink cluster` as the command runs it: `schema` is a path, and only the\n"
          "id column is loaded. A Linker's `cluster` reuses the store it already holds.");

    m.def("merge_predictions", &MergePredictions, py::arg("shards"), py::arg("out"),
          py::arg("format") = py::none(), py::arg("source") = "auto",
          py::arg("threshold") = py::none(), py::arg("probability") = py::none(),
          py::arg("batch_rows") = 65536, py::arg("schema") = "",
          py::arg("files") = std::vector<std::string>{},
          "Fold a shard directory into one csv or parquet file, as\n"
          "`cpplink merge-predictions` does. `schema` (a path) and `files` together\n"
          "turn the rows binary shards name back into ids.");
}

}  // namespace python
}  // namespace cpplink
