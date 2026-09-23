// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/search.hpp"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/common.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/model.hpp"
#include "cpplink/neighbourhood.hpp"
#include "cpplink/pipeline.hpp"
#include "cpplink/score.hpp"

namespace cpplink {
namespace python {

namespace {

// The query as Python gives it: a mapping of column to value, with a list or a
// tuple spread over the repeats a list column takes one element per.
QueryRecord RecordFrom(const py::dict& fields) {
    QueryRecord query;
    for (const auto& item : fields) {
        const std::string column = py::cast<std::string>(py::str(item.first));
        const py::handle value = item.second;
        if (value.is_none()) continue;
        if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
            for (const py::handle element : value) {
                query.Set(column, py::cast<std::string>(py::str(element)));
            }
            continue;
        }
        query.Set(column, py::cast<std::string>(py::str(value)));
    }
    return query;
}

// One search, start to finish, with the query row installed for exactly as long
// as it takes to answer and explain it.
//
// The ledgers are built here rather than handed back as rows to explain later,
// because the query stops being a row when the searcher goes and every report
// that explains a pair takes two rows. A result that outlived its query row
// would be a result nothing could explain.
SearchOutcome SearchStage(Session& session, const py::dict& fields,
                          const py::object& model_object, size_t k,
                          const py::object& threshold, const py::object& probability,
                          unsigned threads, const py::object& expected_matches,
                          const py::object& prior_weight, bool explain,
                          const std::string& clusters, double tf_damping, bool fuzzy_tf,
                          const BallOptions& ball, bool interactions) {
    const Model& model = model_object.cast<const Model&>();
    const QueryRecord query = RecordFrom(fields);
    Check(!query.fields.empty(), "search: give at least one column and value");

    ComparisonSet* comparisons = session.mutable_comparisons();
    RecordStore* store = session.mutable_store();

    ScoreOptions score;
    score.tf_damping = tf_damping;
    score.use_interactions = interactions;
    BallTables balls;
    std::ostringstream ball_text;
    if (fuzzy_tf) {
        py::gil_scoped_release release;
        BuildBallTables(*comparisons, *store, ball, &balls, ball_text);
    }
    Scorer scorer;
    std::string error;
    Check(scorer.Bind(model, *comparisons, *store, score, &error,
                      fuzzy_tf ? &balls : nullptr),
          "search: " + error);

    SearchOptions options;
    options.k = k;
    options.threads = threads;
    if (!threshold.is_none() || !probability.is_none()) {
        options.threshold = ThresholdFrom(threshold, probability, "search");
    }
    if (!expected_matches.is_none() && !prior_weight.is_none()) {
        throw Error("search: give expected_matches or prior_weight, not both");
    }
    if (!expected_matches.is_none()) {
        options.override_prior = true;
        options.prior_weight = PriorWeightForExpected(py::cast<double>(expected_matches),
                                                      store->NumRecords());
    } else if (!prior_weight.is_none()) {
        options.override_prior = true;
        options.prior_weight = py::cast<double>(prior_weight);
    }

    SearchOutcome outcome;
    Searcher searcher;
    Check(searcher.Bind(store, comparisons, &scorer, &error), "search: " + error);
    {
        py::gil_scoped_release release;
        if (!searcher.Search(query, options, &outcome.report, &error)) {
            // The searcher takes the query row back on destruction, so nothing
            // is left behind whichever way this goes.
            error = "search: " + error;
        } else {
            error.clear();
            if (explain) {
                // The store's row is the first side throughout: an exact level's
                // term-frequency move reads the frequency of `a`'s value, and the
                // query's own value was never counted into the table.
                for (const SearchHit& hit : outcome.report.hits) {
                    outcome.waterfalls.push_back(
                        BuildPairWaterfall(*store, *comparisons, scorer, hit.row,
                                           searcher.QueryRow(), &model));
                }
            }
        }
    }
    Check(error.empty(), error);
    if (!clusters.empty()) {
        Check(GroupHitsByCluster(*store, clusters, &outcome.report, &error),
              "search: " + error);
    }
    outcome.text =
        CaptureText([&](std::ostream& out) { PrintSearchReport(outcome.report, out); });
    return outcome;
}

}  // namespace

void BindSearch(py::module_& m, SessionClass* session) {
    py::class_<SearchHit>(m, "SearchHit", "One record a query scored against.")
        .def_readonly("row", &SearchHit::row)
        .def_readonly("id", &SearchHit::id)
        .def_readonly("dataset", &SearchHit::dataset)
        .def_readonly("gamma", &SearchHit::gamma)
        .def_readonly("match_weight", &SearchHit::weight)
        .def_readonly("match_probability", &SearchHit::probability)
        .def_readonly("cluster", &SearchHit::cluster)
        .def_readonly("cluster_best", &SearchHit::cluster_best)
        .def("__repr__", [](const SearchHit& hit) {
            return "SearchHit(id=" + hit.id +
                   ", match_weight=" + std::to_string(hit.weight) + ")";
        });

    py::class_<SearchReport>(m, "SearchReport",
                             "What one search did: the hits, and what each phase cost.")
        .def_readonly("hits", &SearchReport::hits)
        .def_readonly("records", &SearchReport::records)
        .def_readonly("tabulated", &SearchReport::tabled)
        .def_readonly("evaluated", &SearchReport::evaluated)
        .def_readonly("constant", &SearchReport::constant)
        .def_readonly("values_walked", &SearchReport::values_walked)
        .def_readonly("values_adopted", &SearchReport::values_adopted)
        .def_readonly("rescored", &SearchReport::rescored)
        .def_readonly("prior", &SearchReport::prior)
        .def_readonly("model_prior", &SearchReport::model_prior)
        .def_readonly("walk_seconds", &SearchReport::walk_seconds)
        .def_readonly("gather_seconds", &SearchReport::gather_seconds)
        .def_readonly("threads", &SearchReport::threads)
        .def_readonly("clusters", &SearchReport::clusters);

    py::class_<SearchOutcome>(
        m, "SearchResult",
        "The hits of one query, with the ledger behind each where `explain` asked\n"
        "for one. Iterating the result iterates the hits.")
        .def_property_readonly("hits",
                               [](const SearchOutcome& o) { return o.report.hits; })
        .def_readonly("report", &SearchOutcome::report)
        .def_readonly("waterfalls", &SearchOutcome::waterfalls)
        .def_readonly("text", &SearchOutcome::text)
        .def("json", [](const SearchOutcome& o) { return SearchReportJson(o.report); })
        .def("__len__", [](const SearchOutcome& o) { return o.report.hits.size(); })
        .def("__getitem__",
             [](const SearchOutcome& o, size_t index) {
                 if (index >= o.report.hits.size()) throw py::index_error();
                 return o.report.hits[index];
             })
        .def("__repr__", [](const SearchOutcome& o) { return o.text; });

    session->def("search", &SearchStage, py::arg("fields"), py::arg("model"),
                 py::arg("k") = 10, py::arg("threshold") = py::none(),
                 py::arg("probability") = py::none(), py::arg("threads") = 1,
                 py::arg("expected_matches") = py::none(),
                 py::arg("prior_weight") = py::none(), py::arg("explain") = false,
                 py::arg("clusters") = "", py::arg("tf_damping") = 1.0,
                 py::arg("fuzzy_tf") = false, py::arg("ball") = BallOptions{},
                 py::arg("interactions") = true);
}

}  // namespace python
}  // namespace cpplink
