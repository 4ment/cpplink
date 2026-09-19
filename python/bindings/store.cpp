// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/common.hpp"
#include "cpplink/id_index.hpp"
#include "cpplink/inspect.hpp"
#include "cpplink/pipeline.hpp"

namespace cpplink {
namespace python {

Session::Session(const PySchema& schema, std::vector<std::string> paths, PairMode mode,
                 bool all_pairs)
    : schema_(schema.schema),
      paths_(std::move(paths)),
      mode_(mode),
      all_pairs_(all_pairs) {
    Check(!paths_.empty(), "a Linker needs at least one parquet file");
    std::string error;
    Check(ResolveColumnTypes(paths_, &schema_, &error), error);
    store_ = std::make_unique<RecordStore>(schema_);
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = LoadParquetFiles(paths_, schema_, store_.get(), &stats_, &error);
    }
    Check(ok, error);
}

const BlockingPlan& Session::Plan(bool for_estimation) {
    std::unique_ptr<BlockingPlan>& slot =
        for_estimation ? estimation_plan_ : prediction_plan_;
    if (!slot) {
        auto plan = std::make_unique<BlockingPlan>();
        std::string error;
        bool ok = false;
        {
            py::gil_scoped_release release;
            ok = BuildPlan(schema_, *store_, mode_, all_pairs_, for_estimation,
                           plan.get(), &error);
        }
        Check(ok, error);
        slot = std::move(plan);
    }
    return *slot;
}

const ComparisonSet& Session::Comparisons() {
    if (!comparisons_) comparisons_ = BindComparisons(true, true);
    return *comparisons_;
}

std::unique_ptr<ComparisonSet> Session::BindComparisons(bool use_signatures,
                                                        bool use_ladders) {
    Check(!schema_.comparisons.empty(), "the schema declares no \"comparisons\"");
    auto comparisons = std::make_unique<ComparisonSet>();
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = comparisons->Bind(schema_, *store_, &error, use_signatures, use_ladders);
    }
    Check(ok, error);
    return comparisons;
}

namespace {

// What `inspect` reports: the load, and the store's memory line by line.
struct Inspection {
    LoadStats stats;
    MemoryReport memory;
    std::vector<std::string> datasets;
    std::string text;
};

Inspection Inspect(Session& session) {
    Inspection result;
    result.stats = session.stats();
    result.memory = session.store().Memory();
    for (size_t d = 0; d < session.store().NumDatasets(); ++d) {
        result.datasets.push_back(session.store().DatasetName(d));
    }
    result.text = CaptureText([&](std::ostream& out) {
        PrintInspection(session.store(), session.stats(), out);
    });
    return result;
}

// The ids of a store, for the record a row index names.
std::string IdOf(const Session& session, uint64_t row) {
    Check(row < session.store().NumRecords(), "row out of range");
    return std::string(session.store().ids().Get(row));
}

uint64_t RowOf(const Session& session, const std::string& id) {
    uint64_t row = 0;
    std::string error;
    Check(ResolveId(session.store(), id, &row, &error), error);
    return row;
}

}  // namespace

SessionClass BindSession(py::module_& m) {
    py::class_<LoadStats>(m, "LoadStats", "What loading the parquet input cost.")
        .def_readonly("rows", &LoadStats::rows)
        .def_readonly("row_groups", &LoadStats::row_groups)
        .def_readonly("seconds", &LoadStats::seconds)
        .def_readonly("dataset_rows", &LoadStats::dataset_rows);

    py::class_<MemoryLine>(m, "MemoryLine")
        .def_readonly("structure", &MemoryLine::structure)
        .def_readonly("bytes", &MemoryLine::bytes)
        .def_readonly("basis", &MemoryLine::basis);

    py::class_<MemoryReport>(m, "MemoryReport",
                             "The store's resident memory, by structure.")
        .def_readonly("lines", &MemoryReport::lines)
        .def_property_readonly("total", &MemoryReport::Total);

    py::class_<Inspection>(m, "Inspection", "What `cpplink inspect` reports.")
        .def_readonly("stats", &Inspection::stats)
        .def_readonly("memory", &Inspection::memory)
        .def_readonly("datasets", &Inspection::datasets)
        .def_readonly("text", &Inspection::text)
        .def("__repr__", [](const Inspection& i) { return i.text; });

    SessionClass session(m, "_Session",
                         "The loaded store behind a Linker. Use `cpplink.Linker`.");
    session
        .def(py::init([](const PySchema& schema, std::vector<std::string> paths,
                         const py::object& mode, bool all_pairs) {
                 return std::make_unique<Session>(
                     schema, paths, ModeFrom(mode, paths.size()), all_pairs);
             }),
             py::arg("schema"), py::arg("files"), py::arg("mode") = py::none(),
             py::arg("all_pairs") = false)
        .def_property_readonly("records",
                               [](const Session& s) { return s.store().NumRecords(); })
        .def_property_readonly("datasets",
                               [](const Session& s) { return s.store().NumDatasets(); })
        .def_property_readonly("files", &Session::paths)
        .def_property_readonly("mode",
                               [](const Session& s) { return PairModeName(s.mode()); })
        .def_property_readonly("all_pairs", &Session::all_pairs)
        .def_property_readonly("stats", &Session::stats)
        .def_property_readonly(
            "schema",
            [](const Session& s) {
                PySchema schema;
                schema.schema = s.schema();
                return schema;
            },
            "The schema as the store holds it, with the file's column types filled in.")
        .def("inspect", &Inspect)
        .def("id_of", &IdOf, py::arg("row"), "The unique id of the record at this row.")
        .def("dataset_of",
             [](const Session& s, uint64_t row) {
                 Check(row < s.store().NumRecords(), "row out of range");
                 return s.store().DatasetName(s.store().DatasetOf(row));
             })
        .def("qualified_id",
             [](const Session& s, uint64_t row) {
                 Check(row < s.store().NumRecords(), "row out of range");
                 return QualifiedId(s.store(), row);
             })
        .def("row_of", &RowOf, py::arg("id"),
             "The row a record id names, or `dataset:id` where the inputs share ids.");
    return session;
}

}  // namespace python
}  // namespace cpplink
