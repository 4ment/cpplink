// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/common.hpp"
#include "cpplink/app.hpp"
#include "cpplink/parquet_io.hpp"
#include "cpplink/pipeline.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/score.hpp"

#ifndef CPPLINK_VERSION_STRING
#define CPPLINK_VERSION_STRING "0.0.0"
#endif

namespace cpplink {
namespace python {

namespace {

// What `cpplink.run` returns: the exit code and the two streams the command
// wrote, captured rather than printed so a caller can read them.
struct CliResult {
    int code = 0;
    std::string stdout_text;
    std::string stderr_text;
};

CliResult RunCli(const std::vector<std::string>& args) {
    CliResult result;
    std::ostringstream out;
    std::ostringstream err;
    {
        py::gil_scoped_release release;
        result.code = Run(args, out, err);
    }
    result.stdout_text = out.str();
    result.stderr_text = err.str();
    return result;
}

std::string GenSample(const std::string& out, uint64_t rows, uint64_t seed,
                      double duplicate_rate, const std::string& truth,
                      const std::vector<std::string>& out_b) {
    SampleOptions options;
    options.rows = rows;
    options.seed = seed;
    options.duplicate_rate = duplicate_rate;
    options.truth_path = truth;
    options.link_paths = out_b;
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = WriteSampleParquet(out, options, &error);
    }
    Check(ok, error);
    return out;
}

// The same rows with no file written: one table per output `gen-sample` would
// have made, the truth file still written where asked.
std::vector<std::shared_ptr<SampleTable>> GenSampleTables(uint64_t rows, uint64_t seed,
                                                          double duplicate_rate,
                                                          const std::string& truth,
                                                          size_t link_files) {
    SampleOptions options;
    options.rows = rows;
    options.seed = seed;
    options.duplicate_rate = duplicate_rate;
    options.truth_path = truth;
    options.link_paths.assign(link_files, "");
    std::vector<BatchBuilder> builders;
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = GenerateSampleTables(options, &builders, &error);
    }
    Check(ok, error);
    std::vector<std::shared_ptr<SampleTable>> tables;
    for (BatchBuilder& builder : builders) {
        tables.push_back(std::make_shared<SampleTable>(std::move(builder)));
    }
    return tables;
}

}  // namespace

PairMode ModeFrom(const py::object& mode, size_t inputs) {
    if (mode.is_none()) return DefaultMode(false, PairMode::kAll, inputs);
    PairMode parsed = PairMode::kAll;
    std::string error;
    Check(ParseMode(py::cast<std::string>(mode), &parsed, &error), error);
    return parsed;
}

double ThresholdFrom(const py::object& threshold, const py::object& probability,
                     const char* command) {
    if (threshold.is_none() == probability.is_none()) {
        throw Error(std::string(command) +
                    ": give a threshold in bits or a probability, not both");
    }
    if (!threshold.is_none()) return py::cast<double>(threshold);
    const double p = py::cast<double>(probability);
    if (p <= 0.0 || p >= 1.0) {
        throw Error(std::string(command) + ": probability wants a value in (0, 1)");
    }
    return WeightForProbability(p);
}

void BindModule(py::module_& m) {
    m.attr("__version__") = std::string(CPPLINK_VERSION_STRING);
    m.def("parquet_supported", &ParquetSupported,
          "Whether this build reads and writes parquet itself. A build without\n"
          "Arrow takes its input as data frames and reads a file through pyarrow.");
    m.def("dataset_names_for", &DatasetNamesFor, py::arg("paths"),
          "The dataset name each file gets: its stem, made safe for a csv field and\n"
          "a qualified id, and disambiguated. One file gets none.");
    // `kVersion` in the command line and the CMake project version are the same
    // number; the module carries both so a test can say so.
    m.attr("cli_version") = std::string(kVersion);

    py::register_exception<Error>(m, "Error", PyExc_RuntimeError);

    py::class_<CliResult>(m, "CliResult",
                          "The exit code and captured output of one command line run.")
        .def_readonly("code", &CliResult::code)
        .def_readonly("stdout", &CliResult::stdout_text)
        .def_readonly("stderr", &CliResult::stderr_text)
        .def("__repr__", [](const CliResult& r) {
            return "CliResult(code=" + std::to_string(r.code) + ")";
        });

    m.def("run", &RunCli, py::arg("args"),
          "Run the command line in process: `run([\"estimate\", \"--schema\", ...])`.\n"
          "`args` excludes the program name. Returns a CliResult.");

    m.def("gen_sample", &GenSample, py::arg("out"), py::arg("rows") = 1000000,
          py::arg("seed") = 1, py::arg("duplicate_rate") = 0.08, py::arg("truth") = "",
          py::arg("out_b") = std::vector<std::string>{},
          "Write a sample parquet file with planted duplicates, as `cpplink gen-sample`\n"
          "does. `out_b` names the later files of a link fixture.");

    m.def("gen_sample_tables", &GenSampleTables, py::arg("rows") = 1000000,
          py::arg("seed") = 1, py::arg("duplicate_rate") = 0.08, py::arg("truth") = "",
          py::arg("link_files") = 0,
          "The sample as tables rather than files: one per output, the first the\n"
          "originals and each later one a link fixture's duplicates.");

    m.def("weight_for_probability", &WeightForProbability, py::arg("probability"),
          "The match weight, in bits, a match probability corresponds to.");
    m.def("probability_for_weight", &ProbabilityForWeight, py::arg("weight"),
          "The match probability a match weight, in bits, corresponds to.");
}

}  // namespace python
}  // namespace cpplink

PYBIND11_MODULE(_cpplink, m) {
    m.doc() = "The cpplink core, in process. Use the `cpplink` package rather than this.";
    cpplink::python::BindModule(m);
    cpplink::python::BindSchema(m);
    cpplink::python::BindModel(m);
    cpplink::python::BindTables(m);
    cpplink::python::SessionClass session = cpplink::python::BindSession(m);
    cpplink::python::BindStages(m, &session);
    cpplink::python::BindSearch(m, &session);
    cpplink::python::BindDiagnostics(m, &session);
}
