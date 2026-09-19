// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {
namespace python {

namespace py = pybind11;

// What every `bool f(..., std::string* error)` in the core becomes on the Python
// side: `cpplink.Error`, carrying the core's own message.
class Error : public std::runtime_error {
   public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

inline void Check(bool ok, const std::string& error) {
    if (!ok) throw Error(error);
}

// Runs one of the core's `Print*(..., std::ostream&)` into a string. Every
// report's `text` and `__repr__` is the table the command line prints, produced
// by the same function, so the two cannot disagree.
template <typename Printer>
std::string CaptureText(Printer&& printer) {
    std::ostringstream out;
    printer(out);
    return out.str();
}

// A schema as Python holds it: the parsed form beside the text it came from.
// `levels` and `simplify` rewrite the text rather than the struct so that a
// hand-written schema keeps its own layout, and a schema built in Python has
// only the serialised form, which is what `SchemaToJson` gives.
struct PySchema {
    Schema schema;
    std::string text;

    std::string Json() const;
};

// The store behind a `Linker`: one load, shared by every stage, with the plan
// for each purpose built once on first use. The mode is decided at
// construction, as the command line decides it from the file count and
// `--mode`, and `all_pairs` replaces the schema's sources for both plans.
class Session {
   public:
    Session(const PySchema& schema, std::vector<std::string> paths, PairMode mode,
            bool all_pairs);

    const Schema& schema() const { return schema_; }
    const RecordStore& store() const { return *store_; }
    const LoadStats& stats() const { return stats_; }
    const std::vector<std::string>& paths() const { return paths_; }
    PairMode mode() const { return mode_; }
    bool all_pairs() const { return all_pairs_; }

    // The plan for one purpose, built on first use. Estimation and prediction
    // see different unions of sources, and that split is kept here exactly as
    // the command line keeps it.
    const BlockingPlan& Plan(bool for_estimation);
    // The comparisons bound with the signature filter and the ladders on, which
    // is every command's default and the only binding most stages need.
    const ComparisonSet& Comparisons();
    // A binding with either optimisation off, built fresh, for `predict
    // --no-signatures` and `--no-ladders`.
    std::unique_ptr<ComparisonSet> BindComparisons(bool use_signatures, bool use_ladders);

   private:
    Schema schema_;
    std::vector<std::string> paths_;
    PairMode mode_;
    bool all_pairs_;
    std::unique_ptr<RecordStore> store_;
    LoadStats stats_;
    std::unique_ptr<BlockingPlan> estimation_plan_;
    std::unique_ptr<BlockingPlan> prediction_plan_;
    std::unique_ptr<ComparisonSet> comparisons_;
};

// The schema text `levels` and `simplify` rewrite: the source text where the
// schema came from one, else the serialised struct.
std::string SchemaTextOf(const PySchema& schema);

// `--mode` as a keyword: "dedup", "link", "link-and-dedup", or None for what the
// file count decides.
PairMode ModeFrom(const py::object& mode, size_t inputs);

// A threshold in bits, or a probability turned into one; exactly one is given.
double ThresholdFrom(const py::object& threshold, const py::object& probability,
                     const char* command);

// The stages hang off the session class, so the class is made once and handed
// to the files that add methods to it.
using SessionClass = py::class_<Session>;

void BindModule(py::module_& m);
void BindSchema(py::module_& m);
void BindModel(py::module_& m);
SessionClass BindSession(py::module_& m);
void BindStages(py::module_& m, SessionClass* session);
void BindDiagnostics(py::module_& m, SessionClass* session);

}  // namespace python
}  // namespace cpplink
