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

#include "cpplink/arrow_export.hpp"
#include "cpplink/blocking.hpp"
#include "cpplink/cluster.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/predict.hpp"
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
    // The same store from objects that speak the Arrow C stream protocol -- a
    // pyarrow table or reader, a pandas or polars frame -- each named for the
    // dataset it becomes. One input gets no name, as one file gets none. The
    // frames are read through the same decoder a parquet file is, and nothing
    // here links Arrow: the capsule hands over three C structs and a pointer.
    Session(const PySchema& schema,
            std::vector<std::pair<std::string, py::object>> inputs, PairMode mode,
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

    // The predictions of the last `predict`, which `cluster` with no table
    // given reads: the run's own rows, nothing to resolve.
    const std::shared_ptr<EdgeTable>& last_edges() const { return last_edges_; }
    void set_last_edges(std::shared_ptr<EdgeTable> edges) {
        last_edges_ = std::move(edges);
    }

   private:
    std::shared_ptr<EdgeTable> last_edges_;
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

// A table handed to Python through the Arrow PyCapsule protocol: whatever
// `pa.table(x)`, `pl.DataFrame(x)` or `pd.DataFrame(pa.table(x).to_pandas())`
// wants, from one builder of borrowed columns that is exported on every call and
// copies nothing. The record ids leave as a dictionary over the store's id
// arena with the rows as indices, so a table of fifty million predictions
// carries no id text of its own. `payload` is what the columns point into, and
// every live export holds it, and through it the session, until released.
class ArrowTable : public std::enable_shared_from_this<ArrowTable> {
   public:
    virtual ~ArrowTable() = default;
    int64_t Rows() const { return builder_.Rows(); }
    const std::vector<std::string>& Columns() const { return names_; }
    // The C structs, for a stream to hand out. Every column is borrowed, so the
    // batch can be exported any number of times.
    void ExportSchema(ArrowSchema* out) const { builder_.ExportSchema(out); }
    bool ExportBatch(ArrowArray* out, std::string* error) {
        return builder_.ExportBatch(out, error);
    }
    // `__arrow_c_schema__`, `__arrow_c_array__` and `__arrow_c_stream__`.
    py::object SchemaCapsule() const;
    py::tuple ArrayCapsules();
    py::object StreamCapsule();

   protected:
    int Add(const std::string& name, BorrowedColumn column) {
        names_.push_back(name);
        return builder_.AddBorrowed(name, column);
    }

    BatchBuilder builder_;
    std::vector<std::string> names_;
};

// The predictions of one run: `dataset_a, id_a, dataset_b, id_b` (the datasets
// only over several inputs), `gamma`, `match_weight`, `match_probability`, the
// merged file's own columns.
class PredictionTable : public ArrowTable {
   public:
    PredictionTable(py::object session, std::shared_ptr<EdgeTable> edges);
    const std::shared_ptr<EdgeTable>& edges() const;

   private:
    struct Payload;
    std::shared_ptr<Payload> payload_;
};

// The clusters of one partition, the records in clusters of `min_size` or more:
// `dataset` (over several inputs), `unique_id`, `cluster_id`, `cluster_size`,
// the cluster file's own columns.
class ClusterTable : public ArrowTable {
   public:
    ClusterTable(py::object session, const ClusterAssignment& assignment,
                 uint64_t min_size);

   private:
    struct Payload;
    std::shared_ptr<Payload> payload_;
};

// The sample `gen_sample` makes when no file is wanted. Its columns are owned
// rather than borrowed, so the rows move out on the first read and a second
// read finds none.
class SampleTable : public ArrowTable {
   public:
    explicit SampleTable(BatchBuilder builder) {
        names_ = builder.ColumnNames();
        builder_ = std::move(builder);
    }
};

// The stages hang off the session class, so the class is made once and handed
// to the files that add methods to it.
using SessionClass = py::class_<Session>;

void BindModule(py::module_& m);
void BindTables(py::module_& m);
void BindSchema(py::module_& m);
void BindModel(py::module_& m);
SessionClass BindSession(py::module_& m);
void BindStages(py::module_& m, SessionClass* session);
void BindDiagnostics(py::module_& m, SessionClass* session);

}  // namespace python
}  // namespace cpplink
