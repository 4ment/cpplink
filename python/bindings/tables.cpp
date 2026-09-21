// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <cerrno>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/common.hpp"
#include "cpplink/arrow_c.hpp"
#include "cpplink/id_index.hpp"
#include "cpplink/score.hpp"

namespace cpplink {
namespace python {

namespace {

// Strings as an arena: 64-bit offsets over one text, which is the layout a
// `large_string` column and a dictionary's values both want.
struct Arena {
    std::vector<int64_t> offsets{0};
    std::string text;
    void Add(const std::string& value) {
        text += value;
        offsets.push_back(static_cast<int64_t>(text.size()));
    }
    int64_t Size() const { return static_cast<int64_t>(offsets.size() - 1); }
};

BorrowedColumn Dictionary(const std::vector<uint32_t>& indices, const int64_t* offsets,
                          const char* text, int64_t size) {
    BorrowedColumn column;
    column.type = ExportType::kDictionary;
    column.length = static_cast<int64_t>(indices.size());
    column.values = indices.data();
    column.dictionary_offsets = offsets;
    column.dictionary_text = text;
    column.dictionary_size = size;
    return column;
}

// The record ids: the store's arena as the dictionary, `rows` as the indices.
BorrowedColumn IdsOf(const RecordStore& store, const std::vector<uint32_t>& rows) {
    const IdColumn& ids = store.ids();
    return Dictionary(rows, reinterpret_cast<const int64_t*>(ids.offsets.data()),
                      ids.text.data(), static_cast<int64_t>(ids.offsets.size() - 1));
}

BorrowedColumn NamesOf(const Arena& names, const std::vector<uint32_t>& which) {
    return Dictionary(which, names.offsets.data(), names.text.data(), names.Size());
}

template <typename T>
BorrowedColumn Fixed(ExportType type, const std::vector<T>& values) {
    BorrowedColumn column;
    column.type = type;
    column.length = static_cast<int64_t>(values.size());
    column.values = values.data();
    return column;
}

BorrowedColumn Strings(const Arena& arena) {
    BorrowedColumn column;
    column.type = ExportType::kLargeString;
    column.length = arena.Size();
    column.values = arena.offsets.data();
    column.text = arena.text.c_str();
    return column;
}

// What every payload carries: the session, so the store's arena outlives any
// export that points into it. Dropping the last reference to a Python object
// needs the GIL, and the last export may be released from anywhere.
struct SessionHold {
    py::object session;
    explicit SessionHold(py::object s) : session(std::move(s)) {}
    ~SessionHold() {
        py::gil_scoped_acquire acquire;
        session = py::object();
    }
};

const RecordStore& StoreOf(const py::object& session) {
    return py::cast<const Session&>(session).store();
}

Arena DatasetNames(const RecordStore& store) {
    Arena names;
    for (size_t d = 0; d < store.NumDatasets(); ++d) names.Add(store.DatasetName(d));
    return names;
}

// What a capsule frees: the struct's own release, then the struct.
void FreeSchemaCapsule(PyObject* capsule) {
    auto* schema =
        static_cast<ArrowSchema*>(PyCapsule_GetPointer(capsule, "arrow_schema"));
    if (schema == nullptr) return;
    if (schema->release != nullptr) schema->release(schema);
    delete schema;
}

void FreeArrayCapsule(PyObject* capsule) {
    auto* array = static_cast<ArrowArray*>(PyCapsule_GetPointer(capsule, "arrow_array"));
    if (array == nullptr) return;
    if (array->release != nullptr) array->release(array);
    delete array;
}

void FreeStreamCapsule(PyObject* capsule) {
    auto* stream = static_cast<ArrowArrayStream*>(
        PyCapsule_GetPointer(capsule, "arrow_array_stream"));
    if (stream == nullptr) return;
    if (stream->release != nullptr) stream->release(stream);
    delete stream;
}

// A stream of one batch, exported from the table when the consumer pulls it.
// The table lives as long as the stream does.
struct OneBatchStream {
    std::shared_ptr<ArrowTable> table;
    bool sent = false;
    std::string error;

    static int GetSchema(ArrowArrayStream* self, ArrowSchema* out) {
        static_cast<OneBatchStream*>(self->private_data)->table->ExportSchema(out);
        return 0;
    }
    static int GetNext(ArrowArrayStream* self, ArrowArray* out) {
        auto* state = static_cast<OneBatchStream*>(self->private_data);
        if (state->sent) {
            out->release = nullptr;
            return 0;
        }
        state->sent = true;
        return state->table->ExportBatch(out, &state->error) ? 0 : EINVAL;
    }
    static const char* GetLastError(ArrowArrayStream* self) {
        auto* state = static_cast<OneBatchStream*>(self->private_data);
        return state->error.empty() ? nullptr : state->error.c_str();
    }
    static void Release(ArrowArrayStream* self) {
        delete static_cast<OneBatchStream*>(self->private_data);
        self->release = nullptr;
    }
};

}  // namespace

py::object ArrowTable::SchemaCapsule() const {
    auto* schema = new ArrowSchema;
    ExportSchema(schema);
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(schema, "arrow_schema", FreeSchemaCapsule));
}

py::tuple ArrowTable::ArrayCapsules() {
    auto* array = new ArrowArray;
    std::string error;
    if (!ExportBatch(array, &error)) {
        delete array;
        throw Error(error);
    }
    py::object schema = SchemaCapsule();
    py::object batch = py::reinterpret_steal<py::object>(
        PyCapsule_New(array, "arrow_array", FreeArrayCapsule));
    return py::make_tuple(schema, batch);
}

py::object ArrowTable::StreamCapsule() {
    auto* state = new OneBatchStream;
    state->table = shared_from_this();
    auto* stream = new ArrowArrayStream;
    stream->get_schema = OneBatchStream::GetSchema;
    stream->get_next = OneBatchStream::GetNext;
    stream->get_last_error = OneBatchStream::GetLastError;
    stream->release = OneBatchStream::Release;
    stream->private_data = state;
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(stream, "arrow_array_stream", FreeStreamCapsule));
}

struct PredictionTable::Payload : SessionHold {
    std::shared_ptr<EdgeTable> edges;
    std::vector<double> probability;
    // Over several inputs: which dataset each side's record came from.
    Arena names;
    std::vector<uint32_t> dataset_a;
    std::vector<uint32_t> dataset_b;
    Payload(py::object s, std::shared_ptr<EdgeTable> e)
        : SessionHold(std::move(s)), edges(std::move(e)) {}
};

PredictionTable::PredictionTable(py::object session, std::shared_ptr<EdgeTable> edges)
    : payload_(std::make_shared<Payload>(session, std::move(edges))) {
    const RecordStore& store = StoreOf(session);
    const EdgeTable& table = *payload_->edges;
    const bool datasets = store.NumDatasets() > 1;
    payload_->probability.reserve(table.Size());
    for (const double weight : table.weight) {
        payload_->probability.push_back(ProbabilityForWeight(weight));
    }
    if (datasets) {
        payload_->names = DatasetNames(store);
        payload_->dataset_a.reserve(table.Size());
        payload_->dataset_b.reserve(table.Size());
        for (uint64_t i = 0; i < table.Size(); ++i) {
            payload_->dataset_a.push_back(store.DatasetOf(table.a[i]));
            payload_->dataset_b.push_back(store.DatasetOf(table.b[i]));
        }
    }
    // The merged file's columns, in its order.
    if (datasets) Add("dataset_a", NamesOf(payload_->names, payload_->dataset_a));
    Add("id_a", IdsOf(store, table.a));
    if (datasets) Add("dataset_b", NamesOf(payload_->names, payload_->dataset_b));
    Add("id_b", IdsOf(store, table.b));
    Add("gamma", Fixed(ExportType::kUInt32, table.gamma));
    Add("match_weight", Fixed(ExportType::kDouble, table.weight));
    Add("match_probability", Fixed(ExportType::kDouble, payload_->probability));
    builder_.KeepAlive(payload_);
}

const std::shared_ptr<EdgeTable>& PredictionTable::edges() const {
    return payload_->edges;
}

struct ClusterTable::Payload : SessionHold {
    std::vector<uint32_t> rows;   // the records kept, in row order
    std::vector<uint32_t> roots;  // each one's representative row
    std::vector<uint64_t> sizes;
    std::vector<uint32_t> dataset;
    Arena names;
    Arena qualified;  // `dataset:id` per kept row, over several inputs
    explicit Payload(py::object s) : SessionHold(std::move(s)) {}
};

ClusterTable::ClusterTable(py::object session, const ClusterAssignment& assignment,
                           uint64_t min_size)
    : payload_(std::make_shared<Payload>(session)) {
    const RecordStore& store = StoreOf(session);
    const bool datasets = store.NumDatasets() > 1;
    Payload& p = *payload_;
    for (uint64_t row = 0; row < assignment.root.size(); ++row) {
        const uint32_t root = assignment.root[static_cast<size_t>(row)];
        const uint64_t size = assignment.size[root];
        if (size < min_size) continue;
        p.rows.push_back(static_cast<uint32_t>(row));
        p.roots.push_back(root);
        p.sizes.push_back(size);
        if (datasets) {
            p.dataset.push_back(store.DatasetOf(row));
            p.qualified.Add(QualifiedId(store, root));
        }
    }
    if (datasets) p.names = DatasetNames(store);
    // The cluster file's columns, in its order. A cluster is named by its
    // representative's id, so over one input that column is the arena again.
    if (datasets) Add("dataset", NamesOf(p.names, p.dataset));
    Add("unique_id", IdsOf(store, p.rows));
    if (datasets) {
        Add("cluster_id", Strings(p.qualified));
    } else {
        Add("cluster_id", IdsOf(store, p.roots));
    }
    Add("cluster_size", Fixed(ExportType::kUInt64, p.sizes));
    builder_.KeepAlive(payload_);
}

void BindTables(py::module_& m) {
    py::class_<ArrowTable, std::shared_ptr<ArrowTable>>(
        m, "ArrowTable",
        "A table the core holds, readable by anything that speaks the Arrow\n"
        "PyCapsule protocol: `pyarrow.table(t)`, `polars.DataFrame(t)`, and through\n"
        "pyarrow, pandas. Nothing is copied on the way out; the record ids are a\n"
        "dictionary over the store's own ids.")
        .def("__len__", &ArrowTable::Rows)
        .def_property_readonly("columns", &ArrowTable::Columns)
        .def("__arrow_c_schema__", &ArrowTable::SchemaCapsule)
        .def(
            "__arrow_c_array__",
            [](ArrowTable& t, const py::object&) { return t.ArrayCapsules(); },
            py::arg("requested_schema") = py::none())
        .def(
            "__arrow_c_stream__",
            [](ArrowTable& t, const py::object&) { return t.StreamCapsule(); },
            py::arg("requested_schema") = py::none());
    py::class_<PredictionTable, ArrowTable, std::shared_ptr<PredictionTable>>(
        m, "PredictionTable",
        "The predictions of one run, as the merged file's columns.");
    py::class_<ClusterTable, ArrowTable, std::shared_ptr<ClusterTable>>(
        m, "ClusterTable",
        "The clusters of one partition, as the cluster file's columns.");
    py::class_<SampleTable, ArrowTable, std::shared_ptr<SampleTable>>(
        m, "SampleTable",
        "A sample file's rows, before any file: read once, as the rows move out.");
}

}  // namespace python
}  // namespace cpplink
