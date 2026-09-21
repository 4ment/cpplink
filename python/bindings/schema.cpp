// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/schema.hpp"

#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/common.hpp"
#include "cpplink/batch_loader.hpp"
#include "cpplink/init.hpp"

namespace cpplink {
namespace python {

std::string PySchema::Json() const { return SchemaToJson(schema); }

std::string SchemaTextOf(const PySchema& schema) {
    return schema.text.empty() ? schema.Json() : schema.text;
}

namespace {

std::string ReadFile(const std::string& path) {
    std::ifstream file(path);
    Check(static_cast<bool>(file), "cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

PySchema SchemaFromJson(const std::string& text) {
    PySchema result;
    std::string error;
    Check(ParseSchema(text, &result.schema, &error), error);
    result.text = text;
    return result;
}

PySchema SchemaFromFile(const std::string& path) {
    PySchema result;
    std::string error;
    Check(LoadSchema(path, &result.schema, &error), error);
    result.text = ReadFile(path);
    return result;
}

void SaveSchema(const PySchema& schema, const std::string& path) {
    std::ofstream file(path);
    Check(static_cast<bool>(file), "cannot write " + path);
    file << SchemaTextOf(schema);
}

// What `cpplink.init` returns beside the report: the draft as a schema, with
// the text exactly as the command would have written it.
std::pair<PySchema, DraftReport> Init(
    const std::vector<std::string>& paths, const std::string& unique_id,
    const std::vector<std::pair<std::string, std::string>>& roles,
    const std::string& out) {
    DraftOptions options;
    options.unique_id = unique_id;
    options.roles = roles;
    DraftReport report;
    std::string error;
    bool ok = false;
    {
        py::gil_scoped_release release;
        ok = DraftSchema(paths, options, &report, &error);
    }
    Check(ok, error);
    PySchema schema;
    schema.schema = report.schema;
    schema.text = report.json;
    if (!out.empty()) SaveSchema(schema, out);
    return {schema, report};
}

// The draft from inputs described by an Arrow schema each -- what a frame has
// instead of a footer. Each object speaks `__arrow_c_schema__`.
std::pair<PySchema, DraftReport> InitFrom(
    const std::vector<std::pair<std::string, py::object>>& inputs,
    const std::string& unique_id,
    const std::vector<std::pair<std::string, std::string>>& roles,
    const std::string& out) {
    std::vector<DescribedInput> described;
    for (const auto& [name, object] : inputs) {
        if (!py::hasattr(object, "__arrow_c_schema__")) {
            throw Error("input " + name + " has no __arrow_c_schema__");
        }
        py::object capsule = object.attr("__arrow_c_schema__")();
        void* pointer = PyCapsule_GetPointer(capsule.ptr(), "arrow_schema");
        if (pointer == nullptr) {
            PyErr_Clear();
            throw Error("__arrow_c_schema__ did not return an arrow_schema capsule");
        }
        const ArrowSchema& schema = *static_cast<ArrowSchema*>(pointer);
        DescribedInput input;
        input.name = name;
        for (int64_t i = 0; i < schema.n_children; ++i) {
            const ArrowSchema& field = *schema.children[i];
            FileColumn column;
            column.name = field.name == nullptr ? "" : field.name;
            column.arrow_type = ArrowTypeName(field);
            column.readable = ColumnTypeOf(field, &column.type);
            input.columns.push_back(std::move(column));
        }
        described.push_back(std::move(input));
    }
    DraftOptions options;
    options.unique_id = unique_id;
    options.roles = roles;
    DraftReport report;
    std::string error;
    Check(DraftSchemaFrom(described, options, &report, &error), error);
    PySchema schema;
    schema.schema = report.schema;
    schema.text = report.json;
    if (!out.empty()) SaveSchema(schema, out);
    return {schema, report};
}

template <typename Spec>
std::vector<std::string> Describe(const std::vector<Spec>& specs) {
    std::vector<std::string> out;
    out.reserve(specs.size());
    for (const Spec& spec : specs) out.push_back(spec.Describe());
    return out;
}

}  // namespace

void BindSchema(py::module_& m) {
    py::class_<PySchema>(m, "Schema",
                         "The JSON configuration: columns, comparisons and blocking.\n"
                         "Build one with `Schema.from_file`, `Schema.from_json` or\n"
                         "`Schema.from_dict`; edit it by dict round trip.")
        .def_static("from_file", &SchemaFromFile, py::arg("path"))
        .def_static("from_json", &SchemaFromJson, py::arg("text"))
        .def("to_json", &PySchema::Json,
             "The schema serialised by the same writer `cpplink init` uses.")
        .def("save", &SaveSchema, py::arg("path"),
             "Write the schema: its source text where it came from one, else to_json().")
        .def_property_readonly("text", &SchemaTextOf,
                               "The text `levels` and `simplify` rewrite.")
        .def_property_readonly("unique_id",
                               [](const PySchema& s) { return s.schema.unique_id; })
        .def_property_readonly("column_names",
                               [](const PySchema& s) {
                                   std::vector<std::string> names;
                                   for (const ColumnSpec& column : s.schema.columns) {
                                       names.push_back(column.name);
                                   }
                                   return names;
                               })
        .def_property_readonly(
            "input_columns",
            [](const PySchema& s) {
                std::vector<std::string> names;
                if (!s.schema.unique_id.empty()) names.push_back(s.schema.unique_id);
                for (const ColumnSpec& column : s.schema.columns) {
                    if (!column.IsDerived()) names.push_back(column.name);
                }
                return names;
            },
            "The columns an input must hold: the id and every column that is not\n"
            "derived, which is what a reader is asked for.")
        .def_property_readonly(
            "comparison_names",
            [](const PySchema& s) {
                std::vector<std::string> names;
                for (const ComparisonSpec& comparison : s.schema.comparisons) {
                    names.push_back(comparison.name);
                }
                return names;
            })
        .def_property_readonly(
            "comparisons",
            [](const PySchema& s) {
                std::vector<std::vector<std::string>> out;
                for (const ComparisonSpec& comparison : s.schema.comparisons) {
                    out.push_back(Describe(comparison.levels));
                }
                return out;
            },
            "Per comparison, the description of each level, in order.")
        .def_property_readonly(
            "blocking", [](const PySchema& s) { return Describe(s.schema.blocking); },
            "The description of each blocking source, in order.")
        .def_property_readonly(
            "gamma_width",
            [](const PySchema& s) { return static_cast<int>(s.schema.GammaWidth()); },
            "Bits of the packed comparison vector.")
        .def("__repr__", [](const PySchema& s) {
            return "Schema(" + std::to_string(s.schema.columns.size()) + " columns, " +
                   std::to_string(s.schema.comparisons.size()) + " comparisons, " +
                   std::to_string(s.schema.blocking.size()) + " blocking sources)";
        });

    py::class_<DraftColumn>(m, "DraftColumn")
        .def_readonly("name", &DraftColumn::name)
        .def_readonly("arrow_type", &DraftColumn::arrow_type)
        .def_property_readonly(
            "type", [](const DraftColumn& c) { return ColumnTypeName(c.type); })
        .def_readonly("readable", &DraftColumn::readable)
        .def_property_readonly("role",
                               [](const DraftColumn& c) { return RoleName(c.role); })
        .def_readonly("role_given", &DraftColumn::role_given)
        .def_readonly("note", &DraftColumn::note);

    py::class_<DraftReport>(m, "DraftReport",
                            "What `cpplink init` guessed, column by column.")
        .def_readonly("path", &DraftReport::path)
        .def_readonly("columns", &DraftReport::columns)
        .def_readonly("notes", &DraftReport::notes)
        .def_readonly("json", &DraftReport::json)
        .def_property_readonly(
            "text",
            [](const DraftReport& r) {
                return CaptureText([&](std::ostream& out) { PrintDraftReport(r, out); });
            })
        .def("__repr__", [](const DraftReport& r) {
            return CaptureText([&](std::ostream& out) { PrintDraftReport(r, out); });
        });

    m.def("init_from", &InitFrom, py::arg("inputs"), py::arg("id") = "",
          py::arg("roles") = std::vector<std::pair<std::string, std::string>>{},
          py::arg("out") = "",
          "Draft a schema from `(name, object)` pairs, each object speaking\n"
          "`__arrow_c_schema__`: a pyarrow schema or table.");
    m.def("init", &Init, py::arg("files"), py::arg("id") = "",
          py::arg("roles") = std::vector<std::pair<std::string, std::string>>{},
          py::arg("out") = "",
          "Draft a schema from the parquet footer, as `cpplink init` does. Returns\n"
          "(Schema, DraftReport). `roles` is a list of (column, role) overrides.");
    m.def("known_roles", &KnownRoles, "The roles `init` accepts, as one string.");
}

}  // namespace python
}  // namespace cpplink
