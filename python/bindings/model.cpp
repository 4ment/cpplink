// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/model.hpp"

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/common.hpp"

namespace cpplink {
namespace python {

namespace {

Model ModelFromFile(const std::string& path) {
    Model model;
    std::string error;
    Check(LoadModel(path, &model, &error), error);
    return model;
}

Model ModelFromJson(const std::string& text) {
    Model model;
    std::string error;
    Check(ParseModelJson(text, &model, &error), error);
    return model;
}

void SaveModel(const Model& model, const std::string& path) {
    std::string error;
    Check(WriteModelJson(model, path, &error), error);
}

std::string ModelText(const Model& model) {
    return CaptureText([&](std::ostream& out) {
        PrintModel(model, out);
        PrintInteractions(model, out);
    });
}

}  // namespace

void BindModel(py::module_& m) {
    py::class_<ModelLevel>(m, "ModelLevel", "One comparison level's learned m and u.")
        .def_readonly("label", &ModelLevel::label)
        .def_readonly("m", &ModelLevel::m)
        .def_readonly("u", &ModelLevel::u)
        .def_readonly("u_exact", &ModelLevel::u_exact)
        .def_readonly("m_estimated", &ModelLevel::m_estimated)
        .def_readonly("u_observed", &ModelLevel::u_observed)
        .def_readonly("m_support", &ModelLevel::m_support)
        .def("weight", &ModelLevel::Weight, "log2(m / u): the bits this level is worth.")
        .def("__repr__", [](const ModelLevel& l) {
            return "ModelLevel(" + l.label + ", m=" + std::to_string(l.m) +
                   ", u=" + std::to_string(l.u) + ")";
        });

    py::class_<ModelComparison>(m, "ModelComparison")
        .def_readonly("name", &ModelComparison::name)
        .def_readonly("columns", &ModelComparison::columns)
        .def_readonly("term_frequency", &ModelComparison::term_frequency)
        .def_readonly("levels", &ModelComparison::levels)
        .def_readonly("sessions", &ModelComparison::sessions)
        .def("__repr__", [](const ModelComparison& c) {
            return "ModelComparison(" + c.name + ", " + std::to_string(c.levels.size()) +
                   " levels)";
        });

    py::class_<ModelInteraction>(m, "ModelInteraction",
                                 "One fitted two-way correction to the weight.")
        .def_readonly("left", &ModelInteraction::left)
        .def_readonly("right", &ModelInteraction::right)
        .def_readonly("left_levels", &ModelInteraction::left_levels)
        .def_readonly("right_levels", &ModelInteraction::right_levels)
        .def_readonly("table", &ModelInteraction::bits,
                      "left_levels x right_levels corrections, row-major, in bits.")
        .def_readonly("match_bits", &ModelInteraction::match_bits)
        .def_readonly("effect", &ModelInteraction::effect)
        .def_readonly("sessions", &ModelInteraction::sessions)
        .def(
            "bits",
            [](const ModelInteraction& i, int left, int right) {
                return i.Bits(static_cast<uint8_t>(left), static_cast<uint8_t>(right));
            },
            py::arg("left_level"), py::arg("right_level"),
            "The correction when the two comparisons land on these levels.")
        .def("__repr__", [](const ModelInteraction& i) {
            return "ModelInteraction(" + i.left + " x " + i.right + ")";
        });

    py::class_<Model>(m, "Model",
                      "The learned parameters: lambda, m and u per level, and any\n"
                      "two-way corrections. What `estimate` writes and `predict` reads.")
        .def_static("from_file", &ModelFromFile, py::arg("path"))
        .def_static("from_json", &ModelFromJson, py::arg("text"))
        .def("to_json", &ModelJson, "The model as `estimate --out` writes it.")
        .def("save", &SaveModel, py::arg("path"))
        .def_readonly("lambda_", &Model::lambda, "The prior that two records match.")
        .def_readonly("lambda_basis", &Model::lambda_basis)
        .def_readonly("records", &Model::records)
        .def_readonly("comparisons", &Model::comparisons)
        .def_readonly("interactions", &Model::interactions)
        .def("prior_weight", &Model::PriorWeight,
             "log2(lambda / (1 - lambda)): the prior term of the match weight.")
        .def_property_readonly("text", &ModelText,
                               "The parameter table `estimate` prints.")
        .def("__repr__", &ModelText);
}

}  // namespace python
}  // namespace cpplink
