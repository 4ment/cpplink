// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/model.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace cpplink {
namespace {

std::string Truncate(std::string text, size_t width) {
    if (text.size() <= width) return text;
    return text.substr(0, width - 1) + ".";
}

std::string Fixed(double value, int places) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(places) << value;
    return buffer.str();
}

// u runs to 1e-7 on a high-entropy column, which fixed notation cannot show at any
// sane width, so small values switch to scientific.
std::string Probability(double value) {
    std::ostringstream buffer;
    if (value != 0.0 && std::fabs(value) < 1e-4) {
        buffer << std::scientific << std::setprecision(2) << value;
    } else {
        buffer << std::fixed << std::setprecision(6) << value;
    }
    return buffer.str();
}

}  // namespace

double ModelLevel::Weight() const {
    if (m <= 0.0 || u <= 0.0) return 0.0;
    return std::log2(m / u);
}

double Model::PriorWeight() const {
    if (lambda <= 0.0 || lambda >= 1.0) return 0.0;
    return std::log2(lambda / (1.0 - lambda));
}

std::string ModelJson(const Model& model) {
    nlohmann::json root;
    root["lambda"] = model.lambda;
    root["lambda_basis"] = model.lambda_basis;
    root["records"] = model.records;
    root["comparisons"] = nlohmann::json::array();
    for (const ModelComparison& comparison : model.comparisons) {
        nlohmann::json item;
        item["name"] = comparison.name;
        item["columns"] = comparison.columns;
        item["term_frequency"] = comparison.term_frequency;
        item["sessions"] = comparison.sessions;
        item["levels"] = nlohmann::json::array();
        for (const ModelLevel& level : comparison.levels) {
            nlohmann::json entry;
            entry["label"] = level.label;
            entry["m"] = level.m;
            entry["u"] = level.u;
            entry["u_exact"] = level.u_exact;
            entry["m_estimated"] = level.m_estimated;
            entry["u_observed"] = level.u_observed;
            entry["m_support"] = level.m_support;
            item["levels"].push_back(entry);
        }
        root["comparisons"].push_back(item);
    }
    return root.dump(2) + "\n";
}

bool WriteModelJson(const Model& model, const std::string& path, std::string* error) {
    std::ofstream file(path);
    if (!file) {
        *error = "cannot write \"" + path + "\"";
        return false;
    }
    file << ModelJson(model);
    if (!file) {
        *error = "failed while writing \"" + path + "\"";
        return false;
    }
    return true;
}

bool ParseModelJson(const std::string& text, Model* model, std::string* error) {
    nlohmann::json root = nlohmann::json::parse(text, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        *error = "the model is not a JSON object";
        return false;
    }
    if (!root.contains("lambda") || !root["lambda"].is_number()) {
        *error = "the model has no numeric \"lambda\"";
        return false;
    }
    model->lambda = root["lambda"].get<double>();
    if (root.contains("lambda_basis") && root["lambda_basis"].is_string()) {
        model->lambda_basis = root["lambda_basis"].get<std::string>();
    }
    if (root.contains("records") && root["records"].is_number()) {
        model->records = root["records"].get<uint64_t>();
    }
    if (!root.contains("comparisons") || !root["comparisons"].is_array()) {
        *error = "the model has no \"comparisons\" array";
        return false;
    }
    model->comparisons.clear();
    for (const nlohmann::json& item : root["comparisons"]) {
        if (!item.is_object() || !item.contains("levels") || !item["levels"].is_array()) {
            *error = "every comparison needs a \"levels\" array";
            return false;
        }
        ModelComparison comparison;
        if (item.contains("name") && item["name"].is_string()) {
            comparison.name = item["name"].get<std::string>();
        }
        if (item.contains("columns") && item["columns"].is_array()) {
            for (const nlohmann::json& column : item["columns"]) {
                if (column.is_string()) {
                    comparison.columns.push_back(column.get<std::string>());
                }
            }
        }
        comparison.term_frequency = item.contains("term_frequency") &&
                                    item["term_frequency"].is_boolean() &&
                                    item["term_frequency"].get<bool>();
        if (item.contains("sessions") && item["sessions"].is_number()) {
            comparison.sessions = item["sessions"].get<size_t>();
        }
        for (const nlohmann::json& entry : item["levels"]) {
            if (!entry.is_object() || !entry.contains("m") || !entry.contains("u") ||
                !entry["m"].is_number() || !entry["u"].is_number()) {
                *error = "every level needs numeric \"m\" and \"u\"";
                return false;
            }
            ModelLevel level;
            level.m = entry["m"].get<double>();
            level.u = entry["u"].get<double>();
            if (entry.contains("label") && entry["label"].is_string()) {
                level.label = entry["label"].get<std::string>();
            }
            level.u_exact = entry.contains("u_exact") && entry["u_exact"].is_boolean() &&
                            entry["u_exact"].get<bool>();
            level.m_estimated = entry.contains("m_estimated") &&
                                entry["m_estimated"].is_boolean() &&
                                entry["m_estimated"].get<bool>();
            if (entry.contains("u_observed") && entry["u_observed"].is_number()) {
                level.u_observed = entry["u_observed"].get<uint64_t>();
            }
            if (entry.contains("m_support") && entry["m_support"].is_number()) {
                level.m_support = entry["m_support"].get<double>();
            }
            comparison.levels.push_back(std::move(level));
        }
        model->comparisons.push_back(std::move(comparison));
    }
    return true;
}

bool LoadModel(const std::string& path, Model* model, std::string* error) {
    std::ifstream file(path);
    if (!file) {
        *error = "cannot open model \"" + path + "\"";
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (!ParseModelJson(buffer.str(), model, error)) {
        *error = path + ": " + *error;
        return false;
    }
    return true;
}

void PrintModel(const Model& model, std::ostream& out) {
    out << "lambda       " << Probability(model.lambda) << "  (" << model.lambda_basis
        << ")\n"
        << "prior weight " << Fixed(model.PriorWeight(), 3) << " bits\n\n";

    out << std::left << std::setw(18) << "Comparison" << std::setw(26) << "Level"
        << std::right << std::setw(12) << "m" << std::setw(14) << "u" << std::setw(10)
        << "weight" << "  " << std::left << "note" << "\n";
    out << std::string(92, '-') << "\n";

    for (const ModelComparison& comparison : model.comparisons) {
        bool first = true;
        for (const ModelLevel& level : comparison.levels) {
            std::string note;
            if (level.u_exact && level.m == level.u) {
                note = "unreachable: no pair can land here";
            } else if (!level.m_estimated) {
                note = "m not estimated";
            } else if (level.m_support < 0.5) {
                note = "m at the floor: no matching pair reached this level";
            } else if (!level.u_exact && level.u_observed == 0) {
                note = "u below the sampling floor";
            } else if (!level.u_exact && level.u_observed < 10) {
                note = "u from only " + std::to_string(level.u_observed) +
                       (level.u_observed == 1 ? " random pair" : " random pairs");
            } else if (level.u_exact) {
                note = "u exact";
            }
            out << std::left << std::setw(18)
                << Truncate(first ? comparison.name : "", 17) << std::setw(26)
                << Truncate(level.label, 25) << std::right << std::setw(12)
                << Probability(level.m) << std::setw(14) << Probability(level.u)
                << std::setw(10) << Fixed(level.Weight(), 2) << "  " << std::left << note
                << "\n";
            first = false;
        }
    }
    out << std::string(92, '-') << "\n";
    out << "Weight is log2(m/u): the bits of evidence agreeing at that level carries.\n";
}

}  // namespace cpplink
