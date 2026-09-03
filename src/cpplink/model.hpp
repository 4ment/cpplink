// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace cpplink {

// One comparison level's learned parameters. m and u are kept apart from the
// weight they imply, because scoring adjusts them per value for term frequency
// and needs the pieces, not the product.
struct ModelLevel {
    std::string label;
    double m = 0.0;
    double u = 0.0;
    bool u_exact = false;      // u came from the term frequencies, not from sampling
    bool m_estimated = false;  // at least one session learned m for this comparison
    uint64_t u_observed = 0;   // random pairs that landed here
    double m_support = 0.0;    // expected estimation pairs at this level

    // log2 Bayes factor: what agreeing at this level is worth as evidence.
    double Weight() const;
};

struct ModelComparison {
    std::string name;
    std::vector<std::string> columns;
    bool term_frequency = false;
    std::vector<ModelLevel> levels;
    size_t sessions = 0;  // estimation sessions that could learn m here
};

// The learned model: everything scoring needs and nothing it does not. lambda is
// the prior that two random records match, on the full-data pair count -- never on
// a blocked one, which is biased upward by construction.
struct Model {
    double lambda = 0.0;
    std::string lambda_basis;
    uint64_t records = 0;
    std::vector<ModelComparison> comparisons;

    // log2 lambda/(1 - lambda): the prior term of the match weight.
    double PriorWeight() const;
};

std::string ModelJson(const Model& model);
bool WriteModelJson(const Model& model, const std::string& path, std::string* error);
bool ParseModelJson(const std::string& text, Model* model, std::string* error);
bool LoadModel(const std::string& path, Model* model, std::string* error);

// The parameter table in the shape splink users read: one row per level, with the
// weight each level contributes.
void PrintModel(const Model& model, std::ostream& out);

}  // namespace cpplink
