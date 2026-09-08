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

// One fitted two-way interaction: the bits the independent model over- or
// under-counts when these two comparisons land on this pair of levels.
//
// Fellegi-Sunter multiplies m and u across comparisons, which is right only where
// agreement is conditionally independent given the class. Where it is not, the
// correction is the log ratio of the joint to the product of its margins under M,
// less the same quantity under U -- and it is a function of gamma alone, so it
// costs the scorer nothing but a table lookup and costs the tabulated path
// nothing at all.
struct ModelInteraction {
    std::string left;  // comparison names, in schema order
    std::string right;
    uint8_t left_levels = 0;
    uint8_t right_levels = 0;
    std::vector<double> bits;  // left_levels x right_levels, row-major
    double match_bits = 0.0;   // what it moves an average matching pair by
    double effect = 0.0;       // mass-weighted mean |bits| under M
    size_t sessions = 0;       // sessions that left both comparisons free

    double Bits(uint8_t left_level, uint8_t right_level) const;
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
    // Empty unless the run asked for them. An interaction is a correction to the
    // weight and never to m or u, which stay exactly what the margins say.
    std::vector<ModelInteraction> interactions;

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

// The fitted two-way corrections, if the run asked for any. Printed by PrintModel
// and separately by the report that has to show them beside their candidates.
void PrintInteractions(const Model& model, std::ostream& out);

}  // namespace cpplink
