// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/model.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/score.hpp"

namespace cpplink {

// Prints the packed layout: which comparison occupies which bits of the pattern.
void PrintGammaLayout(const ComparisonSet& comparisons, std::ostream& out);

// Prints, for one pair, the level each comparison assigns and the values behind it,
// then the packed pattern. This is how a configuration is debugged before it is
// pointed at billions of pairs.
void PrintPairExplanation(const RecordStore& store, const ComparisonSet& comparisons,
                          uint64_t a, uint64_t b, std::ostream& out);

// One comparison's row of the waterfall: the level it landed on, what it charged
// for it, and the values that put it there.
struct WaterfallStep {
    std::string name;
    uint8_t level = 0;
    std::string label;
    std::string value_a;
    std::string value_b;
    // The level's rates, where the model was given; NaN otherwise.
    double m = 0.0;
    double u = 0.0;
    double bits = 0.0;  // log2(m/u)
    double tf = 0.0;    // the term-frequency move, zero where the level has none
    // Records carrying the shared value, on an exact adjusted level; zero elsewhere.
    uint32_t frequency = 0;
    double running = 0.0;  // the total after this row
};

struct WaterfallInteraction {
    std::string name;
    double bits = 0.0;
    double running = 0.0;
};

// The match weight of one pair as a ledger: the prior, then each comparison's
// contribution in bits and its term-frequency move, then the two-way corrections,
// with a running total that ends at `Scorer::Weight`. The text report and the
// JSON one are both printed from this, so they cannot explain different sums.
//
// A pattern says which levels fired; it does not say why the pair scored what it
// scored. Two exact surname matches carry the same gamma and can differ by ten
// bits, because one of them is "Smith". This is the view that shows that.
struct PairWaterfall {
    uint64_t row_a = 0;
    uint64_t row_b = 0;
    std::string id_a;  // empty where the store carries no ids
    std::string id_b;
    uint64_t records = 0;
    uint32_t gamma = 0;
    double prior = 0.0;
    std::vector<WaterfallStep> steps;
    std::vector<WaterfallInteraction> interactions;
    double weight = 0.0;
    double probability = 0.0;
    double bracket_low = 0.0;  // what any pair with this pattern can score
    double bracket_high = 0.0;
    double threshold = 0.0;
    Zone zone = Zone::kCheck;
    bool emitted = false;
};

// `model` is optional and only supplies each level's m and u for the report; the
// bits come from the scorer either way.
PairWaterfall BuildPairWaterfall(const RecordStore& store,
                                 const ComparisonSet& comparisons, const Scorer& scorer,
                                 uint64_t a, uint64_t b, const Model* model = nullptr);

void PrintPairWaterfall(const PairWaterfall& waterfall, std::ostream& out);
void PrintPairWaterfall(const RecordStore& store, const ComparisonSet& comparisons,
                        const Scorer& scorer, uint64_t a, uint64_t b, std::ostream& out);

// The same ledger as one JSON object on one line, for a tool that draws it.
std::string PairWaterfallJson(const PairWaterfall& waterfall);

// Resolves a unique_id to a row by linear scan. There is no id index: ids are
// almost all distinct, so an index would cost as much as the values and is only
// ever needed for one-off lookups like this one.
bool FindRowById(const RecordStore& store, const std::string& id, uint64_t* row);

}  // namespace cpplink
