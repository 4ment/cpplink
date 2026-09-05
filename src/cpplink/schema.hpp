// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cpplink {

// How a column is stored, which decides what can be asked of it later.
enum class ColumnType {
    kString,      // interned to a dense uint32 id
    kStringList,  // CSR of interned ids, sorted and deduplicated per row
    kDate,        // days since 1970-01-01
    kDouble,      // stored as-is; neither interned nor counted
};

const char* ColumnTypeName(ColumnType type);
bool ParseColumnType(const std::string& name, ColumnType* type);

// Term frequencies are kept only where exact agreement on a value is a discrete
// event worth counting. Two doubles agreeing to the last bit says nothing useful,
// so kDouble carries no counts and cannot drive rare-value blocking.
bool HasTermFrequencies(ColumnType type);

struct ColumnSpec {
    std::string name;
    ColumnType type = ColumnType::kString;
};

// What a comparison level tests. Levels are evaluated top-down, first hit wins,
// so ordering is the model: put the strongest evidence first.
enum class LevelType {
    kNull,           // any of the comparison's columns is missing on either side
    kExact,          // interned id or value equality
    kLevenshtein,    // edit distance <= threshold
    kJaroWinkler,    // similarity >= threshold
    kDateWithin,     // |difference| <= threshold days
    kNumericWithin,  // |difference| <= threshold
    kGeoWithin,      // great-circle distance <= threshold km
    kListOverlap,    // intersection size >= threshold
    kListJaccard,    // Jaccard similarity >= threshold
    kListContains,   // one row's scalar value is an element of the other's list
    kElse,           // always fires; must be last
};

const char* LevelTypeName(LevelType type);
bool ParseLevelType(const std::string& name, LevelType* type);

struct LevelSpec {
    LevelType type = LevelType::kElse;
    double threshold = 0.0;
    std::string label;  // defaults to a description of type and threshold

    std::string Describe() const;
};

// How a blocking source generates candidate pairs.
enum class SourceKind {
    kExactValue,           // every pair sharing a value
    kRareValue,            // pairs sharing a value seen at most max_frequency times
    kMinHash,              // MinHash LSH bands over character n-grams
    kSortedNeighbourhood,  // pairs within a window of a sorted order
};

const char* SourceKindName(SourceKind kind);
bool ParseSourceKind(const std::string& name, SourceKind* kind);

// A blocking source. Every source here selects on a single column, which is what
// makes it usable for estimating m: the selection event factors as a condition on
// that column, so the column is held fixed for the session and every other
// comparison stays identifiable. A source selecting on a whole record would not.
struct BlockingSpec {
    SourceKind kind = SourceKind::kExactValue;
    std::string name;
    std::string column;
    uint32_t max_frequency = 100;  // kRareValue
    uint32_t window = 8;           // kSortedNeighbourhood
    uint32_t bands = 12;           // kMinHash
    uint32_t rows_per_band = 4;
    uint32_t ngram = 3;
    uint64_t seed = 1;

    std::string Describe() const;
};

// One comparison over one logical field, which may span more than one column:
// a coordinate pair is a single comparison over latitude and longitude, not two
// independent ones.
struct ComparisonSpec {
    std::string name;
    std::vector<std::string> columns;
    std::vector<LevelSpec> levels;
    bool term_frequency = false;  // read in a later phase; parsed here

    // Bits needed to hold a level index, and the offset into the packed gamma.
    uint8_t bits = 0;
    uint8_t shift = 0;
};

// The data description: what the columns are and how they are stored. How to
// compare them is a separate concern and is not read here.
struct Schema {
    std::string unique_id;  // optional; empty means the row index is the id
    std::vector<ColumnSpec> columns;
    std::vector<ComparisonSpec> comparisons;  // optional until phase 1 is used
    std::vector<BlockingSpec> blocking;       // optional until phase 2 is used

    const ColumnSpec* Find(const std::string& name) const;
    // Total packed width in bits. Must fit in a uint32.
    uint8_t GammaWidth() const;
};

bool ParseSchema(const std::string& json_text, Schema* schema, std::string* error);
bool LoadSchema(const std::string& path, Schema* schema, std::string* error);

}  // namespace cpplink
