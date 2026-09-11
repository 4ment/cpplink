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

// A value transform applied at load to build a derived column. Each has an input
// type and an output type, so a chain of them type-checks like a pipeline and the
// derived column's type is decided by the last one rather than declared.
enum class Transform {
    kNormalize,      // lowercase, and every byte that is not alphanumeric a space
    kSortedTokens,   // whitespace-separated tokens, sorted and rejoined
    kSoundex,        // the four-character American Soundex key
    kYear,           // a date's year
    kMonth,          // its month, zero-padded
    kDay,            // its day of the month, zero-padded
    kYearMonth,      // its year and month, as "1987-03"
    kEmailUsername,  // the part before the first "@", or the whole value without one
    kEmailDomain,    // the part after the last "@"; nothing without one
};

const char* TransformName(Transform transform);
bool ParseTransform(const std::string& name, Transform* transform);
ColumnType TransformInput(Transform transform);
ColumnType TransformOutput(Transform transform);

// A column computed at load from another column rather than read from the file.
//
// Interning is what makes this the cheap end of feature engineering: the
// transform runs once per distinct value of the source dictionary rather than
// once per row, so a phonetic key over 20M records costs one pass over 658k
// surnames and a gather. What it buys is an exact level -- an integer equality --
// sitting between the exact and the fuzzy level of the column it came from, and a
// blocking source that is EM-safe by the same single-column argument every other
// source here is.
//
// A derivation is a functional dependency the schema declares, which is why
// `SameSource` exists. The estimate's tie hold-out and the profile's anchor
// sessions have to treat a derived column and its source as one piece of
// evidence, and here that is knowledge rather than something a pairwise pass must
// rediscover from the rows and may refuse to.
struct DeriveSpec {
    std::string from;
    std::vector<Transform> transforms;  // applied in order

    std::string Describe() const;
};

struct ColumnSpec {
    std::string name;
    ColumnType type = ColumnType::kString;
    DeriveSpec derive;  // an empty transform list means the column is read

    bool IsDerived() const { return !derive.transforms.empty(); }
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
    // The two pairwise levels: the closest pair of elements over the cross
    // product of the two rows' lists, rather than the elements they share. A set
    // of email addresses agreeing up to a typo is evidence that an intersection
    // reads as no agreement at all, and the exact levels above are the special
    // case where the closest pair is a shared element.
    kListLevenshtein,  // some element pair is within threshold edits
    kListJaroWinkler,  // some element pair is at least this similar
    // The fuzzy half of kListContains, over the same two columns: one row's
    // scalar value against the elements of the other's list, in both directions.
    // It costs |a| + |b| metric evaluations where the pairwise levels above cost
    // |a| x |b|, because only one side of each comparison is a list.
    kContainsLevenshtein,  // some element is within threshold edits of the value
    kContainsJaroWinkler,  // some element is at least this similar to the value
    kElse,                 // always fires; must be last
};

const char* LevelTypeName(LevelType type);
bool ParseLevelType(const std::string& name, LevelType* type);

// A level reads the comparison's columns. Where the comparison names several
// string columns, a single-column level reads the one `column` indexes -- the
// first by default -- which is what lets one comparison rank an exact match on a
// field above an exact match on a key derived from it and a fuzzy match on
// either below both, as splink's email comparison does over the address and its
// username. Levels are ordered evidence about one field, so that has to be one
// comparison and not several: split up, "the addresses agree" and "the usernames
// agree" would be counted as independent evidence about the same characters.
struct LevelSpec {
    LevelType type = LevelType::kElse;
    double threshold = 0.0;
    std::string label;   // defaults to a description of type and threshold
    uint8_t column = 0;  // index into the comparison's columns

    std::string Describe() const;
};

// How a blocking source generates candidate pairs.
enum class SourceKind {
    kExactValue,           // every pair sharing a value
    kRareValue,            // pairs sharing a value seen at most max_frequency times
    kMinHash,              // MinHash LSH bands over character n-grams
    kSortedNeighbourhood,  // pairs within a window of a sorted order
    kAllPairs,             // every pair the mode admits: no blocking at all
};

const char* SourceKindName(SourceKind kind);
bool ParseSourceKind(const std::string& name, SourceKind* kind);

// A blocking source. Every source here selects on a single column, which is what
// makes it usable for estimating m: the selection event factors as a condition on
// that column, so the column is held fixed for the session and every other
// comparison stays identifiable. A source selecting on a whole record would not.
//
// `all_pairs` is the degenerate member and names no column: it selects on nothing,
// which factors on the empty column subset, so it holds nothing out and is the one
// source whose session estimates every m at once. It exists because on a small
// input blocking is a cost with no benefit -- in link mode especially, where the
// admissible space is the cross product rather than a triangle -- and because it
// is the reference a plan's pair completeness is measured against.
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
    // The columns a file is actually read for: everything but the derived ones.
    bool IsDerived(const std::string& name) const;
    // Total packed width in bits. Must fit in a uint32.
    uint8_t GammaWidth() const;
};

// Whether two columns are one piece of evidence by construction: either derives
// from the other, or both derive from the same column. Unlike containment or a
// u-side overlap this needs no rows to see and cannot be refused for want of
// them, which matters because a derived column's collision rate with its source
// is exactly the rate a file's own duplicates swamp.
bool SameSource(const Schema& schema, const std::string& a, const std::string& b);

bool ParseSchema(const std::string& json_text, Schema* schema, std::string* error);
bool LoadSchema(const std::string& path, Schema* schema, std::string* error);

}  // namespace cpplink
