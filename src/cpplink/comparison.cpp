// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/comparison.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "cpplink/string_metrics.hpp"

namespace cpplink {
namespace {

// Intersection size of two rows of a list column. Both are sorted and deduplicated
// at load, so this is a linear merge with no hashing and no allocation.
uint32_t OverlapSize(const StringListColumn& column, uint64_t a, uint64_t b) {
    uint64_t i = column.offsets[a];
    const uint64_t i_end = column.offsets[a + 1];
    uint64_t j = column.offsets[b];
    const uint64_t j_end = column.offsets[b + 1];
    uint32_t shared = 0;
    while (i < i_end && j < j_end) {
        const uint32_t left = column.ids[i];
        const uint32_t right = column.ids[j];
        if (left == right) {
            ++shared;
            ++i;
            ++j;
        } else if (left < right) {
            ++i;
        } else {
            ++j;
        }
    }
    return shared;
}

// Whether one row of a list column holds this value id. The cells are sorted at
// load, so the scan stops at the first id past the one sought and never runs the
// whole list.
bool ListContains(const StringListColumn& column, uint64_t row, uint32_t id) {
    if (id == kNullId) return false;
    const uint64_t end = column.offsets[row + 1];
    for (uint64_t i = column.offsets[row]; i < end; ++i) {
        if (column.ids[i] == id) return true;
        if (column.ids[i] > id) return false;
    }
    return false;
}

bool SameSet(const StringListColumn& column, uint64_t a, uint64_t b) {
    const uint64_t size_a = column.offsets[a + 1] - column.offsets[a];
    const uint64_t size_b = column.offsets[b + 1] - column.offsets[b];
    if (size_a != size_b) return false;
    return std::equal(column.ids.begin() + column.offsets[a],
                      column.ids.begin() + column.offsets[a + 1],
                      column.ids.begin() + column.offsets[b]);
}

}  // namespace

bool ComparisonSet::Bind(const Schema& schema, const RecordStore& store,
                         std::string* error, bool use_signatures) {
    bound_.clear();
    tables_.clear();
    width_ = 0;
    // Several comparisons can read the same column, and the signatures belong to
    // the column's values, so they are built once per dictionary and shared.
    std::vector<std::pair<const Dictionary*, SignatureTable*>> built;
    for (const ComparisonSpec& spec : schema.comparisons) {
        BoundComparison bound;
        bound.spec = &spec;
        bound.bits = spec.bits;
        bound.shift = spec.shift;

        for (size_t i = 0; i < spec.columns.size(); ++i) {
            size_t index = 0;
            bool found = false;
            for (size_t c = 0; c < schema.columns.size(); ++c) {
                if (schema.columns[c].name == spec.columns[i]) {
                    index = c;
                    found = true;
                    break;
                }
            }
            if (!found) {
                *error = "comparison \"" + spec.name + "\" names unknown column \"" +
                         spec.columns[i] + "\"";
                return false;
            }
            const Column& column = store.column(index);
            if (const auto* typed = std::get_if<StringColumn>(&column)) {
                bound.strings = typed;
            } else if (const auto* typed = std::get_if<DateColumn>(&column)) {
                bound.dates = typed;
            } else if (const auto* typed = std::get_if<StringListColumn>(&column)) {
                bound.lists = typed;
            } else if (const auto* typed = std::get_if<DoubleColumn>(&column)) {
                if (i == 0) {
                    bound.numbers = typed;
                } else {
                    bound.numbers2 = typed;
                }
            }
        }
        bool fuzzy = false;
        for (const LevelSpec& level : spec.levels) {
            if (level.type == LevelType::kLevenshtein ||
                level.type == LevelType::kJaroWinkler) {
                fuzzy = true;
                break;
            }
        }
        if (use_signatures && fuzzy && bound.strings != nullptr) {
            const Dictionary* dict = &bound.strings->dict;
            for (const auto& entry : built) {
                if (entry.first == dict) {
                    bound.signatures = entry.second;
                    break;
                }
            }
            if (bound.signatures == nullptr) {
                tables_.push_back(std::make_unique<SignatureTable>());
                tables_.back()->Build(*dict);
                bound.signatures = tables_.back().get();
                built.emplace_back(dict, tables_.back().get());
            }
        }
        bool contains = false;
        for (const LevelSpec& level : spec.levels) {
            if (level.type == LevelType::kListContains) {
                contains = true;
                break;
            }
        }
        if (contains && bound.strings != nullptr && bound.lists != nullptr) {
            // One pass over the list column's dictionary to index it, then one
            // over the scalar column's to translate it. Both are dictionaries,
            // not rows: aligning 150k surnames against 200k nicknames costs
            // nothing beside the pairs the result is then asked about.
            std::unordered_map<std::string_view, uint32_t> index;
            const Dictionary& target = bound.lists->dict;
            index.reserve(target.Size());
            for (uint32_t id = 0; id < target.Size(); ++id) {
                index.emplace(target.Value(id), id);
            }
            const Dictionary& source = bound.strings->dict;
            auto map = std::make_unique<std::vector<uint32_t>>(source.Size(), kNullId);
            for (uint32_t id = 0; id < source.Size(); ++id) {
                const auto found = index.find(source.Value(id));
                if (found != index.end()) (*map)[id] = found->second;
            }
            bound.alias_size = source.Size();
            alias_maps_.push_back(std::move(map));
            bound.alias_ids = alias_maps_.back()->data();
        }

        bound_.push_back(bound);
        width_ = static_cast<uint8_t>(width_ + bound.bits);
    }
    return true;
}

uint64_t ComparisonSet::SignatureBytes() const {
    uint64_t bytes = 0;
    for (const auto& table : tables_) bytes += table->BytesUsed();
    return bytes;
}

bool ComparisonSet::IsNull(const BoundComparison& comparison, uint64_t row) const {
    if (comparison.strings != nullptr && comparison.lists != nullptr) {
        // A list_contains comparison reads two columns and fires in either
        // direction, so a row only makes it unevaluable when it holds neither
        // part: with no value of its own and no list to be searched, no partner
        // can produce a match. Requiring both would be wrong in the other
        // direction -- it would let the null level pre-empt a level that fires.
        return comparison.strings->ids[row] == kNullId &&
               comparison.lists->offsets[row + 1] == comparison.lists->offsets[row];
    }
    if (comparison.strings != nullptr) {
        return comparison.strings->ids[row] == kNullId;
    }
    if (comparison.dates != nullptr) {
        return comparison.dates->values[row] == kNullDate;
    }
    if (comparison.lists != nullptr) {
        return comparison.lists->offsets[row + 1] == comparison.lists->offsets[row];
    }
    if (comparison.numbers != nullptr) {
        if (std::isnan(comparison.numbers->values[row])) return true;
        if (comparison.numbers2 != nullptr &&
            std::isnan(comparison.numbers2->values[row])) {
            return true;
        }
        return false;
    }
    return true;
}

bool ComparisonSet::StringLevelFires(const BoundComparison& comparison,
                                     const LevelSpec& level, uint32_t left,
                                     uint32_t right) const {
    switch (level.type) {
        case LevelType::kExact:
            // A null equals nothing, not even another null.
            return left != kNullId && left == right;

        case LevelType::kLevenshtein: {
            if (left == kNullId || right == kNullId) return false;
            if (left == right) return true;  // identical ids, distance zero
            const int limit = static_cast<int>(level.threshold);
            // Two loads and two popcounts, and the strings are never touched. On a
            // candidate set this rejects the great majority of pairs, which is the
            // only reason the fuzzy levels are affordable at all.
            if (comparison.signatures != nullptr &&
                LevenshteinLowerBound(comparison.signatures->Mask(left),
                                      comparison.signatures->Length(left),
                                      comparison.signatures->Mask(right),
                                      comparison.signatures->Length(right)) > limit) {
                return false;
            }
            return BoundedLevenshtein(comparison.strings->dict.Value(left),
                                      comparison.strings->dict.Value(right),
                                      limit) <= limit;
        }

        case LevelType::kJaroWinkler: {
            if (left == kNullId || right == kNullId) return false;
            if (left == right) return true;
            if (comparison.signatures != nullptr &&
                JaroWinklerUpperBound(comparison.signatures->Mask(left),
                                      comparison.signatures->Length(left),
                                      comparison.signatures->Mask(right),
                                      comparison.signatures->Length(right)) <
                    level.threshold) {
                return false;
            }
            return JaroWinklerAtLeast(comparison.strings->dict.Value(left),
                                      comparison.strings->dict.Value(right),
                                      level.threshold);
        }

        default:
            return false;
    }
}

uint8_t ComparisonSet::LevelForValues(size_t comparison, uint32_t left,
                                      uint32_t right) const {
    const BoundComparison& bound = bound_[comparison];
    const std::vector<LevelSpec>& levels = bound.spec->levels;
    for (size_t i = 0; i < levels.size(); ++i) {
        if (levels[i].type == LevelType::kNull) continue;  // not a value's business
        if (levels[i].type == LevelType::kElse) return static_cast<uint8_t>(i);
        if (StringLevelFires(bound, levels[i], left, right)) {
            return static_cast<uint8_t>(i);
        }
    }
    return static_cast<uint8_t>(levels.size() - 1);
}

bool ComparisonSet::LevelFires(const BoundComparison& comparison, const LevelSpec& level,
                               uint64_t a, uint64_t b) const {
    switch (level.type) {
        case LevelType::kNull:
            return IsNull(comparison, a) || IsNull(comparison, b);

        case LevelType::kElse:
            return true;

        case LevelType::kExact:
            if (comparison.strings != nullptr) {
                return StringLevelFires(comparison, level, comparison.strings->ids[a],
                                        comparison.strings->ids[b]);
            }
            if (comparison.dates != nullptr) {
                const int32_t left = comparison.dates->values[a];
                return left != kNullDate && left == comparison.dates->values[b];
            }
            if (comparison.lists != nullptr) return SameSet(*comparison.lists, a, b);
            return false;

        case LevelType::kLevenshtein:
        case LevelType::kJaroWinkler:
            return StringLevelFires(comparison, level, comparison.strings->ids[a],
                                    comparison.strings->ids[b]);

        case LevelType::kDateWithin: {
            const int32_t left = comparison.dates->values[a];
            const int32_t right = comparison.dates->values[b];
            if (left == kNullDate || right == kNullDate) return false;
            const int64_t difference =
                std::abs(static_cast<int64_t>(left) - static_cast<int64_t>(right));
            return static_cast<double>(difference) <= level.threshold;
        }

        case LevelType::kNumericWithin: {
            const double left = comparison.numbers->values[a];
            const double right = comparison.numbers->values[b];
            if (std::isnan(left) || std::isnan(right)) return false;
            return std::abs(left - right) <= level.threshold;
        }

        case LevelType::kGeoWithin: {
            const double lat_a = comparison.numbers->values[a];
            const double lat_b = comparison.numbers->values[b];
            const double lon_a = comparison.numbers2->values[a];
            const double lon_b = comparison.numbers2->values[b];
            if (std::isnan(lat_a) || std::isnan(lat_b) || std::isnan(lon_a) ||
                std::isnan(lon_b)) {
                return false;
            }
            return HaversineKm(lat_a, lon_a, lat_b, lon_b) <= level.threshold;
        }

        case LevelType::kListOverlap:
            return static_cast<double>(OverlapSize(*comparison.lists, a, b)) >=
                   level.threshold;

        case LevelType::kListJaccard: {
            const uint32_t shared = OverlapSize(*comparison.lists, a, b);
            const uint64_t size_a =
                comparison.lists->offsets[a + 1] - comparison.lists->offsets[a];
            const uint64_t size_b =
                comparison.lists->offsets[b + 1] - comparison.lists->offsets[b];
            const uint64_t together = size_a + size_b - shared;
            if (together == 0) return false;
            return static_cast<double>(shared) / static_cast<double>(together) >=
                   level.threshold;
        }

        case LevelType::kListContains: {
            // Both directions, because the relation is asymmetric: a nickname
            // list is a property of the row that owns it, and "Bill is one of
            // William's aliases" is evidence whichever row was drawn first.
            // Intersecting the two lists instead would agree on two rows that
            // share an alias without either being the other's name.
            if (comparison.alias_ids == nullptr) return false;
            const uint32_t left = comparison.strings->ids[a];
            const uint32_t right = comparison.strings->ids[b];
            if (left != kNullId && left < comparison.alias_size &&
                ListContains(*comparison.lists, b, comparison.alias_ids[left])) {
                return true;
            }
            return right != kNullId && right < comparison.alias_size &&
                   ListContains(*comparison.lists, a, comparison.alias_ids[right]);
        }
    }
    return false;
}

// The cheap half of `LevelFires`. Every case here either evaluates the level
// exactly, because that costs a load and a compare, or answers with an admissible
// bound that can only ever say "maybe" where the truth is "no".
bool ComparisonSet::LevelMaybe(const BoundComparison& comparison, const LevelSpec& level,
                               uint64_t a, uint64_t b) const {
    switch (level.type) {
        case LevelType::kNull:
        case LevelType::kElse:
        case LevelType::kExact:
        case LevelType::kDateWithin:
        case LevelType::kNumericWithin:
        case LevelType::kListContains:
            // Exact already, and cheaper than any bound would be: two integer
            // lookups and a walk over a cell that holds a handful of ids.
            return LevelFires(comparison, level, a, b);

        case LevelType::kLevenshtein: {
            const uint32_t left = comparison.strings->ids[a];
            const uint32_t right = comparison.strings->ids[b];
            if (left == kNullId || right == kNullId) return false;
            if (left == right) return true;
            if (comparison.signatures == nullptr) return true;
            return LevenshteinLowerBound(comparison.signatures->Mask(left),
                                         comparison.signatures->Length(left),
                                         comparison.signatures->Mask(right),
                                         comparison.signatures->Length(right)) <=
                   static_cast<int>(level.threshold);
        }

        case LevelType::kJaroWinkler: {
            const uint32_t left = comparison.strings->ids[a];
            const uint32_t right = comparison.strings->ids[b];
            if (left == kNullId || right == kNullId) return false;
            if (left == right) return true;
            if (comparison.signatures == nullptr) return true;
            return JaroWinklerUpperBound(comparison.signatures->Mask(left),
                                         comparison.signatures->Length(left),
                                         comparison.signatures->Mask(right),
                                         comparison.signatures->Length(right)) >=
                   level.threshold;
        }

        case LevelType::kGeoWithin: {
            const double lat_a = comparison.numbers->values[a];
            const double lat_b = comparison.numbers->values[b];
            const double lon_a = comparison.numbers2->values[a];
            const double lon_b = comparison.numbers2->values[b];
            if (std::isnan(lat_a) || std::isnan(lat_b) || std::isnan(lon_a) ||
                std::isnan(lon_b)) {
                return false;
            }
            // The latitude arc alone is a lower bound on the great-circle
            // distance, and it costs a subtract where the haversine costs four
            // trigonometric calls. The metre of slack absorbs rounding rather
            // than risking a bound that is wrong at the boundary.
            const double arc = kKmPerDegreeLatitude * std::abs(lat_a - lat_b);
            return arc <= level.threshold + 0.001;
        }

        case LevelType::kListOverlap: {
            const uint64_t size_a =
                comparison.lists->offsets[a + 1] - comparison.lists->offsets[a];
            const uint64_t size_b =
                comparison.lists->offsets[b + 1] - comparison.lists->offsets[b];
            // The intersection cannot be larger than the smaller set, and the
            // sizes are two subtractions off the offset array.
            return static_cast<double>(std::min(size_a, size_b)) >= level.threshold;
        }

        case LevelType::kListJaccard: {
            const uint64_t size_a =
                comparison.lists->offsets[a + 1] - comparison.lists->offsets[a];
            const uint64_t size_b =
                comparison.lists->offsets[b + 1] - comparison.lists->offsets[b];
            const uint64_t smaller = std::min(size_a, size_b);
            const uint64_t larger = std::max(size_a, size_b);
            if (larger == 0) return false;
            // shared <= smaller and union >= larger, so smaller/larger is an
            // upper bound on the Jaccard coefficient.
            return static_cast<double>(smaller) / static_cast<double>(larger) >=
                   level.threshold;
        }
    }
    return true;
}

bool ComparisonSet::LevelPossible(size_t comparison, size_t level, uint64_t a,
                                  uint64_t b) const {
    const BoundComparison& bound = bound_[comparison];
    return LevelMaybe(bound, bound.spec->levels[level], a, b);
}

uint8_t ComparisonSet::EvaluateOne(size_t comparison, uint64_t a, uint64_t b) const {
    const BoundComparison& bound = bound_[comparison];
    const std::vector<LevelSpec>& levels = bound.spec->levels;
    for (size_t i = 0; i < levels.size(); ++i) {
        if (LevelFires(bound, levels[i], a, b)) return static_cast<uint8_t>(i);
    }
    // Parsing guarantees a trailing "else", so this is unreachable in practice.
    return static_cast<uint8_t>(levels.size() - 1);
}

uint32_t ComparisonSet::Evaluate(uint64_t a, uint64_t b) const {
    uint32_t gamma = 0;
    for (size_t i = 0; i < bound_.size(); ++i) {
        gamma |= static_cast<uint32_t>(EvaluateOne(i, a, b)) << bound_[i].shift;
    }
    return gamma;
}

bool ComparisonSet::IsNullValue(size_t comparison, uint64_t row) const {
    return IsNull(bound_[comparison], row);
}

uint8_t ComparisonSet::LevelOf(uint32_t gamma, size_t comparison) const {
    const BoundComparison& bound = bound_[comparison];
    const uint32_t mask = (1u << bound.bits) - 1u;
    return static_cast<uint8_t>((gamma >> bound.shift) & mask);
}

}  // namespace cpplink
