// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/comparison.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <variant>

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
                         std::string* error) {
    bound_.clear();
    width_ = 0;
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
        bound_.push_back(bound);
        width_ = static_cast<uint8_t>(width_ + bound.bits);
    }
    return true;
}

bool ComparisonSet::IsNull(const BoundComparison& comparison, uint64_t row) const {
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

bool ComparisonSet::LevelFires(const BoundComparison& comparison, const LevelSpec& level,
                               uint64_t a, uint64_t b) const {
    switch (level.type) {
        case LevelType::kNull:
            return IsNull(comparison, a) || IsNull(comparison, b);

        case LevelType::kElse:
            return true;

        case LevelType::kExact:
            if (comparison.strings != nullptr) {
                const uint32_t left = comparison.strings->ids[a];
                // A null equals nothing, not even another null.
                return left != kNullId && left == comparison.strings->ids[b];
            }
            if (comparison.dates != nullptr) {
                const int32_t left = comparison.dates->values[a];
                return left != kNullDate && left == comparison.dates->values[b];
            }
            if (comparison.lists != nullptr) return SameSet(*comparison.lists, a, b);
            return false;

        case LevelType::kLevenshtein: {
            const uint32_t left = comparison.strings->ids[a];
            const uint32_t right = comparison.strings->ids[b];
            if (left == kNullId || right == kNullId) return false;
            if (left == right) return true;  // identical ids, distance zero
            const int limit = static_cast<int>(level.threshold);
            return BoundedLevenshtein(comparison.strings->dict.Value(left),
                                      comparison.strings->dict.Value(right),
                                      limit) <= limit;
        }

        case LevelType::kJaroWinkler: {
            const uint32_t left = comparison.strings->ids[a];
            const uint32_t right = comparison.strings->ids[b];
            if (left == kNullId || right == kNullId) return false;
            if (left == right) return true;
            return JaroWinklerAtLeast(comparison.strings->dict.Value(left),
                                      comparison.strings->dict.Value(right),
                                      level.threshold);
        }

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
    }
    return false;
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
