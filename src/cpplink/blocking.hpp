// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// A row that no source key applies to: a null, or a value the source excludes.
// Two rows carrying kNoKey never become a candidate pair, which is what keeps
// nulls from blocking together.
inline constexpr uint64_t kNoKey = std::numeric_limits<uint64_t>::max();
inline constexpr uint32_t kNoRank = std::numeric_limits<uint32_t>::max();

// A blocking source bound to a loaded store. A MinHash spec expands into one
// source per band, because each band independently makes a pair a candidate.
struct BoundSource {
    SourceKind kind = SourceKind::kExactValue;
    std::string name;
    std::string column;
    size_t column_index = 0;
    // True when the source's selection event factors as a condition on its column,
    // so EM can hold that column fixed and estimate every other m without bias.
    bool em_safe = true;

    const StringColumn* strings = nullptr;
    const DateColumn* dates = nullptr;
    const StringListColumn* lists = nullptr;
    uint32_t max_frequency = 0;  // 0 means no cap

    std::vector<uint64_t> band_key;  // kMinHash: one key per distinct value id
    std::vector<uint32_t> rank;      // kSortedNeighbourhood: position per row
    std::vector<uint32_t> order;     // kSortedNeighbourhood: rows in sorted order
    uint32_t window = 0;
};

// The ordered union of sources. Order matters: a pair is emitted by the first
// source that produces it, so later sources pay a cheap predicate instead of the
// pipeline paying for a global deduplication pass.
class BlockingPlan {
   public:
    bool Build(const Schema& schema, const RecordStore& store, std::string* error);

    size_t Size() const { return sources_.size(); }
    const BoundSource& at(size_t index) const { return sources_[index]; }

    uint64_t KeyOf(size_t source, uint64_t row) const;
    bool Produces(size_t source, uint64_t a, uint64_t b) const;
    // Whether any source before `source` already produces the pair.
    bool ProducedEarlier(size_t source, uint64_t a, uint64_t b) const;
    // Whether any source at all produces the pair. This is what the recall harness
    // asks, and it costs O(sources) rather than an enumeration.
    bool ProducedByAny(uint64_t a, uint64_t b) const;

    // Exact candidate count for one source, from term frequencies alone: no sort,
    // no enumeration. Summing these over sources bounds the union from above.
    uint64_t CountPairs(size_t source) const;
    // The largest group the source would enumerate, in rows. A single huge group
    // is quadratic and is the usual reason a run never finishes.
    uint64_t LargestGroup(size_t source) const;

    // Enumerates the deduplicated union, calling emit(a, b) for each pair.
    template <typename Emit>
    void ForEachPair(Emit&& emit) const {
        std::vector<std::pair<uint64_t, uint32_t>> keyed;
        for (size_t s = 0; s < sources_.size(); ++s) {
            if (sources_[s].kind == SourceKind::kSortedNeighbourhood) {
                EnumerateWindow(s, emit);
            } else {
                EnumerateGroups(s, &keyed, emit);
            }
        }
    }

    uint64_t CountUnion() const;

   private:
    bool BuildValueSource(const BlockingSpec& spec, const RecordStore& store,
                          std::string* error);
    bool BuildMinHash(const BlockingSpec& spec, const RecordStore& store,
                      std::string* error);
    bool BuildSortedNeighbourhood(const BlockingSpec& spec, const RecordStore& store,
                                  std::string* error);
    const std::vector<uint32_t>* Frequencies(const BoundSource& source) const;

    template <typename Emit>
    void EnumerateGroups(size_t s, std::vector<std::pair<uint64_t, uint32_t>>* keyed,
                         Emit&& emit) const {
        keyed->clear();
        for (uint64_t row = 0; row < rows_; ++row) {
            const uint64_t key = KeyOf(s, row);
            if (key != kNoKey) keyed->emplace_back(key, static_cast<uint32_t>(row));
        }
        std::sort(keyed->begin(), keyed->end());
        size_t begin = 0;
        while (begin < keyed->size()) {
            size_t end = begin + 1;
            while (end < keyed->size() && (*keyed)[end].first == (*keyed)[begin].first) {
                ++end;
            }
            for (size_t i = begin; i < end; ++i) {
                for (size_t j = i + 1; j < end; ++j) {
                    const uint32_t a = (*keyed)[i].second;
                    const uint32_t b = (*keyed)[j].second;
                    if (!ProducedEarlier(s, a, b)) emit(a, b);
                }
            }
            begin = end;
        }
    }

    template <typename Emit>
    void EnumerateWindow(size_t s, Emit&& emit) const {
        const BoundSource& source = sources_[s];
        const size_t count = source.order.size();
        for (size_t i = 0; i < count; ++i) {
            const size_t last = std::min(i + source.window, count - 1);
            for (size_t j = i + 1; j <= last; ++j) {
                const uint32_t a = source.order[i];
                const uint32_t b = source.order[j];
                if (!ProducedEarlier(s, a, b)) emit(a, b);
            }
        }
    }

    std::vector<BoundSource> sources_;
    uint64_t rows_ = 0;
};

}  // namespace cpplink
