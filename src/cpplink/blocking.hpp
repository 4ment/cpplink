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

// One source's rows sorted by blocking key, with the group boundaries alongside.
// Materialised once per source so that threads take whole groups from an atomic
// counter instead of each rebuilding the sort.
struct SourceGroups {
    std::vector<std::pair<uint64_t, uint32_t>> keyed;  // (key, row), sorted
    std::vector<uint64_t> starts;                      // groups + 1 offsets into keyed

    uint64_t GroupCount() const { return starts.empty() ? 0 : starts.size() - 1; }
    uint64_t Size(uint64_t group) const { return starts[group + 1] - starts[group]; }
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

    // Every source index in plan order: the union the prediction path enumerates.
    std::vector<size_t> AllSources() const;
    // Whether a source listed before `position` in `selected` produces the pair.
    // An estimation session is a subset of the plan, so the earlier-source
    // predicate has to be relative to the subset, not to the whole plan.
    bool ProducedEarlierIn(const std::vector<size_t>& selected, size_t position,
                           uint64_t a, uint64_t b) const;

    // Sorts one source's rows by key and records the group boundaries.
    void BuildGroups(size_t source, SourceGroups* groups) const;

    // Enumerates the deduplicated union of `selected`, calling emit(a, b).
    template <typename Emit>
    void ForEachPairIn(const std::vector<size_t>& selected, Emit&& emit) const {
        SourceGroups groups;
        for (size_t position = 0; position < selected.size(); ++position) {
            const BoundSource& source = sources_[selected[position]];
            if (source.kind == SourceKind::kSortedNeighbourhood) {
                EnumerateWindowRange(selected, position, 0, source.order.size(), emit);
            } else {
                BuildGroups(selected[position], &groups);
                for (uint64_t g = 0; g < groups.GroupCount(); ++g) {
                    EnumerateGroupRange(selected, position, groups, g, 0, groups.Size(g),
                                        emit);
                }
            }
        }
    }

    // Enumerates the deduplicated union of every source.
    template <typename Emit>
    void ForEachPair(Emit&& emit) const {
        ForEachPairIn(AllSources(), emit);
    }

    // Rows [begin, end) of one group, paired with everything after them in it.
    // Splitting a group this way is what stops one thread holding "Smith" alone.
    template <typename Emit>
    void EnumerateGroupRange(const std::vector<size_t>& selected, size_t position,
                             const SourceGroups& groups, uint64_t group, uint64_t begin,
                             uint64_t end, Emit&& emit) const {
        const uint64_t first = groups.starts[group];
        const uint64_t last = groups.starts[group + 1];
        for (uint64_t i = first + begin; i < first + end; ++i) {
            const uint32_t a = groups.keyed[i].second;
            for (uint64_t j = i + 1; j < last; ++j) {
                const uint32_t b = groups.keyed[j].second;
                if (!ProducedEarlierIn(selected, position, a, b)) emit(a, b);
            }
        }
    }

    // Window starts [begin, end) of a sorted-neighbourhood source.
    template <typename Emit>
    void EnumerateWindowRange(const std::vector<size_t>& selected, size_t position,
                              uint64_t begin, uint64_t end, Emit&& emit) const {
        const BoundSource& source = sources_[selected[position]];
        const uint64_t count = source.order.size();
        if (count == 0) return;
        for (uint64_t i = begin; i < end; ++i) {
            const uint64_t last = std::min(i + source.window, count - 1);
            const uint32_t a = source.order[i];
            for (uint64_t j = i + 1; j <= last; ++j) {
                const uint32_t b = source.order[j];
                if (!ProducedEarlierIn(selected, position, a, b)) emit(a, b);
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

    std::vector<BoundSource> sources_;
    uint64_t rows_ = 0;
};

}  // namespace cpplink
