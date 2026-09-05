// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/blocking.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "cpplink/minhash.hpp"

namespace cpplink {
namespace {

// Dates are widened into the key space with an offset, so that no real date can
// collide with the kNoKey sentinel.
uint64_t DateKey(int32_t value) {
    return static_cast<uint64_t>(static_cast<int64_t>(value) - kNullDate);
}

uint64_t PairsIn(uint64_t group) { return group * (group - 1) / 2; }

}  // namespace

const std::vector<uint32_t>* BlockingPlan::Frequencies(const BoundSource& source) const {
    if (source.strings != nullptr) return &source.strings->tf;
    if (source.lists != nullptr) return &source.lists->tf;
    if (source.dates != nullptr) return &source.dates->tf;
    return nullptr;
}

uint64_t BlockingPlan::KeyOf(size_t source_index, uint64_t row) const {
    const BoundSource& source = sources_[source_index];
    if (source.kind == SourceKind::kSortedNeighbourhood) return kNoKey;

    if (source.strings != nullptr) {
        const uint32_t id = source.strings->ids[row];
        if (id == kNullId) return kNoKey;
        if (source.kind == SourceKind::kMinHash) return source.band_key[id];
        const uint32_t frequency = source.strings->tf[id];
        // A singleton can pair with nothing, and a value past the cap is excluded.
        if (frequency < 2) return kNoKey;
        if (source.max_frequency != 0 && frequency > source.max_frequency) {
            return kNoKey;
        }
        return id;
    }
    if (source.dates != nullptr) {
        const int32_t value = source.dates->values[row];
        if (value == kNullDate) return kNoKey;
        const size_t bucket = static_cast<size_t>(value - source.dates->tf_origin);
        const uint32_t frequency = source.dates->tf[bucket];
        if (frequency < 2) return kNoKey;
        if (source.max_frequency != 0 && frequency > source.max_frequency) {
            return kNoKey;
        }
        return DateKey(value);
    }
    return kNoKey;
}

bool BlockingPlan::Produces(size_t source_index, uint64_t a, uint64_t b) const {
    const BoundSource& source = sources_[source_index];
    if (a == b) return false;
    // In link mode a pair inside one input is not a candidate at all, so no source
    // produces it. Putting the gate here rather than in the enumerators is what
    // makes the recall harness and the earlier-source predicate agree with what the
    // stream actually emits.
    if (mode_ == PairMode::kCrossDataset && !CrossDataset(a, b)) return false;
    if (source.kind == SourceKind::kSortedNeighbourhood) {
        const uint32_t rank_a = source.rank[a];
        const uint32_t rank_b = source.rank[b];
        if (rank_a == kNoRank || rank_b == kNoRank) return false;
        const uint32_t gap = rank_a > rank_b ? rank_a - rank_b : rank_b - rank_a;
        return gap <= source.window;
    }
    const uint64_t key = KeyOf(source_index, a);
    return key != kNoKey && key == KeyOf(source_index, b);
}

bool BlockingPlan::ProducedEarlier(size_t source_index, uint64_t a, uint64_t b) const {
    for (size_t s = 0; s < source_index; ++s) {
        if (Produces(s, a, b)) return true;
    }
    return false;
}

bool BlockingPlan::ProducedByAny(uint64_t a, uint64_t b) const {
    for (size_t s = 0; s < sources_.size(); ++s) {
        if (Produces(s, a, b)) return true;
    }
    return false;
}

uint64_t BlockingPlan::CountPairs(size_t source_index) const {
    const BoundSource& source = sources_[source_index];
    if (mode_ == PairMode::kCrossDataset) return CountCrossPairs(source_index);

    if (source.kind == SourceKind::kSortedNeighbourhood) {
        const uint64_t count = source.order.size();
        if (count < 2) return 0;
        const uint64_t window = std::min<uint64_t>(source.window, count - 1);
        // Full windows for the first count - window rows, then a tail shrinking
        // from window - 1 down to zero.
        return (count - window) * window + PairsIn(window);
    }

    if (source.kind == SourceKind::kMinHash) {
        // Band keys group values, so counts accumulate rows per band key.
        std::unordered_map<uint64_t, uint64_t> rows_per_key;
        const std::vector<uint32_t>& tf = source.strings->tf;
        for (uint32_t id = 0; id < tf.size(); ++id) {
            const uint64_t key = source.band_key[id];
            if (key == kNoKey) continue;
            rows_per_key[key] += tf[id];
        }
        uint64_t pairs = 0;
        for (const auto& entry : rows_per_key) pairs += PairsIn(entry.second);
        return pairs;
    }

    // Exact and rare value: the count comes from the term frequencies alone.
    const std::vector<uint32_t>* tf = Frequencies(source);
    if (tf == nullptr) return 0;
    uint64_t pairs = 0;
    for (const uint32_t frequency : *tf) {
        if (frequency < 2) continue;
        if (source.max_frequency != 0 && frequency > source.max_frequency) continue;
        pairs += PairsIn(frequency);
    }
    return pairs;
}

// The exact cross-dataset candidate count.
//
// Term frequencies pool the inputs -- they say a value occurs n times, not how
// those n split between the files -- so the closed form the dedup path uses is not
// available here. What is available is still not an enumeration: a group of size n
// splitting n_d per input contributes (n^2 - sum n_d^2) / 2, which one walk over
// the grouped rows adds up. It costs a sort, and the equivalence with enumeration
// is a test either way.
uint64_t BlockingPlan::CountCrossPairs(size_t source_index) const {
    const BoundSource& source = sources_[source_index];
    const size_t datasets = NumDatasets();
    if (datasets < 2) return 0;

    if (source.kind == SourceKind::kSortedNeighbourhood) {
        // A window is not a group, so this counts the window's rows per input with
        // a prefix sum -- linear in rows rather than in rows times window.
        const uint64_t count = source.order.size();
        if (count < 2) return 0;
        std::vector<uint64_t> prefix((count + 1) * datasets, 0);
        for (uint64_t i = 0; i < count; ++i) {
            for (size_t d = 0; d < datasets; ++d) {
                prefix[(i + 1) * datasets + d] = prefix[i * datasets + d];
            }
            ++prefix[(i + 1) * datasets + DatasetOf(source.order[i])];
        }
        uint64_t pairs = 0;
        for (uint64_t i = 0; i < count; ++i) {
            const uint64_t last = std::min(i + source.window, count - 1);
            const size_t own = DatasetOf(source.order[i]);
            const uint64_t span = last - i;
            const uint64_t same =
                prefix[(last + 1) * datasets + own] - prefix[(i + 1) * datasets + own];
            pairs += span - same;
        }
        return pairs;
    }

    SourceGroups groups;
    BuildGroups(source_index, &groups);
    std::vector<uint64_t> per_dataset(datasets, 0);
    uint64_t pairs = 0;
    for (uint64_t g = 0; g < groups.GroupCount(); ++g) {
        std::fill(per_dataset.begin(), per_dataset.end(), 0);
        const uint64_t first = groups.starts[g];
        const uint64_t last = groups.starts[g + 1];
        for (uint64_t i = first; i < last; ++i) {
            ++per_dataset[DatasetOf(groups.keyed[i].second)];
        }
        const uint64_t size = last - first;
        uint64_t within = 0;
        for (const uint64_t count : per_dataset) within += count * count;
        pairs += (size * size - within) / 2;
    }
    return pairs;
}

uint64_t BlockingPlan::LargestGroup(size_t source_index) const {
    const BoundSource& source = sources_[source_index];
    if (source.kind == SourceKind::kSortedNeighbourhood) {
        return std::min<uint64_t>(source.window + 1, source.order.size());
    }
    if (source.kind == SourceKind::kMinHash) {
        std::unordered_map<uint64_t, uint64_t> rows_per_key;
        const std::vector<uint32_t>& tf = source.strings->tf;
        for (uint32_t id = 0; id < tf.size(); ++id) {
            const uint64_t key = source.band_key[id];
            if (key == kNoKey) continue;
            rows_per_key[key] += tf[id];
        }
        uint64_t largest = 0;
        for (const auto& entry : rows_per_key) {
            largest = std::max(largest, entry.second);
        }
        return largest;
    }
    const std::vector<uint32_t>* tf = Frequencies(source);
    if (tf == nullptr) return 0;
    uint64_t largest = 0;
    for (const uint32_t frequency : *tf) {
        if (frequency < 2) continue;
        if (source.max_frequency != 0 && frequency > source.max_frequency) continue;
        largest = std::max<uint64_t>(largest, frequency);
    }
    return largest;
}

std::vector<size_t> BlockingPlan::AllSources() const {
    std::vector<size_t> selected(sources_.size());
    for (size_t s = 0; s < sources_.size(); ++s) selected[s] = s;
    return selected;
}

bool BlockingPlan::ProducedEarlierIn(const std::vector<size_t>& selected, size_t position,
                                     uint64_t a, uint64_t b) const {
    for (size_t p = 0; p < position; ++p) {
        if (Produces(selected[p], a, b)) return true;
    }
    return false;
}

void BlockingPlan::BuildGroups(size_t source_index, SourceGroups* groups) const {
    groups->keyed.clear();
    groups->starts.clear();
    for (uint64_t row = 0; row < rows_; ++row) {
        const uint64_t key = KeyOf(source_index, row);
        if (key != kNoKey) {
            groups->keyed.emplace_back(key, static_cast<uint32_t>(row));
        }
    }
    std::sort(groups->keyed.begin(), groups->keyed.end());

    // Singletons are compacted out rather than recorded and skipped: a group of
    // one produces no pair, and dropping them keeps the groups contiguous, so a
    // group is a range of `keyed` and nothing else has to be carried alongside.
    uint64_t write = 0;
    uint64_t begin = 0;
    while (begin < groups->keyed.size()) {
        uint64_t end = begin + 1;
        while (end < groups->keyed.size() &&
               groups->keyed[end].first == groups->keyed[begin].first) {
            ++end;
        }
        if (end - begin > 1) {
            if (groups->starts.empty()) groups->starts.push_back(0);
            for (uint64_t i = begin; i < end; ++i) {
                groups->keyed[write++] = groups->keyed[i];
            }
            groups->starts.push_back(write);
        }
        begin = end;
    }
    groups->keyed.resize(write);
}

uint64_t BlockingPlan::CountUnion() const {
    uint64_t pairs = 0;
    ForEachPair([&pairs](uint32_t, uint32_t) { ++pairs; });
    return pairs;
}

bool BlockingPlan::BuildValueSource(const BlockingSpec& spec, const RecordStore& store,
                                    std::string* error) {
    BoundSource source;
    source.kind = spec.kind;
    source.name = spec.name;
    source.column = spec.column;
    source.max_frequency = spec.kind == SourceKind::kRareValue ? spec.max_frequency : 0;

    for (size_t i = 0; i < store.schema().columns.size(); ++i) {
        if (store.schema().columns[i].name != spec.column) continue;
        source.column_index = i;
        const Column& column = store.column(i);
        if (const auto* typed = std::get_if<StringColumn>(&column)) {
            source.strings = typed;
        } else if (const auto* typed = std::get_if<DateColumn>(&column)) {
            source.dates = typed;
        } else if (const auto* typed = std::get_if<StringListColumn>(&column)) {
            source.lists = typed;
        }
        break;
    }
    if (source.strings == nullptr && source.dates == nullptr) {
        *error = "blocking source \"" + spec.name + "\" cannot block on column \"" +
                 spec.column + "\"";
        return false;
    }
    sources_.push_back(std::move(source));
    return true;
}

bool BlockingPlan::BuildMinHash(const BlockingSpec& spec, const RecordStore& store,
                                std::string* error) {
    const StringColumn* strings = nullptr;
    size_t column_index = 0;
    for (size_t i = 0; i < store.schema().columns.size(); ++i) {
        if (store.schema().columns[i].name != spec.column) continue;
        column_index = i;
        strings = std::get_if<StringColumn>(&store.column(i));
        break;
    }
    if (strings == nullptr) {
        *error = "minhash blocking needs a string column, but \"" + spec.column +
                 "\" is not one";
        return false;
    }

    // Signatures are computed per distinct value, not per row: for a Zipf column
    // that is far fewer, and the band key of a row is just its value's band key.
    const uint32_t values = strings->dict.Size();
    const size_t hashes = static_cast<size_t>(spec.bands) * spec.rows_per_band;
    std::vector<std::vector<uint64_t>> keys(spec.bands,
                                            std::vector<uint64_t>(values, kNoKey));

    std::vector<uint32_t> signature(hashes);
    std::vector<uint64_t> band(spec.bands);
    std::vector<uint64_t> shingles;
    for (uint32_t id = 0; id < values; ++id) {
        Shingles(strings->dict.Value(id), spec.ngram, &shingles);
        if (shingles.empty()) continue;  // stays kNoKey: an empty value blocks nothing
        MinHashSignature(shingles, spec.seed, hashes, signature.data());
        BandKeys(signature.data(), spec.bands, spec.rows_per_band, band.data());
        for (uint32_t b = 0; b < spec.bands; ++b) {
            // Reserve the sentinel: a genuine key that collides with it is shifted.
            keys[b][id] = band[b] == kNoKey ? band[b] - 1 : band[b];
        }
    }

    for (uint32_t b = 0; b < spec.bands; ++b) {
        BoundSource source;
        source.kind = SourceKind::kMinHash;
        source.name = spec.name + " band " + std::to_string(b);
        source.column = spec.column;
        source.column_index = column_index;
        source.strings = strings;
        source.band_key = std::move(keys[b]);
        sources_.push_back(std::move(source));
    }
    return true;
}

bool BlockingPlan::BuildSortedNeighbourhood(const BlockingSpec& spec,
                                            const RecordStore& store,
                                            std::string* error) {
    const StringColumn* strings = nullptr;
    const DateColumn* dates = nullptr;
    size_t column_index = 0;
    for (size_t i = 0; i < store.schema().columns.size(); ++i) {
        if (store.schema().columns[i].name != spec.column) continue;
        column_index = i;
        strings = std::get_if<StringColumn>(&store.column(i));
        dates = std::get_if<DateColumn>(&store.column(i));
        break;
    }
    if (strings == nullptr && dates == nullptr) {
        *error =
            "sorted_neighbourhood blocking cannot sort column \"" + spec.column + "\"";
        return false;
    }

    BoundSource source;
    source.kind = SourceKind::kSortedNeighbourhood;
    source.name = spec.name;
    source.column = spec.column;
    source.column_index = column_index;
    source.strings = strings;
    source.dates = dates;
    source.window = spec.window;

    // Values are ranked first, then rows are placed by their value's rank: sorting
    // distinct values costs far less than sorting every row by text.
    std::vector<uint64_t> value_rank;
    if (strings != nullptr) {
        const uint32_t values = strings->dict.Size();
        std::vector<uint32_t> by_text(values);
        for (uint32_t id = 0; id < values; ++id) by_text[id] = id;
        std::sort(by_text.begin(), by_text.end(),
                  [strings](uint32_t left, uint32_t right) {
                      return strings->dict.Value(left) < strings->dict.Value(right);
                  });
        value_rank.assign(values, 0);
        for (uint32_t i = 0; i < values; ++i) value_rank[by_text[i]] = i;
    }

    std::vector<std::pair<uint64_t, uint32_t>> keyed;
    keyed.reserve(store.NumRecords());
    for (uint64_t row = 0; row < store.NumRecords(); ++row) {
        if (strings != nullptr) {
            const uint32_t id = strings->ids[row];
            if (id == kNullId) continue;
            keyed.emplace_back(value_rank[id], static_cast<uint32_t>(row));
        } else {
            const int32_t value = dates->values[row];
            if (value == kNullDate) continue;
            keyed.emplace_back(DateKey(value), static_cast<uint32_t>(row));
        }
    }
    std::sort(keyed.begin(), keyed.end());

    source.rank.assign(store.NumRecords(), kNoRank);
    source.order.resize(keyed.size());
    for (size_t i = 0; i < keyed.size(); ++i) {
        source.order[i] = keyed[i].second;
        source.rank[keyed[i].second] = static_cast<uint32_t>(i);
    }
    sources_.push_back(std::move(source));
    return true;
}

bool BlockingPlan::Build(const Schema& schema, const RecordStore& store,
                         std::string* error) {
    return Build(schema, store, PairMode::kAll, error);
}

bool BlockingPlan::Build(const Schema& schema, const RecordStore& store, PairMode mode,
                         std::string* error) {
    sources_.clear();
    mode_ = mode;
    dataset_starts_ = store.dataset_starts();
    if (mode == PairMode::kCrossDataset && store.NumDatasets() < 2) {
        *error = "link mode needs at least two inputs; one file was given";
        return false;
    }
    rows_ = store.NumRecords();
    for (const BlockingSpec& spec : schema.blocking) {
        bool ok = false;
        switch (spec.kind) {
            case SourceKind::kExactValue:
            case SourceKind::kRareValue:
                ok = BuildValueSource(spec, store, error);
                break;
            case SourceKind::kMinHash:
                ok = BuildMinHash(spec, store, error);
                break;
            case SourceKind::kSortedNeighbourhood:
                ok = BuildSortedNeighbourhood(spec, store, error);
                break;
        }
        if (!ok) return false;
    }
    return true;
}

}  // namespace cpplink
