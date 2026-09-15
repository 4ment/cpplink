// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/id_index.hpp"

#include <algorithm>
#include <string>
#include <string_view>

namespace cpplink {

IdIndex::IdIndex(const RecordStore& store) : store_(store) {
    const IdColumn& ids = store.ids();
    order_.resize(static_cast<size_t>(store.NumRecords()));
    for (size_t row = 0; row < order_.size(); ++row) {
        order_[row] = static_cast<uint32_t>(row);
    }
    // By id, then by dataset: the dataset is read only on an id tie, so a store
    // of one input never pays for it and one of several pays it rarely.
    std::sort(order_.begin(), order_.end(), [&](uint32_t a, uint32_t b) {
        const std::string_view id_a = ids.Get(a);
        const std::string_view id_b = ids.Get(b);
        if (id_a != id_b) return id_a < id_b;
        return store.DatasetOf(a) < store.DatasetOf(b);
    });
    // Repeats are adjacent now: the same id in the same input is a duplicate, the
    // same id in another input is shared. Each shared id is counted once.
    bool counted_shared = false;
    for (size_t i = 1; i < order_.size(); ++i) {
        const std::string_view previous = ids.Get(order_[i - 1]);
        const std::string_view current = ids.Get(order_[i]);
        if (previous != current) {
            counted_shared = false;
            continue;
        }
        const size_t before = store.DatasetOf(order_[i - 1]);
        const size_t here = store.DatasetOf(order_[i]);
        if (before == here) {
            if (duplicates_ == 0) {
                first_duplicate_ = std::string(current);
                first_duplicate_dataset_ = here;
            }
            ++duplicates_;
        } else if (!counted_shared) {
            ++shared_;
            counted_shared = true;
        }
    }
}

bool IdIndex::Unique(std::string* error) const {
    if (duplicates_ == 0) return true;
    *error = "the id \"" + first_duplicate_ + "\" names more than one record";
    if (store_.NumDatasets() > 1) {
        *error += " of input " + store_.DatasetName(first_duplicate_dataset_);
    }
    *error += " (" + std::to_string(duplicates_) + " such repeat" +
              (duplicates_ == 1 ? "" : "s") +
              " in all); ids must be unique within an input";
    return false;
}

IdLookup IdIndex::Find(size_t dataset, std::string_view id, uint32_t* row) const {
    const IdColumn& ids = store_.ids();
    const auto at = std::lower_bound(order_.begin(), order_.end(), id,
                                     [&](uint32_t candidate, std::string_view key) {
                                         return ids.Get(candidate) < key;
                                     });
    if (at == order_.end() || ids.Get(*at) != id) return IdLookup::kMissing;
    if (dataset == kAnyDataset) {
        const auto next = at + 1;
        if (next != order_.end() && ids.Get(*next) == id) return IdLookup::kAmbiguous;
        *row = *at;
        return IdLookup::kFound;
    }
    // Within an id the entries are in dataset order, so the wanted one is a short
    // walk; there are at most as many as there are inputs.
    for (auto it = at; it != order_.end() && ids.Get(*it) == id; ++it) {
        if (store_.DatasetOf(*it) != dataset) continue;
        const auto next = it + 1;
        if (next != order_.end() && ids.Get(*next) == id &&
            store_.DatasetOf(*next) == dataset) {
            return IdLookup::kAmbiguous;
        }
        *row = *it;
        return IdLookup::kFound;
    }
    return IdLookup::kMissing;
}

IdLookup IdIndex::FindQualified(std::string_view text, uint32_t* row) const {
    size_t dataset = kAnyDataset;
    std::string_view id;
    store_.SplitQualifiedId(text, &dataset, &id);
    return Find(dataset, id, row);
}

IdLookup FindRowById(const RecordStore& store, std::string_view text, uint64_t* row) {
    const IdColumn& ids = store.ids();
    if (ids.offsets.empty()) return IdLookup::kMissing;
    size_t dataset = kAnyDataset;
    std::string_view id;
    store.SplitQualifiedId(text, &dataset, &id);
    uint64_t matches = 0;
    for (uint64_t i = 0; i < store.NumRecords(); ++i) {
        if (ids.Get(i) != id) continue;
        if (dataset != kAnyDataset && store.DatasetOf(i) != dataset) continue;
        if (matches++ == 0) *row = i;
    }
    if (matches == 0) return IdLookup::kMissing;
    return matches == 1 ? IdLookup::kFound : IdLookup::kAmbiguous;
}

std::string QualifiedId(const RecordStore& store, uint64_t row) {
    std::string text;
    if (store.NumDatasets() > 1) {
        text = store.DatasetName(store.DatasetOf(row));
        text += ':';
    }
    text += store.ids().Get(row);
    return text;
}

}  // namespace cpplink
