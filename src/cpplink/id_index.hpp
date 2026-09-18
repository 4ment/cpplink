// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cpplink/record_store.hpp"

namespace cpplink {

// An id-to-row lookup over the store's id column: one `uint32` per record, kept
// in the order of the id it names. A merged prediction file carries `unique_id`s
// rather than row indices, so reading one back has to map them, and a hash map
// over 20M ids costs an order of magnitude more than the structures it feeds.
// This is 4 bytes a record and a handful of string compares a lookup.
class IdIndex {
   public:
    explicit IdIndex(const RecordStore& store) : ids_(store.ids()) {
        order_.resize(static_cast<size_t>(store.NumRecords()));
        for (size_t row = 0; row < order_.size(); ++row) {
            order_[row] = static_cast<uint32_t>(row);
        }
        std::sort(order_.begin(), order_.end(),
                  [this](uint32_t a, uint32_t b) { return ids_.Get(a) < ids_.Get(b); });
    }

    bool Find(std::string_view id, uint32_t* row) const {
        const auto at =
            std::lower_bound(order_.begin(), order_.end(), id,
                             [this](uint32_t candidate, std::string_view key) {
                                 return ids_.Get(candidate) < key;
                             });
        if (at == order_.end() || ids_.Get(*at) != id) return false;
        *row = *at;
        return true;
    }

   private:
    const IdColumn& ids_;
    std::vector<uint32_t> order_;
};

// What looking an id up can come back with. An id is unique within its input and
// not necessarily across inputs, so an unqualified id can name one record, none,
// or one per input that holds it -- and the last is not a hit on any of them.
enum class IdLookup {
    kFound,
    kMissing,
    kAmbiguous,  // more than one record answers to it; qualify it with its dataset
};

// An id-to-row lookup over the store's id column: one `uint32` per record, kept
// in the order of the id it names and, within an id, of the dataset. A merged
// prediction file names records by id rather than by row, so clustering one has
// to map them back, and a hash map over 20M ids costs an order of magnitude more
// than the union-find it feeds. This is 4 bytes a record and a handful of string
// compares an edge.
//
// Building it is also where duplicate ids are found: two records of one input
// with the same id are adjacent once sorted. Such an id names nothing, so the
// index is not to be read through until `Unique` has said there are none.
class IdIndex {
   public:
    explicit IdIndex(const RecordStore& store);

    // False, naming the id and its input, when an id occurs twice inside one
    // input. Across inputs a repeated id is allowed and is what `Shared` reports.
    bool Unique(std::string* error) const;
    // Whether some id occurs in more than one input, which is when an unqualified
    // lookup can be ambiguous and a prediction file has to carry the dataset.
    bool Shared() const { return shared_ > 0; }
    uint64_t shared_ids() const { return shared_; }

    // The record called `id` in dataset `dataset`, or in whichever dataset holds
    // it when `dataset` is kAnyDataset.
    IdLookup Find(size_t dataset, std::string_view id, uint32_t* row) const;
    IdLookup Find(std::string_view id, uint32_t* row) const {
        return Find(kAnyDataset, id, row);
    }
    // `dataset:id`, or a bare id, as the store's `SplitQualifiedId` reads it.
    IdLookup FindQualified(std::string_view text, uint32_t* row) const;

    uint64_t Size() const { return order_.size(); }

   private:
    const RecordStore& store_;
    std::vector<uint32_t> order_;
    uint64_t duplicates_ = 0;  // ids repeated within one input
    uint64_t shared_ = 0;      // ids held by more than one input
    std::string first_duplicate_;
    size_t first_duplicate_dataset_ = 0;
};

// The same question answered by one linear scan rather than a sort, for a
// command that resolves a single pair and would pay more to build the index
// than to read the column once. Reads `dataset:id` like `FindQualified`.
IdLookup FindRowById(const RecordStore& store, std::string_view text, uint64_t* row);

// How a qualified id is written: `dataset:id` when the store has more than one
// input, the id alone otherwise. This is the one spelling every report and
// output file uses.
std::string QualifiedId(const RecordStore& store, uint64_t row);

}  // namespace cpplink
