// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
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

}  // namespace cpplink
