// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/dictionary.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace cpplink {

std::string_view Dictionary::Store(std::string_view value) {
    if (value.size() > kChunkBytes) {
        // Oversized values get a chunk to themselves rather than splitting.
        auto chunk = std::make_unique<char[]>(value.size());
        std::memcpy(chunk.get(), value.data(), value.size());
        std::string_view stored(chunk.get(), value.size());
        chunks_.push_back(std::move(chunk));
        return stored;
    }
    if (chunks_.empty() || chunk_used_ + value.size() > kChunkBytes) {
        chunks_.push_back(std::make_unique<char[]>(kChunkBytes));
        chunk_used_ = 0;
    }
    char* dest = chunks_.back().get() + chunk_used_;
    std::memcpy(dest, value.data(), value.size());
    chunk_used_ += value.size();
    return std::string_view(dest, value.size());
}

uint32_t Dictionary::Intern(std::string_view value) {
    auto it = index_.find(value);
    if (it != index_.end()) return it->second;

    const std::string_view stored = Store(value);
    const uint32_t id = static_cast<uint32_t>(entries_.size());
    entries_.push_back(stored);
    index_.emplace(stored, id);
    text_bytes_ += stored.size();
    return id;
}

uint64_t Dictionary::BytesUsed() const {
    uint64_t bytes = 0;
    // Chunks are fully allocated whether or not they are fully used.
    bytes += static_cast<uint64_t>(chunks_.size()) * kChunkBytes;
    bytes += static_cast<uint64_t>(entries_.capacity()) * sizeof(std::string_view);
    if (!index_.empty()) {
        // One node plus a bucket pointer per entry, approximately.
        bytes += static_cast<uint64_t>(index_.size()) *
                 (sizeof(std::pair<std::string_view, uint32_t>) + 2 * sizeof(void*));
        bytes += static_cast<uint64_t>(index_.bucket_count()) * sizeof(void*);
    }
    return bytes;
}

void Dictionary::ReleaseIndex() {
    std::unordered_map<std::string_view, uint32_t>().swap(index_);
}

}  // namespace cpplink
