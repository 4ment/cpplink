// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cpplink {

// A missing value. Nulls must never compare equal to one another, so this id is
// never handed out by a dictionary and never forms a blocking key.
inline constexpr uint32_t kNullId = 0xFFFFFFFFu;

// Maps string values to dense uint32 ids.
//
// Interning is what makes an exact-match comparison level a uint32 equality test
// rather than a string compare, which is the difference between minutes and a day
// over billions of pairs. Bytes live in stable chunks so the string_view keys held
// by the index stay valid as the arena grows.
class Dictionary {
   public:
    uint32_t Intern(std::string_view value);
    std::string_view Value(uint32_t id) const { return entries_[id]; }
    uint32_t Size() const { return static_cast<uint32_t>(entries_.size()); }

    // Total bytes of value text, excluding index and offset overhead.
    uint64_t TextBytes() const { return text_bytes_; }
    // Approximate resident bytes: arena chunks, the entry table and the index.
    uint64_t BytesUsed() const;
    // The index is only needed while loading; dropping it frees roughly half.
    void ReleaseIndex();

   private:
    static constexpr size_t kChunkBytes = 1u << 20;

    std::string_view Store(std::string_view value);

    std::vector<std::unique_ptr<char[]>> chunks_;
    size_t chunk_used_ = kChunkBytes;
    uint64_t text_bytes_ = 0;
    std::vector<std::string_view> entries_;
    std::unordered_map<std::string_view, uint32_t> index_;
};

}  // namespace cpplink
