// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "cpplink/dictionary.hpp"

namespace cpplink {

// A 64-bit character-presence mask for one value: bit (c & 63) is set for every
// byte c the string contains.
//
// Folding 256 byte values onto 64 bits collides -- '0' lands on the same bit as
// 'p' -- but a collision only ever sets a bit that would otherwise be clear, which
// makes the bounds below weaker and never wrong. Real columns rarely mix digits
// with the second half of the lowercase alphabet, so the folding costs almost
// nothing in practice and costs one instruction to compute.
uint64_t CharacterMask(std::string_view value);

// Signatures for every value in a dictionary, in a parallel array indexed by
// value id. Twelve bytes a value buys the right to reject a pair without reading
// a single character of either string -- which is the whole point, because the
// characters are what is expensive.
class SignatureTable {
   public:
    void Build(const Dictionary& dict);

    bool Empty() const { return mask_.empty(); }
    uint64_t Mask(uint32_t id) const { return mask_[id]; }
    uint32_t Length(uint32_t id) const { return length_[id]; }
    uint64_t BytesUsed() const {
        return mask_.size() * sizeof(uint64_t) + length_.size() * sizeof(uint32_t);
    }

   private:
    std::vector<uint64_t> mask_;
    std::vector<uint32_t> length_;
};

}  // namespace cpplink
