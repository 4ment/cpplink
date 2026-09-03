// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/signature.hpp"

namespace cpplink {

uint64_t CharacterMask(std::string_view value) {
    uint64_t mask = 0;
    for (const char ch : value) {
        mask |= uint64_t{1} << (static_cast<unsigned char>(ch) & 63u);
    }
    return mask;
}

void SignatureTable::Build(const Dictionary& dict) {
    const uint32_t size = dict.Size();
    mask_.resize(size);
    length_.resize(size);
    for (uint32_t id = 0; id < size; ++id) {
        const std::string_view value = dict.Value(id);
        mask_[id] = CharacterMask(value);
        length_[id] = static_cast<uint32_t>(value.size());
    }
}

}  // namespace cpplink
