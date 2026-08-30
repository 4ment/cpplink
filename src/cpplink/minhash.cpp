// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/minhash.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cpplink {

uint64_t Mix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

void Shingles(std::string_view value, size_t n, std::vector<uint64_t>* out) {
    out->clear();
    if (value.empty() || n == 0) return;
    if (value.size() <= n) {
        uint64_t hash = 1469598103934665603ull;
        for (char c : value) {
            hash = (hash ^ static_cast<unsigned char>(c)) * 1099511628211ull;
        }
        out->push_back(Mix64(hash));
        return;
    }
    out->reserve(value.size() - n + 1);
    for (size_t i = 0; i + n <= value.size(); ++i) {
        uint64_t hash = 1469598103934665603ull;
        for (size_t j = 0; j < n; ++j) {
            hash = (hash ^ static_cast<unsigned char>(value[i + j])) * 1099511628211ull;
        }
        out->push_back(Mix64(hash));
    }
    std::sort(out->begin(), out->end());
    out->erase(std::unique(out->begin(), out->end()), out->end());
}

void MinHashSignature(const std::vector<uint64_t>& shingles, uint64_t seed, size_t hashes,
                      uint32_t* signature) {
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    for (size_t h = 0; h < hashes; ++h) signature[h] = kMax;
    if (shingles.empty()) return;
    for (const uint64_t shingle : shingles) {
        for (size_t h = 0; h < hashes; ++h) {
            const uint32_t hashed =
                static_cast<uint32_t>(Mix64(shingle ^ Mix64(seed + h)) >> 32);
            if (hashed < signature[h]) signature[h] = hashed;
        }
    }
}

void BandKeys(const uint32_t* signature, size_t bands, size_t rows_per_band,
              uint64_t* keys) {
    for (size_t b = 0; b < bands; ++b) {
        uint64_t key = Mix64(b + 1);
        for (size_t r = 0; r < rows_per_band; ++r) {
            key = Mix64(key ^ signature[b * rows_per_band + r]);
        }
        keys[b] = key;
    }
}

double BandingRecall(double similarity, size_t bands, size_t rows_per_band) {
    const double band_hit = std::pow(similarity, static_cast<double>(rows_per_band));
    return 1.0 - std::pow(1.0 - band_hit, static_cast<double>(bands));
}

}  // namespace cpplink
