// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace cpplink {

// A 64-bit mixer, used both for shingle hashing and for deriving the independent
// hash functions a MinHash signature needs.
uint64_t Mix64(uint64_t value);

// Character n-grams of a string, hashed and deduplicated. A string shorter than
// `n` yields itself as a single shingle rather than nothing, so short values still
// participate.
void Shingles(std::string_view value, size_t n, std::vector<uint64_t>* out);

// MinHash signature: for each of `hashes` independent permutations, the smallest
// hashed shingle. Writes `hashes` entries. An empty shingle set yields all-max,
// which callers must treat as "no key" rather than as a value that can agree.
void MinHashSignature(const std::vector<uint64_t>& shingles, uint64_t seed, size_t hashes,
                      uint32_t* signature);

// Folds a signature into `bands` keys of `rows_per_band` entries each. Two records
// become candidates when any band key agrees, which gives the S-curve
// P(candidate) = 1 - (1 - s^rows)^bands for Jaccard similarity s.
void BandKeys(const uint32_t* signature, size_t bands, size_t rows_per_band,
              uint64_t* keys);

// The probability that a pair of the given Jaccard similarity becomes a candidate.
// Recall is a property of the configuration and can be computed before any run.
double BandingRecall(double similarity, size_t bands, size_t rows_per_band);

}  // namespace cpplink
