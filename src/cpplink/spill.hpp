// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"

namespace cpplink {

// The spill format: the pair and its agreement pattern, and nothing else. Twelve
// bytes is the whole point -- gamma is what the string metrics cost to produce, so
// keeping it turns re-scoring under new parameters into a linear read instead of a
// second comparison pass.
inline constexpr char kSpillMagic[8] = {'C', 'P', 'P', 'L', 'N', 'K', 'S', '1'};
inline constexpr size_t kSpillBytes = 12;  // a, b, gamma, all u32

// What the spilling run was, recorded beside the shards. Re-scoring a spill under a
// model that would have kept different pairs is only exact for the pairs the spill
// holds, so the reader needs to know what was thrown away.
struct SpillManifest {
    double threshold = 0.0;    // the bits the spilling run emitted at
    double sample_rate = 0.0;  // 0 when only above-threshold pairs were kept
    uint64_t records = 0;
    uint64_t candidates = 0;  // pairs the blocking plan produced
    uint64_t spilled = 0;
    uint64_t above_threshold = 0;
    uint8_t gamma_width = 0;
    // Comparison names and level counts, so a spill cannot be read back against a
    // schema whose gamma means something else.
    std::string layout;
};

// "name:levels,name:levels,..." -- the part of a schema that gives gamma its
// meaning. Two schemas agreeing here pack the same pattern the same way.
std::string GammaLayout(const ComparisonSet& comparisons);

bool WriteSpillManifest(const std::string& dir, const SpillManifest& manifest,
                        std::string* error);
bool ReadSpillManifest(const std::string& dir, SpillManifest* manifest,
                       std::string* error);

// One thread's spill buffer, flushed in bulk like the edge writer.
class SpillWriter {
   public:
    bool Open(const std::string& path);
    void Write(uint32_t a, uint32_t b, uint32_t gamma);
    bool Close();

   private:
    void Flush();

    std::ofstream file_;
    std::string buffer_;
};

// Reads one spill shard as a stream. Nothing holds the file: pairs are handed to
// the sink and forgotten, exactly as they were written.
class SpillReader {
   public:
    bool Open(const std::string& path, std::string* error);
    // Fills `pairs` with up to `capacity` triples; returns false at end of file or
    // on a malformed tail, which `error` distinguishes.
    bool Next(uint32_t* out, size_t capacity, size_t* got, std::string* error);

   private:
    std::string path_;
    std::ifstream file_;
    std::vector<char> buffer_;
};

// Every spill-*.bin under `dir`, sorted so a run is reproducible.
bool CollectSpillShards(const std::string& dir, std::vector<std::string>* shards,
                        std::string* error);

}  // namespace cpplink
