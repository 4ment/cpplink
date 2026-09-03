// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/spill.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cpplink {
namespace {

constexpr size_t kFlushBytes = 1u << 20;
constexpr const char* kManifestName = "spill.json";

}  // namespace

std::string GammaLayout(const ComparisonSet& comparisons) {
    std::string layout;
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        const BoundComparison& bound = comparisons.at(c);
        if (!layout.empty()) layout.push_back(',');
        layout.append(bound.spec->name);
        layout.push_back(':');
        layout.append(std::to_string(bound.spec->levels.size()));
    }
    return layout;
}

bool WriteSpillManifest(const std::string& dir, const SpillManifest& manifest,
                        std::string* error) {
    nlohmann::json json;
    json["threshold"] = manifest.threshold;
    json["sample_rate"] = manifest.sample_rate;
    json["records"] = manifest.records;
    json["candidates"] = manifest.candidates;
    json["spilled"] = manifest.spilled;
    json["above_threshold"] = manifest.above_threshold;
    json["gamma_width"] = manifest.gamma_width;
    json["layout"] = manifest.layout;

    const std::string path = (std::filesystem::path(dir) / kManifestName).string();
    std::ofstream file(path);
    if (!file) {
        *error = "cannot write \"" + path + "\"";
        return false;
    }
    file << json.dump(2) << "\n";
    file.close();
    if (file.fail()) {
        *error = "failed while writing \"" + path + "\"";
        return false;
    }
    return true;
}

bool ReadSpillManifest(const std::string& dir, SpillManifest* manifest,
                       std::string* error) {
    const std::string path = (std::filesystem::path(dir) / kManifestName).string();
    std::ifstream file(path);
    if (!file) {
        *error = "cannot read \"" + path + "\"; is that a spill directory?";
        return false;
    }
    nlohmann::json json;
    try {
        file >> json;
    } catch (const nlohmann::json::exception& problem) {
        *error = std::string("cannot parse \"") + path + "\": " + problem.what();
        return false;
    }
    try {
        manifest->threshold = json.at("threshold").get<double>();
        manifest->sample_rate = json.at("sample_rate").get<double>();
        manifest->records = json.at("records").get<uint64_t>();
        manifest->candidates = json.at("candidates").get<uint64_t>();
        manifest->spilled = json.at("spilled").get<uint64_t>();
        manifest->above_threshold = json.at("above_threshold").get<uint64_t>();
        manifest->gamma_width = json.at("gamma_width").get<uint8_t>();
        manifest->layout = json.at("layout").get<std::string>();
    } catch (const nlohmann::json::exception& problem) {
        *error = std::string("\"") + path + "\" is missing a field: " + problem.what();
        return false;
    }
    return true;
}

bool SpillWriter::Open(const std::string& path) {
    file_.open(path, std::ios::binary | std::ios::out);
    if (!file_) return false;
    file_.write(kSpillMagic, sizeof(kSpillMagic));
    return static_cast<bool>(file_);
}

void SpillWriter::Write(uint32_t a, uint32_t b, uint32_t gamma) {
    const size_t at = buffer_.size();
    buffer_.resize(at + kSpillBytes);
    char* out = buffer_.data() + at;
    std::memcpy(out, &a, 4);
    std::memcpy(out + 4, &b, 4);
    std::memcpy(out + 8, &gamma, 4);
    if (buffer_.size() >= kFlushBytes) Flush();
}

void SpillWriter::Flush() {
    if (buffer_.empty()) return;
    file_.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
    buffer_.clear();
}

bool SpillWriter::Close() {
    Flush();
    file_.close();
    return !file_.fail();
}

bool SpillReader::Open(const std::string& path, std::string* error) {
    path_ = path;
    file_.open(path, std::ios::binary);
    if (!file_) {
        *error = "cannot open \"" + path + "\"";
        return false;
    }
    char magic[sizeof(kSpillMagic)];
    file_.read(magic, sizeof(magic));
    if (file_.gcount() != static_cast<std::streamsize>(sizeof(magic)) ||
        std::memcmp(magic, kSpillMagic, sizeof(magic)) != 0) {
        *error = "\"" + path + "\" is not a cpplink spill shard";
        return false;
    }
    return true;
}

bool SpillReader::Next(uint32_t* out, size_t capacity, size_t* got, std::string* error) {
    *got = 0;
    if (!file_) return false;
    const size_t want = capacity / 3;
    buffer_.resize(want * kSpillBytes);
    file_.read(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
    const size_t read = static_cast<size_t>(file_.gcount());
    if (read % kSpillBytes != 0) {
        *error = "\"" + path_ + "\" is truncated mid-pair";
        return false;
    }
    const size_t pairs = read / kSpillBytes;
    for (size_t i = 0; i < pairs; ++i) {
        std::memcpy(out + i * 3, buffer_.data() + i * kSpillBytes, 12);
    }
    *got = pairs;
    return pairs > 0;
}

bool CollectSpillShards(const std::string& dir, std::vector<std::string>* shards,
                        std::string* error) {
    std::error_code code;
    if (!std::filesystem::is_directory(dir, code)) {
        *error = "'" + dir + "' is not a directory of spill shards";
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir, code)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("spill-", 0) == 0 && entry.path().extension() == ".bin") {
            shards->push_back(entry.path().string());
        }
    }
    if (code) {
        *error = "could not read '" + dir + "'";
        return false;
    }
    std::sort(shards->begin(), shards->end());
    if (shards->empty()) {
        *error = "no spill-*.bin files under '" + dir + "'";
        return false;
    }
    return true;
}

}  // namespace cpplink
