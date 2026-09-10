// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/predict.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cpplink/merge_edges.hpp"
#include "cpplink/pair_stream.hpp"
#include "cpplink/spill.hpp"

namespace cpplink {
namespace {

constexpr size_t kFlushBytes = 1u << 20;

std::string WithThousands(uint64_t value) {
    std::string digits = std::to_string(value);
    std::string out;
    int count = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (count > 0 && count % 3 == 0) out.push_back(',');
        out.push_back(*it);
        ++count;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

std::string Percent(uint64_t part, uint64_t whole) {
    if (whole == 0) return "-";
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f%%",
                  100.0 * static_cast<double>(part) / static_cast<double>(whole));
    return buffer;
}

uint64_t Mix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

// Per-thread counters, one cache line each: they are touched once per candidate
// pair, and packing them adjacently would put every thread's increment on the
// same line and cost more than the scoring being counted.
struct alignas(64) ThreadTally {
    uint64_t enumerated = 0;
    uint64_t skipped = 0;
    uint64_t dropped = 0;
    uint64_t checked = 0;
    uint64_t certain = 0;
    uint64_t edges = 0;
    uint64_t spilled = 0;
    uint64_t random = 0;
    uint64_t tf_lookups = 0;
};

}  // namespace

std::string EdgeShardName(unsigned thread) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "shard-%03u", thread);
    return buffer;
}

bool EdgeShardWriter::Open(const std::string& path, EdgeFormat format) {
    format_ = format;
    file_.open(path, format == EdgeFormat::kBinary ? std::ios::binary | std::ios::out
                                                   : std::ios::out);
    if (!file_) return false;
    if (format == EdgeFormat::kBinary) {
        file_.write(kEdgeMagic, sizeof(kEdgeMagic));
    } else {
        buffer_ = "id_a,id_b,gamma,match_weight,match_probability\n";
    }
    return static_cast<bool>(file_);
}

void EdgeShardWriter::WriteBinary(uint32_t a, uint32_t b, uint32_t gamma, double weight) {
    const size_t at = buffer_.size();
    buffer_.resize(at + kEdgeBytes);
    char* out = buffer_.data() + at;
    std::memcpy(out, &a, 4);
    std::memcpy(out + 4, &b, 4);
    std::memcpy(out + 8, &gamma, 4);
    std::memcpy(out + 12, &weight, 8);
    if (buffer_.size() >= kFlushBytes) Flush();
}

void EdgeShardWriter::WriteCsv(std::string_view id_a, std::string_view id_b,
                               uint32_t gamma, double weight) {
    char numbers[64];
    std::snprintf(numbers, sizeof(numbers), ",%u,%.6f,%.9f", gamma, weight,
                  ProbabilityForWeight(weight));
    buffer_.append(id_a);
    buffer_.push_back(',');
    buffer_.append(id_b);
    buffer_.append(numbers);
    buffer_.push_back('\n');
    if (buffer_.size() >= kFlushBytes) Flush();
}

void EdgeShardWriter::Flush() {
    if (buffer_.empty()) return;
    file_.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
    buffer_.clear();
}

bool EdgeShardWriter::Close() {
    Flush();
    file_.close();
    return !file_.fail();
}

bool Predict(const RecordStore& store, const ComparisonSet& comparisons,
             const BlockingPlan& plan, const Scorer& scorer,
             const PredictOptions& options, PredictReport* report, std::string* error) {
    const auto started = std::chrono::steady_clock::now();
    const unsigned threads = ResolveThreads(options.threads);

    std::error_code ec;
    std::filesystem::create_directories(options.out_dir, ec);
    if (ec) {
        *error =
            "cannot create output directory \"" + options.out_dir + "\": " + ec.message();
        return false;
    }

    const char* suffix = options.format == EdgeFormat::kBinary ? ".bin" : ".csv";
    std::vector<std::unique_ptr<EdgeShardWriter>> writers;
    writers.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        const std::string path =
            (std::filesystem::path(options.out_dir) / (EdgeShardName(t) + suffix))
                .string();
        auto writer = std::make_unique<EdgeShardWriter>();
        if (!writer->Open(path, options.format)) {
            *error = "cannot write \"" + path + "\"";
            return false;
        }
        writers.push_back(std::move(writer));
        report->shards.push_back(path);
    }

    // The spill is opened alongside the edges, one shard per thread, so a spilling
    // run costs one more sequential write and no extra pass.
    const bool spilling = !options.spill_dir.empty();
    std::vector<std::unique_ptr<SpillWriter>> spills;
    std::vector<std::string> spill_paths;
    if (spilling) {
        std::filesystem::create_directories(options.spill_dir, ec);
        if (ec) {
            *error = "cannot create spill directory \"" + options.spill_dir +
                     "\": " + ec.message();
            return false;
        }
        spills.reserve(threads);
        for (unsigned t = 0; t < threads; ++t) {
            char name[32];
            std::snprintf(name, sizeof(name), "spill-%03u.bin", t);
            const std::string path =
                (std::filesystem::path(options.spill_dir) / name).string();
            auto writer = std::make_unique<SpillWriter>();
            if (!writer->Open(path)) {
                *error = "cannot write \"" + path + "\"";
                return false;
            }
            spills.push_back(std::move(writer));
            spill_paths.push_back(path);
        }
    }
    const bool sampling = spilling && options.spill_sample > 0.0;
    const uint64_t sample_cut =
        sampling ? static_cast<uint64_t>(std::min(1.0, options.spill_sample) *
                                         18446744073709549568.0)
                 : 0;

    std::vector<ThreadTally> tally(threads);
    // Each thread draws from its own stream, so the sample does not depend on how
    // the tasks happened to be handed out.
    for (unsigned t = 0; t < threads; ++t) {
        tally[t].random = Mix64(options.spill_seed + t * 0x9E3779B97F4A7C15ull);
    }
    std::atomic<uint64_t> emitted{0};
    const uint64_t limit = options.max_edges;
    const bool csv = options.format == EdgeFormat::kCsv;
    const IdColumn& ids = store.ids();

    ForEachPairParallel(plan, plan.AllSources(), threads, [&](unsigned t) {
        ThreadTally* counts = &tally[t];
        EdgeShardWriter* writer = writers[t].get();
        SpillWriter* spill = spilling ? spills[t].get() : nullptr;
        return [&, counts, writer, spill](uint32_t a, uint32_t b) {
            ++counts->enumerated;
            // Drawn per candidate, before anything decides the pair's fate, so
            // the sample stays uniform over candidates.
            bool sampled = false;
            if (sampling) {
                counts->random = Mix64(counts->random);
                sampled = counts->random < sample_cut;
            }
            // The ceiling, before the comparison rather than after it. Every
            // level the cheap bounds still admit is granted, and if the best the
            // pair could possibly score is under the threshold there is nothing
            // a string metric could change. A sampled pair is exempt: the spill
            // holds patterns, and a pattern is what this skips producing.
            if (!sampled && !scorer.CanReach(a, b)) {
                ++counts->skipped;
                return;
            }
            const uint32_t gamma = comparisons.Evaluate(a, b);
            const Zone zone = scorer.Classify(gamma);
            if (zone == Zone::kDrop) {
                ++counts->dropped;
                if (sampled) {
                    spill->Write(a, b, gamma);
                    ++counts->spilled;
                }
                return;
            }
            // Drop is where the bracket pays: the pattern cannot clear the
            // threshold however rare its values are, so no term-frequency table is
            // touched. Check and certain both still need the exact weight, because
            // the weight is part of the output.
            if (zone == Zone::kCheck) {
                ++counts->checked;
            } else {
                ++counts->certain;
            }
            const double weight = scorer.Weight(gamma, a, b);
            ++counts->tf_lookups;
            if (weight < scorer.threshold()) {
                if (sampled) {
                    spill->Write(a, b, gamma);
                    ++counts->spilled;
                }
                return;
            }
            if (limit > 0 && emitted.fetch_add(1) >= limit) return;
            ++counts->edges;
            // Above threshold: always spilled, and only once even if also sampled.
            if (spilling) {
                spill->Write(a, b, gamma);
                ++counts->spilled;
            }
            if (csv) {
                writer->WriteCsv(ids.Get(a), ids.Get(b), gamma, weight);
            } else {
                writer->WriteBinary(a, b, gamma, weight);
            }
        };
    });

    for (unsigned t = 0; t < threads; ++t) {
        if (!writers[t]->Close()) {
            *error = "failed while writing \"" + report->shards[t] + "\"";
            return false;
        }
        report->enumerated += tally[t].enumerated;
        report->skipped += tally[t].skipped;
        report->dropped += tally[t].dropped;
        report->checked += tally[t].checked;
        report->certain += tally[t].certain;
        report->edges += tally[t].edges;
        report->spilled += tally[t].spilled;
        report->tf_lookups += tally[t].tf_lookups;
    }
    for (unsigned t = 0; t < threads; ++t) {
        if (spilling && !spills[t]->Close()) {
            *error = "failed while writing \"" + spill_paths[t] + "\"";
            return false;
        }
    }
    report->threads = threads;
    report->mode = plan.mode();
    report->datasets = plan.NumDatasets();
    report->truncated = limit > 0 && emitted.load() > limit;
    report->pattern_space = scorer.PatternSpace();
    if (scorer.Dense()) {
        report->patterns_drop = scorer.PatternsIn(Zone::kDrop);
        report->patterns_check = scorer.PatternsIn(Zone::kCheck);
        report->patterns_certain = scorer.PatternsIn(Zone::kEmit);
    }
    if (spilling) {
        SpillManifest manifest;
        manifest.threshold = scorer.threshold();
        manifest.sample_rate = options.spill_sample;
        manifest.records = store.NumRecords();
        manifest.candidates = report->enumerated;
        manifest.spilled = report->spilled;
        manifest.above_threshold = report->edges;
        manifest.gamma_width = comparisons.Width();
        manifest.layout = GammaLayout(comparisons);
        if (!WriteSpillManifest(options.spill_dir, manifest, error)) return false;
    }
    if (!options.merge_path.empty()) {
        if (!MergeStagedShards(store, options.out_dir, options.merge_path,
                               options.format == EdgeFormat::kBinary,
                               &report->merge_seconds, error)) {
            return false;
        }
        report->shards.clear();
        report->merged_path = options.merge_path;
    }
    report->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return true;
}

void PrintPredictReport(const PredictReport& report, const Scorer& scorer,
                        std::ostream& out) {
    out << "Threshold      " << std::fixed << std::setprecision(3) << scorer.threshold()
        << " bits  (posterior " << std::setprecision(6)
        << ProbabilityForWeight(scorer.threshold()) << ")\n"
        << "Threads        " << report.threads << "\n";
    if (report.datasets > 1) {
        out << "Mode           " << PairModeName(report.mode) << " over "
            << report.datasets << " inputs\n";
    }
    out << "Candidates     " << WithThousands(report.enumerated) << "\n"
        << "Predictions    " << WithThousands(report.edges) << "  ("
        << Percent(report.edges, report.enumerated) << " of candidates)\n"
        << "Elapsed        " << std::setprecision(1) << report.seconds << " s";
    if (report.seconds > 0.0) {
        out << "  (" << std::setprecision(0)
            << static_cast<double>(report.enumerated) / report.seconds
            << " candidates/s)";
    }
    out << "\n\n";

    out << std::left << std::setw(10) << "Zone" << std::right << std::setw(20)
        << "Candidate pairs" << std::setw(12) << "Share" << std::setw(16) << "Patterns"
        << "\n";
    out << std::string(58, '-') << "\n";
    out << std::left << std::setw(10) << "skipped" << std::right << std::setw(20)
        << WithThousands(report.skipped) << std::setw(12)
        << Percent(report.skipped, report.enumerated) << std::setw(16) << "-"
        << "\n";
    const uint64_t zones[3] = {report.dropped, report.checked, report.certain};
    const uint64_t patterns[3] = {report.patterns_drop, report.patterns_check,
                                  report.patterns_certain};
    for (int i = 0; i < 3; ++i) {
        out << std::left << std::setw(10) << ZoneName(static_cast<Zone>(i)) << std::right
            << std::setw(20) << WithThousands(zones[i]) << std::setw(12)
            << Percent(zones[i], report.enumerated) << std::setw(16)
            << (scorer.Dense() ? WithThousands(patterns[i]) : std::string("-")) << "\n";
    }
    out << std::string(58, '-') << "\n";
    out << "Comparisons avoided    " << WithThousands(report.skipped) << " ("
        << Percent(report.skipped, report.enumerated) << " of candidates)\n"
        << "Term-frequency lookups " << WithThousands(report.tf_lookups) << ", avoided "
        << WithThousands(report.dropped) << " ("
        << Percent(report.dropped, report.enumerated) << ").\n"
        << "The ceiling and the bracket are both admissible, so skipping and\n"
        << "dropping on them emit exactly the predictions scoring every pair would\n"
           "have.\n";
    if (report.truncated) {
        out << "\nWARNING: the prediction limit was reached; the output is "
               "incomplete.\n";
    }
    if (!report.merged_path.empty()) {
        out << "\nPredictions\n  " << report.merged_path << "\n"
            << "  " << report.threads << " shard" << (report.threads == 1 ? "" : "s")
            << " merged and removed in " << std::setprecision(2) << report.merge_seconds
            << " s\n";
        return;
    }
    out << "\nPredictions\n";
    for (const std::string& shard : report.shards) out << "  " << shard << "\n";
}

}  // namespace cpplink
