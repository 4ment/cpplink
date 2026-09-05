// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/rescore.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cpplink/pair_stream.hpp"

namespace cpplink {
namespace {

// Pairs pulled from a shard per read. Three u32 each, so this is a 96 KB buffer.
constexpr size_t kBatchPairs = 8192;

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

struct alignas(64) ThreadTally {
    uint64_t pairs = 0;
    uint64_t edges = 0;
    uint64_t dropped = 0;
    uint64_t tf_lookups = 0;
};

}  // namespace

bool Rescore(const RecordStore& store, const ComparisonSet& comparisons,
             const Scorer& scorer, const RescoreOptions& options, RescoreReport* report,
             std::string* error) {
    const auto started = std::chrono::steady_clock::now();

    if (!ReadSpillManifest(options.spill_dir, &report->manifest, error)) {
        *error = "cpplink rescore: " + *error;
        return false;
    }
    // A spill is only meaningful against the schema that packed its gamma.
    const std::string layout = GammaLayout(comparisons);
    if (layout != report->manifest.layout) {
        *error =
            "cpplink rescore: this spill was written with a different set of "
            "comparisons, so its gamma means something else\n  spill:  " +
            report->manifest.layout + "\n  schema: " + layout;
        return false;
    }
    if (report->manifest.records != store.NumRecords()) {
        *error = "cpplink rescore: this spill was written over " +
                 std::to_string(report->manifest.records) + " records but the data has " +
                 std::to_string(store.NumRecords());
        return false;
    }

    std::vector<std::string> shards;
    if (!CollectSpillShards(options.spill_dir, &shards, error)) {
        *error = "cpplink rescore: " + *error;
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(options.out_dir, ec);
    if (ec) {
        *error =
            "cannot create output directory \"" + options.out_dir + "\": " + ec.message();
        return false;
    }

    const unsigned threads = std::min<unsigned>(ResolveThreads(options.threads),
                                                static_cast<unsigned>(shards.size()));
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

    // One shard at a time off an atomic counter: the shards are the natural unit
    // and there are as many of them as the run that wrote them had threads.
    std::vector<ThreadTally> tally(threads);
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> emitted{0};
    std::vector<std::string> failures(threads);
    const uint64_t limit = options.max_edges;
    const bool csv = options.format == EdgeFormat::kCsv;
    const IdColumn& ids = store.ids();

    auto work = [&](unsigned t) {
        ThreadTally* counts = &tally[t];
        EdgeShardWriter* writer = writers[t].get();
        std::vector<uint32_t> batch(kBatchPairs * 3);
        for (;;) {
            const size_t index = next.fetch_add(1);
            if (index >= shards.size()) return;
            SpillReader reader;
            std::string problem;
            if (!reader.Open(shards[index], &problem)) {
                failures[t] = problem;
                return;
            }
            size_t got = 0;
            while (reader.Next(batch.data(), batch.size(), &got, &problem)) {
                for (size_t i = 0; i < got; ++i) {
                    const uint32_t a = batch[i * 3];
                    const uint32_t b = batch[i * 3 + 1];
                    const uint32_t gamma = batch[i * 3 + 2];
                    ++counts->pairs;
                    // The bracket still applies: a pattern that cannot clear the
                    // threshold at its rarest values needs no term-frequency read.
                    if (scorer.Classify(gamma) == Zone::kDrop) {
                        ++counts->dropped;
                        continue;
                    }
                    const double weight = scorer.Weight(gamma, a, b);
                    ++counts->tf_lookups;
                    if (weight < scorer.threshold()) {
                        ++counts->dropped;
                        continue;
                    }
                    if (limit > 0 && emitted.fetch_add(1) >= limit) continue;
                    ++counts->edges;
                    if (csv) {
                        writer->WriteCsv(ids.Get(a), ids.Get(b), gamma, weight);
                    } else {
                        writer->WriteBinary(a, b, gamma, weight);
                    }
                }
            }
            if (!problem.empty()) {
                failures[t] = problem;
                return;
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work, t);
    work(0);
    for (std::thread& thread : pool) thread.join();

    for (unsigned t = 0; t < threads; ++t) {
        if (!failures[t].empty()) {
            *error = "cpplink rescore: " + failures[t];
            return false;
        }
    }
    for (unsigned t = 0; t < threads; ++t) {
        if (!writers[t]->Close()) {
            *error = "failed while writing \"" + report->shards[t] + "\"";
            return false;
        }
        report->pairs += tally[t].pairs;
        report->edges += tally[t].edges;
        report->dropped += tally[t].dropped;
        report->tf_lookups += tally[t].tf_lookups;
    }
    report->threads = threads;
    report->truncated = limit > 0 && emitted.load() > limit;
    report->below_spill_threshold = report->manifest.sample_rate <= 0.0 &&
                                    scorer.threshold() < report->manifest.threshold;
    report->seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return true;
}

void PrintRescoreReport(const RescoreReport& report, const Scorer& scorer,
                        std::ostream& out) {
    out << "Spill          " << WithThousands(report.manifest.spilled) << " pairs from "
        << WithThousands(report.manifest.candidates) << " candidates, written at "
        << std::fixed << std::setprecision(3) << report.manifest.threshold << " bits";
    if (report.manifest.sample_rate > 0.0) {
        out << " plus a " << std::setprecision(4) << report.manifest.sample_rate
            << " sample of the rest";
    }
    out << "\n";
    out << "Threshold      " << std::setprecision(3) << scorer.threshold() << " bits\n"
        << "Threads        " << report.threads << "\n"
        << "Pairs read     " << WithThousands(report.pairs) << "\n"
        << "Edges          " << WithThousands(report.edges) << "\n"
        << "Elapsed        " << std::setprecision(2) << report.seconds << " s";
    if (report.seconds > 0.0) {
        out << "  (" << std::setprecision(0)
            << static_cast<double>(report.pairs) / report.seconds << " pairs/s)";
    }
    out << "\n";
    for (const std::string& shard : report.shards) out << "  " << shard << "\n";

    if (report.truncated) {
        out << "\nThe edge limit was reached, so this is a sample of the edges and not "
               "all of them.\n";
    }
    // The honest limitation, printed rather than buried: a spill holds what one
    // model kept, and a different model would have kept a different set.
    out << "\nRe-scoring reads only the pairs the spilling run retained. Every pair here "
           "is\nscored exactly, but a pair that run discarded cannot come back";
    if (report.below_spill_threshold) {
        out << " -- and this\nthreshold is below the one the spill was written at, so "
               "there are certainly such\npairs. Treat this as tuning, not as a run.\n";
    } else if (report.manifest.sample_rate > 0.0) {
        out << ".\nThe uniform sample in this spill is what shows how much that is.\n";
    } else {
        out << ".\n";
    }
}

}  // namespace cpplink
