// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/predict.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
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

#include "cpplink/format.hpp"
#include "cpplink/merge_edges.hpp"
#include "cpplink/pair_stream.hpp"
#include "cpplink/spill.hpp"

namespace cpplink {
namespace {

constexpr size_t kFlushBytes = 1u << 20;

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
//
// The two the progress line reads are atomic so that reading them from another
// thread is defined, and they are bumped with a relaxed load and store rather
// than a read-modify-write: one thread owns each tally, so that is a plain
// increment to the compiler and to the hardware, not a locked instruction.
struct alignas(64) ThreadTally {
    std::atomic<uint64_t> enumerated{0};
    uint64_t skipped = 0;
    uint64_t dropped = 0;
    uint64_t checked = 0;
    uint64_t certain = 0;
    std::atomic<uint64_t> edges{0};
    uint64_t spilled = 0;
    uint64_t random = 0;
    uint64_t tf_lookups = 0;
};

inline void Bump(std::atomic<uint64_t>* counter) {
    counter->store(counter->load(std::memory_order_relaxed) + 1,
                   std::memory_order_relaxed);
}

inline uint64_t Peek(const std::atomic<uint64_t>& counter) {
    return counter.load(std::memory_order_relaxed);
}

// 10.09bn, 8.7M, 120k: the progress line is redrawn in the width of a terminal,
// where a ten-digit count with its commas is a third of the room.
std::string Short(double value) {
    char buffer[32];
    if (value >= 1e9) {
        std::snprintf(buffer, sizeof(buffer), "%.2fbn", value / 1e9);
    } else if (value >= 1e6) {
        std::snprintf(buffer, sizeof(buffer), "%.2fM", value / 1e6);
    } else if (value >= 1e3) {
        std::snprintf(buffer, sizeof(buffer), "%.1fk", value / 1e3);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
    }
    return buffer;
}

std::string Clock(double seconds) {
    const auto whole = static_cast<uint64_t>(seconds);
    char buffer[32];
    if (whole >= 3600) {
        std::snprintf(buffer, sizeof(buffer), "%" PRIu64 ":%02" PRIu64 ":%02" PRIu64,
                      whole / 3600, whole / 60 % 60, whole % 60);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%" PRIu64 ":%02" PRIu64, whole / 60,
                      whole % 60);
    }
    return buffer;
}

// The progress line. Its denominator is the sum over the sources of what each is
// priced at, `ApproximatePairs`; a finished source contributes its whole price and
// the running one a share proportional to the tasks done, so the fraction reaches
// one when the walk ends whether the prices were exact or not.
// The candidate and prediction counts are the tallies as they stand, read
// relaxed: a number that is a few hundred pairs stale is what a progress line is.
class PredictProgress : public PairProgress {
   public:
    PredictProgress(const BlockingPlan& plan, const std::vector<size_t>& sources,
                    const std::vector<ThreadTally>& tally, std::ostream& out,
                    unsigned columns)
        : plan_(plan),
          sources_(sources),
          tally_(tally),
          out_(out),
          columns_(columns),
          started_(std::chrono::steady_clock::now()),
          last_line_(started_) {
        before_.reserve(sources.size() + 1);
        before_.push_back(0);
        for (const size_t s : sources) {
            // An estimate paces the sources against one another; the fraction
            // still reaches one, because a source contributes what it was priced
            // at once it is done.
            bool exact = false;
            before_.push_back(before_.back() + plan.ApproximatePairs(s, &exact));
        }
    }

    void BeginSource(size_t position, size_t tasks) override {
        // One permanent line per source, so a log says which one was running when
        // the run slowed down; on a terminal it goes above the redrawn line.
        if (sources_.size() < 2) return;
        ClearLine();
        const BoundSource& source = plan_.at(sources_[position]);
        out_ << "  source " << (position + 1) << "/" << sources_.size() << "  "
             << source.name << "  "
             << WithThousands(before_[position + 1] - before_[position]) << " pairs, "
             << WithThousands(tasks) << (tasks == 1 ? " task" : " tasks") << "\n";
        out_.flush();
    }

    void Tick(size_t position, size_t done, size_t tasks) override {
        const double total = static_cast<double>(before_.back());
        double fraction = 1.0;
        if (total > 0.0) {
            const double share =
                tasks == 0 ? 1.0 : static_cast<double>(done) / static_cast<double>(tasks);
            fraction =
                (static_cast<double>(before_[position]) +
                 share * static_cast<double>(before_[position + 1] - before_[position])) /
                total;
        }
        fraction = std::min(1.0, std::max(0.0, fraction));
        const bool last = position + 1 == sources_.size() && done == tasks;
        const auto now = std::chrono::steady_clock::now();
        if (columns_ == 0 && !last) {
            // A log line every few percent or every minute, whichever comes
            // first, and never two for the same reading.
            const double since = std::chrono::duration<double>(now - last_line_).count();
            if (fraction - last_fraction_ < 0.05 && since < 60.0) return;
        }
        last_fraction_ = fraction;
        last_line_ = now;
        Render(fraction, std::chrono::duration<double>(now - started_).count(), last);
    }

    // The redrawn line is left on screen, ended so the report starts under it.
    void Finish() {
        if (columns_ > 0) out_ << "\n";
        out_.flush();
    }

   private:
    void ClearLine() {
        if (columns_ > 0) out_ << "\r\033[K";
    }

    void Render(double fraction, double elapsed, bool last) {
        uint64_t candidates = 0;
        uint64_t edges = 0;
        for (const ThreadTally& t : tally_) {
            candidates += Peek(t.enumerated);
            edges += Peek(t.edges);
        }
        const bool terminal = columns_ > 0;
        char percent[16];
        std::snprintf(percent, sizeof(percent), "%5.1f%%", 100.0 * fraction);
        std::string line = percent;
        line += "  " +
                (terminal ? Short(static_cast<double>(candidates))
                          : WithThousands(candidates)) +
                " candidates";
        if (elapsed > 0.0) {
            line += "  " + Short(static_cast<double>(candidates) / elapsed) + "/s";
        }
        line += "  " +
                (terminal ? Short(static_cast<double>(edges)) : WithThousands(edges)) +
                " predictions";
        line += "  " + Clock(elapsed) + " elapsed";
        // The first few seconds of a run are a source's first tasks, which say
        // little about the rest of the stream, so the estimate waits for them.
        if (!last && fraction > 0.0 && fraction < 1.0 && elapsed >= 5.0) {
            line += "  ~" + Clock(elapsed * (1.0 - fraction) / fraction) + " left";
        }
        if (!terminal) {
            out_ << line << "\n";
            out_.flush();
            return;
        }
        // The bar takes the room the numbers leave, and a terminal too narrow
        // for both keeps the numbers: a line wider than the terminal wraps and
        // the redraw stacks up instead of replacing itself.
        const size_t room = columns_ - 1;
        if (line.size() + 4 <= room) {
            const size_t width = std::min<size_t>(20, room - line.size() - 3);
            const size_t filled = static_cast<size_t>(fraction * width + 0.5);
            line = "[" + std::string(filled, '=') + std::string(width - filled, ' ') +
                   "] " + line;
        } else if (line.size() > room) {
            line.resize(room);
        }
        ClearLine();
        out_ << line;
        out_.flush();
    }

    const BlockingPlan& plan_;
    const std::vector<size_t>& sources_;
    const std::vector<ThreadTally>& tally_;
    std::ostream& out_;
    const unsigned columns_;
    std::vector<uint64_t> before_;  // pairs in the sources before each position
    std::chrono::steady_clock::time_point started_;
    std::chrono::steady_clock::time_point last_line_;
    double last_fraction_ = -1.0;
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

    const std::vector<size_t> sources = plan.AllSources();
    std::unique_ptr<PredictProgress> progress;
    if (options.progress != nullptr) {
        progress = std::make_unique<PredictProgress>(
            plan, sources, tally, *options.progress, options.progress_columns);
    }

    ForEachPairParallel(
        plan, sources, threads,
        [&](unsigned t) {
            ThreadTally* counts = &tally[t];
            EdgeShardWriter* writer = writers[t].get();
            SpillWriter* spill = spilling ? spills[t].get() : nullptr;
            return [&, counts, writer, spill](uint32_t a, uint32_t b) {
                Bump(&counts->enumerated);
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
                Bump(&counts->edges);
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
        },
        progress.get());
    if (progress) progress->Finish();

    for (unsigned t = 0; t < threads; ++t) {
        if (!writers[t]->Close()) {
            *error = "failed while writing \"" + report->shards[t] + "\"";
            return false;
        }
        report->enumerated += Peek(tally[t].enumerated);
        report->skipped += tally[t].skipped;
        report->dropped += tally[t].dropped;
        report->checked += tally[t].checked;
        report->certain += tally[t].certain;
        report->edges += Peek(tally[t].edges);
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
        if (options.progress != nullptr) {
            *options.progress << "  merging " << threads
                              << (threads == 1 ? " shard" : " shards") << " into "
                              << options.merge_path << "\n";
            options.progress->flush();
        }
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

void PrintPredictPlan(const RecordStore& store, const BlockingPlan& plan,
                      const ComparisonSet& comparisons, const Scorer& scorer,
                      const PredictOptions& options, double load_seconds,
                      std::ostream& out) {
    out << "Records        " << WithThousands(store.NumRecords());
    if (load_seconds > 0.0) {
        out << "  (loaded in " << std::fixed << std::setprecision(1) << load_seconds
            << " s)";
    }
    out << "\n";
    if (store.NumDatasets() > 1) {
        out << "Inputs         " << store.NumDatasets() << "  (";
        for (size_t d = 0; d < store.NumDatasets(); ++d) {
            if (d > 0) out << " + ";
            out << WithThousands(store.DatasetEnd(d) - store.DatasetStart(d));
        }
        out << " rows), mode " << PairModeName(plan.mode()) << "\n";
    }
    out << "Threshold      " << std::fixed << std::setprecision(3) << scorer.threshold()
        << " bits  (posterior " << std::setprecision(6)
        << ProbabilityForWeight(scorer.threshold()) << ")\n"
        << "Threads        " << ResolveThreads(options.threads) << "\n"
        << "Gamma          " << static_cast<unsigned>(comparisons.Width()) << " bits";
    if (scorer.Dense()) {
        out << ", " << WithThousands(scorer.PatternSpace())
            << " patterns: " << WithThousands(scorer.PatternsIn(Zone::kDrop)) << " drop, "
            << WithThousands(scorer.PatternsIn(Zone::kCheck)) << " check, "
            << WithThousands(scorer.PatternsIn(Zone::kEmit)) << " certain";
    }
    out << "\n";
    const unsigned threads = ResolveThreads(options.threads);
    if (!options.merge_path.empty()) {
        out << "Predictions    " << options.merge_path << "  (" << threads
            << (threads == 1 ? " shard" : " shards") << " merged at the end)\n";
    } else {
        out << "Predictions    " << options.out_dir << "  (" << threads
            << (threads == 1 ? " shard" : " shards")
            << (options.format == EdgeFormat::kCsv ? ", csv)" : ", binary)") << "\n";
    }
    if (!options.spill_dir.empty()) {
        out << "Spill          " << options.spill_dir;
        if (options.spill_sample > 0.0) {
            out << "  (sampling " << std::setprecision(4) << options.spill_sample
                << " of every candidate)";
        }
        out << "\n";
    }
    if (options.max_edges > 0) {
        out << "Limit          " << WithThousands(options.max_edges) << " predictions\n";
    }
    out << "\n";

    out << std::left << std::setw(36) << "Source" << std::right << std::setw(20)
        << "Candidate pairs" << std::setw(16) << "Largest group" << "\n";
    out << std::string(72, '-') << "\n";
    uint64_t sum = 0;
    bool all_exact = true;
    for (size_t s = 0; s < plan.Size(); ++s) {
        bool exact = false;
        const uint64_t pairs = plan.ApproximatePairs(s, &exact);
        all_exact = all_exact && exact;
        sum += pairs;
        out << std::left << std::setw(36) << Truncate(plan.at(s).name, 35) << std::right
            << std::setw(20) << ((exact ? "" : "~") + WithThousands(pairs))
            << std::setw(16) << WithThousands(plan.LargestGroup(s)) << "\n";
    }
    out << std::string(72, '-') << "\n";
    out << std::left << std::setw(36) << "Sum over sources" << std::right << std::setw(20)
        << ((all_exact ? "" : "~") + WithThousands(sum)) << "\n";
    const double space = store.PairSpace(plan.mode());
    if (space > 0.0) {
        out << std::left << std::setw(36) << "Pairs unblocked" << std::right
            << std::setw(20) << WithThousands(static_cast<uint64_t>(space)) << "\n";
    }
    if (all_exact) {
        out << "The sum bounds the candidates from above: a pair an earlier source "
               "produces\nis not produced again, and the progress line measures "
               "against the sum.\n\n";
    } else {
        out << "A count marked ~ is the pooled count scaled to the cross-dataset "
               "share of the\npair space, since the exact one costs a sort per "
               "source; explain-blocking pays\nit. The progress line paces the "
               "sources against these.\n\n";
    }
    out.flush();
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
