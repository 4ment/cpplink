// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "cpplink/blocking.hpp"

namespace cpplink {

// One slice of one source's pair stream. Whole groups when row_end is zero,
// otherwise rows [row_begin, row_end) of the single group at `begin`.
struct PairTask {
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t row_begin = 0;
    uint64_t row_end = 0;
};

// Groups are batched until a task is worth taking, and an oversized group is cut
// into row ranges of its own triangle -- otherwise one "Smith" block is a whole
// thread's run while the others idle.
// `target` is how many tasks the caller wants at least, so that a source too small
// to fill a task of the default size still spreads over the threads.
void PlanGroupTasks(const SourceGroups& groups, uint64_t target,
                    std::vector<PairTask>* tasks);
void PlanWindowTasks(uint64_t starts, uint32_t window, uint64_t target,
                     std::vector<PairTask>* tasks);

// The requested thread count, or what the hardware reports.
unsigned ResolveThreads(unsigned requested);

// What a caller may watch while a pair stream is walked. The workers pay one
// relaxed atomic increment per task -- once per about a million pairs, nothing
// per pair -- and everything here runs on the calling thread, which would
// otherwise be idle at the join. Positions index `selected`, not the plan.
class PairProgress {
   public:
    virtual ~PairProgress() = default;
    // A source's tasks are planned and its workers are about to start.
    virtual void BeginSource(size_t position, size_t tasks) = 0;
    // Every `Interval()` while the source's workers run, and once more when its
    // last task is done. Tasks are planned to hold about the same number of
    // pairs, so `done / tasks` is how far through the source the walk is.
    virtual void Tick(size_t position, size_t done, size_t tasks) = 0;
    virtual std::chrono::milliseconds Interval() const {
        return std::chrono::milliseconds(250);
    }
};

// Walks the deduplicated pair stream of `selected` in parallel. `make_sink(t)`
// returns the callable that thread t folds its pairs into; sinks are per-thread
// and never shared, which is what keeps the hot path lock-free.
//
// Sources are processed one at a time because a source's groups are materialised
// once and shared read-only by the threads working it; the parallelism is within
// a source, not across them.
template <typename MakeSink>
void ForEachPairParallel(const BlockingPlan& plan, const std::vector<size_t>& selected,
                         unsigned threads, MakeSink&& make_sink,
                         PairProgress* progress = nullptr) {
    const unsigned count = ResolveThreads(threads);
    SourceGroups groups;
    std::vector<PairTask> tasks;
    for (size_t position = 0; position < selected.size(); ++position) {
        const BoundSource& source = plan.at(selected[position]);
        const bool window = source.kind == SourceKind::kSortedNeighbourhood;
        // Four tasks a thread: enough that a thread finishing early can take more
        // work, few enough that the atomic counter stays uncontended.
        const uint64_t target = static_cast<uint64_t>(count) * 4;
        if (window) {
            PlanWindowTasks(source.order.size(), source.window, target, &tasks);
        } else {
            plan.BuildGroups(selected[position], &groups);
            PlanGroupTasks(groups, target, &tasks);
        }

        std::atomic<size_t> next{0};
        std::atomic<size_t> done{0};
        // Signalled by whichever worker finishes the last task, so the watcher
        // below wakes then rather than at the end of its interval: with a source
        // per wait that slack was adding up to a second a run.
        std::mutex finished_mutex;
        std::condition_variable finished;
        std::vector<std::thread> workers;
        workers.reserve(count);
        for (unsigned t = 0; t < count; ++t) {
            workers.emplace_back([&, t] {
                auto sink = make_sink(t);
                for (;;) {
                    const size_t index = next.fetch_add(1);
                    if (index >= tasks.size()) break;
                    const PairTask& task = tasks[index];
                    if (window) {
                        plan.EnumerateWindowRange(selected, position, task.begin,
                                                  task.end, sink);
                    } else if (task.row_end != 0) {
                        plan.EnumerateGroupRange(selected, position, groups, task.begin,
                                                 task.row_begin, task.row_end, sink);
                    } else {
                        for (uint64_t g = task.begin; g < task.end; ++g) {
                            plan.EnumerateGroupRange(selected, position, groups, g, 0,
                                                     groups.Size(g), sink);
                        }
                    }
                    if (done.fetch_add(1, std::memory_order_relaxed) + 1 ==
                        tasks.size()) {
                        // Taken so the notify cannot slip between the watcher's
                        // check and its wait.
                        std::lock_guard<std::mutex> lock(finished_mutex);
                        finished.notify_all();
                    }
                }
            });
        }
        if (progress != nullptr) {
            progress->BeginSource(position, tasks.size());
            std::unique_lock<std::mutex> lock(finished_mutex);
            const auto all_done = [&] {
                return done.load(std::memory_order_relaxed) >= tasks.size();
            };
            // The finished reading is reported once, after the join below.
            while (!finished.wait_for(lock, progress->Interval(), all_done)) {
                progress->Tick(position, done.load(std::memory_order_relaxed),
                               tasks.size());
            }
        }
        for (std::thread& worker : workers) worker.join();
        if (progress != nullptr) progress->Tick(position, tasks.size(), tasks.size());
    }
}

}  // namespace cpplink
