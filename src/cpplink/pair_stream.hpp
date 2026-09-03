// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>
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

// Walks the deduplicated pair stream of `selected` in parallel. `make_sink(t)`
// returns the callable that thread t folds its pairs into; sinks are per-thread
// and never shared, which is what keeps the hot path lock-free.
//
// Sources are processed one at a time because a source's groups are materialised
// once and shared read-only by the threads working it; the parallelism is within
// a source, not across them.
template <typename MakeSink>
void ForEachPairParallel(const BlockingPlan& plan, const std::vector<size_t>& selected,
                         unsigned threads, MakeSink&& make_sink) {
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
                }
            });
        }
        for (std::thread& worker : workers) worker.join();
    }
}

}  // namespace cpplink
