// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/pair_stream.hpp"

#include <algorithm>
#include <vector>

namespace cpplink {
namespace {

// The ceiling on task size. A source smaller than this would otherwise be one
// task and one thread, however many threads are waiting, so the actual size is
// the smaller of this and an even split across the target task count.
constexpr uint64_t kMaxPairsPerTask = 1u << 20;

uint64_t PairsIn(uint64_t group) { return group * (group - 1) / 2; }

uint64_t TaskSize(uint64_t total, uint64_t target) {
    if (target == 0) return kMaxPairsPerTask;
    return std::max<uint64_t>(1, std::min(kMaxPairsPerTask, total / target));
}

}  // namespace

unsigned ResolveThreads(unsigned requested) {
    if (requested > 0) return requested;
    const unsigned available = std::thread::hardware_concurrency();
    return available > 0 ? available : 1;
}

void PlanGroupTasks(const SourceGroups& groups, uint64_t target,
                    std::vector<PairTask>* tasks) {
    tasks->clear();
    const uint64_t count = groups.GroupCount();
    uint64_t total = 0;
    for (uint64_t g = 0; g < count; ++g) total += PairsIn(groups.Size(g));
    const uint64_t per_task = TaskSize(total, target);

    uint64_t batch = 0;
    uint64_t pending = 0;
    for (uint64_t g = 0; g < count; ++g) {
        const uint64_t rows = groups.Size(g);
        if (PairsIn(rows) > per_task) {
            if (g > batch) tasks->push_back({batch, g, 0, 0});
            uint64_t low = 0;
            uint64_t chunk = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                chunk += rows - 1 - i;
                if (chunk >= per_task) {
                    tasks->push_back({g, g + 1, low, i + 1});
                    low = i + 1;
                    chunk = 0;
                }
            }
            if (low < rows) tasks->push_back({g, g + 1, low, rows});
            batch = g + 1;
            pending = 0;
            continue;
        }
        pending += PairsIn(rows);
        if (pending >= per_task) {
            tasks->push_back({batch, g + 1, 0, 0});
            batch = g + 1;
            pending = 0;
        }
    }
    if (batch < count) tasks->push_back({batch, count, 0, 0});
}

void PlanWindowTasks(uint64_t starts, uint32_t window, uint64_t target,
                     std::vector<PairTask>* tasks) {
    tasks->clear();
    const uint64_t span = std::max<uint32_t>(1, window);
    const uint64_t per_task = TaskSize(starts * span, target);
    const uint64_t step = std::max<uint64_t>(1, per_task / span);
    for (uint64_t i = 0; i < starts; i += step) {
        tasks->push_back({i, std::min(i + step, starts), 0, 0});
    }
}

}  // namespace cpplink
