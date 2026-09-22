// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace cpplink_test {

// Removes a temporary directory a test owns, retrying while the platform says a
// file in it is still in use. Windows refuses the delete until the last handle
// on a file is gone, and the last handle is not always one the test can see: a
// reader's read-ahead can outlive the object that owns it, and a virus scanner
// opening a file the test has just written is enough on its own. Posix unlinks
// an open file without complaint, so the same lingering handle costs nothing
// there and this loop returns on its first attempt.
//
// A handle the process really leaked never clears, and that is a bug worth
// failing on, so giving up names the entry that is still held rather than the
// directory the caller passed -- which is all the platform's own message says.
inline void RemoveAll(const std::filesystem::path& dir) {
    constexpr int kAttempts = 40;
    constexpr auto kPause = std::chrono::milliseconds(50);
    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        if (!ec) return;
        std::this_thread::sleep_for(kPause);
    }
    // Still there after the whole budget: remove the entries one at a time, so
    // the failure names the file rather than its directory. The paths are taken
    // first and removed deepest first, because the walk cannot outlive the
    // entries it is walking.
    std::error_code ec;
    std::vector<std::filesystem::path> entries;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        entries.push_back(entry.path());
    }
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        std::error_code removed;
        std::filesystem::remove(*it, removed);
        if (removed) {
            throw std::filesystem::filesystem_error(
                "still held after " + std::to_string(kAttempts * kPause.count()) + " ms",
                *it, removed);
        }
    }
    std::filesystem::remove_all(dir);
}

}  // namespace cpplink_test
