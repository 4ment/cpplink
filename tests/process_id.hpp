// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace cpplink_test {

// The process id as text, for naming a temporary directory that no other run of
// the test binary shares. Windows spells the call with a leading underscore and
// declares it in a different header, which is the whole reason this exists.
inline std::string ProcessId() {
#ifdef _WIN32
    return std::to_string(_getpid());
#else
    return std::to_string(::getpid());
#endif
}

}  // namespace cpplink_test
