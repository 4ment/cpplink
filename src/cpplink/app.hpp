// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace cpplink {

extern const char* const kVersion;

// Runs the command line application. `args` excludes the program name.
// Returns the process exit code; diagnostics go to `err`, results to `out`.
int Run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace cpplink
