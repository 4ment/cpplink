// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace cpplink {

// Report formatting shared by every command that prints a table. Candidate and
// pair counts run to ten digits, and a run is compared against another run by
// eye, so grouping is not decoration.
std::string WithThousands(uint64_t value);

// Marked with an ASCII dot rather than an ellipsis: std::setw pads by bytes, and
// a multi-byte marker silently costs the column its alignment.
std::string Truncate(std::string text, size_t width);

}  // namespace cpplink
