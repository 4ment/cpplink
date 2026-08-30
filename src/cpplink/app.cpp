// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/app.hpp"

#include <ostream>

namespace cpplink {

const char* const kVersion = "0.1.0";

namespace {

void PrintUsage(std::ostream& out) {
    out << "usage: cpplink [options]\n"
        << "\n"
        << "options:\n"
        << "  -h, --help       show this message and exit\n"
        << "  -v, --version    show the version and exit\n";
}

}  // namespace

int Run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    for (const std::string& arg : args) {
        if (arg == "-h" || arg == "--help") {
            PrintUsage(out);
            return 0;
        }
        if (arg == "-v" || arg == "--version") {
            out << kVersion << '\n';
            return 0;
        }
        err << "cpplink: unknown argument '" << arg << "'\n";
        PrintUsage(err);
        return 1;
    }

    PrintUsage(out);
    return 0;
}

}  // namespace cpplink
