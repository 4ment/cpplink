// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>
#include <vector>

#include "cpplink/app.hpp"

int main(int argc, char* argv[]) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    return cpplink::Run(args, std::cout, std::cerr);
}
