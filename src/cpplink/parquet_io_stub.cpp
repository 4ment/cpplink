// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

// `parquet_io.hpp` without Arrow: every call fails and says why. Compiled under
// CPPLINK_WITH_ARROW=OFF, which is how the Python wheel is built -- its input is
// a data frame and its output a data frame, and neither needs a file.

#include <string>
#include <vector>

#include "cpplink/parquet_io.hpp"

namespace cpplink {
namespace {

const char* kUnsupported =
    "this build of cpplink cannot read or write parquet: it was built without Arrow";

}  // namespace

bool ParquetSupported() { return false; }

bool ReadParquetSchema(const std::string&, ArrowSchema*, std::string* error) {
    *error = kUnsupported;
    return false;
}

bool OpenParquetStream(const std::string&, const std::vector<std::string>&,
                       ArrowArrayStream*, std::string* error) {
    *error = kUnsupported;
    return false;
}

struct ParquetWriter::Impl {};

ParquetWriter::ParquetWriter() = default;
ParquetWriter::~ParquetWriter() = default;

bool ParquetWriter::Open(const std::string&, const ArrowSchema&, std::string* error) {
    *error = kUnsupported;
    return false;
}

bool ParquetWriter::Write(ArrowArray* batch, std::string* error) {
    if (batch->release != nullptr) batch->release(batch);
    *error = kUnsupported;
    return false;
}

bool ParquetWriter::Close(std::string* error) {
    *error = kUnsupported;
    return false;
}

}  // namespace cpplink
