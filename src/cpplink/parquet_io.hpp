// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpplink/arrow_c.hpp"

namespace cpplink {

// The whole of what cpplink asks of Arrow C++: a parquet file read as a stream
// of C Data batches, and a stream of C Data batches written as a parquet file.
// Everything above this line reads and writes the C structs, so the parquet
// library is a leaf that `CPPLINK_WITH_ARROW` swaps for a stub, and a wheel that
// takes its input from a data frame links no Arrow at all. The stub fails every
// call with the reason.

// Whether this build can read and write parquet.
bool ParquetSupported();

// The file's schema, as the C struct. The caller releases it.
bool ReadParquetSchema(const std::string& path, ArrowSchema* out, std::string* error);

// The file as a stream of batches, one row group at a time, holding only the
// named columns (all of them where `columns` is empty). A name the file lacks
// is an error here rather than a missing child later. The caller releases the
// stream, which closes the file.
bool OpenParquetStream(const std::string& path, const std::vector<std::string>& columns,
                       ArrowArrayStream* out, std::string* error);

// A parquet file written batch by batch; each batch is one row group.
class ParquetWriter {
   public:
    ParquetWriter();
    ~ParquetWriter();

    // Creates the file for batches of this schema. The schema is read, not
    // taken: the caller still releases it.
    bool Open(const std::string& path, const ArrowSchema& schema, std::string* error);
    // Writes and releases the batch, whichever way it goes.
    bool Write(ArrowArray* batch, std::string* error);
    bool Close(std::string* error);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpplink
