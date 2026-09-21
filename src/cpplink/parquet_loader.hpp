// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "cpplink/batch_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// One column of a parquet file as its footer describes it: the name, the Arrow
// type spelled out, and the column type it reads as, where one does.
struct FileColumn {
    std::string name;
    std::string arrow_type;
    ColumnType type = ColumnType::kString;
    bool readable = false;  // false where no column type reads the Arrow type
};

// Reads the footer of one file and lists its columns in file order.
bool ReadFileColumns(const std::string& path, std::vector<FileColumn>* columns,
                     std::string* error);

// Fills in the type of every column the schema left undeclared from the Arrow
// type the first file holds, then runs the checks those types decide. A parquet
// file already knows what its columns are, so the schema need not say again;
// what it says is checked against the file, not used instead of it. Reads only
// the footer. Must run before a store is built from the schema, because the
// store's column layout is the types.
bool ResolveColumnTypes(const std::vector<std::string>& paths, Schema* schema,
                        std::string* error);

// Reads a parquet file into the record store one row group at a time, so the
// Arrow buffers for a group are released before the next is read and the two
// representations are never both resident for the whole file. The file arrives
// as a C Data stream from `parquet_io` and is decoded by `batch_loader`, so a
// file and a frame are read by the same code and nothing here links Arrow.
bool LoadParquet(const std::string& path, const Schema& schema, RecordStore* store,
                 LoadStats* stats, std::string* error);

// The same, over several inputs read in order into one store. Each file becomes a
// dataset, and because rows are appended one file at a time a dataset is a
// contiguous row range -- which is what makes a link mode's cross-product a nested
// loop over two ranges rather than a per-row test. Everything downstream sees one
// store; only the pair mode knows there was more than one file.
//
// Each dataset is named by its file's stem (`DatasetNamesFor`), which is what the
// prediction and cluster files qualify a record's id with when there is more
// than one input.
bool LoadParquetFiles(const std::vector<std::string>& paths, const Schema& schema,
                      RecordStore* store, LoadStats* stats, std::string* error);

// The dataset names a list of inputs gets: each file's stem, with a comma or a
// colon replaced so the name can sit in a csv field and before the `:` of a
// qualified id, and a stem that repeats an earlier one suffixed with `#` and its
// position so two inputs never share a name. One file gets no name at all.
std::vector<std::string> DatasetNamesFor(const std::vector<std::string>& paths);

}  // namespace cpplink
