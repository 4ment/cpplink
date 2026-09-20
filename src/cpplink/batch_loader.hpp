// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "cpplink/arrow_c.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// Arrow record batches, through the C Data Interface, appended to the store.
//
// This is the one decoder of columns into the store. The parquet loader exports
// each row group through it, and a data frame handed over by Python arrives here
// directly, so the two front ends cannot read a value differently: a `large_string`
// or an integer column is text either way, a timestamp is days either way. Nothing
// here links Arrow C++; the structs are the specification's and the buffers are
// read where the producer left them, which is what keeps a frame from being
// copied on its way in. Every string is interned, so the store is a re-encoding
// rather than a view, and the only bytes duplicated are the dictionary's.

struct LoadStats {
    uint64_t rows = 0;
    int row_groups = 0;  // row groups of parquet, or batches of a stream
    double seconds = 0.0;
    std::vector<uint64_t> dataset_rows;  // rows contributed by each input, in order
};

// The column type an Arrow field reads as, or false where none does. An integer
// is text: a postcode, a year or a phone number that was numeric in the frame is
// still the value it spells, and a writer who wanted a double wrote one. A
// dictionary-encoded field reads as its values' type.
bool ColumnTypeOf(const ArrowSchema& field, ColumnType* out);

// The field's type spelled the way Arrow C++ prints it (`int64`, `large_string`,
// `list<item: string>`, `timestamp[ns, tz=UTC]`), for messages and reports.
std::string ArrowTypeName(const ArrowSchema& field);

// Which child of a batch each store column reads, by name. `children[i]` is the
// child for `spec.columns[i]`, -1 for a derived column, which no input holds;
// `id` is the child for `spec.unique_id`, -1 where the schema names none. Fails
// naming the first column the batch lacks.
struct BatchColumns {
    int id = -1;
    std::vector<int> children;
};
bool ResolveBatchColumns(const ArrowSchema& schema, const Schema& spec,
                         BatchColumns* columns, std::string* error);

// Fills in the type of every column the schema left undeclared from the field
// the Arrow schema holds, then runs the checks those types decide. `source`
// names the file or frame in the message. Must run before a store is built from
// the schema, because the store's column layout is the types.
bool ResolveColumnTypesFrom(const ArrowSchema& schema, const std::string& source,
                            Schema* spec, std::string* error);

// Appends one batch's rows to the store, column by column, without finalizing it:
// more batches may be coming, and the term frequencies are only right once every
// row is in. The batch's fields are matched to the store's columns by name. A
// sliced array (a non-zero `offset`) is read at its slice, and a dictionary-
// encoded column interns each dictionary value once and maps the indices.
bool AppendRecordBatch(const ArrowSchema& schema, const ArrowArray& batch,
                       const Schema& spec, RecordStore* store, std::string* error);

// Drains a stream into the store, batch by batch, releasing each before the next
// is fetched so only one batch's buffers are ever resident beside the store. The
// stream itself is released on return, success or not. `rows` and `batches`
// count what was read.
bool AppendArrayStream(ArrowArrayStream* stream, const Schema& spec, RecordStore* store,
                       uint64_t* rows, int* batches, std::string* error);

// One input of a run: a stream of batches and the name its dataset gets.
struct InputStream {
    std::string name;
    ArrowArrayStream* stream = nullptr;
};

// Loads every input in order into one store and finalizes it. Each input becomes
// a dataset, and because rows are appended one input at a time a dataset is a
// contiguous row range -- which is what makes a link mode's cross-product a
// nested loop over two ranges rather than a per-row test. A single input gets
// no dataset name, because nothing needs qualifying. Every stream is released,
// success or not.
bool LoadStreams(const std::vector<InputStream>& inputs, const Schema& spec,
                 RecordStore* store, LoadStats* stats, std::string* error);

// The child of a struct schema called `name`, or -1.
int FieldIndex(const ArrowSchema& schema, std::string_view name);

// One text column of a batch read by row, for the files that are not the store:
// a prediction file's ids and datasets. Binds a string, string view, integer or
// dictionary-encoded column, sliced as the batch is.
class TextReader {
   public:
    TextReader();
    ~TextReader();
    bool Bind(const ArrowSchema& schema, const ArrowArray& batch, int column,
              std::string* error);
    // The value at `row`, or false where it is null. An integer's digits are
    // written into `scratch`, which the view then points at.
    bool At(int64_t row, std::string_view* out, char (*scratch)[24]) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One numeric column of a batch read by row: a float of either width or an
// integer of any, as a double.
class NumberReader {
   public:
    NumberReader();
    ~NumberReader();
    bool Bind(const ArrowSchema& schema, const ArrowArray& batch, int column,
              std::string* error);
    bool At(int64_t row, double* out) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpplink
