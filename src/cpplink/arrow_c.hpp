// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

// The Arrow C Data Interface and C Stream Interface, as the specification defines
// them: https://arrow.apache.org/docs/format/CDataInterface.html
//
// Three plain C structs with a frozen ABI are the whole contract between cpplink
// and anything that holds Arrow-shaped memory -- pyarrow, pandas, polars, duckdb,
// or Arrow C++ itself. Vendoring them here is what lets the record store be filled
// from a data frame without linking libarrow: a consumer reads the buffers the
// producer points at, and calls `release` when it is done. The include guards are
// the specification's own, so this header and Arrow's `arrow/c/abi.h` can meet in
// one translation unit without redefining anything.

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
    // Array type description
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;

    // Release callback
    void (*release)(struct ArrowSchema*);
    // Opaque producer-specific data
    void* private_data;
};

struct ArrowArray {
    // Array data description
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;

    // Release callback
    void (*release)(struct ArrowArray*);
    // Opaque producer-specific data
    void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
    // Callback to get the stream type (the same for every array in the stream).
    // Returns 0 on success, an errno-compatible code otherwise. On success the
    // ArrowSchema must be released independently of the stream.
    int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);

    // Callback to get the next array. A released `out` with no error means the
    // stream has ended. On success the ArrowArray must be released independently
    // of the stream.
    int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);

    // Optional detail on the last failure, valid until the next operation on the
    // stream. Only to be called after a non-zero return.
    const char* (*get_last_error)(struct ArrowArrayStream*);

    // Release callback: release the stream's own resources.
    void (*release)(struct ArrowArrayStream*);
    // Opaque producer-specific data
    void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE

#ifdef __cplusplus
}
#endif
