// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "cpplink/blocking.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/neighbourhood.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

// The glue every data command runs between parsing its arguments and calling a
// stage. It lives here rather than in the command dispatch so that a second
// front end -- the Python package -- runs exactly what the command line runs,
// and cannot drift from it: a schema is resolved the same way, a plan is built
// from the same union, an output path means the same thing.

// The schema for a data command: parsed, then given the column types it left to
// the file. Every command that builds a store from a schema goes through here,
// because the store's layout is the types and they must be settled first.
bool LoadSchemaFor(const std::string& schema_path,
                   const std::vector<std::string>& data_paths, Schema* schema,
                   std::string* error);

// "dedup" is every pair the store holds, which over more than one input is
// link-and-dedup; "link" is the cross-product of the inputs alone.
bool ParseMode(const std::string& text, PairMode* mode, std::string* error);

// A second file with nothing said about it means linking the two, which is what a
// second file is for; asking for the within-file pairs as well is --mode
// link-and-dedup.
PairMode DefaultMode(bool given, PairMode mode, size_t inputs);

// The schema one purpose sees. Estimation keeps the sources declared `"use":
// "estimate"` and drops the `"use": "predict"` ones, prediction the other way
// round, so the two purposes see only their own sources. `all_pairs` then
// replaces whatever is left with the one source that blocks on nothing, so the
// same schema can be run blocked and unblocked without being edited. It is what
// makes the schema's "blocking" section optional: on an input small enough to
// enumerate, there is nothing for it to say. Fails where the result declares no
// source at all.
bool SchemaForPlan(const Schema& full, bool all_pairs, bool for_estimation,
                   Schema* schema, std::string* error);

// The plan for one purpose over a store already loaded. The store's layout is
// its columns, which the purpose filter leaves alone, so one store serves both
// the estimation and the prediction plan.
bool BuildPlan(const Schema& full, const RecordStore& store, PairMode mode,
               bool all_pairs, bool for_estimation, BlockingPlan* plan,
               std::string* error);

// Loads a schema and a parquet file, the opening move of every blocking command.
// `schema` comes back as the purpose's schema, which is what the command then
// binds its comparisons against.
bool LoadForBlocking(const std::string& schema_path,
                     const std::vector<std::string>& data_paths, PairMode mode,
                     Schema* schema, std::unique_ptr<RecordStore>* store,
                     BlockingPlan* plan, std::string* error, bool all_pairs = false,
                     LoadStats* stats = nullptr, bool for_estimation = false);

// Clustering needs only the identifiers: the edges already carry every row index
// and weight, so the comparison columns are left on disk rather than interned.
bool LoadIdsOnly(const std::string& schema_path,
                 const std::vector<std::string>& data_paths,
                 std::unique_ptr<RecordStore>* store, std::string* error);

// `--out` names either a directory of shards or the single file they are to be
// merged into, and the extension is what says which. The staging directory sits
// beside the file so a run that dies mid-merge leaves its shards somewhere
// obvious rather than in a temporary directory nobody looks in. `command` names
// the caller in the message a `--format` with a single-file `--out` gets.
bool ResolveEdgeOutput(const std::string& command, const std::string& out,
                       bool format_given, std::string* out_dir, std::string* merge_path,
                       std::string* error);

// Builds the neighbourhood masses a fuzzy term-frequency adjustment needs, and
// says which columns got one. A column too large for the budget keeps today's
// behaviour, which is worth saying out loud rather than degrading quietly.
void BuildBallTables(const ComparisonSet& comparisons, const RecordStore& store,
                     const BallOptions& options, BallTables* balls, std::ostream& out);

// One record named by its id, or by `<dataset>:<id>` where the inputs share ids.
// Found is a row; missing and ambiguous are each an error that says so, the
// latter naming the datasets to qualify with.
bool ResolveId(const RecordStore& store, const std::string& text, uint64_t* row,
               std::string* error);

// One `<a>,<b>` line of `explain` resolved to two rows, by id or by row index.
bool ResolvePair(const RecordStore& store, const std::string& text, bool by_row,
                 uint64_t* row_a, uint64_t* row_b, std::string* error);

}  // namespace cpplink
