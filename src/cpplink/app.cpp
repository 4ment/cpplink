// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/app.hpp"

#include <ostream>

#include "cpplink/inspect.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/schema.hpp"

namespace cpplink {

const char* const kVersion = "0.1.0";

namespace {

void PrintUsage(std::ostream& out) {
    out << "usage: cpplink <command> [options]\n"
        << "\n"
        << "commands:\n"
        << "  inspect     load a parquet file and report cardinality and memory\n"
        << "  gen-sample  write a sample parquet file with planted duplicates\n"
        << "\n"
        << "options:\n"
        << "  -h, --help       show this message and exit\n"
        << "  -v, --version    show the version and exit\n"
        << "\n"
        << "cpplink inspect --schema <schema.json> <file.parquet>\n"
        << "cpplink gen-sample --out <file.parquet> [--rows N] [--seed N]\n"
        << "                   [--duplicate-rate F] [--truth <file.csv>]\n";
}

// Reads "--name value" pairs. Returns false and reports on a missing value.
bool TakeValue(const std::vector<std::string>& args, size_t* index, std::string* value,
               std::ostream& err) {
    if (*index + 1 >= args.size()) {
        err << "cpplink: " << args[*index] << " needs a value\n";
        return false;
    }
    *value = args[++(*index)];
    return true;
}

int RunInspect(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::string data_path;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink inspect: unknown option '" << args[i] << "'\n";
            return 1;
        } else if (data_path.empty()) {
            data_path = args[i];
        } else {
            err << "cpplink inspect: unexpected argument '" << args[i] << "'\n";
            return 1;
        }
    }
    if (schema_path.empty() || data_path.empty()) {
        err << "cpplink inspect: --schema <schema.json> and a parquet file are "
               "required\n";
        return 1;
    }

    Schema schema;
    std::string error;
    if (!LoadSchema(schema_path, &schema, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    RecordStore store(schema);
    LoadStats stats;
    if (!LoadParquet(data_path, schema, &store, &stats, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    out << "File         " << data_path << "\n";
    PrintInspection(store, stats, out);
    return 0;
}

int RunGenSample(const std::vector<std::string>& args, std::ostream& out,
                 std::ostream& err) {
    SampleOptions options;
    std::string out_path;
    std::string value;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--out") {
            if (!TakeValue(args, &i, &out_path, err)) return 1;
        } else if (args[i] == "--rows") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.rows = std::stoull(value);
        } else if (args[i] == "--seed") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.seed = std::stoull(value);
        } else if (args[i] == "--duplicate-rate") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.duplicate_rate = std::stod(value);
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &options.truth_path, err)) return 1;
        } else {
            err << "cpplink gen-sample: unknown option '" << args[i] << "'\n";
            return 1;
        }
    }
    if (out_path.empty()) {
        err << "cpplink gen-sample: --out <file.parquet> is required\n";
        return 1;
    }

    std::string error;
    if (!WriteSampleParquet(out_path, options, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    out << "Wrote " << options.rows << " rows to " << out_path << "\n";
    if (!options.truth_path.empty()) {
        out << "Planted duplicate pairs listed in " << options.truth_path << "\n";
    }
    return 0;
}

}  // namespace

int Run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (args.empty()) {
        PrintUsage(out);
        return 0;
    }
    const std::string& first = args[0];
    if (first == "-h" || first == "--help") {
        PrintUsage(out);
        return 0;
    }
    if (first == "-v" || first == "--version") {
        out << kVersion << "\n";
        return 0;
    }

    const std::vector<std::string> rest(args.begin() + 1, args.end());
    if (first == "inspect") return RunInspect(rest, out, err);
    if (first == "gen-sample") return RunGenSample(rest, out, err);

    err << "cpplink: unknown command '" << first << "'\n";
    PrintUsage(err);
    return 1;
}

}  // namespace cpplink
