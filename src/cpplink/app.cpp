// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/app.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <ostream>

#include "cpplink/blocking.hpp"
#include "cpplink/cluster.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/completeness.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/explain_blocking.hpp"
#include "cpplink/inspect.hpp"
#include "cpplink/levels.hpp"
#include "cpplink/merge_edges.hpp"
#include "cpplink/model.hpp"
#include "cpplink/neighbourhood.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/profile.hpp"
#include "cpplink/recall.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/rescore.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"
#include "cpplink/simplify.hpp"

namespace cpplink {

const char* const kVersion = "0.1.0";

namespace {

void PrintUsage(std::ostream& out) {
    out << "usage: cpplink <command> [options]\n"
        << "\n"
        << "commands:\n"
        << "  inspect     load a parquet file and report cardinality and memory\n"
        << "  profile     what the columns can be worth, what a matching pair "
           "will\n"
        << "              score, and which pairs of them are the same evidence "
           "twice\n"
        << "  levels      check the schema's fuzzy thresholds against the column "
           "they\n"
        << "              run on, and propose better ones\n"
        << "  simplify    merge the levels a scored stream cannot tell apart\n"
        << "  explain     show the levels a single pair lands on, and with a "
           "model\n"
        << "              the waterfall of bits behind its score\n"
        << "  explain-blocking  price every blocking source without enumerating\n"
        << "  recall      measure what fraction of known pairs blocking reaches,\n"
        << "              and with --why, diagnose the ones it does not\n"
        << "  estimate    learn m, u and lambda and write the model\n"
        << "  completeness  estimate blocking recall with no known pairs at all\n"
        << "  predict     score the candidate pairs and write the predictions "
           "above a\n"
        << "              threshold, as one file or as one shard per thread\n"
        << "  cluster     join the predictions into duplicate clusters, from "
           "either a\n"
        << "              merged prediction file or a shard directory\n"
        << "  merge-predictions  combine the prediction shards into one csv or "
           "parquet file\n"
        << "  rescore     re-score a spilled run under a new model, without "
           "comparing again\n"
        << "  gen-sample  write a sample parquet file with planted duplicates\n"
        << "\n"
        << "options:\n"
        << "  -h, --help       show this message and exit\n"
        << "  -v, --version    show the version and exit\n"
        << "\n"
        << "Every command taking data accepts more than one parquet file. Files\n"
        << "are read in order into one store and each becomes a dataset, so two\n"
        << "files mean linking: --mode link scores only pairs that cross the two,\n"
        << "--mode dedup (or link-and-dedup) scores every pair of the whole store.\n"
        << "One file is a deduplication and needs no --mode.\n"
        << "\n"
        << "--all-pairs runs the five plan-building commands with no blocking at\n"
        << "all: every pair the mode admits becomes a candidate. On an input small\n"
        << "enough to enumerate that is cheap, and it leaves blocking nothing to\n"
        << "miss. Estimating from it holds no column out; see the documentation.\n"
        << "\n"
        << "cpplink inspect --schema <schema.json> <file.parquet>...\n"
        << "cpplink profile --schema <schema.json> [--sample-rows N] [--no-pairs]\n"
        << "                [--expected-matches N] [--threads N] [--json] "
           "[--mode MODE]\n"
        << "                [--no-anchors] [--anchor-rows N] [--anchor-margin BITS]\n"
        << "                [--anchor-pairs N] [--truth <pairs.csv>] "
           "<file.parquet>...\n"
        << "cpplink levels --schema <schema.json> [--out <schema.json>] [--json]\n"
        << "               [--truth <pairs.csv>]\n"
        << "               [--levels N] [--max-levels N] [--jaro-floor F]\n"
        << "               [--jaro-step F] [--edit-max N] [--min-match-pairs N]\n"
        << "               [--anchor-margin BITS] [--anchor-rows N]\n"
        << "               [--anchor-pairs N] [--ball-budget N] [--threads N]\n"
        << "               [--mode MODE] <file.parquet>...\n"
        << "cpplink simplify --schema <schema.json> --model <model.json>\n"
        << "                 [--out <schema.json>] [--alpha F] [--min-gap BITS]\n"
        << "                 [--pair-cap N] [--threads N] [--seed N] [--json]\n"
        << "                 [--mode MODE] <file.parquet>...\n"
        << "cpplink explain --schema <schema.json> --pair <id_a>,<id_b>\n"
        << "                [--rows <i>,<j>] [--model <model.json>] "
           "[--threshold BITS]\n"
        << "                [--tf-damping F] <file.parquet>...\n"
        << "cpplink explain-blocking --schema <schema.json> [--count] "
           "[--mode MODE]\n"
        << "                [--all-pairs] <file.parquet>...\n"
        << "cpplink recall --schema <schema.json> --truth <truth.csv> [--why]\n"
        << "               [--show-misses N] [--count] [--json] [--mode MODE]\n"
        << "               [--all-pairs] <file.parquet>...\n"
        << "cpplink estimate --schema <schema.json> [--out <model.json>]\n"
        << "                 [--u-sample N] [--session-pairs N] [--threads N]\n"
        << "                 [--iterations N] [--lambda F] [--seed N] "
           "[--mode MODE]\n"
        << "                 [--fuzzy-u] [--ball-budget N] [--no-tie-holdout]\n"
        << "                 [--tied-bits F] [--tie-sample-rows N]\n"
        << "                 [--interactions] [--max-interactions N]\n"
        << "                 [--interaction-bits F] [--all-pairs] "
           "<file.parquet>...\n"
        << "cpplink predict --schema <schema.json> --model <model.json>\n"
        << "                --out <dir|file.csv|file.parquet>\n"
        << "                [--threshold BITS | --probability P] [--format bin|csv]\n"
        << "                [--threads N] [--limit N] [--no-bounds] [--no-ceiling]\n"
        << "                [--tf-damping F] [--no-signatures] [--spill <dir>]\n"
        << "                [--spill-sample R] [--fuzzy-tf] [--ball-budget N]\n"
        << "                [--no-interactions] [--mode MODE] [--all-pairs]\n"
        << "                <file.parquet>...\n"
        << "cpplink completeness --schema <schema.json> --model <model.json>\n"
        << "                [--truth <pairs.csv>] [--sample R] [--threads N]\n"
        << "                [--value-weighting records|pairs] [--min-observed N]\n"
        << "                [--bound-only] [--json] [--mode MODE] [--all-pairs]\n"
        << "                <file.parquet>...\n"
        << "cpplink rescore --schema <schema.json> --model <model.json> --spill <dir>\n"
        << "                --out <dir|file.csv|file.parquet>\n"
        << "                [--threshold BITS | --probability P]\n"
        << "                [--format bin|csv] [--threads N] [--limit N] "
           "[--mode MODE]\n"
        << "                <file.parquet>...\n"
        << "cpplink cluster --schema <schema.json>\n"
        << "                --predictions <dir|file.csv|file.parquet>\n"
        << "                [--out <file.csv>]\n"
        << "                [--threshold BITS | --probability P] [--truth <file.csv>]\n"
        << "                [--min-size N] <file.parquet>...\n"
        << "cpplink merge-predictions --shards <dir> "
           "--out <file.csv|file.parquet>\n"
        << "                          [--format csv|parquet] [--from bin|csv]\n"
        << "                          [--threshold BITS | --probability P]\n"
        << "                          [--schema <schema.json>] "
           "[<file.parquet>...]\n"
        << "cpplink gen-sample --out <file.parquet> [--rows N] [--seed N]\n"
        << "                   [--duplicate-rate F] [--truth <file.csv>]\n"
        << "                   [--out-b <file.parquet>]\n";
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

// "dedup" is every pair the store holds, which over more than one input is
// link-and-dedup; "link" is the cross-product of the inputs alone.
bool ParseMode(const std::string& text, PairMode* mode, std::ostream& err) {
    if (text == "dedup" || text == "link-and-dedup") {
        *mode = PairMode::kAll;
        return true;
    }
    if (text == "link") {
        *mode = PairMode::kCrossDataset;
        return true;
    }
    err << "cpplink: --mode must be dedup, link or link-and-dedup, not '" << text
        << "'\n";
    return false;
}

// A second file with nothing said about it means linking the two, which is what a
// second file is for; asking for the within-file pairs as well is --mode
// link-and-dedup.
PairMode DefaultMode(bool given, PairMode mode, size_t inputs) {
    if (given) return mode;
    return inputs > 1 ? PairMode::kCrossDataset : PairMode::kAll;
}

int RunInspect(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink inspect: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty()) {
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
    if (!LoadParquetFiles(data_paths, schema, &store, &stats, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    for (size_t i = 0; i < data_paths.size(); ++i) {
        out << (i == 0 ? "File         " : "             ") << data_paths[i];
        if (data_paths.size() > 1) {
            out << "  (dataset " << i << ", " << stats.dataset_rows[i] << " rows)";
        }
        out << "\n";
    }
    PrintInspection(store, stats, out);
    return 0;
}

int RunProfile(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::string truth_path;
    std::string value;
    std::vector<std::string> data_paths;
    ProfileOptions options;
    bool as_json = false;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--sample-rows") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.sample_rows = std::stoull(value);
        } else if (args[i] == "--expected-matches") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.expected_matches = std::stoull(value);
        } else if (args[i] == "--threads") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threads = static_cast<unsigned>(std::stoul(value));
        } else if (args[i] == "--anchor-rows") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.anchor_rows = std::stoull(value);
        } else if (args[i] == "--anchor-pairs") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.anchor_pairs = std::stoull(value);
        } else if (args[i] == "--anchor-margin") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.anchor_margin = std::stod(value);
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &truth_path, err)) return 1;
        } else if (args[i] == "--no-pairs") {
            options.pairs = false;
        } else if (args[i] == "--no-anchors") {
            options.anchors = false;
        } else if (args[i] == "--json") {
            as_json = true;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink profile: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty()) {
        err << "cpplink profile: --schema <schema.json> and a parquet file are "
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
    if (!LoadParquetFiles(data_paths, schema, &store, &stats, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    TruthPairs truth;
    if (!truth_path.empty() && !LoadTruthPairs(truth_path, store, &truth, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    const ProfileReport report =
        BuildProfile(store, DefaultMode(mode_given, mode, data_paths.size()), options,
                     truth_path.empty() ? nullptr : &truth);
    if (as_json) {
        WriteProfileJson(report, out);
    } else {
        PrintProfileReport(report, out);
    }
    return 0;
}

int RunLevels(const std::vector<std::string>& args, std::ostream& out,
              std::ostream& err) {
    std::string schema_path;
    std::string out_path;
    std::string truth_path;
    std::string value;
    std::vector<std::string> data_paths;
    LevelsOptions options;
    bool as_json = false;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--out") {
            if (!TakeValue(args, &i, &out_path, err)) return 1;
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &truth_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--jaro-floor") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.jaro_floor = std::stod(value);
        } else if (args[i] == "--jaro-step") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.jaro_step = std::stod(value);
        } else if (args[i] == "--edit-max") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.edit_max = static_cast<uint32_t>(std::stoul(value));
        } else if (args[i] == "--max-levels") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.max_levels = static_cast<size_t>(std::stoul(value));
        } else if (args[i] == "--levels") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.levels = static_cast<size_t>(std::stoul(value));
        } else if (args[i] == "--min-match-pairs") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.min_match_pairs = std::stoull(value);
        } else if (args[i] == "--anchor-margin") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.profile.anchor_margin = std::stod(value);
        } else if (args[i] == "--anchor-rows") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.profile.anchor_rows = std::stoull(value);
        } else if (args[i] == "--anchor-pairs") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.profile.anchor_pairs = std::stoull(value);
        } else if (args[i] == "--expected-matches") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.profile.expected_matches = std::stoull(value);
        } else if (args[i] == "--ball-budget") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.ball.budget = std::stoull(value);
        } else if (args[i] == "--threads") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threads = static_cast<unsigned>(std::stoul(value));
            options.profile.threads = options.threads;
        } else if (args[i] == "--json") {
            as_json = true;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink levels: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty()) {
        err << "cpplink levels: --schema <schema.json> and a parquet file are "
               "required\n";
        return 1;
    }
    if (options.max_levels < 2) {
        err << "cpplink levels: --max-levels must be at least 2\n";
        return 1;
    }

    Schema schema;
    std::string error;
    if (!LoadSchema(schema_path, &schema, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    if (schema.comparisons.empty()) {
        err << "cpplink levels: the schema declares no comparisons\n";
        return 1;
    }

    RecordStore store(schema);
    LoadStats stats;
    if (!LoadParquetFiles(data_paths, schema, &store, &stats, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    ComparisonSet comparisons;
    if (!comparisons.Bind(schema, store, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    TruthPairs truth;
    if (!truth_path.empty() && !LoadTruthPairs(truth_path, store, &truth, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    const LevelsReport report =
        BuildLevels(store, comparisons, DefaultMode(mode_given, mode, data_paths.size()),
                    options, truth_path.empty() ? nullptr : &truth);
    if (as_json) {
        WriteLevelsJson(report, out);
    } else {
        PrintLevelsReport(report, out);
    }

    if (out_path.empty()) return 0;
    std::ifstream source(schema_path);
    if (!source) {
        err << "cpplink levels: cannot reread " << schema_path << "\n";
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(source)),
                           std::istreambuf_iterator<char>());
    std::string rewritten;
    if (!RewriteSchema(text, report, &rewritten, &error)) {
        err << "cpplink levels: " << error << "\n";
        return 1;
    }
    std::ofstream target(out_path);
    if (!target) {
        err << "cpplink levels: cannot write " << out_path << "\n";
        return 1;
    }
    target << rewritten;
    if (!as_json) out << "\nWrote the proposal to " << out_path << "\n";
    return 0;
}

int RunSimplify(const std::vector<std::string>& args, std::ostream& out,
                std::ostream& err) {
    std::string schema_path;
    std::string model_path;
    std::string out_path;
    std::string value;
    std::vector<std::string> data_paths;
    SimplifyOptions options;
    bool as_json = false;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--model") {
            if (!TakeValue(args, &i, &model_path, err)) return 1;
        } else if (args[i] == "--out") {
            if (!TakeValue(args, &i, &out_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--alpha") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.alpha = std::stod(value);
        } else if (args[i] == "--min-gap") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.min_gap = std::stod(value);
        } else if (args[i] == "--pair-cap") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.pair_cap = std::stoull(value);
        } else if (args[i] == "--threads") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threads = static_cast<unsigned>(std::stoul(value));
        } else if (args[i] == "--seed") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.seed = std::stoull(value);
        } else if (args[i] == "--json") {
            as_json = true;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink simplify: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || model_path.empty() || data_paths.empty()) {
        err << "cpplink simplify: --schema <schema.json>, --model <model.json> and a "
               "parquet file are required\n";
        return 1;
    }
    if (options.alpha <= 0.0 || options.alpha >= 1.0) {
        err << "cpplink simplify: --alpha must be between 0 and 1\n";
        return 1;
    }

    Schema schema;
    std::string error;
    if (!LoadSchema(schema_path, &schema, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    if (schema.comparisons.empty()) {
        err << "cpplink simplify: the schema declares no comparisons\n";
        return 1;
    }
    Model model;
    if (!LoadModel(model_path, &model, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    RecordStore store(schema);
    LoadStats stats;
    if (!LoadParquetFiles(data_paths, schema, &store, &stats, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    ComparisonSet comparisons;
    if (!comparisons.Bind(schema, store, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    if (!ModelMatches(model, comparisons, &error)) {
        err << "cpplink simplify: " << error << "\n";
        return 1;
    }
    BlockingPlan plan;
    if (!plan.Build(schema, store, DefaultMode(mode_given, mode, data_paths.size()),
                    &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    const SimplifyReport report = BuildSimplify(store, comparisons, plan, model, options);
    if (as_json) {
        WriteSimplifyJson(report, out);
    } else {
        PrintSimplifyReport(report, out);
    }

    if (out_path.empty()) return 0;
    if (report.merged_levels == 0) {
        if (!as_json) out << "\nNothing to write: the schema is already simple.\n";
        return 0;
    }
    std::ifstream source(schema_path);
    if (!source) {
        err << "cpplink simplify: cannot reread " << schema_path << "\n";
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(source)),
                           std::istreambuf_iterator<char>());
    std::string rewritten;
    if (!RewriteSchema(text, report, &rewritten, &error)) {
        err << "cpplink simplify: " << error << "\n";
        return 1;
    }
    std::ofstream target(out_path);
    if (!target) {
        err << "cpplink simplify: cannot write " << out_path << "\n";
        return 1;
    }
    target << rewritten;
    if (!as_json) out << "\nWrote the simplified schema to " << out_path << "\n";
    return 0;
}

// Splits "a,b" into its two halves.
bool SplitPair(const std::string& text, std::string* first, std::string* second) {
    const size_t comma = text.find(',');
    if (comma == std::string::npos || comma == 0 || comma + 1 >= text.size()) {
        return false;
    }
    *first = text.substr(0, comma);
    *second = text.substr(comma + 1);
    return true;
}

int RunExplain(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string pair;
    std::string rows;
    std::string model_path;
    std::string value;
    ScoreOptions score;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--model") {
            if (!TakeValue(args, &i, &model_path, err)) return 1;
        } else if (args[i] == "--threshold") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            score.threshold = std::stod(value);
        } else if (args[i] == "--tf-damping") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            score.tf_damping = std::stod(value);
        } else if (args[i] == "--pair") {
            if (!TakeValue(args, &i, &pair, err)) return 1;
        } else if (args[i] == "--rows") {
            if (!TakeValue(args, &i, &rows, err)) return 1;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink explain: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty()) {
        err << "cpplink explain: --schema <schema.json> and a parquet file are "
               "required\n";
        return 1;
    }
    if (pair.empty() == rows.empty()) {
        err << "cpplink explain: give exactly one of --pair <id_a>,<id_b> or "
               "--rows <i>,<j>\n";
        return 1;
    }

    Schema schema;
    std::string error;
    if (!LoadSchema(schema_path, &schema, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    if (schema.comparisons.empty()) {
        err << "cpplink explain: the schema declares no \"comparisons\"\n";
        return 1;
    }

    RecordStore store(schema);
    if (!LoadParquetFiles(data_paths, schema, &store, nullptr, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    ComparisonSet comparisons;
    if (!comparisons.Bind(schema, store, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    uint64_t row_a = 0;
    uint64_t row_b = 0;
    std::string first;
    std::string second;
    if (!pair.empty()) {
        if (!SplitPair(pair, &first, &second)) {
            err << "cpplink explain: --pair wants <id_a>,<id_b>\n";
            return 1;
        }
        if (!FindRowById(store, first, &row_a)) {
            err << "cpplink explain: no record with id '" << first << "'\n";
            return 1;
        }
        if (!FindRowById(store, second, &row_b)) {
            err << "cpplink explain: no record with id '" << second << "'\n";
            return 1;
        }
    } else {
        if (!SplitPair(rows, &first, &second)) {
            err << "cpplink explain: --rows wants <i>,<j>\n";
            return 1;
        }
        row_a = std::stoull(first);
        row_b = std::stoull(second);
        if (row_a >= store.NumRecords() || row_b >= store.NumRecords()) {
            err << "cpplink explain: row out of range; the file has "
                << store.NumRecords() << " records\n";
            return 1;
        }
    }

    PrintGammaLayout(comparisons, out);
    out << "\n";
    PrintPairExplanation(store, comparisons, row_a, row_b, out);

    // Without a model there is no weight to explain: the levels are the whole
    // story, and the waterfall is simply not printed.
    if (!model_path.empty()) {
        Model model;
        if (!LoadModel(model_path, &model, &error)) {
            err << "cpplink: " << error << "\n";
            return 1;
        }
        Scorer scorer;
        if (!scorer.Bind(model, comparisons, store, score, &error)) {
            err << "cpplink explain: " << error << "\n";
            return 1;
        }
        PrintPairWaterfall(store, comparisons, scorer, row_a, row_b, out);
    }
    return 0;
}

// Builds the neighbourhood masses a fuzzy term-frequency adjustment needs, and
// says which columns got one. A column too large for the budget keeps today's
// behaviour, which is worth saying out loud rather than degrading quietly.
void BuildBallTables(const ComparisonSet& comparisons, const RecordStore& store,
                     const BallOptions& options, BallTables* balls, std::ostream& out) {
    balls->Build(comparisons, store.NumRecords(), options);
    out << "Neighbourhood masses in " << std::fixed << std::setprecision(1)
        << balls->seconds << " s\n";
    for (size_t c = 0; c < comparisons.Size(); ++c) {
        out << "  " << comparisons.at(c).spec->name << ": ";
        if (balls->Has(c)) {
            out << balls->tables[c].Values() << " values, "
                << balls->tables[c].ValuePairs() << " value pairs, "
                << std::setprecision(2) << balls->tables[c].Seconds() << " s\n";
        } else {
            out << balls->reasons[c] << "\n";
        }
    }
    out << "\n";
}

// Loads a schema and a parquet file, the opening move of every blocking command.
// `--all-pairs` replaces whatever the schema declares with the one source that
// blocks on nothing, so the same schema can be run blocked and unblocked without
// being edited. It is what makes the schema's "blocking" section optional: on an
// input small enough to enumerate, there is nothing for it to say.
bool LoadForBlocking(const std::string& schema_path,
                     const std::vector<std::string>& data_paths, PairMode mode,
                     Schema* schema, std::unique_ptr<RecordStore>* store,
                     BlockingPlan* plan, std::ostream& err, bool all_pairs = false) {
    std::string error;
    if (!LoadSchema(schema_path, schema, &error)) {
        err << "cpplink: " << error << "\n";
        return false;
    }
    if (all_pairs) {
        BlockingSpec spec;
        spec.kind = SourceKind::kAllPairs;
        spec.name = "all pairs";
        schema->blocking.assign(1, spec);
    } else if (schema->blocking.empty()) {
        err << "cpplink: the schema declares no \"blocking\" sources; pass "
               "--all-pairs to\n         enumerate every pair instead\n";
        return false;
    }
    *store = std::make_unique<RecordStore>(*schema);
    if (!LoadParquetFiles(data_paths, *schema, store->get(), nullptr, &error)) {
        err << "cpplink: " << error << "\n";
        return false;
    }
    if (!plan->Build(*schema, **store, mode, &error)) {
        err << "cpplink: " << error << "\n";
        return false;
    }
    return true;
}

int RunExplainBlocking(const std::vector<std::string>& args, std::ostream& out,
                       std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string value;
    bool count_union = false;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    bool all_pairs = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--all-pairs") {
            all_pairs = true;
        } else if (args[i] == "--count") {
            count_union = true;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink explain-blocking: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty()) {
        err << "cpplink explain-blocking: --schema <schema.json> and a parquet file "
               "are required\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_paths,
                         DefaultMode(mode_given, mode, data_paths.size()), &schema,
                         &store, &plan, err, all_pairs)) {
        return 1;
    }
    PrintBlockingReport(plan, *store, count_union, out);
    return 0;
}

int RunRecall(const std::vector<std::string>& args, std::ostream& out,
              std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string truth_path;
    std::string value;
    bool why = false;
    size_t show_misses = 0;
    bool count_union = false;
    bool as_json = false;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    bool all_pairs = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--all-pairs") {
            all_pairs = true;
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &truth_path, err)) return 1;
        } else if (args[i] == "--why") {
            why = true;
        } else if (args[i] == "--count") {
            count_union = true;
        } else if (args[i] == "--json") {
            as_json = true;
        } else if (args[i] == "--show-misses") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            show_misses = static_cast<size_t>(std::stoul(value));
            why = true;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink recall: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty() || truth_path.empty()) {
        err << "cpplink recall: --schema <schema.json>, --truth <truth.csv> and a "
               "parquet file are required\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_paths,
                         DefaultMode(mode_given, mode, data_paths.size()), &schema,
                         &store, &plan, err, all_pairs)) {
        return 1;
    }

    TruthPairs truth;
    std::string error;
    if (!LoadTruthPairs(truth_path, *store, &truth, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    const RecallMetrics metrics = MeasureRecall(plan, *store, truth, count_union);
    if (as_json) {
        WriteRecallJson(metrics, out);
    } else {
        PrintRecallReport(metrics, out);
    }

    if (why) {
        if (schema.comparisons.empty()) {
            err << "cpplink recall: --why needs the schema to declare "
                   "\"comparisons\", to say what still agrees on a missed pair\n";
            return 1;
        }
        ComparisonSet comparisons;
        if (!comparisons.Bind(schema, *store, &error)) {
            err << "cpplink: " << error << "\n";
            return 1;
        }
        MissReport report;
        report.example_limit = show_misses;
        DiagnoseMisses(plan, comparisons, truth, &report);
        PrintMissReport(plan, report, out);
    }
    return 0;
}

int RunCompleteness(const std::vector<std::string>& args, std::ostream& out,
                    std::ostream& err) {
    std::string schema_path;
    std::string model_path;
    std::string truth_path;
    std::vector<std::string> data_paths;
    std::string value;
    CompletenessOptions options;
    bool as_json = false;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    bool all_pairs = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--model") {
            if (!TakeValue(args, &i, &model_path, err)) return 1;
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &truth_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--all-pairs") {
            all_pairs = true;
        } else if (args[i] == "--threads") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threads = static_cast<unsigned>(std::stoul(value));
        } else if (args[i] == "--sample") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.sample = std::stod(value);
            if (options.sample <= 0.0 || options.sample > 1.0) {
                err << "cpplink completeness: --sample wants a rate in (0, 1]\n";
                return 1;
            }
        } else if (args[i] == "--min-observed") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.min_observed = std::stoull(value);
        } else if (args[i] == "--value-weighting") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (value == "pairs") {
                options.pair_weighting = true;
            } else if (value == "records") {
                options.pair_weighting = false;
            } else {
                err << "cpplink completeness: --value-weighting wants "
                       "\"records\" or \"pairs\"\n";
                return 1;
            }
        } else if (args[i] == "--bound-only") {
            options.skip_observed = true;
        } else if (args[i] == "--json") {
            as_json = true;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink completeness: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty() || model_path.empty()) {
        err << "cpplink completeness: --schema <schema.json>, --model <model.json> "
               "and a parquet file are required\n";
        return 1;
    }

    Model model;
    std::string error;
    if (!LoadModel(model_path, &model, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_paths,
                         DefaultMode(mode_given, mode, data_paths.size()), &schema,
                         &store, &plan, err, all_pairs)) {
        return 1;
    }
    if (schema.comparisons.empty()) {
        err << "cpplink completeness: the schema declares no \"comparisons\"\n";
        return 1;
    }
    ComparisonSet comparisons;
    if (!comparisons.Bind(schema, *store, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    CompletenessReport report;
    if (!EstimateCompleteness(*store, comparisons, plan, model, options, &report,
                              &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    // With a truth file the estimator is scored against the thing it exists to
    // replace, which is the only way to know whether to believe it without one.
    if (!truth_path.empty()) {
        TruthPairs truth;
        if (!LoadTruthPairs(truth_path, *store, &truth, &error)) {
            err << "cpplink: " << error << "\n";
            return 1;
        }
        const RecallMetrics metrics = MeasureRecall(plan, *store, truth, false);
        if (metrics.truth_pairs > 0) {
            report.measured = true;
            report.truth_pairs = metrics.truth_pairs;
            report.pc_measured = static_cast<double>(metrics.union_found) /
                                 static_cast<double>(metrics.truth_pairs);
        }
    }

    if (as_json) {
        WriteCompletenessJson(report, out);
    } else {
        PrintCompletenessReport(report, out);
    }
    return 0;
}

int RunEstimate(const std::vector<std::string>& args, std::ostream& out,
                std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string model_path;
    std::string value;
    EstimateOptions options;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    bool all_pairs = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--all-pairs") {
            all_pairs = true;
        } else if (args[i] == "--out") {
            if (!TakeValue(args, &i, &model_path, err)) return 1;
        } else if (args[i] == "--u-sample") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.u_sample = std::stoull(value);
        } else if (args[i] == "--session-pairs") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.session_pairs = std::stoull(value);
        } else if (args[i] == "--threads") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threads = static_cast<unsigned>(std::stoul(value));
        } else if (args[i] == "--iterations") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.max_iterations = std::stoi(value);
        } else if (args[i] == "--lambda") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.lambda = std::stod(value);
        } else if (args[i] == "--seed") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.seed = std::stoull(value);
        } else if (args[i] == "--no-tie-holdout") {
            options.exclude_tied = false;
        } else if (args[i] == "--tied-bits") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.tied_bits = std::stod(value);
        } else if (args[i] == "--tie-sample-rows") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.tie_sample_rows = std::stoull(value);
        } else if (args[i] == "--interactions") {
            options.interactions.enabled = true;
        } else if (args[i] == "--max-interactions") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.interactions.enabled = true;
            options.interactions.max_terms = std::stoull(value);
        } else if (args[i] == "--interaction-clamp") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.interactions.clamp_bits = std::stod(value);
        } else if (args[i] == "--min-interaction-sessions") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.interactions.min_sessions = std::stoull(value);
        } else if (args[i] == "--min-pairs-per-parameter") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.interactions.min_pairs_per_parameter = std::stod(value);
        } else if (args[i] == "--interaction-bits") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.interactions.enabled = true;
            options.interactions.min_bits = std::stod(value);
        } else if (args[i] == "--fuzzy-u") {
            options.fuzzy_u = true;
        } else if (args[i] == "--ball-budget") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.ball.budget = std::stoull(value);
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink estimate: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty()) {
        err << "cpplink estimate: --schema <schema.json> and a parquet file are "
               "required\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_paths,
                         DefaultMode(mode_given, mode, data_paths.size()), &schema,
                         &store, &plan, err, all_pairs)) {
        return 1;
    }
    if (schema.comparisons.empty()) {
        err << "cpplink estimate: the schema declares no \"comparisons\"\n";
        return 1;
    }

    ComparisonSet comparisons;
    std::string error;
    if (!comparisons.Bind(schema, *store, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    Model model;
    EstimateReport report;
    if (!Estimate(*store, comparisons, plan, options, &model, &report, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    PrintEstimateReport(report, out);
    PrintModel(model, out);
    if (!model_path.empty()) {
        if (!WriteModelJson(model, model_path, &error)) {
            err << "cpplink: " << error << "\n";
            return 1;
        }
        out << "\nWrote " << model_path << "\n";
    }
    return 0;
}

// `--out` names either a directory of shards or the single file they are to be
// merged into, and the extension is what says which. The staging directory sits
// beside the file so a run that dies mid-merge leaves its shards somewhere
// obvious rather than in a temporary directory nobody looks in.
bool ResolveEdgeOutput(const std::string& command, const std::string& out,
                       bool format_given, std::string* out_dir, std::string* merge_path,
                       std::ostream& err) {
    MergeFormat format = MergeFormat::kCsv;
    if (!MergedFormatOf(out, &format)) {
        *out_dir = out;
        return true;
    }
    if (format_given) {
        err << "cpplink " << command << ": --format names the shard format, and --out "
            << out
            << " asks for a single file; the file's extension picks csv or "
               "parquet\n";
        return false;
    }
    *merge_path = out;
    *out_dir = out + ".shards";
    return true;
}

int RunPredict(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string model_path;
    std::string value;
    PredictOptions options;
    ScoreOptions score;
    BallOptions ball;
    bool fuzzy_tf = false;
    bool have_threshold = false;
    bool use_signatures = true;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    bool all_pairs = false;
    bool format_given = false;
    std::string out_path;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--all-pairs") {
            all_pairs = true;
        } else if (args[i] == "--model") {
            if (!TakeValue(args, &i, &model_path, err)) return 1;
        } else if (args[i] == "--out") {
            if (!TakeValue(args, &i, &out_path, err)) return 1;
        } else if (args[i] == "--threshold") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            score.threshold = std::stod(value);
            have_threshold = true;
        } else if (args[i] == "--probability") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            const double probability = std::stod(value);
            if (probability <= 0.0 || probability >= 1.0) {
                err << "cpplink predict: --probability wants a value in (0, 1)\n";
                return 1;
            }
            score.threshold = WeightForProbability(probability);
            have_threshold = true;
        } else if (args[i] == "--format") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (value == "bin") {
                options.format = EdgeFormat::kBinary;
            } else if (value == "csv") {
                options.format = EdgeFormat::kCsv;
            } else {
                err << "cpplink predict: --format wants bin or csv\n";
                return 1;
            }
            format_given = true;
        } else if (args[i] == "--threads") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threads = static_cast<unsigned>(std::stoul(value));
        } else if (args[i] == "--limit") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.max_edges = std::stoull(value);
        } else if (args[i] == "--tf-damping") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            score.tf_damping = std::stod(value);
        } else if (args[i] == "--no-bounds") {
            score.use_bounds = false;
        } else if (args[i] == "--no-ceiling") {
            score.use_ceiling = false;
        } else if (args[i] == "--fuzzy-tf") {
            fuzzy_tf = true;
        } else if (args[i] == "--ball-budget") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            ball.budget = std::stoull(value);
        } else if (args[i] == "--spill") {
            if (!TakeValue(args, &i, &options.spill_dir, err)) return 1;
        } else if (args[i] == "--spill-sample") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.spill_sample = std::stod(value);
            if (options.spill_sample <= 0.0 || options.spill_sample > 1.0) {
                err << "cpplink predict: --spill-sample wants a rate in (0, 1]\n";
                return 1;
            }
        } else if (args[i] == "--no-signatures") {
            use_signatures = false;
        } else if (args[i] == "--no-interactions") {
            score.use_interactions = false;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink predict: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty() || model_path.empty() ||
        out_path.empty()) {
        err << "cpplink predict: --schema <schema.json>, --model <model.json>, "
               "--out <dir|file.csv|file.parquet> and a parquet file are "
               "required\n";
        return 1;
    }
    if (!ResolveEdgeOutput("predict", out_path, format_given, &options.out_dir,
                           &options.merge_path, err)) {
        return 1;
    }
    if (!have_threshold) {
        err << "cpplink predict: give a --threshold in bits or a --probability\n";
        return 1;
    }

    Model model;
    std::string error;
    if (!LoadModel(model_path, &model, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_paths,
                         DefaultMode(mode_given, mode, data_paths.size()), &schema,
                         &store, &plan, err, all_pairs)) {
        return 1;
    }
    if (schema.comparisons.empty()) {
        err << "cpplink predict: the schema declares no \"comparisons\"\n";
        return 1;
    }

    ComparisonSet comparisons;
    if (!comparisons.Bind(schema, *store, &error, use_signatures)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    BallTables balls;
    if (fuzzy_tf) {
        ball.threads = options.threads;
        BuildBallTables(comparisons, *store, ball, &balls, out);
    }

    Scorer scorer;
    if (!scorer.Bind(model, comparisons, *store, score, &error,
                     fuzzy_tf ? &balls : nullptr)) {
        err << "cpplink predict: " << error << "\n";
        return 1;
    }

    PredictReport report;
    if (!Predict(*store, comparisons, plan, scorer, options, &report, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    PrintPredictReport(report, scorer, out);
    return 0;
}

// Clustering needs only the identifiers: the edges already carry every row index
// and weight, so the comparison columns are left on disk rather than interned.
bool LoadIdsOnly(const std::string& schema_path,
                 const std::vector<std::string>& data_paths,
                 std::unique_ptr<RecordStore>* store, std::ostream& err) {
    Schema schema;
    std::string error;
    if (!LoadSchema(schema_path, &schema, &error)) {
        err << "cpplink: " << error << "\n";
        return false;
    }
    schema.columns.clear();
    schema.comparisons.clear();
    schema.blocking.clear();
    *store = std::make_unique<RecordStore>(schema);
    if (!LoadParquetFiles(data_paths, schema, store->get(), nullptr, &error)) {
        err << "cpplink: " << error << "\n";
        return false;
    }
    return true;
}

int RunCluster(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string truth_path;
    std::string value;
    ClusterOptions options;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--predictions") {
            if (!TakeValue(args, &i, &options.edge_path, err)) return 1;
        } else if (args[i] == "--out") {
            if (!TakeValue(args, &i, &options.out_path, err)) return 1;
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &truth_path, err)) return 1;
        } else if (args[i] == "--threshold") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threshold = std::stod(value);
        } else if (args[i] == "--probability") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            const double probability = std::stod(value);
            if (probability <= 0.0 || probability >= 1.0) {
                err << "cpplink cluster: --probability wants a value in (0, 1)\n";
                return 1;
            }
            options.threshold = WeightForProbability(probability);
        } else if (args[i] == "--min-size") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.min_size = std::stoull(value);
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink cluster: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty() || options.edge_path.empty()) {
        err << "cpplink cluster: --schema <schema.json>, --predictions <dir|file> "
               "and a parquet file are required\n";
        return 1;
    }

    std::unique_ptr<RecordStore> store;
    if (!LoadIdsOnly(schema_path, data_paths, &store, err)) return 1;

    std::string error;
    ClusterAssignment assignment;
    ClusterReport report;
    if (!Cluster(*store, options, &assignment, &report, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    if (!WriteClusters(assignment, *store, options, &report.written, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    PrintClusterReport(report, out);

    if (!truth_path.empty()) {
        TruthPairs truth;
        if (!LoadTruthPairs(truth_path, *store, &truth, &error)) {
            err << "cpplink: " << error << "\n";
            return 1;
        }
        PrintClusterQuality(MeasureClusters(assignment, truth), out);
    }
    if (!options.out_path.empty()) {
        out << "\nWrote " << options.out_path << "\n";
    }
    return 0;
}

int RunMergeEdges(const std::vector<std::string>& args, std::ostream& out,
                  std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string value;
    MergeOptions options;
    bool format_given = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--shards") {
            if (!TakeValue(args, &i, &options.edge_dir, err)) return 1;
        } else if (args[i] == "--out") {
            if (!TakeValue(args, &i, &options.out_path, err)) return 1;
        } else if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--format") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (value == "csv") {
                options.format = MergeFormat::kCsv;
            } else if (value == "parquet") {
                options.format = MergeFormat::kParquet;
            } else {
                err << "cpplink merge-predictions: --format wants csv or parquet\n";
                return 1;
            }
            format_given = true;
        } else if (args[i] == "--from") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (value == "bin") {
                options.source = MergeSource::kBinary;
            } else if (value == "csv") {
                options.source = MergeSource::kCsv;
            } else {
                err << "cpplink merge-predictions: --from wants bin or csv\n";
                return 1;
            }
        } else if (args[i] == "--threshold") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threshold = std::stod(value);
        } else if (args[i] == "--probability") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            const double probability = std::stod(value);
            if (probability <= 0.0 || probability >= 1.0) {
                err << "cpplink merge-predictions: --probability wants a value in (0, "
                       "1)\n";
                return 1;
            }
            options.threshold = WeightForProbability(probability);
        } else if (args[i] == "--batch-rows") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.batch_rows = static_cast<size_t>(std::stoull(value));
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink merge-predictions: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (options.edge_dir.empty() || options.out_path.empty()) {
        err << "cpplink merge-predictions: --shards <dir> and --out <file> are "
               "required\n";
        return 1;
    }
    // The extension is what a user means by the format; --format is for a name
    // that does not carry one.
    if (!format_given) {
        const std::string suffix = std::filesystem::path(options.out_path).extension();
        if (suffix == ".parquet" || suffix == ".pq") {
            options.format = MergeFormat::kParquet;
        }
    }
    if (schema_path.empty() != data_paths.empty()) {
        err << "cpplink merge-predictions: naming the ids takes both --schema and the "
               "parquet input\n";
        return 1;
    }

    // Binary shards name rows; only the store can turn one back into an id.
    std::unique_ptr<RecordStore> store;
    if (!schema_path.empty()) {
        if (!LoadIdsOnly(schema_path, data_paths, &store, err)) return 1;
    }

    std::string error;
    MergeReport report;
    if (!MergeEdges(store.get(), options, &report, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    PrintMergeReport(report, out);
    return 0;
}

int RunRescore(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string model_path;
    std::string value;
    RescoreOptions options;
    ScoreOptions score;
    bool have_threshold = false;
    bool format_given = false;
    std::string out_path;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--model") {
            if (!TakeValue(args, &i, &model_path, err)) return 1;
        } else if (args[i] == "--spill") {
            if (!TakeValue(args, &i, &options.spill_dir, err)) return 1;
        } else if (args[i] == "--out") {
            if (!TakeValue(args, &i, &out_path, err)) return 1;
        } else if (args[i] == "--threshold") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            score.threshold = std::stod(value);
            have_threshold = true;
        } else if (args[i] == "--probability") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            const double probability = std::stod(value);
            if (probability <= 0.0 || probability >= 1.0) {
                err << "cpplink rescore: --probability wants a value in (0, 1)\n";
                return 1;
            }
            score.threshold = WeightForProbability(probability);
            have_threshold = true;
        } else if (args[i] == "--format") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (value == "bin") {
                options.format = EdgeFormat::kBinary;
            } else if (value == "csv") {
                options.format = EdgeFormat::kCsv;
            } else {
                err << "cpplink rescore: --format wants bin or csv\n";
                return 1;
            }
            format_given = true;
        } else if (args[i] == "--threads") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.threads = static_cast<unsigned>(std::stoul(value));
        } else if (args[i] == "--limit") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            options.max_edges = std::stoull(value);
        } else if (args[i] == "--tf-damping") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            score.tf_damping = std::stod(value);
        } else if (args[i] == "--no-bounds") {
            score.use_bounds = false;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink rescore: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty() || model_path.empty() ||
        options.spill_dir.empty() || out_path.empty()) {
        err << "cpplink rescore: --schema <schema.json>, --model <model.json>, "
               "--spill <dir>, --out <dir|file.csv|file.parquet> and a parquet "
               "file are required\n";
        return 1;
    }
    if (!ResolveEdgeOutput("rescore", out_path, format_given, &options.out_dir,
                           &options.merge_path, err)) {
        return 1;
    }
    if (!have_threshold) {
        err << "cpplink rescore: give a --threshold in bits or a --probability\n";
        return 1;
    }

    Model model;
    std::string error;
    if (!LoadModel(model_path, &model, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_paths,
                         DefaultMode(mode_given, mode, data_paths.size()), &schema,
                         &store, &plan, err)) {
        return 1;
    }
    ComparisonSet comparisons;
    if (!comparisons.Bind(schema, *store, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    Scorer scorer;
    if (!scorer.Bind(model, comparisons, *store, score, &error)) {
        err << "cpplink rescore: " << error << "\n";
        return 1;
    }

    RescoreReport report;
    if (!Rescore(*store, comparisons, scorer, options, &report, &error)) {
        err << error << "\n";
        return 1;
    }
    PrintRescoreReport(report, scorer, out);
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
        } else if (args[i] == "--out-b") {
            if (!TakeValue(args, &i, &options.link_path, err)) return 1;
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
    if (first == "profile") return RunProfile(rest, out, err);
    if (first == "levels") return RunLevels(rest, out, err);
    if (first == "simplify") return RunSimplify(rest, out, err);
    if (first == "explain") return RunExplain(rest, out, err);
    if (first == "explain-blocking") return RunExplainBlocking(rest, out, err);
    if (first == "recall") return RunRecall(rest, out, err);
    if (first == "estimate") return RunEstimate(rest, out, err);
    if (first == "completeness") return RunCompleteness(rest, out, err);
    if (first == "predict") return RunPredict(rest, out, err);
    if (first == "rescore") return RunRescore(rest, out, err);
    if (first == "cluster") return RunCluster(rest, out, err);
    if (first == "merge-predictions") return RunMergeEdges(rest, out, err);
    if (first == "gen-sample") return RunGenSample(rest, out, err);

    err << "cpplink: unknown command '" << first << "'\n";
    PrintUsage(err);
    return 1;
}

}  // namespace cpplink
