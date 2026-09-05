// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/app.hpp"

#include <memory>
#include <ostream>

#include "cpplink/blocking.hpp"
#include "cpplink/cluster.hpp"
#include "cpplink/comparison.hpp"
#include "cpplink/estimate.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/explain_blocking.hpp"
#include "cpplink/inspect.hpp"
#include "cpplink/model.hpp"
#include "cpplink/parquet_loader.hpp"
#include "cpplink/predict.hpp"
#include "cpplink/recall.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/rescore.hpp"
#include "cpplink/sample_data.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"

namespace cpplink {

const char* const kVersion = "0.1.0";

namespace {

void PrintUsage(std::ostream& out) {
    out << "usage: cpplink <command> [options]\n"
        << "\n"
        << "commands:\n"
        << "  inspect     load a parquet file and report cardinality and memory\n"
        << "  explain     show the levels a single pair lands on, and with a "
           "model\n"
        << "              the waterfall of bits behind its score\n"
        << "  explain-blocking  price every blocking source without enumerating\n"
        << "  recall      measure what fraction of known pairs blocking reaches,\n"
        << "              and with --why, diagnose the ones it does not\n"
        << "  estimate    learn m, u and lambda and write the model\n"
        << "  predict     score the candidate pairs and write the edges above a "
           "threshold\n"
        << "  cluster     join the scored edges into duplicate clusters\n"
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
        << "cpplink inspect --schema <schema.json> <file.parquet>...\n"
        << "cpplink explain --schema <schema.json> --pair <id_a>,<id_b>\n"
        << "                [--rows <i>,<j>] [--model <model.json>] "
           "[--threshold BITS]\n"
        << "                [--tf-damping F] <file.parquet>...\n"
        << "cpplink explain-blocking --schema <schema.json> [--count] "
           "[--mode MODE]\n"
        << "                <file.parquet>...\n"
        << "cpplink recall --schema <schema.json> --truth <truth.csv> [--why]\n"
        << "               [--show-misses N] [--mode MODE] <file.parquet>...\n"
        << "cpplink estimate --schema <schema.json> [--out <model.json>]\n"
        << "                 [--u-sample N] [--session-pairs N] [--threads N]\n"
        << "                 [--iterations N] [--lambda F] [--seed N] "
           "[--mode MODE]\n"
        << "                 <file.parquet>...\n"
        << "cpplink predict --schema <schema.json> --model <model.json> --out <dir>\n"
        << "                [--threshold BITS | --probability P] [--format bin|csv]\n"
        << "                [--threads N] [--limit N] [--no-bounds] [--tf-damping F]\n"
        << "                [--no-signatures] [--spill <dir>] [--spill-sample R]\n"
        << "                [--mode MODE] <file.parquet>...\n"
        << "cpplink rescore --schema <schema.json> --model <model.json> --spill <dir>\n"
        << "                --out <dir> [--threshold BITS | --probability P]\n"
        << "                [--format bin|csv] [--threads N] [--limit N] "
           "[--mode MODE]\n"
        << "                <file.parquet>...\n"
        << "cpplink cluster --schema <schema.json> --edges <dir> [--out <file.csv>]\n"
        << "                [--threshold BITS | --probability P] [--truth <file.csv>]\n"
        << "                [--min-size N] <file.parquet>...\n"
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

// Loads a schema and a parquet file, the opening move of every blocking command.
bool LoadForBlocking(const std::string& schema_path,
                     const std::vector<std::string>& data_paths, PairMode mode,
                     Schema* schema, std::unique_ptr<RecordStore>* store,
                     BlockingPlan* plan, std::ostream& err) {
    std::string error;
    if (!LoadSchema(schema_path, schema, &error)) {
        err << "cpplink: " << error << "\n";
        return false;
    }
    if (schema->blocking.empty()) {
        err << "cpplink: the schema declares no \"blocking\" sources\n";
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
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
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
                         &store, &plan, err)) {
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
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &truth_path, err)) return 1;
        } else if (args[i] == "--why") {
            why = true;
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
                         &store, &plan, err)) {
        return 1;
    }

    TruthPairs truth;
    std::string error;
    if (!LoadTruthPairs(truth_path, *store, &truth, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    PrintRecallReport(plan, truth, out);

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

int RunEstimate(const std::vector<std::string>& args, std::ostream& out,
                std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string model_path;
    std::string value;
    EstimateOptions options;
    PairMode mode = PairMode::kAll;
    bool mode_given = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--mode") {
            if (!TakeValue(args, &i, &value, err)) return 1;
            if (!ParseMode(value, &mode, err)) return 1;
            mode_given = true;
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
                         &store, &plan, err)) {
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

int RunPredict(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string model_path;
    std::string value;
    PredictOptions options;
    ScoreOptions score;
    bool have_threshold = false;
    bool use_signatures = true;
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
        } else if (args[i] == "--out") {
            if (!TakeValue(args, &i, &options.out_dir, err)) return 1;
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
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink predict: unknown option '" << args[i] << "'\n";
            return 1;
        } else {
            data_paths.push_back(args[i]);
        }
    }
    if (schema_path.empty() || data_paths.empty() || model_path.empty() ||
        options.out_dir.empty()) {
        err << "cpplink predict: --schema <schema.json>, --model <model.json>, "
               "--out <dir> and a parquet file are required\n";
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
                         &store, &plan, err)) {
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

    Scorer scorer;
    if (!scorer.Bind(model, comparisons, *store, score, &error)) {
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
        } else if (args[i] == "--edges") {
            if (!TakeValue(args, &i, &options.edge_dir, err)) return 1;
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
    if (schema_path.empty() || data_paths.empty() || options.edge_dir.empty()) {
        err << "cpplink cluster: --schema <schema.json>, --edges <dir> and a parquet "
               "file are required\n";
        return 1;
    }

    std::unique_ptr<RecordStore> store;
    if (!LoadIdsOnly(schema_path, data_paths, &store, err)) return 1;

    std::string error;
    ClusterAssignment assignment;
    ClusterReport report;
    if (!Cluster(store->NumRecords(), options, &assignment, &report, &error)) {
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

int RunRescore(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
    std::string schema_path;
    std::vector<std::string> data_paths;
    std::string model_path;
    std::string value;
    RescoreOptions options;
    ScoreOptions score;
    bool have_threshold = false;
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
            if (!TakeValue(args, &i, &options.out_dir, err)) return 1;
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
        options.spill_dir.empty() || options.out_dir.empty()) {
        err << "cpplink rescore: --schema <schema.json>, --model <model.json>, "
               "--spill <dir>, --out <dir> and a parquet file are required\n";
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
    if (first == "explain") return RunExplain(rest, out, err);
    if (first == "explain-blocking") return RunExplainBlocking(rest, out, err);
    if (first == "recall") return RunRecall(rest, out, err);
    if (first == "estimate") return RunEstimate(rest, out, err);
    if (first == "predict") return RunPredict(rest, out, err);
    if (first == "rescore") return RunRescore(rest, out, err);
    if (first == "cluster") return RunCluster(rest, out, err);
    if (first == "gen-sample") return RunGenSample(rest, out, err);

    err << "cpplink: unknown command '" << first << "'\n";
    PrintUsage(err);
    return 1;
}

}  // namespace cpplink
