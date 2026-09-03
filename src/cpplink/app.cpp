// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/app.hpp"

#include <memory>
#include <ostream>

#include "cpplink/blocking.hpp"
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
        << "  explain     show the comparison levels a single pair lands on\n"
        << "  explain-blocking  price every blocking source without enumerating\n"
        << "  recall      measure what fraction of known pairs blocking reaches\n"
        << "  estimate    learn m, u and lambda and write the model\n"
        << "  predict     score the candidate pairs and write the edges above a "
           "threshold\n"
        << "  gen-sample  write a sample parquet file with planted duplicates\n"
        << "\n"
        << "options:\n"
        << "  -h, --help       show this message and exit\n"
        << "  -v, --version    show the version and exit\n"
        << "\n"
        << "cpplink inspect --schema <schema.json> <file.parquet>\n"
        << "cpplink explain --schema <schema.json> --pair <id_a>,<id_b> "
           "<file.parquet>\n"
        << "cpplink explain-blocking --schema <schema.json> [--count] "
           "<file.parquet>\n"
        << "cpplink recall --schema <schema.json> --truth <truth.csv> "
           "<file.parquet>\n"
        << "cpplink estimate --schema <schema.json> [--out <model.json>]\n"
        << "                 [--u-sample N] [--session-pairs N] [--threads N]\n"
        << "                 [--iterations N] [--lambda F] [--seed N] "
           "<file.parquet>\n"
        << "cpplink predict --schema <schema.json> --model <model.json> --out <dir>\n"
        << "                [--threshold BITS | --probability P] [--format bin|csv]\n"
        << "                [--threads N] [--limit N] [--no-bounds] [--tf-damping F]\n"
        << "                <file.parquet>\n"
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
    std::string data_path;
    std::string pair;
    std::string rows;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--pair") {
            if (!TakeValue(args, &i, &pair, err)) return 1;
        } else if (args[i] == "--rows") {
            if (!TakeValue(args, &i, &rows, err)) return 1;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink explain: unknown option '" << args[i] << "'\n";
            return 1;
        } else if (data_path.empty()) {
            data_path = args[i];
        } else {
            err << "cpplink explain: unexpected argument '" << args[i] << "'\n";
            return 1;
        }
    }
    if (schema_path.empty() || data_path.empty()) {
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
    if (!LoadParquet(data_path, schema, &store, nullptr, &error)) {
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
    return 0;
}

// Loads a schema and a parquet file, the opening move of every blocking command.
bool LoadForBlocking(const std::string& schema_path, const std::string& data_path,
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
    if (!LoadParquet(data_path, *schema, store->get(), nullptr, &error)) {
        err << "cpplink: " << error << "\n";
        return false;
    }
    if (!plan->Build(*schema, **store, &error)) {
        err << "cpplink: " << error << "\n";
        return false;
    }
    return true;
}

int RunExplainBlocking(const std::vector<std::string>& args, std::ostream& out,
                       std::ostream& err) {
    std::string schema_path;
    std::string data_path;
    bool count_union = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--count") {
            count_union = true;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink explain-blocking: unknown option '" << args[i] << "'\n";
            return 1;
        } else if (data_path.empty()) {
            data_path = args[i];
        } else {
            err << "cpplink explain-blocking: unexpected argument '" << args[i] << "'\n";
            return 1;
        }
    }
    if (schema_path.empty() || data_path.empty()) {
        err << "cpplink explain-blocking: --schema <schema.json> and a parquet file "
               "are required\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_path, &schema, &store, &plan, err)) {
        return 1;
    }
    PrintBlockingReport(plan, *store, count_union, out);
    return 0;
}

int RunRecall(const std::vector<std::string>& args, std::ostream& out,
              std::ostream& err) {
    std::string schema_path;
    std::string data_path;
    std::string truth_path;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
        } else if (args[i] == "--truth") {
            if (!TakeValue(args, &i, &truth_path, err)) return 1;
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink recall: unknown option '" << args[i] << "'\n";
            return 1;
        } else if (data_path.empty()) {
            data_path = args[i];
        } else {
            err << "cpplink recall: unexpected argument '" << args[i] << "'\n";
            return 1;
        }
    }
    if (schema_path.empty() || data_path.empty() || truth_path.empty()) {
        err << "cpplink recall: --schema <schema.json>, --truth <truth.csv> and a "
               "parquet file are required\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_path, &schema, &store, &plan, err)) {
        return 1;
    }

    TruthPairs truth;
    std::string error;
    if (!LoadTruthPairs(truth_path, *store, &truth, &error)) {
        err << "cpplink: " << error << "\n";
        return 1;
    }
    PrintRecallReport(plan, truth, out);
    return 0;
}

int RunEstimate(const std::vector<std::string>& args, std::ostream& out,
                std::ostream& err) {
    std::string schema_path;
    std::string data_path;
    std::string model_path;
    std::string value;
    EstimateOptions options;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
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
        } else if (data_path.empty()) {
            data_path = args[i];
        } else {
            err << "cpplink estimate: unexpected argument '" << args[i] << "'\n";
            return 1;
        }
    }
    if (schema_path.empty() || data_path.empty()) {
        err << "cpplink estimate: --schema <schema.json> and a parquet file are "
               "required\n";
        return 1;
    }

    Schema schema;
    std::unique_ptr<RecordStore> store;
    BlockingPlan plan;
    if (!LoadForBlocking(schema_path, data_path, &schema, &store, &plan, err)) {
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
    std::string data_path;
    std::string model_path;
    std::string value;
    PredictOptions options;
    ScoreOptions score;
    bool have_threshold = false;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--schema") {
            if (!TakeValue(args, &i, &schema_path, err)) return 1;
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
        } else if (!args[i].empty() && args[i][0] == '-') {
            err << "cpplink predict: unknown option '" << args[i] << "'\n";
            return 1;
        } else if (data_path.empty()) {
            data_path = args[i];
        } else {
            err << "cpplink predict: unexpected argument '" << args[i] << "'\n";
            return 1;
        }
    }
    if (schema_path.empty() || data_path.empty() || model_path.empty() ||
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
    if (!LoadForBlocking(schema_path, data_path, &schema, &store, &plan, err)) {
        return 1;
    }
    if (schema.comparisons.empty()) {
        err << "cpplink predict: the schema declares no \"comparisons\"\n";
        return 1;
    }

    ComparisonSet comparisons;
    if (!comparisons.Bind(schema, *store, &error)) {
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
    if (first == "explain") return RunExplain(rest, out, err);
    if (first == "explain-blocking") return RunExplainBlocking(rest, out, err);
    if (first == "recall") return RunRecall(rest, out, err);
    if (first == "estimate") return RunEstimate(rest, out, err);
    if (first == "predict") return RunPredict(rest, out, err);
    if (first == "gen-sample") return RunGenSample(rest, out, err);

    err << "cpplink: unknown command '" << first << "'\n";
    PrintUsage(err);
    return 1;
}

}  // namespace cpplink
