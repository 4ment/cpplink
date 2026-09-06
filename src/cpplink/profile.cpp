// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/profile.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>  // NOLINT(build/c++11)
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "cpplink/format.hpp"
#include "cpplink/pair_stream.hpp"

namespace cpplink {
namespace {

// A pair is a suspect once one column is all but determined by the other, or one
// value all but always occurs inside the other, or the u-side overlap is worth a
// whole bit. A bit is the unit the score is in, so it is the unit the threshold
// belongs in too.
constexpr double kDeterminedShare = 0.99;
constexpr double kContainedShare = 0.90;
constexpr double kRedundantBits = 1.0;
// A column with this many distinct values per row determines whatever it is set
// against, by being a key rather than by carrying its information.
constexpr double kUniqueShare = 0.90;
// Collisions the joint table needs before phi is read off it rather than called
// unresolved. Below this the estimate is Poisson noise around a small number.
constexpr double kMinCollisions = 32.0;
// How far above the match rate an independent collision rate has to sit before
// what is measured is the columns rather than the duplicates the file holds.
constexpr double kMatchMargin = 8.0;

double Log2(double value) { return std::log(value) / std::log(2.0); }

// Which rows the pairwise pass reads. Selecting by a hash of the row index rather
// than by a stride keeps the sample independent of the order rows were loaded in,
// which matters because a link-mode store holds one input after another and a
// stride would weight them by nothing but their lengths.
uint64_t Mix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

const std::vector<uint32_t>* TermFrequencies(const Column& column) {
    if (const auto* col = std::get_if<StringColumn>(&column)) return &col->tf;
    if (const auto* col = std::get_if<StringListColumn>(&column)) return &col->tf;
    if (const auto* col = std::get_if<DateColumn>(&column)) return &col->tf;
    return nullptr;
}

// One scalar column, read by row without materialising a key array per column:
// ten columns of 18M keys is 720 MB to answer a question that is one pass wide.
struct ScalarView {
    const std::vector<uint32_t>* ids = nullptr;
    const std::vector<int32_t>* dates = nullptr;
    const Dictionary* dict = nullptr;
    int32_t origin = 0;

    bool Valid() const { return ids != nullptr || dates != nullptr; }

    uint32_t Key(uint64_t row) const {
        if (ids != nullptr) return (*ids)[row];
        const int32_t day = (*dates)[row];
        if (day == kNullDate) return kNullId;
        return static_cast<uint32_t>(day - origin);
    }
};

ScalarView ViewOf(const Column& column) {
    ScalarView view;
    if (const auto* col = std::get_if<StringColumn>(&column)) {
        view.ids = &col->ids;
        view.dict = &col->dict;
    } else if (const auto* col = std::get_if<DateColumn>(&column)) {
        view.dates = &col->values;
        view.origin = col->tf_origin;
    }
    return view;
}

std::string Percent(double share) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%.2f%%", 100.0 * share);
    return buffer;
}

std::string Fixed(double value, int decimals) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return buffer;
}

std::string Signed(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%+.2f", value);
    return buffer;
}

// Effective cardinality runs from 1 to the distinct count, which spans six orders
// of magnitude across the columns of one file. Three significant digits is the
// only formatting that stays readable at both ends.
std::string Compact(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.4g", value);
    return buffer;
}

// The joint value table of two scalar columns, over the rows carrying both.
struct JointTable {
    std::unordered_map<uint64_t, uint32_t> joint;
    std::unordered_map<uint32_t, uint32_t> left;
    std::unordered_map<uint32_t, uint32_t> right;
    uint64_t rows = 0;
    uint64_t left_inside_right = 0;
    uint64_t right_inside_left = 0;
};

std::vector<uint32_t> ValuesOf(const std::unordered_map<uint32_t, uint32_t>& counts) {
    std::vector<uint32_t> out;
    out.reserve(counts.size());
    for (const auto& entry : counts) out.push_back(entry.second);
    return out;
}

std::vector<uint32_t> ValuesOf(const std::unordered_map<uint64_t, uint32_t>& counts) {
    std::vector<uint32_t> out;
    out.reserve(counts.size());
    for (const auto& entry : counts) out.push_back(entry.second);
    return out;
}

// Ordered collisions: how many ordered pairs of distinct rows share a value.
double CollisionsIn(const std::vector<uint32_t>& counts) {
    double collisions = 0.0;
    for (uint32_t count : counts) {
        const double c = static_cast<double>(count);
        collisions += c * (c - 1.0);
    }
    return collisions;
}

// The probability that two *distinct* rows collide, which is what u is, estimated
// over draws without replacement.
//
// The plug-in sum of squared shares is not this and must not be used here. With a
// table of mostly singletons it reads 1/n, where the truth is near zero, so it
// calls two independent high-cardinality columns almost perfectly redundant: on
// `historical_50k` it put 9.4 spurious bits on a pair that shares nothing. This
// form reads exactly zero on that table, which is right, and no larger sample
// would have rescued the other one.
double Collision(const std::vector<uint32_t>& counts, uint64_t total) {
    if (total < 2) return 0.0;
    return CollisionsIn(counts) /
           (static_cast<double>(total) * static_cast<double>(total - 1));
}

void FoldPair(const RecordStore& store, const std::vector<uint64_t>& rows, bool sampled,
              const ScalarView& left, const ScalarView& right, JointTable* table) {
    const uint64_t count = sampled ? rows.size() : store.NumRecords();
    const bool strings = left.dict != nullptr && right.dict != nullptr;
    for (uint64_t k = 0; k < count; ++k) {
        const uint64_t row = sampled ? rows[k] : k;
        const uint32_t a = left.Key(row);
        if (a == kNullId) continue;
        const uint32_t b = right.Key(row);
        if (b == kNullId) continue;
        ++table->rows;
        ++table->joint[(static_cast<uint64_t>(a) << 32) | b];
        ++table->left[a];
        ++table->right[b];
        if (!strings) continue;
        const std::string_view first = left.dict->Value(a);
        const std::string_view second = right.dict->Value(b);
        if (!first.empty() && second.find(first) != std::string_view::npos) {
            ++table->left_inside_right;
        }
        if (!second.empty() && first.find(second) != std::string_view::npos) {
            ++table->right_inside_left;
        }
    }
}

// The share of rows kept by mapping each value of `keyed` to its commonest
// partner: 1 exactly when `keyed` functionally determines the other column.
double DeterminationShare(const std::unordered_map<uint64_t, uint32_t>& joint,
                          bool by_left, uint64_t rows) {
    if (rows == 0) return 0.0;
    std::unordered_map<uint32_t, uint32_t> best;
    for (const auto& entry : joint) {
        const uint32_t key = by_left ? static_cast<uint32_t>(entry.first >> 32)
                                     : static_cast<uint32_t>(entry.first);
        uint32_t& slot = best[key];
        slot = std::max(slot, entry.second);
    }
    uint64_t kept = 0;
    for (const auto& entry : best) kept += entry.second;
    return static_cast<double>(kept) / static_cast<double>(rows);
}

void Summarise(const JointTable& table, double match_rate, ColumnPairProfile* pair) {
    pair->rows = table.rows;
    if (table.rows == 0) return;
    const std::vector<uint32_t> left = ValuesOf(table.left);
    const std::vector<uint32_t> right = ValuesOf(table.right);
    const std::vector<uint32_t> joint = ValuesOf(table.joint);

    const double rows = static_cast<double>(table.rows);
    pair->determines_right = DeterminationShare(table.joint, true, table.rows);
    pair->determines_left = DeterminationShare(table.joint, false, table.rows);
    pair->left_distinct_share = static_cast<double>(table.left.size()) / rows;
    pair->right_distinct_share = static_cast<double>(table.right.size()) / rows;
    pair->baseline_right =
        right.empty() ? 0.0 : *std::max_element(right.begin(), right.end()) / rows;
    pair->baseline_left =
        left.empty() ? 0.0 : *std::max_element(left.begin(), left.end()) / rows;
    // Reading the left column's determination of the right needs a left that is
    // not a key and a right that is not a constant; either way round the share
    // reaches one for a reason that has nothing to do with the two columns.
    pair->left_informative = pair->left_distinct_share < kUniqueShare &&
                             pair->baseline_right < kDeterminedShare;
    pair->right_informative = pair->right_distinct_share < kUniqueShare &&
                              pair->baseline_left < kDeterminedShare;

    pair->u_left = Collision(left, table.rows);
    pair->u_right = Collision(right, table.rows);
    pair->u_joint = Collision(joint, table.rows);
    pair->joint_collisions = CollisionsIn(joint);
    const double independent = pair->u_left * pair->u_right;
    pair->expected_collisions = independent * rows * (rows - 1.0);
    if (independent >= kMatchMargin * match_rate &&
        pair->expected_collisions >= kMinCollisions && pair->u_joint > 0.0) {
        pair->resolved = true;
        pair->redundant_bits = Log2(pair->u_joint / independent);
    }

    const double inverse = 1.0 / rows;
    const double left_in = static_cast<double>(table.left_inside_right) * inverse;
    const double right_in = static_cast<double>(table.right_inside_left) * inverse;
    pair->left_inside_right = left_in >= right_in;
    pair->containment = std::max(left_in, right_in);
}

}  // namespace

bool ColumnPairProfile::Suspect() const {
    if (rows == 0) return false;
    if (left_informative && determines_right >= kDeterminedShare) return true;
    if (right_informative && determines_left >= kDeterminedShare) return true;
    if (containment >= kContainedShare) return true;
    return resolved && redundant_bits >= kRedundantBits;
}

std::string ColumnPairProfile::Verdict() const {
    const bool left_determines = left_informative && determines_right >= kDeterminedShare;
    const bool right_determines =
        right_informative && determines_left >= kDeterminedShare;
    if (left_determines && right_determines) {
        return left_name + " and " + right_name +
               " determine each other: drop one, or make the two one comparison";
    }
    if (left_determines) {
        return left_name + " determines " + right_name + ": drop " + right_name +
               ", or make the two one comparison";
    }
    if (right_determines) {
        return right_name + " determines " + left_name + ": drop " + left_name +
               ", or make the two one comparison";
    }
    if (containment >= kContainedShare) {
        const std::string& inner = left_inside_right ? left_name : right_name;
        const std::string& outer = left_inside_right ? right_name : left_name;
        return inner + " occurs inside " + outer + ": make the two one comparison";
    }
    return left_name + " and " + right_name +
           " are correlated under u: " + Fixed(redundant_bits, 2) + " bits counted twice";
}

ProfileReport BuildProfile(const RecordStore& store, PairMode mode,
                           const ProfileOptions& options) {
    const auto started = std::chrono::steady_clock::now();
    ProfileReport report;
    report.records = store.NumRecords();
    report.mode = mode;
    report.pair_space = store.PairSpace(mode);
    report.expected_matches =
        options.expected_matches > 0 ? options.expected_matches : report.records;
    if (report.pair_space > 0.0) {
        report.match_rate =
            static_cast<double>(report.expected_matches) / report.pair_space;
    }

    for (size_t i = 0; i < store.NumColumns(); ++i) {
        const ColumnSpec& spec = store.schema().columns[i];
        ColumnProfile column;
        column.name = spec.name;
        column.type = spec.type;
        column.nulls = store.NullCount(i);
        column.present = report.records - column.nulls;
        column.coverage = report.records > 0 ? static_cast<double>(column.present) /
                                                   static_cast<double>(report.records)
                                             : 0.0;
        const std::vector<uint32_t>* tf = TermFrequencies(store.column(i));
        if (tf != nullptr) {
            column.distinct = store.DistinctValues(i);
            const uint32_t top =
                tf->empty() ? 0 : *std::max_element(tf->begin(), tf->end());
            column.top_share =
                column.present > 0
                    ? static_cast<double>(top) / static_cast<double>(column.present)
                    : 0.0;
        }
        // A list column's agreement is not a single-value event, so its collision
        // entropy would not be the ceiling this table's heading claims.
        const bool scalar =
            spec.type == ColumnType::kString || spec.type == ColumnType::kDate;
        if (scalar && tf != nullptr && column.present > 1) {
            column.collision = Collision(*tf, column.present);
            // A column no two rows share collides at a rate this file cannot
            // resolve, which is a lower bound on its worth and not an absence of
            // one. Floor it at one collision so the ceiling is reported as the
            // bound it is rather than dropped out of the ledger.
            if (column.collision <= 0.0) {
                const double present = static_cast<double>(column.present);
                column.collision = 1.0 / (present * (present - 1.0));
                column.bits_floored = true;
            }
            if (column.collision > 0.0) {
                column.scored = true;
                column.effective_values = 1.0 / column.collision;
                column.bits = -Log2(column.collision);
                column.bits_floored = column.bits_floored ||
                                      column.collision < kMatchMargin * report.match_rate;
                column.covered_bits = column.coverage * column.coverage * column.bits;
                report.available_bits += column.covered_bits;
            }
        }
        report.columns.push_back(std::move(column));
    }

    if (options.pairs) {
        std::vector<size_t> scalar;
        for (size_t i = 0; i < report.columns.size(); ++i) {
            if (ViewOf(store.column(i)).Valid()) scalar.push_back(i);
        }
        for (size_t a = 0; a + 1 < scalar.size(); ++a) {
            for (size_t b = a + 1; b < scalar.size(); ++b) {
                ColumnPairProfile pair;
                pair.left = scalar[a];
                pair.right = scalar[b];
                pair.left_name = report.columns[scalar[a]].name;
                pair.right_name = report.columns[scalar[b]].name;
                report.pairs.push_back(std::move(pair));
            }
        }
    }

    std::vector<uint64_t> rows;
    report.sampled = options.sample_rows > 0 && options.sample_rows < report.records;
    if (report.sampled) {
        rows.reserve(options.sample_rows + options.sample_rows / 8);
        for (uint64_t row = 0; row < report.records; ++row) {
            if (Mix(row ^ options.seed) % report.records < options.sample_rows) {
                rows.push_back(row);
            }
        }
    }
    report.sampled_rows = report.sampled ? rows.size() : report.records;

    if (!report.pairs.empty() && report.records > 0) {
        report.walked = true;
        report.threads = std::min<unsigned>(ResolveThreads(options.threads),
                                            static_cast<unsigned>(report.pairs.size()));
        std::atomic<size_t> next{0};
        const auto worker = [&]() {
            for (;;) {
                const size_t index = next.fetch_add(1);
                if (index >= report.pairs.size()) return;
                ColumnPairProfile& pair = report.pairs[index];
                JointTable table;
                FoldPair(store, rows, report.sampled, ViewOf(store.column(pair.left)),
                         ViewOf(store.column(pair.right)), &table);
                Summarise(table, report.match_rate, &pair);
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 1; t < report.threads; ++t) pool.emplace_back(worker);
        worker();
        for (std::thread& thread : pool) thread.join();

        for (const ColumnPairProfile& pair : report.pairs) {
            if (!pair.resolved) {
                ++report.unresolved_pairs;
            } else if (pair.redundant_bits > 0.0) {
                report.redundant_bits -= pair.redundant_bits;
            }
        }
        // Suspects first, then whatever evidence each pair does carry: a refused
        // joint still has a determination share and a containment rate, and those
        // are what the top of the table should be sorted on when it has nothing
        // else.
        std::sort(
            report.pairs.begin(), report.pairs.end(),
            [](const ColumnPairProfile& a, const ColumnPairProfile& b) {
                const double left = std::max({a.containment, a.LeftLift(), a.RightLift(),
                                              a.resolved ? a.redundant_bits : 0.0});
                const double right = std::max({b.containment, b.LeftLift(), b.RightLift(),
                                               b.resolved ? b.redundant_bits : 0.0});
                return left > right;
            });
    }

    if (report.pair_space > 0.0) report.space_bits = Log2(report.pair_space);
    if (report.pair_space > 0.0 && report.expected_matches > 0) {
        report.prior_bits =
            Log2(static_cast<double>(report.expected_matches) / report.pair_space);
    }
    report.margin_bits =
        report.prior_bits + report.available_bits + report.redundant_bits;
    report.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return report;
}

void PrintProfileReport(const ProfileReport& report, std::ostream& out) {
    out << "Records      " << WithThousands(report.records) << "\n"
        << "Mode         " << PairModeName(report.mode) << "\n";
    if (report.walked) {
        out << "Rows read    " << WithThousands(report.sampled_rows)
            << (report.sampled ? "  (sampled)" : "  (all)") << "\n"
            << "Time         " << Fixed(report.seconds, 1) << " s  (" << report.threads
            << " threads)\n";
    }
    out << "\n";

    out << "Evidence ledger\n";
    out << "  " << std::left << std::setw(14) << "Pair space" << std::right
        << std::setw(12) << Compact(report.pair_space) << " pairs" << std::setw(10)
        << Fixed(report.space_bits, 2) << " bits\n";
    out << "  " << std::left << std::setw(14) << "Match rate" << std::right
        << std::setw(12) << Compact(report.match_rate) << std::setw(16) << "" << "   ("
        << WithThousands(report.expected_matches) << " matching pairs assumed)\n";
    out << "  " << std::left << std::setw(14) << "Prior odds" << std::right
        << std::setw(18) << "" << std::setw(10) << Signed(report.prior_bits) << " bits\n";
    out << "  " << std::left << std::setw(14) << "Available" << std::right
        << std::setw(18) << "" << std::setw(10) << Signed(report.available_bits)
        << " bits\n";
    if (report.walked) {
        out << "  " << std::left << std::setw(14) << "Redundant" << std::right
            << std::setw(18) << "" << std::setw(10) << Signed(report.redundant_bits)
            << " bits   (pairwise, u side only)\n";
    }
    out << "  " << std::left << std::setw(14) << "Margin" << std::right << std::setw(18)
        << "" << std::setw(10) << Signed(report.margin_bits) << " bits\n\n";

    out << std::left << std::setw(18) << "Column" << std::setw(12) << "Type" << std::right
        << std::setw(12) << "Distinct" << std::setw(9) << "Null" << std::setw(9)
        << "Top val" << std::setw(12) << "Eff vals" << std::setw(9) << "Bits"
        << std::setw(10) << "Cov bits" << "\n";
    out << std::string(91, '-') << "\n";
    for (const ColumnProfile& column : report.columns) {
        out << std::left << std::setw(18) << Truncate(column.name, 17) << std::setw(12)
            << ColumnTypeName(column.type) << std::right << std::setw(12)
            << (column.distinct > 0 ? WithThousands(column.distinct) : "-")
            << std::setw(9) << Percent(1.0 - column.coverage) << std::setw(9)
            << (column.distinct > 0 ? Percent(column.top_share) : "-") << std::setw(12)
            << (column.scored ? Compact(column.effective_values) : "-") << std::setw(9)
            << (column.scored ? Fixed(column.bits, 2) : "-") << std::setw(10)
            << (column.scored ? Fixed(column.covered_bits, 2) : "-") << "\n";
    }
    out << "\n";

    std::string floored;
    for (const ColumnProfile& column : report.columns) {
        if (!column.bits_floored) continue;
        if (!floored.empty()) floored += ", ";
        floored += column.name;
    }
    if (!floored.empty()) {
        out << "Bits is a floor for these columns, whose collision rate is within reach "
               "of\n"
            << "the match rate, so the duplicates the file holds inflate u and the "
               "ceiling\n"
            << "reads low:\n  " << floored << "\n\n";
    }

    if (!report.walked) return;

    out << std::left << std::setw(18) << "Column" << std::setw(18) << "Against"
        << std::right << std::setw(12) << "Rows" << std::setw(9) << "L->R" << std::setw(9)
        << "R->L" << std::setw(9) << "Substr" << std::setw(9) << "Redund" << "\n";
    out << std::string(84, '-') << "\n";
    for (const ColumnPairProfile& pair : report.pairs) {
        out << std::left << std::setw(18) << Truncate(pair.left_name, 17) << std::setw(18)
            << Truncate(pair.right_name, 17) << std::right << std::setw(12)
            << WithThousands(pair.rows) << std::setw(9)
            << (pair.left_informative ? Fixed(pair.determines_right, 3) : "-")
            << std::setw(9)
            << (pair.right_informative ? Fixed(pair.determines_left, 3) : "-")
            << std::setw(9) << Fixed(pair.containment, 3) << std::setw(9)
            << (pair.resolved ? Fixed(pair.redundant_bits, 2) : "-") << "\n";
    }
    out << "\n";

    if (report.unresolved_pairs > 0) {
        out << "Redund is refused on " << report.unresolved_pairs << " of "
            << report.pairs.size() << " pairs: independence puts their joint collision\n"
            << "rate at or below the match rate, so what the joint holds is the file's "
               "own\n"
            << "duplicates rather than a dependence between the columns.\n\n";
    }

    size_t shown = 0;
    for (const ColumnPairProfile& pair : report.pairs) {
        if (!pair.Suspect()) continue;
        if (shown == 0) out << "Suspects\n";
        ++shown;
        out << "  " << pair.Verdict() << "\n";
    }
    if (shown == 0) {
        out << "No column pair is redundant enough to act on.\n";
    }
    out << "\n";

    out << "Bits is the most an exact-match level can be worth on that column, which "
           "takes\n"
        << "m = 1 and no real column reaches it, so a positive margin is necessary and "
           "not\n"
        << "sufficient. Eff vals is 1/u, and on a near-unique column it runs above the\n"
        << "distinct count: two distinct rows agreeing is rarer there than one value in\n"
        << "however many the column holds. L->R is the share of rows kept by mapping "
           "each\n"
        << "left value to its commonest right partner, and is 1 when the left column\n"
        << "determines the right; a determinant that is near-unique, or a target with "
           "one\n"
        << "dominant value, reads 1 for saying nothing and is shown as -. Redund is the\n"
        << "u-side overlap alone; agreement among matches is the larger correlation and\n"
        << "needs matching pairs to measure. Both the prior odds and what Redund can be\n"
        << "read on move with --expected-matches, which defaults to one duplicate per\n"
        << "record.\n";
}

void WriteProfileJson(const ProfileReport& report, std::ostream& out) {
    nlohmann::json root;
    root["records"] = report.records;
    root["mode"] = PairModeName(report.mode);
    root["pair_space"] = report.pair_space;
    root["expected_matches"] = report.expected_matches;
    root["match_rate"] = report.match_rate;
    root["unresolved_pairs"] = report.unresolved_pairs;
    root["space_bits"] = report.space_bits;
    root["prior_bits"] = report.prior_bits;
    root["available_bits"] = report.available_bits;
    root["redundant_bits"] = report.redundant_bits;
    root["margin_bits"] = report.margin_bits;
    root["walked"] = report.walked;
    root["sampled"] = report.sampled;
    root["sampled_rows"] = report.sampled_rows;
    root["seconds"] = report.seconds;

    root["columns"] = nlohmann::json::array();
    for (const ColumnProfile& column : report.columns) {
        nlohmann::json item;
        item["name"] = column.name;
        item["type"] = ColumnTypeName(column.type);
        item["scored"] = column.scored;
        item["distinct"] = column.distinct;
        item["nulls"] = column.nulls;
        item["coverage"] = column.coverage;
        item["top_share"] = column.top_share;
        item["collision"] = column.collision;
        item["effective_values"] = column.effective_values;
        item["bits"] = column.bits;
        item["covered_bits"] = column.covered_bits;
        item["bits_floored"] = column.bits_floored;
        root["columns"].push_back(std::move(item));
    }

    root["pairs"] = nlohmann::json::array();
    for (const ColumnPairProfile& pair : report.pairs) {
        nlohmann::json item;
        item["left"] = pair.left_name;
        item["right"] = pair.right_name;
        item["rows"] = pair.rows;
        item["determines_right"] = pair.determines_right;
        item["determines_left"] = pair.determines_left;
        item["left_distinct_share"] = pair.left_distinct_share;
        item["right_distinct_share"] = pair.right_distinct_share;
        item["left_informative"] = pair.left_informative;
        item["right_informative"] = pair.right_informative;
        item["baseline_right"] = pair.baseline_right;
        item["baseline_left"] = pair.baseline_left;
        item["containment"] = pair.containment;
        item["left_inside_right"] = pair.left_inside_right;
        item["u_left"] = pair.u_left;
        item["u_right"] = pair.u_right;
        item["u_joint"] = pair.u_joint;
        item["redundant_bits"] = pair.redundant_bits;
        item["joint_collisions"] = pair.joint_collisions;
        item["expected_collisions"] = pair.expected_collisions;
        item["resolved"] = pair.resolved;
        item["suspect"] = pair.Suspect();
        if (pair.Suspect()) item["verdict"] = pair.Verdict();
        root["pairs"].push_back(std::move(item));
    }
    out << root.dump(2) << "\n";
}

}  // namespace cpplink
