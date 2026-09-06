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
#include "cpplink/recall.hpp"

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

// A key group larger than this, under a key chosen for being near-unique, is a
// placeholder value rather than a cluster of duplicates, and the pairs it holds are
// the one thing the anchor was picked to exclude.
constexpr size_t kMaxAnchorGroup = 64;
// Anchor pairs a session, or a column within one, needs before anything is read off
// it. m to a hundredth wants ten thousand of them; this is the point below which
// the number is not evidence at all.
constexpr uint64_t kMinAnchorPairs = 100;
// Columns in one anchor. Every one costs coverage on the pairs the anchor selects,
// and the bits are there long before the coverage runs out.
constexpr size_t kMaxAnchorColumns = 4;
// Distinct anchors built. Sessions run one after another over the whole row budget,
// so this is what bounds what the M side costs.
constexpr size_t kMaxAnchorSessions = 4;
// Agreement coupling between a column and an anchor column, in bits, past which
// the session may not be read for that column. This is determination's pairwise
// twin: the row-level check catches a column derived from another, and this one
// catches a column corrupted by whatever corrupted another, which is invisible in
// the rows and shows up only among matches. The threshold is well under the bit a
// redundancy has to be worth to be acted on, because the question here is not what
// to spend the bits on but whether the number means anything at all.
constexpr double kAnchorCoupling = 0.1;

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

std::string Signed(double value, int decimals = 2) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%+.*f", decimals, value);
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

// ---------------------------------------------------------------------------
// The M side: matching pairs, from anchors rather than from a model.

// Whether one column has already said whatever the other would say. An anchor
// column standing in this relation to a column the session would estimate has
// forced the very agreement being measured, so the session may not be read for it.
bool SameEvidence(const ColumnPairProfile& pair) {
    if (pair.left_informative && pair.determines_right >= kDeterminedShare) return true;
    if (pair.right_informative && pair.determines_left >= kDeterminedShare) return true;
    return pair.containment >= kContainedShare;
}

// The same question answered from the schema rather than from the rows. A derived
// column is a functional dependency its declaration already states, and the test
// above can miss one: the determination share refuses a near-unique determinant,
// and the u-side overlap is refused wherever the file's own duplicates swamp the
// joint the two columns would have been read from. Neither refusal makes the
// derivation any less of one, and an anchor that holds a column beside the key it
// was derived from has counted the same agreement twice.
bool SameEvidence(const Schema& schema, const ProfileReport& report, size_t a, size_t b) {
    return SameSource(schema, report.columns[a].name, report.columns[b].name);
}

// `report.pairs` by the two column indices it was built from, which is how the
// anchor pass reaches the U-side results. Valid only before the table is sorted.
class PairIndex {
   public:
    static constexpr size_t kNoPair = static_cast<size_t>(-1);

    explicit PairIndex(const std::vector<ColumnPairProfile>& pairs) {
        for (size_t i = 0; i < pairs.size(); ++i) {
            index_[KeyFor(pairs[i].left, pairs[i].right)] = i;
        }
    }

    // kNoPair when the two are the same column, or either is not scalar.
    size_t Find(size_t a, size_t b) const {
        const auto found = index_.find(KeyFor(a, b));
        return found == index_.end() ? kNoPair : found->second;
    }

   private:
    static uint64_t KeyFor(size_t a, size_t b) {
        const uint64_t low = std::min(a, b);
        const uint64_t high = std::max(a, b);
        return (low << 32) | high;
    }
    std::unordered_map<uint64_t, size_t> index_;
};

// The strongest anchor that holds `target` out: columns by their own bits, skipping
// anything the target or an already-chosen column has said, and charging every
// addition the pairwise overlap it brings with it so the bits are not counted twice
// in the very check that is meant to make them trustworthy.
std::vector<size_t> BuildAnchor(const Schema& schema, const ProfileReport& report,
                                const PairIndex& index, const std::vector<size_t>& order,
                                size_t target, double need, double* bits) {
    std::vector<size_t> anchor;
    *bits = 0.0;
    for (size_t column : order) {
        if (column == target) continue;
        if (SameEvidence(schema, report, column, target)) continue;
        const size_t against = index.Find(column, target);
        if (against != PairIndex::kNoPair && SameEvidence(report.pairs[against])) {
            continue;
        }
        double overlap = 0.0;
        bool shared = false;
        for (size_t chosen : anchor) {
            if (SameEvidence(schema, report, column, chosen)) {
                shared = true;
                break;
            }
            const size_t entry = index.Find(column, chosen);
            if (entry == PairIndex::kNoPair) continue;
            const ColumnPairProfile& pair = report.pairs[entry];
            if (SameEvidence(pair)) {
                shared = true;
                break;
            }
            if (pair.resolved && pair.redundant_bits > 0.0)
                overlap += pair.redundant_bits;
        }
        if (shared) continue;
        const double gain = report.columns[column].bits - overlap;
        if (gain <= 0.0) continue;
        anchor.push_back(column);
        *bits += gain;
        if (*bits >= need || anchor.size() >= kMaxAnchorColumns) break;
    }
    return anchor;
}

// One row of an anchor's key, ready to be sorted into groups.
struct AnchorRow {
    uint64_t key = 0;
    uint64_t row = 0;
};

// What one session counts as it walks its pairs, indexed by column and by position
// in `report.pairs` so the merge is an addition and needs no lookup.
struct AnchorCounts {
    std::vector<uint64_t> both;   // pairs carrying the column on both rows
    std::vector<uint64_t> agree;  // of those, pairs whose values are equal
    std::vector<uint64_t> pair_both;
    std::vector<uint64_t> pair_left;
    std::vector<uint64_t> pair_right;
    std::vector<uint64_t> pair_joint;

    AnchorCounts(size_t columns, size_t pairs)
        : both(columns, 0),
          agree(columns, 0),
          pair_both(pairs, 0),
          pair_left(pairs, 0),
          pair_right(pairs, 0),
          pair_joint(pairs, 0) {}
};

bool AgreesOn(const std::vector<ScalarView>& views, const std::vector<size_t>& anchor,
              uint64_t a, uint64_t b) {
    for (size_t column : anchor) {
        if (views[column].Key(a) != views[column].Key(b)) return false;
    }
    return true;
}

// Two columns of a session's learnable set, and where their counts belong.
struct LearnPair {
    size_t first = 0;   // position in `learns`, and so the pair's left column
    size_t second = 0;  // position in `learns`, and so its right
    size_t entry = 0;   // position in `report.pairs`
};

// Folds every learnable column's agreement over one anchor's pairs into `counts`.
void WalkAnchor(const RecordStore& store, const std::vector<ScalarView>& views,
                const std::vector<uint64_t>& rows, PairMode mode,
                const std::vector<LearnPair>& learn_pairs, uint64_t budget,
                AnchorSession* session, AnchorCounts* counts) {
    const std::vector<size_t>& learns = session->learns;
    std::vector<uint8_t> has(learns.size(), 0);
    std::vector<uint8_t> same(learns.size(), 0);
    const AnchorWalk walk = WalkAnchorPairs(
        store, mode, session->anchor, rows, budget, [&](uint64_t a, uint64_t b) {
            for (size_t k = 0; k < learns.size(); ++k) {
                const ScalarView& view = views[learns[k]];
                const uint32_t first = view.Key(a);
                const uint32_t second = view.Key(b);
                has[k] = first != kNullId && second != kNullId;
                same[k] = has[k] && first == second;
                if (has[k] == 0) continue;
                ++counts->both[learns[k]];
                if (same[k] != 0) ++counts->agree[learns[k]];
            }
            for (const LearnPair& learn : learn_pairs) {
                if (has[learn.first] == 0 || has[learn.second] == 0) continue;
                ++counts->pair_both[learn.entry];
                if (same[learn.first] != 0) ++counts->pair_left[learn.entry];
                if (same[learn.second] != 0) ++counts->pair_right[learn.entry];
                if (same[learn.first] != 0 && same[learn.second] != 0) {
                    ++counts->pair_joint[learn.entry];
                }
            }
        });
    session->groups = walk.groups;
    session->pairs = walk.pairs;
    session->oversized = walk.oversized;
    session->capped = walk.capped;
}

bool Holds(const std::vector<size_t>& values, size_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

// The whole M side: choose the anchors, walk their pairs, and turn the counts into
// m per column, the M half of the dependence map, and the ledger's second half.
void BuildMatchSide(const RecordStore& store, const ProfileOptions& options,
                    ProfileReport* report) {
    if (!options.anchors) return;
    if (!report->walked) {
        report->anchor_refusal =
            "the pairwise pass is what tells an anchor from a column it determines, "
            "and --no-pairs turned it off";
        return;
    }

    std::vector<size_t> order;
    for (size_t i = 0; i < report->columns.size(); ++i) {
        if (report->columns[i].scored && ViewOf(store.column(i)).Valid()) {
            order.push_back(i);
        }
    }
    report->scored_columns = order.size();
    if (order.empty()) {
        report->anchor_refusal = "no column carries an exact-match ceiling to anchor on";
        return;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return report->columns[a].bits > report->columns[b].bits;
    });

    // An anchor is admissible when agreement on it alone would put the posterior
    // this far above even odds, which is the whole of the claim that its pairs are
    // matches. m <= 1 on every column, so the bits are an upper bound on the
    // posterior and the margin is what pays for the overlap the pairwise table
    // refused to resolve.
    const PairIndex index(report->pairs);
    const double need = -report->prior_bits + options.anchor_margin;
    double best = 0.0;
    for (size_t target : order) {
        double bits = 0.0;
        std::vector<size_t> anchor =
            BuildAnchor(store.schema(), *report, index, order, target, need, &bits);
        best = std::max(best, bits);
        if (anchor.empty() || bits < need) continue;
        std::sort(anchor.begin(), anchor.end());
        bool seen = false;
        for (const AnchorSession& existing : report->sessions) {
            seen = seen || existing.anchor == anchor;
        }
        if (seen) continue;
        AnchorSession session;
        session.anchor = std::move(anchor);
        session.bits = bits;
        session.posterior_bits = report->prior_bits + bits;
        for (size_t column : session.anchor) {
            session.anchor_names.push_back(report->columns[column].name);
        }
        // Every scored column the anchor neither contains nor has already spoken
        // for. `learns` stays in column order, which is the order `report.pairs`
        // was built in, so a learnable pair's first column is that pair's left.
        for (size_t column : order) {
            if (Holds(session.anchor, column)) continue;
            bool forced = false;
            for (size_t chosen : session.anchor) {
                if (SameEvidence(store.schema(), *report, column, chosen)) {
                    forced = true;
                    break;
                }
                const size_t entry = index.Find(column, chosen);
                forced = forced || (entry != PairIndex::kNoPair &&
                                    SameEvidence(report->pairs[entry]));
            }
            if (!forced) session.learns.push_back(column);
        }
        std::sort(session.learns.begin(), session.learns.end());
        if (!session.learns.empty()) report->sessions.push_back(std::move(session));
        if (report->sessions.size() >= kMaxAnchorSessions) break;
    }

    if (report->sessions.empty()) {
        report->anchor_refusal = "the strongest admissible column set is worth " +
                                 Fixed(best, 2) + " bits against the " + Fixed(need, 2) +
                                 " a pair needs to be a match on agreement alone";
        return;
    }

    const std::vector<uint64_t> rows = AnchorRows(report->records, options);
    report->anchor_rows = rows.empty() ? report->records : rows.size();

    std::vector<ScalarView> views;
    views.reserve(store.NumColumns());
    for (size_t i = 0; i < store.NumColumns(); ++i)
        views.push_back(ViewOf(store.column(i)));

    // One session at a time: each holds a keyed copy of the row budget, and running
    // them side by side would multiply the only large allocation the command makes.
    std::vector<AnchorCounts> counts;
    for (AnchorSession& session : report->sessions) {
        std::vector<LearnPair> learn_pairs;
        for (size_t i = 0; i + 1 < session.learns.size(); ++i) {
            for (size_t j = i + 1; j < session.learns.size(); ++j) {
                const size_t entry = index.Find(session.learns[i], session.learns[j]);
                if (entry != PairIndex::kNoPair) learn_pairs.push_back({i, j, entry});
            }
        }
        counts.emplace_back(report->columns.size(), report->pairs.size());
        WalkAnchor(store, views, rows, report->mode, learn_pairs, options.anchor_pairs,
                   &session, &counts.back());
        session.used = session.pairs >= kMinAnchorPairs;
        if (session.used) report->anchor_pairs += session.pairs;
    }

    if (report->anchor_pairs == 0) {
        report->anchor_refusal = "the anchors selected fewer than " +
                                 std::to_string(kMinAnchorPairs) +
                                 " pairs, which is too few to read an agreement rate off";
        return;
    }
    report->anchored = true;

    // Round one: what agrees with what among matches, pooled over the sessions that
    // measured it. A pair is only ever counted in a session that anchors on neither
    // of its columns, so these readings are already clear of the conditioning that
    // is about to be checked against them.
    std::vector<double> coupling(report->pairs.size(), 0.0);
    std::vector<char> known(report->pairs.size(), 0);
    for (size_t p = 0; p < report->pairs.size(); ++p) {
        uint64_t both = 0;
        uint64_t left = 0;
        uint64_t right = 0;
        uint64_t joint = 0;
        for (size_t sess = 0; sess < report->sessions.size(); ++sess) {
            if (!report->sessions[sess].used) continue;
            both += counts[sess].pair_both[p];
            left += counts[sess].pair_left[p];
            right += counts[sess].pair_right[p];
            joint += counts[sess].pair_joint[p];
        }
        if (both < kMinAnchorPairs || left == 0 || right == 0 || joint == 0) continue;
        const double total = static_cast<double>(both);
        known[p] = 1;
        coupling[p] = Log2((static_cast<double>(joint) * total) /
                           (static_cast<double>(left) * static_cast<double>(right)));
    }

    // A column whose agreement moves with an anchor column's has had that agreement
    // forced, and the session cannot be read for it however independent the two
    // columns look row by row.
    for (AnchorSession& session : report->sessions) {
        std::vector<size_t> kept;
        for (size_t column : session.learns) {
            bool forced = false;
            for (size_t chosen : session.anchor) {
                const size_t entry = index.Find(column, chosen);
                forced = forced || (entry != PairIndex::kNoPair && known[entry] != 0 &&
                                    std::abs(coupling[entry]) >= kAnchorCoupling);
            }
            if (!forced) kept.push_back(column);
        }
        session.learns = std::move(kept);
    }

    for (size_t column : order) {
        ColumnMatchProfile match;
        match.column = column;
        match.name = report->columns[column].name;
        double sum = 0.0;
        double low = 1.0;
        double high = 0.0;
        uint64_t present = 0;
        uint64_t walked = 0;
        for (size_t s = 0; s < report->sessions.size(); ++s) {
            const AnchorSession& session = report->sessions[s];
            if (!session.used || !Holds(session.learns, column)) continue;
            const uint64_t both = counts[s].both[column];
            if (both < kMinAnchorPairs) continue;
            const double rate =
                static_cast<double>(counts[s].agree[column]) / static_cast<double>(both);
            sum += rate;
            low = std::min(low, rate);
            high = std::max(high, rate);
            ++match.sessions;
            present += both;
            walked += session.pairs;
        }
        if (match.sessions > 0) {
            match.estimated = true;
            match.pairs = present;
            match.coverage =
                walked > 0 ? static_cast<double>(present) / static_cast<double>(walked)
                           : 0.0;
            match.m = sum / match.sessions;
            match.m_low = low;
            match.m_high = high;
            // A session where every pair agreed, or none did, is a rate of exactly
            // one or zero, and log2(m/u) is not finite at either. Half a pair is
            // the usual continuity correction and it is reported rather than
            // quietly applied, because a floored m is a statement about the sample
            // size and not about the column.
            const double edge = 0.5 / static_cast<double>(present);
            if (match.m < edge) {
                match.m = edge;
                match.floored = true;
            } else if (match.m > 1.0 - edge) {
                match.m = 1.0 - edge;
                match.floored = true;
            }
            const double u = report->columns[column].collision;
            match.weight = Log2(match.m / u);
            // What the column contributes to a matching pair on average: the
            // agreement weight when it agrees, the disagreement penalty when it
            // does not, and nothing at all when one side is missing, which is the
            // same convention the ceiling's coverage term uses.
            match.expected_bits =
                match.coverage * (match.m * match.weight +
                                  (1.0 - match.m) * Log2((1.0 - match.m) / (1.0 - u)));
            report->expected_bits += match.expected_bits;
            ++report->estimated_columns;
        }
        report->matches.push_back(std::move(match));
    }

    // Anchors that select pairs but leave no column with enough of them are the
    // shape a narrow file takes: every strong column ends up inside an anchor, and
    // what is left is read off too few pairs to be a rate.
    if (report->estimated_columns == 0) {
        report->anchored = false;
        report->matches.clear();
        report->anchor_refusal =
            "an agreement rate needs " + std::to_string(kMinAnchorPairs) +
            " anchor pairs carrying the column on both rows, and no column has them "
            "among the " +
            WithThousands(report->anchor_pairs) + " the anchors selected";
        return;
    }

    // Round two, over what survived the prune: the counts a pruned session gathered
    // for a column are exactly the ones its anchor forced, and they have no more
    // business in the dependence table than they had in m.
    for (size_t p = 0; p < report->pairs.size(); ++p) {
        ColumnPairProfile& pair = report->pairs[p];
        uint64_t both = 0;
        uint64_t left = 0;
        uint64_t right = 0;
        uint64_t joint = 0;
        for (size_t s = 0; s < report->sessions.size(); ++s) {
            const AnchorSession& session = report->sessions[s];
            if (!session.used) continue;
            if (!Holds(session.learns, pair.left) || !Holds(session.learns, pair.right)) {
                continue;
            }
            both += counts[s].pair_both[p];
            left += counts[s].pair_left[p];
            right += counts[s].pair_right[p];
            joint += counts[s].pair_joint[p];
        }
        pair.m_pairs = both;
        if (both < kMinAnchorPairs) continue;
        const double total = static_cast<double>(both);
        pair.m_left = static_cast<double>(left) / total;
        pair.m_right = static_cast<double>(right) / total;
        pair.m_joint = static_cast<double>(joint) / total;
        if (pair.m_left > 0.0 && pair.m_right > 0.0 && pair.m_joint > 0.0) {
            pair.m_resolved = true;
            pair.m_redundant_bits = Log2(pair.m_joint / (pair.m_left * pair.m_right));
        }
    }

    // The score adds log2(m/u) for both columns, so only the difference between the
    // two overlaps is wrong, and it is only wrong on the pairs where both agree.
    for (const ColumnPairProfile& pair : report->pairs) {
        const double net = pair.NetRedundantBits();
        if (net > 0.0) report->double_counted_bits -= pair.m_joint * net;
    }
    report->estimated_margin_bits =
        report->prior_bits + report->expected_bits + report->double_counted_bits;
}

// The same agreement rates, over known pairs rather than anchor pairs. Nothing
// above depends on this: it is a second reading of the number the anchor estimate
// already produced, so the difference between them is the bias the anchor carries
// and the report can print it rather than claim there is none.
void BuildTruthSide(const RecordStore& store, const TruthPairs& truth,
                    ProfileReport* report) {
    if (report->matches.empty()) return;
    std::vector<ScalarView> views;
    views.reserve(report->columns.size());
    for (size_t i = 0; i < report->columns.size(); ++i) {
        views.push_back(ViewOf(store.column(i)));
    }

    std::vector<uint64_t> both(report->columns.size(), 0);
    std::vector<uint64_t> agree(report->columns.size(), 0);
    uint64_t pairs = 0;
    for (const std::pair<uint32_t, uint32_t>& known : truth.rows) {
        // Link mode's pair space is the cross product, so a known pair inside one
        // input is not a pair the model will ever be asked about.
        if (report->mode == PairMode::kCrossDataset &&
            store.DatasetOf(known.first) == store.DatasetOf(known.second)) {
            continue;
        }
        ++pairs;
        for (ColumnMatchProfile& match : report->matches) {
            const ScalarView& view = views[match.column];
            if (!view.Valid()) continue;
            const uint32_t left = view.Key(known.first);
            const uint32_t right = view.Key(known.second);
            if (left == kNullId || right == kNullId) continue;
            ++both[match.column];
            if (left == right) ++agree[match.column];
        }
    }
    if (pairs == 0) return;

    report->truthed = true;
    report->truth_pairs = pairs;
    const double walked = static_cast<double>(pairs);
    double error = 0.0;
    size_t scored = 0;
    for (ColumnMatchProfile& match : report->matches) {
        const uint64_t present = both[match.column];
        if (present < kMinAnchorPairs) continue;
        match.truthed = true;
        match.truth_pairs = present;
        match.truth_coverage = static_cast<double>(present) / walked;
        match.truth_m =
            static_cast<double>(agree[match.column]) / static_cast<double>(present);
        const double edge = 0.5 / static_cast<double>(present);
        match.truth_m = std::min(std::max(match.truth_m, edge), 1.0 - edge);
        const double u = report->columns[match.column].collision;
        match.truth_weight = Log2(match.truth_m / u);
        match.truth_expected_bits =
            match.truth_coverage *
            (match.truth_m * match.truth_weight +
             (1.0 - match.truth_m) * Log2((1.0 - match.truth_m) / (1.0 - u)));
        // The margin is summed over the columns the anchor estimate covers, so
        // the two numbers answer the same question about the same columns.
        if (match.estimated) {
            report->truth_expected_bits += match.truth_expected_bits;
            error += std::abs(match.m - match.truth_m);
            ++scored;
        }
    }
    if (scored > 0) report->truth_mean_error = error / static_cast<double>(scored);
    report->truth_margin_bits =
        report->prior_bits + report->truth_expected_bits + report->double_counted_bits;
}

}  // namespace

std::vector<uint64_t> AnchorRows(uint64_t records, const ProfileOptions& options) {
    std::vector<uint64_t> rows;
    if (options.anchor_rows == 0 || options.anchor_rows >= records) return rows;
    rows.reserve(options.anchor_rows + options.anchor_rows / 8);
    for (uint64_t row = 0; row < records; ++row) {
        if (Mix(row ^ options.seed) % records < options.anchor_rows) rows.push_back(row);
    }
    return rows;
}

// The key is near-unique by construction, so almost every group is one row long and
// the sort is what the pass costs. Hash equality is not taken as agreement: a group
// is a candidate list and every pair in it is verified against the anchor columns,
// because a 64-bit collision over 18M rows is not rare enough to ignore when what it
// would corrupt is the estimate itself.
AnchorWalk WalkAnchorPairs(const RecordStore& store, PairMode mode,
                           const std::vector<size_t>& anchor,
                           const std::vector<uint64_t>& rows, uint64_t budget,
                           const std::function<void(uint64_t, uint64_t)>& visit) {
    AnchorWalk walk;
    if (anchor.empty() || budget == 0) return walk;
    std::vector<ScalarView> views;
    views.reserve(store.NumColumns());
    for (size_t i = 0; i < store.NumColumns(); ++i) {
        views.push_back(ViewOf(store.column(i)));
    }

    const bool sampled = !rows.empty();
    const uint64_t count = sampled ? rows.size() : store.NumRecords();
    std::vector<AnchorRow> keyed;
    keyed.reserve(count);
    for (uint64_t k = 0; k < count; ++k) {
        const uint64_t row = sampled ? rows[k] : k;
        uint64_t key = 0x243f6a8885a308d3ull;
        bool present = true;
        for (size_t column : anchor) {
            const uint32_t id = views[column].Key(row);
            if (id == kNullId) {
                present = false;
                break;
            }
            key = Mix(key ^ (static_cast<uint64_t>(id) + 0x9e3779b97f4a7c15ull));
        }
        if (present) keyed.push_back({key, row});
    }
    std::sort(keyed.begin(), keyed.end(), [](const AnchorRow& a, const AnchorRow& b) {
        return a.key != b.key ? a.key < b.key : a.row < b.row;
    });

    size_t start = 0;
    while (start < keyed.size() && walk.pairs < budget) {
        size_t end = start + 1;
        while (end < keyed.size() && keyed[end].key == keyed[start].key) ++end;
        const size_t size = end - start;
        if (size < 2) {
            start = end;
            continue;
        }
        if (size > kMaxAnchorGroup) {
            ++walk.oversized;
            start = end;
            continue;
        }
        ++walk.groups;
        for (size_t i = start; i + 1 < end && walk.pairs < budget; ++i) {
            for (size_t j = i + 1; j < end && walk.pairs < budget; ++j) {
                const uint64_t a = keyed[i].row;
                const uint64_t b = keyed[j].row;
                if (mode == PairMode::kCrossDataset && store.DatasetEndFor(a) > b) {
                    continue;
                }
                if (!AgreesOn(views, anchor, a, b)) continue;
                ++walk.pairs;
                visit(a, b);
            }
        }
        start = end;
    }
    walk.capped = walk.pairs >= budget;
    return walk;
}

bool ColumnPairProfile::Suspect() const {
    if (rows == 0) return false;
    if (left_informative && determines_right >= kDeterminedShare) return true;
    if (right_informative && determines_left >= kDeterminedShare) return true;
    if (containment >= kContainedShare) return true;
    if (NetRedundantBits() >= kRedundantBits) return true;
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
    if (NetRedundantBits() >= kRedundantBits) {
        return left_name + " and " + right_name +
               " agree together among matches beyond what agreeing apart would give: " +
               Fixed(NetRedundantBits(), 2) + " bits counted twice, u side netted off";
    }
    return left_name + " and " + right_name +
           " are correlated under u: " + Fixed(redundant_bits, 2) + " bits counted twice";
}

ProfileReport BuildProfile(const RecordStore& store, PairMode mode,
                           const ProfileOptions& options, const TruthPairs* truth) {
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
        report.space_bits = Log2(report.pair_space);
        if (report.expected_matches > 0) {
            report.prior_bits =
                Log2(static_cast<double>(report.expected_matches) / report.pair_space);
        }
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
    }

    // The M side reads the pairwise results by column index, so it runs while the
    // table is still in the order it was built in.
    BuildMatchSide(store, options, &report);
    if (truth != nullptr && !truth->rows.empty()) {
        BuildTruthSide(store, *truth, &report);
    }

    // Suspects first, then whatever evidence each pair does carry: a refused joint
    // still has a determination share and a containment rate, and those are what
    // the top of the table should be sorted on when it has nothing else.
    std::sort(
        report.pairs.begin(), report.pairs.end(),
        [](const ColumnPairProfile& a, const ColumnPairProfile& b) {
            const double left =
                std::max({a.containment, a.LeftLift(), a.RightLift(),
                          a.resolved ? a.redundant_bits : 0.0, a.NetRedundantBits()});
            const double right =
                std::max({b.containment, b.LeftLift(), b.RightLift(),
                          b.resolved ? b.redundant_bits : 0.0, b.NetRedundantBits()});
            return left > right;
        });

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
        << "" << std::setw(10) << Signed(report.margin_bits) << " bits\n";
    if (report.anchored) {
        out << "\n";
        out << "  " << std::left << std::setw(14) << "Expected" << std::right
            << std::setw(18) << "" << std::setw(10) << Signed(report.expected_bits)
            << " bits   (m over " << WithThousands(report.anchor_pairs)
            << " anchor pairs)\n";
        out << "  " << std::left << std::setw(14) << "Double counted" << std::right
            << std::setw(18) << "" << std::setw(10) << Signed(report.double_counted_bits)
            << " bits   (pairwise, both sides)\n";
        out << "  " << std::left << std::setw(14) << "Est. margin" << std::right
            << std::setw(18) << "" << std::setw(10)
            << Signed(report.estimated_margin_bits) << " bits\n";
    }
    if (report.truthed) {
        out << "  " << std::left << std::setw(14) << "Truth margin" << std::right
            << std::setw(18) << "" << std::setw(10) << Signed(report.truth_margin_bits)
            << " bits   (m over " << WithThousands(report.truth_pairs)
            << " known pairs)\n";
    }
    out << "\n";

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

    if (report.anchored) {
        out << std::left << std::setw(18) << "Column" << std::right << std::setw(9)
            << "Sessions" << std::setw(12) << "Pairs" << std::setw(8) << "Cov"
            << std::setw(8) << "m" << std::setw(13) << "Spread" << std::setw(9)
            << "Weight" << std::setw(10) << "Exp bits";
        if (report.truthed) out << std::setw(9) << "True m" << std::setw(8) << "Err";
        out << "\n";
        out << std::string(report.truthed ? 104 : 87, '-') << "\n";
        for (const ColumnMatchProfile& match : report.matches) {
            out << std::left << std::setw(18) << Truncate(match.name, 17) << std::right;
            if (!match.estimated) {
                out << std::setw(9) << 0 << std::setw(12) << "-" << std::setw(8) << "-"
                    << std::setw(8) << "-" << std::setw(13) << "-" << std::setw(9) << "-"
                    << std::setw(10) << "-";
                if (report.truthed) {
                    out << std::setw(9) << (match.truthed ? Fixed(match.truth_m, 3) : "-")
                        << std::setw(8) << "-";
                }
                out << "\n";
                continue;
            }
            const std::string spread =
                match.sessions > 1 ? Fixed(match.m_low, 3) + "-" + Fixed(match.m_high, 3)
                                   : std::string("-");
            out << std::setw(9) << match.sessions << std::setw(12)
                << WithThousands(match.pairs) << std::setw(8) << Percent(match.coverage)
                << std::setw(8) << Fixed(match.m, 3) << std::setw(13) << spread
                << std::setw(9) << Fixed(match.weight, 2) << std::setw(10)
                << Fixed(match.expected_bits, 2);
            if (report.truthed) {
                out << std::setw(9) << (match.truthed ? Fixed(match.truth_m, 3) : "-")
                    << std::setw(8)
                    << (match.truthed ? Signed(match.m - match.truth_m, 3) : "-");
            }
            out << "\n";
        }
        if (report.truthed) {
            out << std::left << std::setw(18) << "mean |error|" << std::right
                << std::setw(86) << Fixed(report.truth_mean_error, 3) << "\n";
        }
        out << "\n";

        out << "Anchors, over " << WithThousands(report.anchor_rows) << " rows\n";
        for (const AnchorSession& session : report.sessions) {
            std::string names;
            for (const std::string& name : session.anchor_names) {
                if (!names.empty()) names += " + ";
                names += name;
            }
            out << "  " << std::left << std::setw(46) << Truncate(names, 45) << std::right
                << std::setw(8) << Fixed(session.bits, 2) << " bits" << std::setw(9)
                << Signed(session.posterior_bits) << " post" << std::setw(12)
                << WithThousands(session.pairs) << " pairs";
            if (!session.used) out << "   (too few to read)";
            if (session.capped) out << "   (budget reached)";
            out << "\n";
        }
        out << "\n";

        std::string floored_m;
        for (const ColumnMatchProfile& match : report.matches) {
            if (!match.floored) continue;
            if (!floored_m.empty()) floored_m += ", ";
            floored_m += match.name;
        }
        if (!floored_m.empty()) {
            out << "m hit the continuity floor for these columns, where every anchor "
                   "pair agreed\nor none did, so what is reported is half a pair off "
                   "the edge rather than a rate\nthe sample could resolve:\n  "
                << floored_m << "\n\n";
        }

        if (report.estimated_columns < report.scored_columns) {
            out << "m is not estimated for "
                << report.scored_columns - report.estimated_columns << " of "
                << report.scored_columns
                << " scored columns: every anchor built either\ncontains the column or "
                   "determines it. Those contribute nothing to Expected, so\nthe "
                   "estimated margin is a floor.\n\n";
        }
    }

    if (!report.anchored && !report.anchor_refusal.empty()) {
        out << "No m: " << report.anchor_refusal
            << ".\nThe ledger above is the "
               "ceiling alone, which takes m = 1 on every column.\n\n";
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

    if (report.anchored) {
        size_t rows = 0;
        for (const ColumnPairProfile& pair : report.pairs) {
            if (!pair.m_resolved) continue;
            if (rows == 0) {
                out << std::left << std::setw(18) << "Column" << std::setw(18)
                    << "Against" << std::right << std::setw(11) << "M pairs"
                    << std::setw(9) << "Both" << std::setw(9) << "M redu" << std::setw(9)
                    << "U redu" << std::setw(9) << "Net" << "\n";
                out << std::string(83, '-') << "\n";
            }
            ++rows;
            out << std::left << std::setw(18) << Truncate(pair.left_name, 17)
                << std::setw(18) << Truncate(pair.right_name, 17) << std::right
                << std::setw(11) << WithThousands(pair.m_pairs) << std::setw(9)
                << Fixed(pair.m_joint, 3) << std::setw(9)
                << Fixed(pair.m_redundant_bits, 2) << std::setw(9)
                << (pair.resolved ? Fixed(pair.redundant_bits, 2) : "-") << std::setw(9)
                << (pair.resolved ? Signed(pair.NetRedundantBits()) : "-") << "\n";
        }
        if (rows > 0) out << "\n";
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
        << "u-side overlap alone, and Net is what the weight actually double-counts "
           "once\n"
        << "the m-side overlap is set against it. Both the prior odds and what Redund "
           "can\n"
        << "be read on move with --expected-matches, which defaults to one duplicate "
           "per\n"
        << "record.\n";
    if (report.anchored) {
        out << "\nm comes from anchor pairs: two rows agreeing on every column of an "
               "anchor,\nwhich the anchor's bits make a match on that agreement alone. "
               "The anchor is\nheld out of what the session is read for, and so is every "
               "column it\ndetermines. The bias runs one way and is not removed by any "
               "of "
               "that: anchors\nare the matches that happened to agree on a clean "
               "identifier, and agreement is\ncorrelated among matches, so every m here "
               "reads high. Spread is the same m\nover different anchors, which is what "
               "that bias looks like from the inside.\n";
    }

    if (report.truthed) {
        out << "\nTrue m is the same agreement rate over the known pairs, which nothing "
               "above it\never read. Err is how far the anchor estimate sits from it, "
               "and it is expected\nto be positive: what that column of numbers "
               "measures is the selection an anchor\nmakes, not an error in the "
               "arithmetic.\n";
    }
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
    root["anchored"] = report.anchored;
    root["anchor_refusal"] = report.anchor_refusal;
    root["anchor_rows"] = report.anchor_rows;
    root["anchor_pairs"] = report.anchor_pairs;
    root["expected_bits"] = report.expected_bits;
    root["double_counted_bits"] = report.double_counted_bits;
    root["estimated_margin_bits"] = report.estimated_margin_bits;
    root["estimated_columns"] = report.estimated_columns;
    root["scored_columns"] = report.scored_columns;
    root["truthed"] = report.truthed;
    root["truth_pairs"] = report.truth_pairs;
    root["truth_expected_bits"] = report.truth_expected_bits;
    root["truth_margin_bits"] = report.truth_margin_bits;
    root["truth_mean_error"] = report.truth_mean_error;
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

    root["sessions"] = nlohmann::json::array();
    for (const AnchorSession& session : report.sessions) {
        nlohmann::json item;
        item["anchor"] = session.anchor_names;
        item["bits"] = session.bits;
        item["posterior_bits"] = session.posterior_bits;
        item["groups"] = session.groups;
        item["pairs"] = session.pairs;
        item["oversized_groups"] = session.oversized;
        item["capped"] = session.capped;
        item["used"] = session.used;
        item["learns"] = nlohmann::json::array();
        for (size_t column : session.learns) {
            item["learns"].push_back(report.columns[column].name);
        }
        root["sessions"].push_back(std::move(item));
    }

    root["matches"] = nlohmann::json::array();
    for (const ColumnMatchProfile& match : report.matches) {
        nlohmann::json item;
        item["name"] = match.name;
        item["estimated"] = match.estimated;
        item["sessions"] = match.sessions;
        item["pairs"] = match.pairs;
        item["coverage"] = match.coverage;
        item["m"] = match.m;
        item["m_low"] = match.m_low;
        item["m_high"] = match.m_high;
        item["weight"] = match.weight;
        item["expected_bits"] = match.expected_bits;
        item["floored"] = match.floored;
        item["truthed"] = match.truthed;
        if (match.truthed) {
            item["truth_pairs"] = match.truth_pairs;
            item["truth_coverage"] = match.truth_coverage;
            item["truth_m"] = match.truth_m;
            item["truth_weight"] = match.truth_weight;
            item["truth_expected_bits"] = match.truth_expected_bits;
        }
        root["matches"].push_back(std::move(item));
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
        item["m_pairs"] = pair.m_pairs;
        item["m_left"] = pair.m_left;
        item["m_right"] = pair.m_right;
        item["m_joint"] = pair.m_joint;
        item["m_redundant_bits"] = pair.m_redundant_bits;
        item["m_resolved"] = pair.m_resolved;
        item["net_redundant_bits"] = pair.NetRedundantBits();
        item["suspect"] = pair.Suspect();
        if (pair.Suspect()) item["verdict"] = pair.Verdict();
        root["pairs"].push_back(std::move(item));
    }
    out << root.dump(2) << "\n";
}

}  // namespace cpplink
