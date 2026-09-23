// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#include "cpplink/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "cpplink/arrow_c.hpp"
#include "cpplink/batch_loader.hpp"
#include "cpplink/derive.hpp"
#include "cpplink/format.hpp"
#include "cpplink/pair_stream.hpp"
#include "cpplink/parquet_io.hpp"

namespace cpplink {
namespace {

double Seconds(std::chrono::steady_clock::time_point from,
               std::chrono::steady_clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

// Howard Hinnant's days_from_civil, the inverse of the conversion `derive` uses
// to take a stored date apart. A query types a date; the store holds days.
int32_t DaysFromCivil(int64_t year, int64_t month, int64_t day) {
    year -= month <= 2;
    const int64_t era = (year >= 0 ? year : year - 399) / 400;
    const int64_t yoe = year - era * 400;
    const int64_t doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<int32_t>(era * 146097 + doe - 719468);
}

// A date as the file would have held it. Only the ISO form is accepted, because
// a query that means one thing to the store and another to the reader is worse
// than a query that is refused.
bool ParseDate(const std::string& text, int32_t* days) {
    if (text.size() != 10 || text[4] != '-' || text[7] != '-') return false;
    for (size_t i = 0; i < text.size(); ++i) {
        if (i == 4 || i == 7) continue;
        if (text[i] < '0' || text[i] > '9') return false;
    }
    const int64_t year = std::stol(text.substr(0, 4));
    const int64_t month = std::stol(text.substr(5, 2));
    const int64_t day = std::stol(text.substr(8, 2));
    if (month < 1 || month > 12 || day < 1 || day > 31) return false;
    *days = DaysFromCivil(year, month, day);
    return true;
}

bool ParseBoolean(const std::string& text, int8_t* value) {
    std::string lower;
    lower.reserve(text.size());
    for (const char ch : text) lower.push_back(static_cast<char>(std::tolower(ch)));
    if (lower == "1" || lower == "true" || lower == "t" || lower == "yes" ||
        lower == "y") {
        *value = 1;
        return true;
    }
    if (lower == "0" || lower == "false" || lower == "f" || lower == "no" ||
        lower == "n") {
        *value = 0;
        return true;
    }
    return false;
}

// The id a dictionary already holds this text under, by a walk over its values.
//
// The load-time index is released when the store is finalized and rebuilding it
// would cost more memory than the dictionary itself, so the lookup is a scan:
// a length test and a memcmp per value, which on the columns a fuzzy level reads
// is a fraction of the walk that follows it anyway.
bool FindValue(const Dictionary& dict, uint32_t limit, std::string_view text,
               uint32_t* id) {
    for (uint32_t v = 0; v < limit; ++v) {
        if (dict.Value(v) == text) {
            *id = v;
            return true;
        }
    }
    return false;
}

// A hit is better than another when it scores higher, and where two tie, when it
// sits at the lower row. Without the second half the answer would depend on which
// thread saw which row first, and "the same k rows in the same order as scoring
// every pair" would not be a property anything could assert.
bool Better(const SearchHit& left, const SearchHit& right) {
    if (left.weight != right.weight) return left.weight > right.weight;
    return left.row < right.row;
}

void Offer(std::vector<SearchHit>* best, size_t k, const SearchHit& hit) {
    // A max-heap under "better" keeps the worst kept hit at the front, which is
    // the one a new hit has to beat and the one the bound is tested against.
    if (best->size() < k) {
        best->push_back(hit);
        std::push_heap(best->begin(), best->end(), Better);
        return;
    }
    if (!Better(hit, best->front())) return;
    std::pop_heap(best->begin(), best->end(), Better);
    best->back() = hit;
    std::push_heap(best->begin(), best->end(), Better);
}

// Whether every level of a comparison is a predicate over two values of one
// string column -- not necessarily the *same* column for every level, which is
// the address-against-its-username shape. Such a comparison can be tabulated:
// each level's verdict against the query is one byte per value id, and a row
// then costs a load and a compare instead of a string metric.
//
// A comparison reading a date, a number, a list or a coordinate pair is not of
// this shape and keeps the pair path's own evaluation against the query row.
bool TabulatableLevels(const BoundComparison& bound) {
    if (bound.slots.empty()) return false;
    if (bound.dates != nullptr || bound.booleans != nullptr || bound.lists != nullptr ||
        bound.numbers != nullptr) {
        return false;
    }
    for (const LevelSpec& level : bound.spec->levels) {
        switch (level.type) {
            case LevelType::kNull:
            case LevelType::kElse:
            case LevelType::kExact:
            case LevelType::kLevenshtein:
            case LevelType::kJaroWinkler:
                break;
            default:
                return false;
        }
    }
    return true;
}

// And whether it reads one column throughout, in which case the whole
// comparison collapses to a single table and a row costs one lookup rather than
// one per level.
bool TabulatableAsOne(const BoundComparison& bound) {
    return bound.slots.size() == 1 && TabulatableLevels(bound);
}

// The level a comparison of this shape lands on where no level fires: its else,
// which every comparison is required to end in.
uint8_t OtherwiseLevel(const BoundComparison& bound) {
    return static_cast<uint8_t>(bound.spec->levels.size() - 1);
}

// A dictionary walk split over threads. It is trivially parallel -- every value
// is an independent question about the query -- and on a near-unique column it is
// the whole of a query's latency, so it is the first thing worth splitting.
template <typename Body>
void Walk(uint32_t values, unsigned threads, Body&& body) {
    if (threads <= 1 || values <= 4096) {
        body(0u, values);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(threads);
    const uint32_t span = (values + threads - 1) / threads;
    for (unsigned t = 0; t < threads; ++t) {
        const uint32_t from = std::min(values, span * t);
        const uint32_t to = std::min(values, from + span);
        if (from >= to) break;
        workers.emplace_back([&body, from, to] { body(from, to); });
    }
    for (std::thread& worker : workers) worker.join();
}

// The level index of the comparison's null level, or -1 where it declares none.
int NullLevel(const BoundComparison& bound) {
    for (size_t i = 0; i < bound.spec->levels.size(); ++i) {
        if (bound.spec->levels[i].type == LevelType::kNull) return static_cast<int>(i);
    }
    return -1;
}

}  // namespace

void QueryRecord::Set(const std::string& column, const std::string& value) {
    fields.push_back({column, value});
}

const std::string* QueryRecord::Find(const std::string& column) const {
    for (const QueryField& field : fields) {
        if (field.column == column) return &field.value;
    }
    return nullptr;
}

bool ParseQueryField(const std::string& text, QueryField* field, std::string* error) {
    const size_t split = text.find('=');
    if (split == std::string::npos || split == 0) {
        *error = "'" + text + "' is not <column>=<value>";
        return false;
    }
    field->column = text.substr(0, split);
    field->value = text.substr(split + 1);
    return true;
}

double PriorWeightForExpected(double expected, uint64_t records) {
    if (records == 0) return 0.0;
    const double n = static_cast<double>(records);
    // Clamped away from both ends: zero expected matches is minus infinity, and
    // expecting every record to be the query's subject is plus infinity.
    const double p = std::min(std::max(expected / n, 1e-300), 1.0 - 1e-12);
    return std::log2(p / (1.0 - p));
}

Searcher::~Searcher() { Uninstall(); }

bool Searcher::Bind(RecordStore* store, ComparisonSet* comparisons, const Scorer* scorer,
                    std::string* error) {
    if (store->NumDatasets() > 1) {
        *error =
            "search reads one input: a query is a record the store does not hold, "
            "so it is appended as a row, and over more than one input that row "
            "would have to be a dataset of its own -- which the levels reading "
            "which input a row came from were bound before it existed";
        return false;
    }
    store_ = store;
    comparisons_ = comparisons;
    scorer_ = scorer;
    query_row_ = store->NumRecords();
    return true;
}

void Searcher::Uninstall() {
    if (!installed_ && written_ == 0 && !wrote_id_) return;
    // Only as far as the install got: a query refused half way through has
    // appended to the columns before the offending one and to none after it, and
    // popping a column that was never written to would take a record's value off
    // the store.
    const size_t columns = written_;
    for (size_t i = 0; i < columns; ++i) {
        Column& column = store_->mutable_column(i);
        if (auto* typed = std::get_if<StringColumn>(&column)) {
            typed->ids.pop_back();
            typed->dict.Truncate(dictionary_sizes_[i]);
            if (!typed->tf.empty()) typed->tf.resize(typed->dict.Size());
        } else if (auto* typed = std::get_if<StringListColumn>(&column)) {
            typed->ids.resize(typed->offsets[query_row_]);
            typed->offsets.pop_back();
            typed->dict.Truncate(dictionary_sizes_[i]);
            if (!typed->tf.empty()) typed->tf.resize(typed->dict.Size());
        } else if (auto* typed = std::get_if<DateColumn>(&column)) {
            typed->values.pop_back();
        } else if (auto* typed = std::get_if<DoubleColumn>(&column)) {
            typed->values.pop_back();
        } else if (auto* typed = std::get_if<BooleanColumn>(&column)) {
            typed->values.pop_back();
        }
    }
    if (wrote_id_) {
        IdColumn& ids = store_->mutable_ids();
        ids.offsets.pop_back();
        ids.text.resize(ids.offsets.back());
    }
    comparisons_->ResizeTables();
    plans_.clear();
    written_ = 0;
    wrote_id_ = false;
    installed_ = false;
}

bool Searcher::Install(const QueryRecord& query, std::string* error) {
    const Schema& schema = store_->schema();
    const size_t columns = store_->NumColumns();
    dictionary_sizes_.assign(columns, 0);
    written_ = 0;
    wrote_id_ = false;

    // Every column's text, so a derived column can be computed from the source
    // the query gave rather than having to be given itself. A derived column the
    // query names explicitly keeps what it was given.
    std::vector<const std::string*> given(columns, nullptr);
    std::unordered_map<std::string, size_t> by_name;
    for (size_t i = 0; i < columns; ++i) {
        by_name.emplace(schema.columns[i].name, i);
        given[i] = query.Find(schema.columns[i].name);
    }
    std::vector<std::string> derived(columns);
    for (size_t i = 0; i < columns; ++i) {
        const ColumnSpec& spec = schema.columns[i];
        if (!spec.IsDerived() || given[i] != nullptr) continue;
        const auto source = by_name.find(spec.derive.from);
        if (source == by_name.end()) continue;
        const std::string* text = given[source->second];
        if (text == nullptr || text->empty()) continue;
        if (schema.columns[source->second].type == ColumnType::kDate) {
            int32_t days = 0;
            if (!ParseDate(*text, &days)) continue;
            ApplyDateTransforms(spec.derive.transforms, days, &derived[i]);
        } else {
            ApplyTransforms(spec.derive.transforms, *text, &derived[i]);
        }
        if (!derived[i].empty()) given[i] = &derived[i];
    }

    size_t adopted = 0;
    for (size_t i = 0; i < columns; ++i) {
        const ColumnSpec& spec = schema.columns[i];
        Column& column = store_->mutable_column(i);
        const std::string* text = given[i];
        const bool missing = text == nullptr || text->empty();

        if (auto* typed = std::get_if<StringColumn>(&column)) {
            dictionary_sizes_[i] = typed->dict.Size();
            uint32_t id = kNullId;
            if (!missing) {
                if (!FindValue(typed->dict, dictionary_sizes_[i], *text, &id)) {
                    // A value the store never saw still has to be a value id: a
                    // fuzzy level reads its text and its signature, and both are
                    // addressed by id. It agrees with nothing exactly, which is
                    // right, because nothing here holds it.
                    id = typed->dict.Adopt(*text);
                    if (!typed->tf.empty()) typed->tf.push_back(0);
                    ++adopted;
                }
            }
            typed->ids.push_back(id);
        } else if (auto* typed = std::get_if<StringListColumn>(&column)) {
            dictionary_sizes_[i] = typed->dict.Size();
            // A list column takes one element per repeat of the field, which is
            // unambiguous where a delimiter inside the values would not be.
            std::vector<uint32_t> cell;
            for (const QueryField& field : query.fields) {
                if (field.column != spec.name || field.value.empty()) continue;
                uint32_t id = kNullId;
                if (!FindValue(typed->dict, dictionary_sizes_[i], field.value, &id)) {
                    id = typed->dict.Adopt(field.value);
                    if (!typed->tf.empty()) typed->tf.push_back(0);
                    ++adopted;
                }
                cell.push_back(id);
            }
            // The cells are sorted and deduplicated at load, and every level over
            // them is a linear merge that assumes it.
            std::sort(cell.begin(), cell.end());
            cell.erase(std::unique(cell.begin(), cell.end()), cell.end());
            typed->ids.insert(typed->ids.end(), cell.begin(), cell.end());
            typed->offsets.push_back(typed->ids.size());
        } else if (auto* typed = std::get_if<DateColumn>(&column)) {
            int32_t days = kNullDate;
            if (!missing && !ParseDate(*text, &days)) {
                *error = "column \"" + spec.name + "\" is a date and '" + *text +
                         "' is not one: write it as YYYY-MM-DD";
                return false;
            }
            typed->values.push_back(days);
        } else if (auto* typed = std::get_if<DoubleColumn>(&column)) {
            double value = std::nan("");
            if (!missing) {
                try {
                    value = std::stod(*text);
                } catch (const std::exception&) {
                    *error = "column \"" + spec.name + "\" is a number and '" + *text +
                             "' is not one";
                    return false;
                }
            }
            typed->values.push_back(value);
        } else if (auto* typed = std::get_if<BooleanColumn>(&column)) {
            int8_t value = kNullBoolean;
            if (!missing && !ParseBoolean(*text, &value)) {
                *error = "column \"" + spec.name + "\" is a boolean and '" + *text +
                         "' is not one";
                return false;
            }
            typed->values.push_back(value);
        }
        written_ = i + 1;
    }
    store_->mutable_ids().Append("query");
    wrote_id_ = true;
    query_row_ = store_->NumRecords();
    installed_ = true;
    adopted_ = adopted;
    // Signatures are indexed by value id, so any dictionary the query grew has to
    // be followed before a level reads one.
    comparisons_->ResizeTables();
    return true;
}

void Searcher::BuildPlans(SearchReport* report) {
    const size_t count = comparisons_->Size();
    plans_.assign(count, Plan());
    const unsigned threads = ResolveThreads(report->threads);
    for (size_t c = 0; c < count; ++c) {
        const BoundComparison& bound = comparisons_->at(c);
        Plan& plan = plans_[c];
        plan.shift = bound.shift;
        const int null_level = NullLevel(bound);
        // A comparison the query says nothing about is the same level for every
        // row, whatever its shape: the null level is the first the walk reaches
        // and it fires on the query's side alone. That is most of a query, since
        // a caller who knows ten fields about the person is not searching, and it
        // is why a nine-comparison schema costs three or four of them.
        if (null_level == 0 && comparisons_->IsNullValue(c, query_row_)) {
            plan.kind = Plan::Kind::kConstant;
            plan.constant = 0;
            ++report->constant;
            continue;
        }
        plan.has_null = null_level >= 0;
        plan.null_level = null_level >= 0 ? static_cast<uint8_t>(null_level) : 0;
        if (!TabulatableLevels(bound)) {
            plan.kind = Plan::Kind::kEvaluate;
            ++report->evaluated;
            continue;
        }
        if (!TabulatableAsOne(bound)) {
            // The levels read different columns, so each gets its own table over
            // its own dictionary and a row walks them in order, exactly as the
            // pair path walks the levels themselves.
            plan.kind = Plan::Kind::kLevels;
            plan.otherwise = OtherwiseLevel(bound);
            for (size_t i = 0; i < bound.spec->levels.size(); ++i) {
                const LevelType type = bound.spec->levels[i].type;
                if (type == LevelType::kNull || type == LevelType::kElse) continue;
                Plan::LevelTable table;
                table.level = static_cast<uint8_t>(i);
                const BoundComparison::StringSlot& slot =
                    bound.slots[bound.spec->levels[i].column];
                table.strings = slot.strings;
                const uint32_t self = slot.strings->ids[query_row_];
                const uint32_t values = slot.strings->dict.Size();
                table.fires.assign(values, 0);
                if (self != kNullId) {
                    const size_t level = i;
                    Walk(values, threads, [&](uint32_t from, uint32_t to) {
                        for (uint32_t v = from; v < to; ++v) {
                            table.fires[v] =
                                comparisons_->StringLevelForValues(c, level, self, v) ? 1
                                                                                      : 0;
                        }
                    });
                    report->values_walked += values;
                }
                plan.levels.push_back(std::move(table));
            }
            ++report->tabled;
            continue;
        }
        const StringColumn& strings = *bound.slots.front().strings;
        const uint32_t self = strings.ids[query_row_];
        // A column the query leaves out whose comparison declares no null level
        // at the top is still a constant, since no level a value reaches can fire
        // against a value that is not there.
        if (self == kNullId) {
            plan.kind = Plan::Kind::kConstant;
            plan.constant = null_level >= 0
                                ? static_cast<uint8_t>(null_level)
                                : comparisons_->LevelForValues(c, kNullId, kNullId);
            ++report->constant;
            continue;
        }
        plan.kind = Plan::Kind::kTable;
        plan.strings = &strings;
        if (null_level < 0) {
            plan.null_level = comparisons_->LevelForValues(c, self, kNullId);
        }
        const uint32_t values = strings.dict.Size();
        plan.table.resize(values);
        // This is where every string metric of the query runs: once per distinct
        // value of the column rather than once per row, which is the whole of the
        // economy and the reason the row pass below touches no character.
        Walk(values, threads, [&](uint32_t from, uint32_t to) {
            for (uint32_t v = from; v < to; ++v) {
                plan.table[v] = comparisons_->LevelForValues(c, self, v);
            }
        });
        report->values_walked += values;
        ++report->tabled;
    }
}

void Searcher::ScanRange(uint64_t begin, uint64_t end, const SearchOptions& options,
                         double shift, std::vector<SearchHit>* best,
                         uint64_t* rescored) const {
    uint32_t constant = 0;
    for (const Plan& plan : plans_) {
        if (plan.kind == Plan::Kind::kConstant) {
            constant |= static_cast<uint32_t>(plan.constant) << plan.shift;
        }
    }
    for (uint64_t row = begin; row < end; ++row) {
        uint32_t gamma = constant;
        for (size_t c = 0; c < plans_.size(); ++c) {
            const Plan& plan = plans_[c];
            uint8_t level = 0;
            switch (plan.kind) {
                case Plan::Kind::kConstant:
                    continue;
                case Plan::Kind::kTable: {
                    const uint32_t value = plan.strings->ids[row];
                    level = value == kNullId ? plan.null_level : plan.table[value];
                    break;
                }
                case Plan::Kind::kLevels: {
                    // The comparison is null wherever *any* of its columns is,
                    // which is not a question one slot's table can answer.
                    if (plan.has_null && comparisons_->IsNullValue(c, row)) {
                        level = plan.null_level;
                        break;
                    }
                    level = plan.otherwise;
                    for (const Plan::LevelTable& table : plan.levels) {
                        const uint32_t value = table.strings->ids[row];
                        if (value != kNullId && table.fires[value] != 0) {
                            level = table.level;
                            break;
                        }
                    }
                    break;
                }
                case Plan::Kind::kEvaluate:
                    level = comparisons_->EvaluateOne(c, row, query_row_);
                    break;
            }
            gamma |= static_cast<uint32_t>(level) << plan.shift;
        }
        const double bound =
            scorer_->BaseWeight(gamma) + scorer_->DeltaMax(gamma) + shift;
        if (bound < options.threshold) continue;
        if (best->size() == options.k && bound < best->front().weight) continue;
        // The store's row is the first argument throughout: an exact level's
        // adjustment reads the frequency of `a`'s value, and the query's own
        // value was never counted into the table.
        SearchHit hit;
        hit.row = row;
        hit.gamma = gamma;
        hit.weight = scorer_->Weight(gamma, row, query_row_) + shift;
        ++*rescored;
        if (hit.weight < options.threshold) continue;
        Offer(best, options.k, hit);
    }
}

bool Searcher::Search(const QueryRecord& query, const SearchOptions& options,
                      SearchReport* report, std::string* error) {
    if (store_ == nullptr) {
        *error = "the searcher is not bound to a store";
        return false;
    }
    if (options.k == 0) {
        *error = "search needs a k of at least one";
        return false;
    }
    Uninstall();
    *report = SearchReport();
    report->threads = ResolveThreads(options.threads);
    report->records = store_->NumRecords();
    report->model_prior = scorer_->PriorWeight();
    report->prior = options.override_prior ? options.prior_weight : report->model_prior;

    const auto started = std::chrono::steady_clock::now();
    if (!Install(query, error)) {
        Uninstall();
        return false;
    }
    report->values_adopted = adopted_;
    BuildPlans(report);
    const auto walked = std::chrono::steady_clock::now();
    report->walk_seconds = Seconds(started, walked);

    const double shift = report->prior - report->model_prior;
    const uint64_t rows = store_->NumRecords();
    std::vector<std::vector<SearchHit>> per_thread(report->threads);
    std::vector<uint64_t> rescored(report->threads, 0);
    if (report->threads > 1 && rows > 0) {
        std::vector<std::thread> workers;
        workers.reserve(report->threads);
        const uint64_t span = (rows + report->threads - 1) / report->threads;
        for (unsigned t = 0; t < report->threads; ++t) {
            const uint64_t from = std::min(rows, span * t);
            const uint64_t to = std::min(rows, from + span);
            if (from >= to) break;
            workers.emplace_back([&, t, from, to] {
                ScanRange(from, to, options, shift, &per_thread[t], &rescored[t]);
            });
        }
        for (std::thread& worker : workers) worker.join();
    } else {
        ScanRange(0, rows, options, shift, &per_thread[0], &rescored[0]);
    }
    report->gather_seconds = Seconds(walked, std::chrono::steady_clock::now());

    std::vector<SearchHit> merged;
    for (size_t t = 0; t < per_thread.size(); ++t) {
        report->rescored += rescored[t];
        merged.insert(merged.end(), per_thread[t].begin(), per_thread[t].end());
    }
    std::sort(merged.begin(), merged.end(), Better);
    if (merged.size() > options.k) merged.resize(options.k);
    for (SearchHit& hit : merged) {
        hit.id = std::string(store_->ids().Get(hit.row));
        hit.dataset = store_->NumDatasets() > 1
                          ? store_->DatasetName(store_->DatasetOf(hit.row))
                          : std::string();
        hit.probability = ProbabilityForWeight(hit.weight);
    }
    report->hits = std::move(merged);
    return true;
}

void PrintSearchReport(const SearchReport& report, std::ostream& out) {
    out << "Records        " << WithThousands(report.records) << "\n"
        << "Comparisons    " << report.tabled << " tabulated over their dictionary, "
        << report.evaluated << " evaluated per row, " << report.constant
        << " constant because the query is missing the column\n"
        << "Values walked  " << WithThousands(report.values_walked);
    if (report.values_adopted > 0) {
        out << "  (" << report.values_adopted << " query value"
            << (report.values_adopted == 1 ? "" : "s") << " the store never held)";
    }
    out << "\n"
        << "Rows scored    " << WithThousands(report.rescored) << " of "
        << WithThousands(report.records) << " exactly; the rest were dropped on the "
        << "bracket\n"
        << "Prior          " << std::fixed << std::setprecision(3) << report.prior
        << " bits";
    if (report.prior != report.model_prior) {
        out << "  (the model's lambda says " << report.model_prior << ")";
    }
    out << "\n"
        << "Elapsed        " << std::setprecision(4) << report.walk_seconds
        << " s walking the dictionaries, " << report.gather_seconds
        << " s over the rows on " << report.threads
        << (report.threads == 1 ? " thread" : " threads") << "\n\n";

    if (report.hits.empty()) {
        out << "Nothing scores above the threshold: no record here is this one.\n";
        return;
    }
    out << std::left << std::setw(26) << "Record" << std::right << std::setw(12)
        << "weight" << std::setw(14) << "posterior";
    if (report.clusters > 0) out << "   cluster";
    out << "\n";
    for (const SearchHit& hit : report.hits) {
        std::string name = hit.id;
        if (!hit.dataset.empty()) name = hit.dataset + ":" + name;
        out << std::left << std::setw(26) << Truncate(name, 25) << std::right
            << std::setw(12) << std::fixed << std::setprecision(3) << hit.weight
            << std::setw(14) << std::setprecision(6) << hit.probability;
        if (report.clusters > 0) {
            out << "   " << hit.cluster;
            if (!hit.cluster_best) out << " (same cluster)";
        }
        out << "\n";
    }
    out << "\nA weight is bits of evidence for the query and the record being the "
           "same\nperson, and the posterior is what that means under the prior above. "
           "Zero bits\nis even odds, so a hit below it is evidence against.\n";
}

std::string SearchReportJson(const SearchReport& report) {
    nlohmann::json out;
    out["records"] = report.records;
    out["tabulated"] = report.tabled;
    out["evaluated"] = report.evaluated;
    out["constant"] = report.constant;
    out["values_walked"] = report.values_walked;
    out["values_adopted"] = report.values_adopted;
    out["rescored"] = report.rescored;
    out["prior"] = report.prior;
    out["model_prior"] = report.model_prior;
    out["walk_seconds"] = report.walk_seconds;
    out["gather_seconds"] = report.gather_seconds;
    out["threads"] = report.threads;
    nlohmann::json hits = nlohmann::json::array();
    for (const SearchHit& hit : report.hits) {
        nlohmann::json one;
        one["row"] = hit.row;
        one["id"] = hit.id;
        if (!hit.dataset.empty()) one["dataset"] = hit.dataset;
        one["gamma"] = hit.gamma;
        one["match_weight"] = hit.weight;
        one["match_probability"] = hit.probability;
        if (!hit.cluster.empty()) {
            one["cluster_id"] = hit.cluster;
            one["cluster_best"] = hit.cluster_best;
        }
        hits.push_back(one);
    }
    out["hits"] = hits;
    return out.dump();
}

namespace {

// The cluster file as a map from record id to cluster id, read from either shape
// `cluster --out` writes. Only the ids the hits name are kept, so a 20M-row
// cluster file costs the hits and not the file.
bool ReadClusterCsv(const std::string& path,
                    std::unordered_map<std::string, std::string>* clusters,
                    std::string* error) {
    std::ifstream file(path);
    if (!file) {
        *error = "cannot open " + path;
        return false;
    }
    std::string line;
    if (!std::getline(file, line)) {
        *error = path + " is empty";
        return false;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> header;
    for (size_t at = 0; at <= line.size();) {
        const size_t comma = std::min(line.find(',', at), line.size());
        header.push_back(line.substr(at, comma - at));
        at = comma + 1;
    }
    size_t id_at = header.size();
    size_t cluster_at = header.size();
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == "unique_id") id_at = i;
        if (header[i] == "cluster_id") cluster_at = i;
    }
    if (id_at == header.size() || cluster_at == header.size()) {
        *error = path + " has no \"unique_id\" and \"cluster_id\" columns, so it is " +
                 "not a cpplink cluster file";
        return false;
    }
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<std::string> cells;
        for (size_t at = 0; at <= line.size();) {
            const size_t comma = std::min(line.find(',', at), line.size());
            cells.push_back(line.substr(at, comma - at));
            at = comma + 1;
        }
        if (cells.size() <= std::max(id_at, cluster_at)) continue;
        clusters->emplace(cells[id_at], cells[cluster_at]);
    }
    return true;
}

bool ReadClusterParquet(const std::string& path,
                        std::unordered_map<std::string, std::string>* clusters,
                        std::string* error) {
    ArrowArrayStream stream;
    stream.release = nullptr;
    if (!OpenParquetStream(path, {"unique_id", "cluster_id"}, &stream, error)) {
        return false;
    }
    ArrowSchema schema;
    schema.release = nullptr;
    const auto release = [&] {
        if (schema.release != nullptr) schema.release(&schema);
        if (stream.release != nullptr) stream.release(&stream);
    };
    if (stream.get_schema(&stream, &schema) != 0) {
        release();
        *error = "cannot read the schema of " + path;
        return false;
    }
    const int id_at = FieldIndex(schema, "unique_id");
    const int cluster_at = FieldIndex(schema, "cluster_id");
    if (id_at < 0 || cluster_at < 0) {
        release();
        *error = path + " has no \"unique_id\" and \"cluster_id\" columns, so it is " +
                 "not a cpplink cluster file";
        return false;
    }
    while (true) {
        ArrowArray batch;
        batch.release = nullptr;
        if (stream.get_next(&stream, &batch) != 0) {
            release();
            *error = "cannot read the next batch of " + path;
            return false;
        }
        if (batch.release == nullptr) break;
        TextReader ids, names;
        if (!ids.Bind(schema, batch, id_at, error) ||
            !names.Bind(schema, batch, cluster_at, error)) {
            batch.release(&batch);
            release();
            return false;
        }
        char left[24], right[24];
        for (int64_t row = 0; row < batch.length; ++row) {
            std::string_view id;
            std::string_view cluster;
            if (!ids.At(row, &id, &left)) continue;
            if (!names.At(row, &cluster, &right)) continue;
            clusters->emplace(std::string(id), std::string(cluster));
        }
        batch.release(&batch);
    }
    release();
    return true;
}

}  // namespace

bool GroupHitsByCluster(const RecordStore& store, const std::string& cluster_path,
                        SearchReport* report, std::string* error) {
    (void)store;
    std::unordered_map<std::string, std::string> clusters;
    const bool parquet =
        cluster_path.size() > 8 &&
        cluster_path.compare(cluster_path.size() - 8, 8, ".parquet") == 0;
    if (parquet) {
        if (!ReadClusterParquet(cluster_path, &clusters, error)) return false;
    } else {
        if (!ReadClusterCsv(cluster_path, &clusters, error)) return false;
    }
    // The hits are already in descending weight, so the first hit of a cluster is
    // its best row and everything after it is another row of the same entity.
    std::unordered_map<std::string, size_t> seen;
    for (SearchHit& hit : report->hits) {
        const auto found = clusters.find(hit.id);
        if (found == clusters.end()) continue;
        hit.cluster = found->second;
        hit.cluster_best = seen.emplace(hit.cluster, 1).second;
    }
    report->clusters = seen.size();
    return true;
}

}  // namespace cpplink
