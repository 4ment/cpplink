// Copyright 2026 Mathieu Fourment
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

#include "cpplink/comparison.hpp"
#include "cpplink/explain.hpp"
#include "cpplink/record_store.hpp"
#include "cpplink/schema.hpp"
#include "cpplink/score.hpp"

namespace cpplink {

// Top-k retrieval over the match weight: one query record against a whole store.
//
// The weight is a sum of per-column terms, which is the shape a search engine's
// ranking function has, so this is top-k retrieval over an additive score rather
// than a linkage problem. What makes it cheap is the interning invariant: a
// string level is a function of two *value ids*, so the level every distinct
// value of a column lands on against the query is one table, computed once per
// query, and a row's pattern is then one gather per column and one lookup into
// the tabulated base weight. No string metric runs per row.
//
// The result is exact with respect to the model. Ranking is on
// `BaseWeight + DeltaMax`, which brackets every pair that pattern can produce,
// and the exact weight is computed only for the rows that bound admits -- the
// same admissibility `predict` rests on, used to skip rows instead of pairs.

// One column of a query, as text: the same form the file the store was loaded
// from holds, because a query value has to be interned against that column's
// dictionary to mean anything.
struct QueryField {
    std::string column;
    std::string value;
};

// What a caller knows about the record being looked for.
//
// A column the query does not name is *missing*, not empty, which is the null
// level -- so a query carrying a name and a date of birth against a store of ten
// columns is scored as a record whose other eight fields were not recorded,
// which is what it is.
struct QueryRecord {
    std::vector<QueryField> fields;

    void Set(const std::string& column, const std::string& value);
    // The text given for a column, or null where the query does not name it.
    const std::string* Find(const std::string& column) const;
};

// Parses `column=value`, which is how the command line takes one field.
bool ParseQueryField(const std::string& text, QueryField* field, std::string* error);

// The prior odds in bits for "I expect this store to hold `expected` records of
// the person I am asking about".
//
// The model's own prior is lambda, the match rate over the *pair* space, which is
// the question deduplication asks. A query against N records asks a different one
// and the two differ by orders of magnitude, so the search prior is a per-query
// option rather than something inherited. It shifts every hit by the same
// constant, so it changes the posterior and the threshold and never the order.
double PriorWeightForExpected(double expected, uint64_t records);

struct SearchOptions {
    size_t k = 10;
    // Rows scoring below this are not returned however few hits there are, which
    // is what makes "nobody in here is this person" expressible.
    double threshold = -std::numeric_limits<double>::infinity();
    unsigned threads = 1;
    // Replaces the model's prior with this many bits. Off by default, so a search
    // scores exactly what `predict` would score for the same pair.
    bool override_prior = false;
    double prior_weight = 0.0;
};

struct SearchHit {
    uint64_t row = 0;
    std::string id;
    std::string dataset;
    uint32_t gamma = 0;
    double weight = 0.0;
    double probability = 0.0;
    // The cluster the row belongs to, where the caller gave a cluster file, and
    // whether this hit is the best row of it.
    std::string cluster;
    bool cluster_best = true;
};

struct SearchReport {
    std::vector<SearchHit> hits;
    uint64_t records = 0;
    // Comparisons answered from a level table, and those evaluated per row
    // because their shape has no table: a comparison reading a date, a list, a
    // coordinate pair or two string columns at once is not a function of one
    // value id, so it keeps the pair path's own evaluation.
    size_t tabled = 0;
    size_t evaluated = 0;
    // Comparisons over a column the query does not carry, which are the same
    // level for every row and cost nothing at all.
    size_t constant = 0;
    // Distinct values phase one walked, summed over the tabled comparisons.
    uint64_t values_walked = 0;
    // Values the query carried that the store's dictionaries had never seen.
    size_t values_adopted = 0;
    // Rows whose exact term-frequency-adjusted weight had to be computed. The
    // rest were dropped on the bracket alone.
    uint64_t rescored = 0;
    double prior = 0.0;        // the prior in force, in bits
    double model_prior = 0.0;  // what the model's own lambda says
    double walk_seconds = 0.0;
    double gather_seconds = 0.0;
    unsigned threads = 1;
    // Set where a cluster file was read: clusters the hits fall in.
    size_t clusters = 0;
};

void PrintSearchReport(const SearchReport& report, std::ostream& out);
std::string SearchReportJson(const SearchReport& report);

// A query held against a store for as long as it takes to answer it and explain
// the answer.
//
// The query is installed as one row past the store's loaded rows. That is the one
// place the store is written to after load, and it is deliberate: a query that is
// a row is scored by the same `Evaluate`, bounded by the same brackets and
// explained by the same `BuildPairWaterfall` as any pair, with no second
// implementation of any of it to drift from the first. The row is written before
// any worker starts and removed when the next query replaces it or the searcher
// is destroyed, so nothing concurrent ever observes the store change and the
// invariant that earns its keep -- no locking in the hot path -- is untouched.
//
// One searcher answers one query at a time. Two searchers over one store, or two
// queries at once, are not supported and would be a second writer.
class Searcher {
   public:
    Searcher() = default;
    ~Searcher();
    Searcher(const Searcher&) = delete;
    Searcher& operator=(const Searcher&) = delete;

    // The store and the comparison set are written to, so both are taken
    // mutable; the scorer is not. Fails where the store holds more than one
    // input: a query row would have to be a dataset of its own, and the levels
    // that read which input a row came from were bound against a boundary list
    // that does not know about it.
    bool Bind(RecordStore* store, ComparisonSet* comparisons, const Scorer* scorer,
              std::string* error);

    bool Search(const QueryRecord& query, const SearchOptions& options,
                SearchReport* report, std::string* error);

    // The row the current query occupies, valid until the next `Search` or the
    // searcher's destruction. This is what makes a hit explainable: the pair is
    // `(hit.row, QueryRow())` and every report that takes a pair takes it.
    uint64_t QueryRow() const { return query_row_; }
    bool HasQuery() const { return installed_; }

   private:
    // What one comparison does per row: read a level out of a table indexed by
    // the column's value id, take a level no row can change, or evaluate the
    // comparison against the query row the way the pair path does.
    struct Plan {
        enum class Kind : uint8_t { kConstant, kTable, kLevels, kEvaluate };
        // One string level of a kLevels comparison: whether it fires against the
        // query, by value id of the column *that level* reads.
        struct LevelTable {
            uint8_t level = 0;
            const StringColumn* strings = nullptr;
            std::vector<uint8_t> fires;
        };
        Kind kind = Kind::kEvaluate;
        uint8_t shift = 0;
        uint8_t constant = 0;    // kConstant: the level every row lands on
        uint8_t null_level = 0;  // the level a row missing the value takes
        bool has_null = false;   // whether the comparison declares one
        uint8_t otherwise = 0;   // kLevels: the level a row reaching none takes
        const StringColumn* strings = nullptr;
        std::vector<uint8_t> table;  // kTable: the level, by value id
        std::vector<LevelTable> levels;
    };

    void Uninstall();
    bool Install(const QueryRecord& query, std::string* error);
    void BuildPlans(SearchReport* report);
    void ScanRange(uint64_t begin, uint64_t end, const SearchOptions& options,
                   double shift, std::vector<SearchHit>* best, uint64_t* rescored) const;

    RecordStore* store_ = nullptr;
    ComparisonSet* comparisons_ = nullptr;
    const Scorer* scorer_ = nullptr;
    uint64_t query_row_ = 0;
    bool installed_ = false;
    size_t adopted_ = 0;
    // How much of the query row was written before the install returned, so a
    // refusal takes back exactly what it added and no more.
    size_t written_ = 0;
    bool wrote_id_ = false;
    // What each string column's dictionary held before the query adopted
    // anything, so the adoptions can be taken back.
    std::vector<uint32_t> dictionary_sizes_;
    std::vector<Plan> plans_;
};

// Reads a cluster file -- what `cluster` writes -- and labels each hit with the
// cluster its row belongs to, marking the best-scoring row of each. The hits are
// rows and a caller looking for a person wants entities, which is the store's
// clusters; grouping at the end rather than indexing one row per cluster is what
// keeps the recall of a cluster whose members disagree.
bool GroupHitsByCluster(const RecordStore& store, const std::string& cluster_path,
                        SearchReport* report, std::string* error);

}  // namespace cpplink
