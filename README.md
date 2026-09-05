# cpplink

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

**cpplink** is a fast, streaming probabilistic record linker in C++. It implements the
Fellegi–Sunter model — EM parameter estimation, term-frequency adjustments, and prediction
— without ever materializing a table of candidate pairs.

It exists because tools like [splink](https://moj-analytical-services.github.io/splink/)
build the intermediate pair table as a real relation, which does not fit in memory at tens
of millions of records. cpplink folds pairs into a histogram of agreement patterns as they
are generated and discards them, so peak memory is set by the number of *records*, not the
number of *pairs*.

> **Status: phases 0–6 complete.** The record store, parquet loader, comparison levels,
> blocking sources, the recall harness, parameter estimation, scoring, clustering, the
> signature filter, spill and re-scoring, and record linkage across two inputs are all in
> and measured, so the pipeline runs end to end from parquet to duplicate clusters in both
> dedup and link mode.

## Approach

A pair enters the Fellegi–Sunter likelihood only through its agreement pattern γ — the
vector of comparison-level indices. Two pairs with the same γ are indistinguishable to the
model, so estimation needs only `count[γ]`, a histogram bounded by the number of *distinct*
patterns rather than the number of pairs. On a 18M-record deduplication:

|  | pair table | pattern histogram |
|---|---:|---:|
| Candidate pairs | 5×10⁹ | 5×10⁹ |
| Stored rows | 5×10⁹ | ~10⁵ |
| Resident memory | ~130 GB | 3.1 GB *(measured)* |
| Cost of one EM re-fit | re-read everything | < 0.1 s |

Blocking is a lazy iterator rather than a join, candidate pairs are deduplicated across
sources by a cheap predicate instead of a global `DISTINCT`, and term-frequency adjustment
is made affordable by admissible score bounds that let most patterns skip the TF tables
entirely.

Blocking is also **automatic**. Rather than hand-written rules, candidates come from the
term-frequency tables the model already needs — pairs are generated from agreement on
*rare* values, which is exactly the high-evidence event the model scores — supplemented by
MinHash LSH and sorted-neighbourhood passes. Sources that select on a whole record rather
than a column (an ANN index, for instance) are used for prediction only, because they break
the conditional-independence argument that makes EM's parameter estimates unbiased.

None of those sources is a new idea — see [prior art](#prior-art) — so the useful question
is not which to believe in but which earns its candidates, and that is measured rather than
argued. `recall` reports pair completeness, pair quality and the reduction ratio per source
and for the plan; [`bench/sweep_blocking.py`](bench/) sweeps each method over its own knob
and reports the frontier, because every method can buy recall with candidates and a single
operating point per method compares nothing.

## Usage

cpplink is told what is in the data with a JSON schema — see
[examples/sample_schema.json](examples/sample_schema.json):

```json
{
  "unique_id": "id",
  "columns": [
    {"name": "last_name",      "type": "string"},
    {"name": "dob",            "type": "date"},
    {"name": "latitude",       "type": "double"},
    {"name": "address_tokens", "type": "string_list"}
  ]
}
```

`string` columns are interned to dense `uint32` ids, `string_list` columns are stored as
CSR with each row sorted, `date` is days since the epoch, and `double` is stored as-is.
Only the first three carry term frequencies: exact agreement between two doubles is not a
discrete event worth counting, so a `double` column cannot drive rare-value blocking.

Comparisons are declared in the same file. Levels are evaluated top-down and the first hit
wins, so their order is the model — put the strongest evidence first. A comparison can span
more than one column: a coordinate pair is one comparison, not two.

```json
{"name": "location", "columns": ["latitude", "longitude"], "levels": [
  {"type": "null"},
  {"type": "geo_within", "threshold": 1},
  {"type": "geo_within", "threshold": 25},
  {"type": "else"}]}
```

Available level types: `null`, `exact`, `levenshtein`, `jaro_winkler`, `date_within`,
`numeric_within`, `geo_within`, `list_overlap`, `list_jaccard`, `list_contains`, `else`. A
configuration that applies a level to a column type it cannot read, omits a trailing `else`, or
overflows the 32-bit packed pattern is rejected at parse time, before a file is opened.

`list_contains` is the one level whose two columns have different types: a scalar string against
a list of aliases, firing when either row's value is an element of the other row's list. It is
how a `first_name` is checked against a `nicknames` column, and it sits in the same comparison as
the name's own levels, ordered between them:

```json
{"name": "forename", "columns": ["first_name", "nicknames"], "levels": [
  {"type": "null"},
  {"type": "exact"},
  {"type": "list_contains"},
  {"type": "jaro_winkler", "threshold": 0.9},
  {"type": "else"}]}
```

Blocking sources are declared in the same file and unioned in order. Every source here
selects on a single column, which is what makes it usable for estimating `m`:

```json
"blocking": [
  {"type": "exact_value", "column": "email"},
  {"type": "rare_value", "column": "last_name", "max_frequency": 100},
  {"type": "minhash", "column": "last_name", "bands": 10, "rows_per_band": 4},
  {"type": "sorted_neighbourhood", "column": "last_name", "window": 20}
]
```

```sh
# Report cardinality, null rates and the memory each structure costs
cpplink inspect --schema examples/sample_schema.json data.parquet

# Show which level each comparison assigns to one pair, and the packed pattern
cpplink explain --schema examples/sample_schema.json --pair r17,r19 data.parquet

# Price every blocking source exactly, without enumerating a single pair
cpplink explain-blocking --schema examples/sample_schema.json data.parquet

# Measure what fraction of known duplicate pairs blocking actually reaches
cpplink recall --schema examples/sample_schema.json --truth truth.csv data.parquet

# Learn m, u and lambda and write the model
cpplink estimate --schema examples/sample_schema.json --out model.json data.parquet

# Score every candidate pair and write the edges above a threshold
cpplink predict --schema examples/sample_schema.json --model model.json \
                --out edges/ --threshold 20 data.parquet

# Re-score a spilled run under a new model, without comparing anything again
cpplink predict --schema examples/sample_schema.json --model model.json \
                --out edges/ --spill spill/ --threshold 20 data.parquet
cpplink rescore --schema examples/sample_schema.json --model tuned.json \
                --spill spill/ --out edges2/ --threshold 20 data.parquet

# Join those edges into duplicate clusters, and score the result against known pairs
cpplink cluster --schema examples/sample_schema.json --edges edges/ \
                --out clusters.csv --truth truth.csv data.parquet

# Write a sample file with realistic cardinalities and planted duplicates
cpplink gen-sample --out sample.parquet --rows 18000000 --truth sample.truth.csv
```

`gen-sample` exists because the memory claims above are only worth making if they are
measured. It plants corrupted copies of earlier rows and records them, so the file also
serves as ground truth for the recall harness in a later phase.

## Planned features

- Parquet input with a JSON schema, including list-valued columns *(done)*
- Configurable comparisons and ordered comparison levels *(done)*
- Fellegi–Sunter model over the resulting agreement patterns *(done)*
- EM estimation of `m` and `λ`; exact closed-form `u` for exact-match levels *(done)*
- Term-frequency adjustments, with admissible bounds for pruning *(done)*
- Per-value character signatures bounding the string metrics, and Myers' bit-vector
  edit distance *(done)*
- Automatic blocking: exact-value and rare-value inverted indexes, MinHash LSH and sorted
  neighbourhood, with exact candidate-count reporting and recall measurement *(done)*
- ~~Optional ANN blocking for prediction~~ — **retired by measurement, not built.** `recall
  --why` shows no missed pair that fails to agree, exactly or fuzzily, on a column already
  in the schema, so an ANN index would have nothing to find; the remaining misses are
  columns that are simply not blocked on
- Multicore, shared-memory parallelism *(estimation and scoring done)*
- Connected-component clustering of the scored edges *(done)*
- Spill of (a, b, γ) and re-scoring under a new model without a second comparison pass
  *(done)*
- Deduplication first; record linkage across datasets through the same interfaces

## Getting Started

### Prerequisites

- CMake 3.15 or higher
- A C++17 compatible compiler
- GoogleTest (for the test suite)

The dependencies are declared in [environment.yml](environment.yml):

```sh
conda env create -f environment.yml
conda activate cpplink
```

### Building

```sh
cmake -S . -B build
cmake --build build
```

### Testing

```sh
cmake -S . -B build -DBUILD_TESTING=on -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build
ctest --test-dir build
```

## Prior art

The blocking methods here are drawn from the record linkage and entity resolution
literature rather than invented for this tool, and it is worth being explicit about which
is which.

| Source | Prior art |
|---|---|
| sorted neighbourhood | Hernández & Stolfo 1995 |
| rare-value inverted index | IDF-weighted canopies (McCallum, Nigam & Ungar 2000); rare-token ordering in prefix-filtering similarity joins (Bayardo, Ma & Srikant 2007; Xiao et al. 2008) |
| MinHash LSH | Broder 1997; Indyk & Motwani 1998; benchmarked for record linkage by Steorts, Ventura, Sadinle & Fienberg 2014 |
| ANN over embeddings | DeepER (2018), AutoBlock (2020), DeepBlocker (2021), Sparkly (2023) |
| automatic blocking as such | Michelson & Knoblock 2006; Bilenko, Kamath & Mooney 2006; Kejriwal & Miranker 2013; the `dedupe` library; token blocking and meta-blocking (Papadakis et al.) |

Christen's 2012 survey of indexing techniques covers most of these and is the reference for
the pair-completeness, pair-quality and reduction-ratio metrics `recall` reports.

What this project claims is narrower, and none of it is a blocking method:

1. **The EM-safety criterion** — a pair source may feed estimation only if its selection
   event factors as a condition on an excludable column subset. A whole-record source
   biases *every* `m_c` with no column left to repair it, so estimation and prediction run
   over different unions of sources.
2. **Admissible per-pattern term-frequency brackets** that let most patterns be emitted or
   dropped without touching the TF tables, emitting exactly the edges a full scoring pass
   would.
3. **Exact closed-form candidate pricing** from the term-frequency tables, which prices
   8.25 billion pairs without enumerating one.
4. **The streaming implementation.** γ's sufficiency is Fellegi & Sunter 1969; that a full
   Fellegi–Sunter pipeline with TF adjustment fits in memory at 18M records, because
   nothing in it holds a row per pair, is an engineering result rather than a statistical
   one.

[DESIGN.md](DESIGN.md) carries the measurements behind each, and [bench/](bench/) the
comparison against splink and the blocking-method frontier.

## Documentation

The full documentation — the model and the EM algorithm, the blocking sources, and a page per
command explaining its goal and how to read its output — is built with
[MkDocs](https://www.mkdocs.org/) from [docs/](docs/):

```sh
conda activate cpplink
mkdocs serve     # live preview on http://127.0.0.1:8000
mkdocs build     # render to site/
```

## Code Style

This project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
with minor customizations (4-space indent, 90-column limit).

Tools used:

- clang-format for formatting — `cmake --build build --target format`
- cpplint for static analysis — `cpplint --recursive src tests`

## License

This project is licensed under the MIT License.
