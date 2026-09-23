# `search`

**Goal:** given a query record, return the records that score highest against it.

This is not linkage, it is retrieval. The match weight is a sum of per-column terms, which is
the shape a search engine's ranking function has, so "find this person in 20M records" is
top-k retrieval over an additive score. What makes it cheap here is interning: a string level
is a function of two *value ids*, so the level every distinct value of a column lands on
against the query is one table, computed once per query, after which a row costs a load and a
compare rather than a string metric.

The answer is exact with respect to the model. It is the same k records, in the same order,
that scoring the query against every record one at a time would return.

## Synopsis

```sh
cpplink search --schema <schema.json> --model <model.json>
               --field <column>=<value> [--field ...]
               [-k N] [--threshold BITS] [--threads N]
               [--expected-matches N | --prior-weight BITS]
               [--clusters <file>] [--explain] [--json] <file.parquet>
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required |
| `--model <file>` | — | required; the model whose `m`, `u` and levels the weight is built from |
| `--field <column>=<value>` | — | required, repeatable; one field of the query record |
| `-k N`, `--top N` | 10 | how many records to return |
| `--threshold BITS` | none | drop records below this match weight, however few are left |
| `--threads N` | 1 | splits both the dictionary walk and the row pass |
| `--expected-matches N` | — | the prior, as the number of records in the store you expect to be the person asked about |
| `--prior-weight BITS` | — | the same thing stated directly, in bits of prior odds |
| `--clusters <file>` | — | a file written by [`cluster`](cluster.md); labels each hit with its cluster |
| `--explain` | off | print the [waterfall](explain.md) behind every hit |
| `--json` | off | the whole report as one JSON object |
| `--tf-damping F` | 1.0 | scale the term-frequency adjustment |
| `--no-interactions` | off | score the plain conditionally-independent model |
| *(positional)* | — | required; the parquet file |

A column the query does not name is **missing**, not empty. So a query carrying a name and a
date of birth against a ten-column schema is scored as a record whose other eight fields were
never recorded, which is what it is, and those comparisons land on their null level.

A list column takes one element per repeat of its field: `--field alias=bill --field
alias=will`. A date is written `YYYY-MM-DD` and is refused in any other form.

## What it prints

```text
Records        1,000,000
Comparisons    3 tabulated over their dictionary, 1 evaluated per row, 5 constant because the query is missing the column
Values walked  174,418  (1 query value the store never held)
Rows scored    81 of 1,000,000 exactly; the rest were dropped on the bracket
Prior          -22.619 bits
Elapsed        0.0079 s walking the dictionaries, 0.0449 s over the rows on 1 thread

Record                          weight     posterior
r0                              38.684      1.000000
r259138                         -7.916      0.004124
```

The three counts on the `Comparisons` line are the three things a comparison can cost.

- **constant** is free. The query does not carry the column, so every record lands on the
  same level and nothing is evaluated at all. Most of a real query's schema is here, because
  a caller who knows ten fields about someone is not searching for them.
- **tabulated** is one walk over the column's dictionary, then one byte read per record. This
  is where every string metric in the query runs, and it runs once per *distinct value*
  rather than once per record.
- **evaluated** is the pair path's own evaluation, once per record. A comparison over a date,
  a number, a list or a coordinate pair is not a function of a single value id, so it gets no
  table.

`Rows scored` is how many records reached the exact weight. The rest were dropped on
`BaseWeight + Δ_max`, the same admissible bracket [`predict`](predict.md) uses, so they could
not have entered the answer whatever the term-frequency tables said.

## The prior is not the model's, and it is the one thing you have to choose

λ is the match rate over the **pair** space, which is the question deduplication asks. A query
against N records asks a different one, roughly "I expect this person to be in here about
once", and the two differ by orders of magnitude. By default `search` inherits the model's
prior, so a hit scores exactly what [`predict`](predict.md) would have given that pair.
`--expected-matches 1` states the search question instead:

```sh
cpplink search --schema s.json --model model.json data.parquet \
    --field last_name=zolnerowich --field postcode=4508 --expected-matches 1
```

The prior is a constant added to every hit, so it moves the posterior and where a sensible
threshold sits, and it never changes the order.

## Why the weight is worth more than a ranking

A search engine returns a list. This returns a list with a number attached, and the number is
calibrated, so "nobody in here is this person" is expressible rather than being an
empty-looking list of bad matches. That is what `--threshold` is for: zero bits is even odds,
so a hit below it is evidence *against*.

And a hit explains itself through the same ledger as any other pair:

```sh
cpplink search ... -k 1 --explain
```

```text
Comparison        Level                       bits        tf     running
------------------------------------------------------------------------
(prior)           lambda                    -22.62                -22.62
last_name         jaro_winkler >= 0.92      +15.81                 -6.80
first_name        exact                      +9.31     +3.35       +5.86
gender            null                       +0.36                 +6.22
dob               exact                     +14.07     +0.01      +20.30
email             null                       +2.79                +23.09
phone             null                       +2.77                +25.87
postcode          exact                     +12.90     -0.08      +38.68
location          null                       +0.00                +38.68
address           null                       +0.00                +38.68
------------------------------------------------------------------------
Match weight   38.684 bits    posterior 1.000000000

Term frequency, for the comparisons that moved the weight:
  first_name        zoudcut                          115 rows    1.15e-04     +3.35 bits
  dob               3504d                             48 rows    4.80e-05     +0.01 bits
  postcode          4508                             115 rows    1.15e-04     -0.08 bits
```

This is [`explain`](explain.md)'s waterfall, over the same pair the search scored, from the
same scorer. Nothing here recomputes a bit of the weight.

## Entities rather than rows

A deduplicated store holds several records of the same person, so the top ten records can be
three people. `--clusters <file>`, given the file [`cluster`](cluster.md) wrote, labels each
hit with its cluster and marks the ones that are another record of an entity already listed.

Grouping at the end rather than indexing one record per cluster is deliberate: indexing a
representative would shrink the scan by the duplicate rate and lose every cluster whose
members disagree on the columns the query happens to carry.

## What it costs

Per query on the `gen-sample` sample against the nine-comparison sample schema. `exact` names a
date and a postcode, `fuzzy` one name column, `full` seven columns including the email address.

| Records | Query | Walk | Gather | Total, 1 thread | Total, 8 threads | Values walked |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 250,000 | exact | 0.2 ms | 10.9 ms | 11.1 ms | 3.0 ms | 9,000 |
| 250,000 | fuzzy | 4.8 ms | 7.9 ms | 12.6 ms | 6.7 ms | 100,787 |
| 250,000 | full | 58.3 ms | 14.9 ms | 73.2 ms | 16.7 ms | 1,274,236 |
| 1,000,000 | exact | 0.4 ms | 44.2 ms | 44.7 ms | 11.3 ms | 9,000 |
| 1,000,000 | fuzzy | 7.5 ms | 28.3 ms | 35.8 ms | 16.0 ms | 149,192 |
| 1,000,000 | full | 215.6 ms | 58.3 ms | 273.9 ms | 60.5 ms | 4,796,686 |
| 4,000,000 | exact | 1.3 ms | 174.7 ms | 176.0 ms | 39.9 ms | 9,000 |
| 4,000,000 | fuzzy | 12.1 ms | 109.0 ms | 121.2 ms | 37.8 ms | 237,310 |
| 4,000,000 | full | 861.1 ms | 242.9 ms | 1,104 ms | 259.4 ms | 18,718,486 |
| 20,000,000 | exact | 13.0 ms | 848.9 ms | 862 ms | 195 ms | 9,000 |
| 20,000,000 | fuzzy | 42.4 ms | 555.8 ms | 598 ms | 166 ms | 657,528 |
| 20,000,000 | full | 4,194 ms | 1,308 ms | 5,502 ms | 1,194 ms | 91,645,846 |

The two phases scale with different things, which is the point of reporting them apart.

- The **row pass** grows with the records, at 27 to 65 ns each depending on how many
  comparisons the query engages, and that rate holds from 250k to 20M.
- The **dictionary walk** grows with the queried columns' *distinct values*, at a flat 46 ns
  each. On a near-unique column with a fuzzy level — an email address — that is the whole of
  the latency: at 20M the `full` query walks 91.6M values and spends 76% of itself there,
  while the same query without the email address is answered in 598 ms.

So the thing to watch is not the size of the store but whether the query names a near-unique
column that a fuzzy level reads. Both phases split 4–5× over eight threads.

The measurements are `bench/scale/search_latency.json`, produced by
`bench/scale/search_latency.py`, which is how to reproduce them on other data. The load is not
in these numbers: `search` is a scan over a resident store, and the sense in which that is a
per-query cost is that the process stays up.

## Limits

!!! warning "One input"
    `search` reads a single parquet file. The query is appended to the store as a row, and
    over more than one input that row would have to be a dataset of its own, which the levels
    reading which input a record came from were bound before it existed. Linking a file of
    queries against a store is the throughput question, and that is what
    [`--mode link`](../linking.md) already answers.

Two more, both narrow:

- The query record is held as a row on the store for the length of the query, so one search
  runs at a time against one store. Threads split the work *within* a query.
- A query value the store's dictionary never held is adopted for the length of the query, so
  it compares fuzzily against everything as it should. It gets no entry in the
  `list_contains` alias map, so a nickname the store never saw is not looked up as one.

## See also

- [`predict`](predict.md) — the same weight over candidate pairs rather than one query
- [`explain`](explain.md) — the waterfall `--explain` prints
- [`cluster`](cluster.md) — what `--clusters` reads
