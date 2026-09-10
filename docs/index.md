# cpplink

**cpplink** is a C++17 command line tool for probabilistic record linkage. It implements the
Fellegi–Sunter model — EM parameter estimation, term-frequency adjustment and prediction —
as a *streaming* pipeline that never materializes a table of candidate pairs.

It is heavily inspired by [splink](https://moj-analytical-services.github.io/splink/) and
reuses many of the ideas and design choices that make splink such a useful record linkage
tool. The motivation was practical: to run probabilistic record linkage on data that does not
fit comfortably in memory on a laptop. Splink relies on a SQL engine to build and process
intermediate candidate-pair tables, and those grow with the data. Rather than materializing
the pairs, cpplink generates candidates as a stream and reduces each one to its
Fellegi–Sunter agreement pattern immediately, storing *records* and *counts* and nothing in
between.

## The load-bearing idea

A pair enters the Fellegi–Sunter likelihood only through its **agreement pattern** γ — the
vector of comparison-level indices. Two pairs with the same γ are indistinguishable to the
model, so estimation does not need pairs; it needs `count[γ]`, a histogram over *distinct
patterns*.

On the 20M-record deduplication this project is built for:

|                        | pair table | pattern histogram |
| ---------------------- | ---------: | ----------------: |
| Candidate pairs        |   1.01×10¹⁰ |          1.01×10¹⁰ |
| Stored rows            |   1.01×10¹⁰ |             ~10⁵  |
| Resident memory        | 121 GB *(at 12 B a pair)* | 4.49 GB *(measured)* |
| Cost of one EM re-fit  | re-read everything | < 0.1 s   |

Pairs are produced by a lazy blocking iterator, folded into a per-thread histogram, and
discarded. EM reads the histogram, so **its cost is independent of the data size**.

On one thread that run takes **38.1 minutes**, and the two figures it belongs to (wall time
against candidate pairs, peak memory against records) are in the
[README](https://github.com/4ment/cpplink#scaling).

Run against splink on the same data, the same schema and the same machine,
[`bench/scale`](https://github.com/4ment/cpplink/tree/main/bench/scale) deduplicates 4M
records over 1.44bn candidate pairs in **89.2 s at 1.16 GB resident, with no scratch file**.
At 1M records, where both tools finish, cpplink is 4.4× faster end to end than splink's best
configuration here and uses 8.9× less memory, for the same quality: F1 0.9974 against
0.9961. Splink did not finish at 2M on that
machine, and what it ran out of was scratch space rather than memory.

## The pipeline

```
0 Profile  what the columns can be worth, what a     cpplink profile
           match will score, and what is the same
           evidence twice; where the fuzzy           cpplink levels
           thresholds should sit; and which levels   cpplink simplify
           a run cannot tell apart
1 Load     parquet → interned columnar store         cpplink inspect
2 Block    lazy pair iterator, no table              cpplink explain-blocking / recall
                                                     cpplink completeness
3 Compare  pair → packed γ                           cpplink explain
4 Estimate u in closed form, m and λ by EM           cpplink estimate
5 Score    TF-adjusted, bound-pruned                 cpplink predict
                                                     cpplink rescore
6 Cluster  union–find over the edge stream           cpplink cluster
```

Every data command takes one or more parquet files. One file deduplicates, two link, and
`--mode` says which when it is not obvious. See [Linking two files](linking.md).

Four invariants hold the design together:

- **Nothing in the pipeline holds a row per pair.** Anything that does defeats the point.
- **The record store is immutable after load** and shared `const` across threads. Per-thread
  histograms and output buffers merge at join; there is no locking in the hot path.
- **Values are interned to dense `uint32` ids per column**, which turns an exact-match
  comparison level into an integer equality rather than a string compare.
- **A pair source may only feed EM if its selection event factors as a condition on an
  excludable column subset.** Per-column sources qualify; whole-record ones (embeddings,
  concatenated-record signatures) do not. See [Estimation and EM](em.md#em-safety).

## Where to go next

- [Getting started](getting-started.md) — build it and run the whole pipeline on a sample file.
- [The model](model.md) — γ, match weights, term-frequency adjustment, the admissible bracket.
- [Comparisons](comparisons.md): the level ladder, what each level reads, and what it costs.
- [Estimation and EM](em.md) — closed-form `u`, EM over the histogram, per-column sessions.
- [Blocking](blocking.md) — the four pair sources, exact costing, and measured recall.
- [Linking two files](linking.md): what changes when the pair space is a cross-product.
- [Commands](commands/index.md) — what each subcommand is for and how to read its output.
- [The schema file](reference/schema.md): every field of the JSON that configures a run.

## Status

**The pipeline runs end to end, in both deduplication and link mode.** The record store,
parquet loader, schema, comparison levels, γ packing, blocking sources, the recall harness,
parameter estimation, scoring, clustering, the signature filter, spill and re-scoring, the
score waterfall and the miss diagnostic are all in and measured.

So is everything built on top of them: the pair-global ceiling,
[`completeness`](commands/completeness.md), term frequency for the fuzzy levels,
[`profile`](commands/profile.md), [`levels`](commands/levels.md), derived columns,
[`simplify`](commands/simplify.md), and
[`estimate --interactions`](commands/estimate.md#relaxing-conditional-independence), which
relaxes conditional independence inside the scoring model.

Two things are worth stating plainly.

The **approximate-nearest-neighbour source was retired by measurement rather than built**.
[`recall --why`](commands/recall.md) classifies every missed pair, and on this data no missed
pair lacks a column-wise signal, so an ANN source would have nothing left to find.

The **20M-record target has been run end to end**, and that is the size this design exists
for: [`estimate`](commands/estimate.md) → [`predict`](commands/predict.md) →
[`cluster`](commands/cluster.md) over 20,000,000 records and 1.01×10¹⁰ candidate pairs, in
**38.1 minutes on one thread at 4.49 GB resident**, with no scratch file and F1 0.9946
against the planted duplicates.
What that run does *not* establish is the two things it was never going to: the data is
generated rather than real, and the
[splink comparison](https://github.com/4ment/cpplink/tree/main/bench/scale) is still at 1M
records, where both tools finish.

Every measurement quoted in these pages was produced by the commands documented here.
Unless stated otherwise, they come from one of four sources: the synthetic samples written by
[`gen-sample`](commands/gen-sample.md), the three public deduplication datasets in
[`bench/`](https://github.com/4ment/cpplink/tree/main/bench) (`fake_1000`, `febrl3` and
`historical_50k`, the last being the one with real messy variation), the scale runs above, or
a 20M-row store where the page says so. Each says how many threads it ran on; the scale sweep
is single-threaded and the splink comparison gives both tools all eight cores.
