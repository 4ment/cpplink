# cpplink

**cpplink** is a C++17 command line tool for probabilistic record linkage. It implements the
Fellegi–Sunter model — EM parameter estimation, term-frequency adjustment and prediction —
as a *streaming* pipeline that never materializes a table of candidate pairs.

It exists because tools like [splink](https://moj-analytical-services.github.io/splink/)
build the blocked-pair and comparison-vector tables as real relations. At 18 million records
a plausible blocking scheme yields ~5×10⁹ candidate pairs, and the comparison-vector table
alone is about 130 GB before a single score is written. cpplink holds the same model in
roughly 3 GB, because it stores *records* and *counts* and nothing in between.

## The load-bearing idea

A pair enters the Fellegi–Sunter likelihood only through its **agreement pattern** γ — the
vector of comparison-level indices. Two pairs with the same γ are indistinguishable to the
model, so estimation does not need pairs; it needs `count[γ]`, a histogram over *distinct
patterns*.

|                        | pair table | pattern histogram |
| ---------------------- | ---------: | ----------------: |
| Candidate pairs        |      5×10⁹ |             5×10⁹ |
| Stored rows            |      5×10⁹ |             ~10⁵  |
| Resident memory        |    ~130 GB | 3.1 GB *(measured)* |
| Cost of one EM re-fit  | re-read everything | < 0.1 s   |

Pairs are produced by a lazy blocking iterator, folded into a per-thread histogram, and
discarded. EM reads the histogram, so **its cost is independent of the data size**.

## The pipeline

```
0 Profile  what the columns can be worth, what a     cpplink profile
           match will score, and what is the same
           evidence twice; and where the fuzzy       cpplink levels
           thresholds should actually sit
1 Load     parquet → interned columnar store        cpplink inspect
2 Block    lazy pair iterator, no table             cpplink explain-blocking / recall
3 Compare  pair → packed γ                          cpplink explain
4 Estimate u in closed form, m and λ by EM          cpplink estimate
5 Score    TF-adjusted, bound-pruned                cpplink predict
6 Cluster  union–find over the edge stream          cpplink cluster
```

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
- [Estimation and EM](em.md) — closed-form `u`, EM over the histogram, per-column sessions.
- [Blocking](blocking.md) — the four pair sources, exact costing, and measured recall.
- [Commands](commands/index.md) — what each subcommand is for and how to read its output.

## Status

Phases 0–5 of the design are complete:
the record store, parquet loader, comparison levels, blocking sources, the recall harness,
parameter estimation, scoring and clustering are all in and measured, so the pipeline runs
end to end from parquet to duplicate clusters. Phase 6 (spill and fast re-scoring, waterfall
explain, the ANN source, then the link path) is in progress; the per-value signature filter
and Myers' bit-vector edit distance from that phase have shipped.

Every measurement quoted in these pages was produced by the commands documented here.
Unless stated otherwise, they come from the 1.8M-row synthetic sample written by
[`gen-sample`](commands/gen-sample.md), on eight threads.
