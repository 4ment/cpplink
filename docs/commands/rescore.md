# `rescore`

**Goal:** re-score a spilled run under a new model, without comparing anything again.

The agreement pattern γ is everything the comparison produced — every string metric, every
threshold test, collapsed into 20 bits. Keep it, and changing `m`, `u`, λ or the
term-frequency damping becomes a linear read over a file instead of a second pass over the
data.

## Synopsis

```sh
cpplink rescore --schema <schema.json> --model <model.json> --spill <dir>
                --out <dir|file.csv|file.parquet>
                [--threshold BITS | --probability P]
                [--format bin|csv] [--threads N] [--limit N]
                [--mode MODE] <file.parquet>...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required; must be the schema the spill was written with |
| `--model <file>` | — | required; the **new** model to score under |
| `--spill <dir>` | — | required; a directory written by [`predict --spill`](predict.md) |
| `--out <path>` | — | required. As in [`predict`](predict.md#one-file-or-one-shard-per-thread): a name ending `.csv`, `.parquet` or `.pq` is one merged file, anything else a directory of shards |
| `--threshold BITS` | — | keep predictions with match weight ≥ this many bits |
| `--probability P` | — | same, expressed as a posterior in (0, 1) |
| `--format bin\|csv` | `bin` | as in [`predict`](predict.md) |
| `--threads N` | hardware | capped at the number of spill shards |
| `--limit N` | unlimited | stop after N predictions |
| `--tf-damping F` | 1.0 | scale the term-frequency adjustment |
| `--no-bounds` | off | score every pair exactly instead of using the bracket. **Verification only** |
| *(positional)* | — | required; the parquet file |
| `--mode dedup\|link\|link-and-dedup` | link with more than one file, else dedup | which pairs to enumerate; see [linking](../linking.md) |

The parquet file is still needed — the term-frequency adjustment reads the value each row
carries, and the output names records by their `unique_id`. What is *not* done again is
blocking, pair enumeration, and every string metric.

## Why it is worth having

On the 1M-row sample, 44.9M candidates at threshold 0:

| Run | Scoring pass | Wall | CPU |
| --- | ---: | ---: | ---: |
| `predict`, no spill | 2.2 s | 3.93 s | 15.8 s |
| `predict --spill` | 2.2 s | 3.77 s | 15.8 s |
| `rescore` | **0.01 s** | **1.58 s** | **1.6 s** |

Writing the spill costs no measurable CPU. Replaying it is **200× faster** than the scoring
pass that produced it, and all but 0.01 s of the 1.58 s wall is reading the parquet.

The ratio was larger before the pair-global ceiling landed: the scoring pass it is being
compared against used to take 14 s rather than 2.2 s, so `rescore` looked 700× faster. What
changed is the numerator, not `rescore` — replay was never doing the work, and now the pass it
replaces is not doing most of it either.

## What a spill can and cannot tell you

!!! warning "Raising the threshold is exact. Lowering it is not."
    A spill holds the pairs **one model kept**. Re-scoring at or above the threshold it was
    written at is exact, because everything that could qualify is already in the file. Below
    that threshold, pairs the original run discarded are simply absent, and no amount of
    re-scoring brings them back. `rescore` detects this and says so in its report.

The fix, when you need to explore downward, is to spill a uniform sample of *every*
candidate alongside the above-threshold pairs:

```sh
cpplink predict --schema examples/sample_schema.json --model model.json \
                --out predictions/ --spill spill/ --spill-sample 0.01 \
                --threshold 20 examples/sample.parquet
```

That keeps 1% of the pairs the threshold discarded — 6.1 MB rather than 948 KB on the 1M
sample — which is enough to see what a lower threshold would have found without storing all
44.9M candidates.

## What is checked before anything is read

`spill.json` records the threshold, the sample rate, the candidate count and the **γ
layout** — the comparison names and their level counts. A packed γ means nothing without the
layout that produced it, so `rescore` refuses a spill written with a different set of
comparisons, or over a different number of records, rather than silently scoring nonsense.
A shard truncated mid-pair is refused too.

## Example

```sh
cpplink predict --schema examples/sample_schema.json --model model.json \
                --out predictions/ --spill spill/ --threshold 0 examples/sample.parquet

cpplink rescore --schema examples/sample_schema.json --model tuned.json \
                --spill spill/ --out predictions2/ --threshold 20 examples/sample.parquet
```

```text
Spill          93,816 pairs from 44,925,919 candidates, written at 0.000 bits
Threshold      20.000 bits
Threads        8
Pairs read     93,816
Predictions    93,816
Elapsed        0.02 s  (5836052 pairs/s)

Re-scoring reads only the pairs the spilling run retained. Every pair here is
scored exactly, but a pair that run discarded cannot come back.
```

## See also

- [`predict`](predict.md) — writes the spill
- [`cluster`](cluster.md) — the other place a threshold can be swept without re-scoring
