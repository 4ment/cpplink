# `rescore`

**Goal:** re-score a spilled run under a new model, without comparing anything again.

The agreement pattern γ is everything the comparison produced — every string metric, every
threshold test, collapsed into 20 bits. Keep it, and changing `m`, `u`, λ or the
term-frequency damping becomes a linear read over a file instead of a second pass over the
data.

## Synopsis

```sh
cpplink rescore --schema <schema.json> --model <model.json> --spill <dir>
                --out <dir> [--threshold BITS | --probability P]
                [--format bin|csv] [--threads N] [--limit N]
                [--mode MODE] <file.parquet>...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required; must be the schema the spill was written with |
| `--model <file>` | — | required; the **new** model to score under |
| `--spill <dir>` | — | required; a directory written by [`predict --spill`](predict.md) |
| `--out <dir>` | — | required; directory for the edge shards |
| `--threshold BITS` | — | keep edges with match weight ≥ this many bits |
| `--probability P` | — | same, expressed as a posterior in (0, 1) |
| `--format bin\|csv` | `bin` | as in [`predict`](predict.md) |
| `--threads N` | hardware | capped at the number of spill shards |
| `--limit N` | unlimited | stop after N edges |
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
| `predict`, no spill | 14.0 s | 16.7 s | 95.7 s |
| `predict --spill` | 14.1 s | 16.1 s | 96.3 s |
| `rescore` | **0.02 s** | **1.7 s** | **1.6 s** |

Writing the spill costs under 1% of CPU. Replaying it is 700× faster than the pass that
produced it, and all but 0.02 s of the 1.7 s is reading the parquet.

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
                --out edges/ --spill spill/ --spill-sample 0.01 \
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
                --out edges/ --spill spill/ --threshold 0 examples/sample.parquet

cpplink rescore --schema examples/sample_schema.json --model tuned.json \
                --spill spill/ --out edges2/ --threshold 20 examples/sample.parquet
```

```text
Spill          79,413 pairs from 44,908,142 candidates, written at 0.000 bits
Threshold      20.000 bits
Threads        8
Pairs read     79,413
Edges          79,195
Elapsed        0.02 s  (4193906 pairs/s)

Re-scoring reads only the pairs the spilling run retained. Every pair here is
scored exactly, but a pair that run discarded cannot come back.
```

## See also

- [`predict`](predict.md) — writes the spill
- [`cluster`](cluster.md) — the other place a threshold can be swept without re-scoring
