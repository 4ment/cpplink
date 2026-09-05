# `predict`

**Goal:** score every candidate pair the blocking plan produces and write the ones above a
threshold, as one shard file per thread.

This is where the pipeline spends its time. Nothing holds a row per candidate pair: a pair is
enumerated, compared, scored, then written or dropped, and forgotten.

## Synopsis

```sh
cpplink predict --schema <schema.json> --model <model.json> --out <dir>
                [--threshold BITS | --probability P] [--format bin|csv]
                [--threads N] [--limit N] [--no-bounds] [--tf-damping F]
                [--no-signatures] [--spill <dir>] [--spill-sample R]
                [--mode MODE] <file.parquet>...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required; must declare `comparisons` and `blocking` |
| `--model <file>` | — | required; the `model.json` from [`estimate`](estimate.md) |
| `--out <dir>` | — | required; directory for the edge shards |
| `--threshold BITS` | — | keep edges with match weight ≥ this many bits |
| `--probability P` | — | same, expressed as a posterior in (0, 1). Converted to bits internally |
| `--format bin\|csv` | `bin` | `bin` is what [`cluster`](cluster.md) reads back; `csv` is for reading with your eyes |
| `--threads N` | hardware | worker threads; one output shard each |
| `--limit N` | unlimited | stop after N edges. For sampling the output, not for production |
| `--tf-damping F` | 1.0 | scale the term-frequency adjustment. 0 disables it |
| `--no-bounds` | off | score every pair exactly instead of using the admissible bracket. **Verification only** |
| `--no-signatures` | off | disable the per-value character-mask filter. **Verification only** |
| `--spill <dir>` | none | also write `(a, b, γ)` at 12 bytes a pair, so the run can be re-scored by [`rescore`](rescore.md) without comparing again |
| `--spill-sample R` | 0 | additionally spill a uniform fraction R of *every* candidate, not just those above the threshold |
| *(positional)* | — | required; the parquet file |
| `--mode dedup\|link\|link-and-dedup` | link with more than one file, else dedup | which pairs to enumerate; see [linking](../linking.md) |

Exactly one of `--threshold` or `--probability` is required. Prefer bits — see
[choosing a threshold](../model.md#choosing-a-threshold).

## Example

```sh
cpplink predict --schema examples/sample_schema.json --model model.json \
                --out edges/ --threshold 20 examples/sample.parquet
```

```text
Threshold      20.000 bits  (posterior 0.999999)
Threads        8
Candidates     116,936,544
Edges          142,541  (0.12% of candidates)
Elapsed        36.0 s  (3246923 candidates/s)

Zone           Candidate pairs       Share        Patterns
----------------------------------------------------------
drop               116,793,569      99.88%          60,973
check                      434       0.00%          14,358
emit                   142,541       0.12%          44,669
----------------------------------------------------------
Term-frequency lookups 142,975, avoided 116,793,569 (99.88%).
The bracket is admissible, so dropping on it emits exactly the edges
scoring every pair would have emitted.

Shards
  edges/shard-000.bin
  edges/shard-001.bin
  ...
  edges/shard-007.bin
```

## Reading the output

### The header

| Field | Meaning |
| --- | --- |
| `Threshold` | in bits, with the posterior it corresponds to. A threshold of 20 bits is a posterior of 0.999999 — which is why bits, not probability, is the usable knob |
| `Threads` | workers, and therefore shard count |
| `Candidates` | the deduplicated union of every source. Should match `explain-blocking --count` exactly |
| `Edges` | pairs written, and their share of candidates |
| `Elapsed` | wall time and candidate throughput |

`Candidates` matching [`explain-blocking`](explain-blocking.md)'s union is a useful check that
the plan you priced is the plan that ran.

### The zone table

This is the [admissible bracket](../model.md#the-admissible-bracket) doing its work. Every
distinct pattern is classified once at bind time from its base weight and the extreme
term-frequency adjustments its comparisons can produce:

| Zone | Meaning |
| --- | --- |
| `drop` | below threshold **even at its rarest possible values** → skipped without any TF lookup |
| `check` | the bracket straddles the threshold → the exact TF-adjusted weight is computed |
| `emit` | above threshold **even at its commonest values** → emitted without any TF lookup |

Two columns, and they mean different things:

- **`Candidate pairs`** — how many actual pairs landed in each zone. 99.88% dropped.
- **`Patterns`** — how many *distinct γ values* fall in each zone, out of the reachable
  pattern space. These do not track each other: `check` covers 14,358 patterns but only 434
  real pairs, because bracket-straddling patterns are rare in the data even though there are
  many of them in principle.

!!! note "The bracket is exact, not a heuristic"
    Both bounds are admissible, so the emitted edge set is **identical** to scoring every pair
    exactly. `--no-bounds` runs it the slow way, and `tests/predict_test.cpp` compares the two
    edge sets at five thresholds so the property cannot rot silently.

!!! info "Only `drop` actually saves anything"
    The three-way split is really two: `check` and `emit` both compute the exact weight,
    because the weight is part of the output. `emit` saves the threshold comparison and
    nothing else. Measured, bounded and exhaustive runs differ by ~0.5% of wall time — the
    bracket removes the TF lookup, and the TF lookup was never the expensive part. **The
    comparison is**, at roughly 3.5 µs of CPU per pair.

### The shard list

One file per thread, `shard-000.bin` upward. Binary shards are 20 bytes a row —
`a`, `b`, `gamma` as `u32`, then the weight as `f64` — behind an 8-byte magic. The format
lives in `predict.hpp` as `kEdgeMagic`/`kEdgeBytes` so the writer and the reader in
`cluster.cpp` cannot drift apart. See
[Output formats](../reference/output-formats.md#edge-shards).

With `--format csv` you get `shard-NNN.csv` instead:

```csv
id_a,id_b,gamma,match_weight,match_probability
r0,r20,174729,106.288416,1.000000000
r1,r119514,174665,108.510385,1.000000000
```

Note the saturated probability: 106 bits of evidence is a posterior of 1 to more decimal places
than a double carries. `cluster` reads only the binary form.

## Performance notes

- **Throughput is the binding constraint of the whole pipeline.** 3.2M candidates/s on eight
  threads here. Blocking enumeration runs at ~17 ns/pair single-threaded, so candidate
  generation is about 240× cheaper than the comparison it feeds — optimizing blocking is
  effort wasted.
- **Where the time goes:** string metrics, on pairs that end at `else`. That is *every*
  non-matching pair, and non-matches outnumber matches by orders of magnitude in any candidate
  set. The fuzzy path is not the exception; it is the common case.
- **The signature filter recovers 24% of comparison CPU** — a per-value length and 64-bit
  character-presence mask that rejects a pair with two loads and two popcounts. `--no-signatures`
  turns it off, which is only useful for confirming that it changes no γ.
- **Threads scale cleanly.** Work is handed out from an atomic counter over blocking groups,
  oversized groups are split into row ranges, per-thread counters are `alignas(64)`, and the
  record store is `const` and shared.

## Choosing the threshold

Because edges carry their weight, **the threshold can be raised later without re-scoring** —
`cluster --threshold` re-reads the shards in seconds. So write at a low threshold and sweep in
`cluster` rather than re-running `predict`. Sweeping on this data shows nothing above 40 bits
is worth having, and everything from 0 to 40 bits is the same answer.
