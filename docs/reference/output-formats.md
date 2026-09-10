# Output formats

Everything cpplink writes to disk, and how to read it back.

## `model.json`

Written by [`estimate --out`](../commands/estimate.md), read by
[`predict --model`](../commands/predict.md). Small, human-readable and hand-editable.

```json
{
  "records": 1800000,
  "lambda": 7.728471673711284e-08,
  "lambda_basis": "lower bound: session last_name implies 125,201 matches over 1,619,999,100,000 pairs",
  "comparisons": [
    {
      "name": "last_name",
      "columns": ["last_name"],
      "term_frequency": true,
      "sessions": 3,
      "levels": [
        {
          "label": "exact",
          "m": 0.6095589501884875,
          "m_estimated": true,
          "m_support": 129023.0,
          "u": 2.3818564197530862e-05,
          "u_exact": true,
          "u_observed": 22
        }
      ]
    }
  ]
}
```

### Top level

| Field | Meaning |
| --- | --- |
| `records` | rows the model was estimated from |
| `lambda` | prior share of pairs that are matches |
| `lambda_basis` | how λ was arrived at — a session's implied match count, or `"given on the command line"` |
| `comparisons` | one entry per comparison, in schema order |

### Per comparison

| Field | Meaning |
| --- | --- |
| `name`, `columns`, `term_frequency` | echoed from the schema |
| `sessions` | how many EM sessions contributed to this comparison's `m`. **0 means its `m` is a starting value, not an estimate** |
| `levels` | one entry per level, in schema order |

### Per level

| Field | Meaning |
| --- | --- |
| `label` | the level's description, matching the printed report |
| `m` | \(P(\text{level} \mid \text{match})\) |
| `m_estimated` | `false` means `m` fell back to a starting value |
| `m_support` | the EM responsibility mass behind `m`. Small values mean a thinly supported level |
| `u` | \(P(\text{level} \mid \text{non-match})\) |
| `u_exact` | `true` when `u` came from the closed form rather than sampling. **Trust these** |
| `u_observed` | how many sampled random pairs landed on this level. `0` with `u_exact: false` means `u` is at the sampling floor |

The level order in the file **must** match the schema's — levels are positional, so editing
the schema's levels invalidates an existing model. The file is a fine thing to edit by hand:
overriding a suspect `u` needs no re-run.

## Prediction shards

Written by [`predict --out <dir>`](../commands/predict.md), one file per thread.

### Binary — `shard-NNN.bin` *(default)*

What [`cluster`](../commands/cluster.md) reads back.

```text
offset 0    8 bytes    magic "CPPLNKE1"
then        20 bytes   per prediction, packed, little-endian:
              u32  a         row index of the first record
              u32  b         row index of the second record
              u32  gamma     the packed agreement pattern
              f64  weight    TF-adjusted match weight, in bits
```

The constants live in
[`predict.hpp`](https://github.com/4ment/cpplink/blob/main/src/cpplink/predict.hpp) as
`kEdgeMagic` and `kEdgeBytes`, shared with the reader in `cluster.cpp` so the writer and the
reader cannot drift apart.

Two things follow from the format:

- **Rows are row indices, not `unique_id`s.** They are only meaningful against the same
  parquet file, loaded through the same schema — which is why `cluster` takes the data file as
  well as the shard directory.
- **The weight is stored**, which is why `cluster --threshold` can raise the threshold with a
  re-read and no re-scoring.

### CSV — `shard-NNN.csv`

For reading with your eyes. `cluster` does not read this form.

```csv
id_a,id_b,gamma,match_weight,match_probability
r0,r20,174729,106.288416,1.000000000
r1,r119514,174665,108.510385,1.000000000
```

| Column | Meaning |
| --- | --- |
| `id_a`, `id_b` | the records' `unique_id` values |
| `gamma` | the packed pattern, decimal. Feed it back through [`explain`](../commands/explain.md)'s layout table to decode |
| `match_weight` | bits of evidence, TF-adjusted |
| `match_probability` | \(1/(1+2^{-W})\). Saturates at 1.0 above ~50 bits, which is why the threshold is in bits |

## The merged prediction file

Written by [`predict --out <file>`](../commands/predict.md#one-file-or-one-shard-per-thread),
by `rescore --out <file>`, or by [`merge-predictions`](../commands/merge-predictions.md) from a shard
directory that already exists. It holds the same five columns whichever format it is written
in, and it is what [`cluster --predictions`](../commands/cluster.md#shards-or-one-file) reads when
it is not given a shard directory.

```csv
id_a,id_b,gamma,match_weight,match_probability
r0,r20,174729,106.288416,1.000000000
r1,r119514,174665,108.510385,1.000000000
```

As parquet, the same columns typed: `id_a` and `id_b` as `string`, `gamma` as `uint32`,
`match_weight` and `match_probability` as `double`, Snappy-compressed, 65,536 rows a row
group. The csv rounds the weight to six decimals as the shards do; the parquet keeps the
`f64` the scorer produced.

Ids are the records' `unique_id` values when the merge was given `--schema` and the parquet
file, or **row indices** when it was not, because binary shards carry no ids. Csv shards carry
their own and need neither. `predict` and `rescore` always have the store in hand, so a file
they write themselves always carries ids.

Because a merged file names records by `unique_id` rather than by row, clustering one costs an
index over the store's id column: a `uint32` per record in the order of the id it names, so a
lookup is a binary search. That is the price of a file that means something outside the run
that wrote it. Ids that resolve to no loaded record are counted and skipped, and the report
says so.

## `clusters.csv`

Written by [`cluster --out`](../commands/cluster.md).

```csv
unique_id,cluster_id,cluster_size
r0,r0,2
r1,r1,3
r3,r3,2
```

| Column | Meaning |
| --- | --- |
| `unique_id` | the record |
| `cluster_id` | the **representative record's own `unique_id`** — the record the others collapse onto |
| `cluster_size` | how many records are in that cluster |

Only records in a cluster of at least `--min-size` (2 by default) are written, so singletons do
not appear. Two records are duplicates exactly when their `cluster_id` values are equal.

## Truth files

Read by [`recall --truth`](../commands/recall.md) and
[`cluster --truth`](../commands/cluster.md); written by
[`gen-sample --truth`](../commands/gen-sample.md).

```csv
id_a,id_b
r5,r9
r17,r19
```

Two `unique_id` values per line. A header line beginning `id_a` is optional and skipped. Ids
that do not resolve to a loaded record are counted and reported as `unresolved` rather than
failing the run.

The pairs are taken as given, not closed transitively: if a truth file lists a–b and b–c, that
is two known pairs, and a–c is not one unless it is also listed. `MeasureClusters` compares
those pairs against the **transitive closure of the partition**, which is why cluster precision
is stricter than the pair-level precision `predict` reports.
