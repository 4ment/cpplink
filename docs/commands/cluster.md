# `cluster`

**Goal:** join the scored edges into duplicate clusters with a union–find, write the partition,
and — given ground truth — score it.

For deduplication, the connected components at a threshold τ *are* the answer.

## Synopsis

```sh
cpplink cluster --schema <schema.json> --edges <dir> [--out <file.csv>]
                [--threshold BITS | --probability P] [--truth <file.csv>]
                [--min-size N] <file.parquet>...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required. Only `unique_id` is used |
| `--edges <dir>` | — | required; the directory of `shard-*.bin` from [`predict`](predict.md) |
| `--out <file.csv>` | none | write the partition. Without it the report is printed and nothing is saved |
| `--threshold BITS` | −∞ | keep only edges at or above this weight. **Re-filters without re-scoring** |
| `--probability P` | — | the same threshold as a posterior in (0, 1) |
| `--truth <file.csv>` | none | known duplicate pairs; adds a quality section |
| `--min-size N` | 2 | smallest cluster written to the output file |
| *(positional)* | — | required; the parquet file |

!!! tip "Clustering is where you sweep the threshold"
    The shards carry each edge's weight, so clustering above the threshold `predict` wrote at
    costs a re-read and **no re-scoring**. Write edges at a low threshold once, then sweep here
    in seconds.

`cluster` loads only the id column — it clears `schema.columns` before handing the schema to
the loader, so no dictionary is built and no comparison column is interned.

## Example

```sh
cpplink cluster --schema examples/sample_schema.json --edges edges/ \
                --out clusters.csv --truth examples/sample.truth.csv \
                examples/sample.parquet
```

```text
Read 142,541 edges from 8 shards in 0.0161131 s
Weights 29.7894 to 115.974 bits
132,440 of them merged two components (92.91%)

122,778 clusters of two or more, covering 255,218 records (14.18%)
1,544,782 records stayed alone
Largest cluster 6 records; the partition asserts 142,901 duplicate pairs
255,218 rows written

Size            Clusters          Records
-------------------------------------------
1 (singleton)   1,544,782        1,544,782
2                 113,858          227,716
3                   8,232           24,696
4                     637            2,548
5                      48              240
6-10                    3               18
-------------------------------------------

Against 143,730 known duplicate pairs, over the transitive closure:
  recovered  131,628
  asserted   142,901
  precision  0.9211
  recall     0.9158
  f1          0.9184

Wrote clusters.csv
```

## Reading the output

### The edge pass

```text
Read 142,541 edges from 8 shards in 0.0161131 s
Weights 29.7894 to 115.974 bits
132,440 of them merged two components (92.91%)
```

| Line | Meaning |
| --- | --- |
| `Read ... edges` | edges above the clustering threshold, and the shards they came from |
| `Weights` | the range of weights actually used. The minimum tells you whether `--threshold` bit |
| `... merged two components` | edges that changed the partition. The rest were redundant — both endpoints were already connected |

**The merge rate is a structural read on the edge set.** 92.91% means the edges are nearly a
forest: the duplicate groups are small and there is little redundancy for union–find to absorb.
A low merge rate means many edges inside already-connected groups, which is what a dense
cluster looks like — and also what a runaway chain looks like just before it swallows the file.

### The partition

| Line | Meaning |
| --- | --- |
| `N clusters of two or more, covering R records` | the answer, and its coverage of the file |
| `records stayed alone` | singletons — records with no above-threshold edge |
| `Largest cluster` | **watch this.** Union–find is transitive, so one bad edge between two correct clusters merges both entirely. A largest cluster in the thousands means a chain ran away |
| `asserts N duplicate pairs` | \(\sum \binom{\text{size}}{2}\) over clusters — the pairs the partition *claims*, which is more than the edges scored |
| `rows written` | rows in the output file, subject to `--min-size` |

Here 14.18% of records land in a cluster of two or more against a planted duplicate rate of 8%
of records, and the largest cluster is 6 — nothing chained.

### The size histogram

Bucketed cluster sizes with the record counts they account for. Its job is to make a runaway
component obvious: a healthy dedup run is dominated by clusters of 2 and 3, and any bucket at
the far right holding records is worth investigating with [`explain`](explain.md) on a pair
inside it.

### The quality section

Given `--truth`, pairwise precision and recall of the partition — **over the transitive
closure, not the edges**:

| Field | Meaning |
| --- | --- |
| `recovered` | known duplicate pairs whose rows share a cluster |
| `asserted` | pairs the partition claims, i.e. \(\sum \binom{\text{size}}{2}\) |
| `precision` | `recovered / asserted` |
| `recall` | `recovered / truth pairs` |
| `f1` | their harmonic mean |

!!! warning "Cluster precision is stricter than edge precision, and it is the one that matters"
    A chain a–b–c asserts a–c whether or not that pair was ever scored. On a 1M-row sample,
    edge precision at threshold 0 was 0.9203 while cluster precision was 0.9166 — the partition
    asserted 331 pairs that were never scored, of which 18 were real.

    **That gap is the number to watch when blocking widens.** One wrong edge between two
    correct clusters turns into every cross pair between them, and it is completely invisible
    in the edge-level numbers `predict` reports.

## Sweeping the threshold

Because re-clustering costs a re-read, sweeping is cheap. On a 1M-row sample over a fixed edge
set:

| Threshold (bits) | Edges kept | Clusters | Asserted pairs | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 79,413 | 68,208 | 79,744 | 0.9166 | 0.9161 | 0.9164 |
| 20 | 79,195 | 68,123 | 79,392 | 0.9206 | 0.9161 | **0.9184** |
| 40 | 79,181 | 68,118 | 79,385 | 0.9206 | 0.9161 | 0.9183 |
| 60 | 77,995 | 67,424 | 78,474 | 0.9215 | 0.9064 | 0.9139 |
| 80 | 62,994 | 56,428 | 63,958 | 0.9352 | 0.7497 | 0.8323 |

Between 0 and 40 bits precision moves 0.4 points and recall does not move at all; above 60,
recall collapses while precision gains 1.9. **There is no threshold at which this pipeline is
materially better than it is at 20 bits**, and the ceiling is the 0.916 blocking recall
measured by [`recall`](recall.md).

## Output file

```csv
unique_id,cluster_id,cluster_size
r0,r0,2
r1,r1,3
r3,r3,2
```

The `cluster_id` is the representative record's own `unique_id`, so the output says which
record the others collapse onto. Only records in a cluster of at least `--min-size` are
written; singletons are excluded by default.

## Cost

The union–find is `uint32` parent plus `uint8` rank — **5 bytes a record**, 90 MB at 18M rows —
and the shards are streamed past it with **no edge retained**. The pass over 142k edges and
1.8M records took 0.016 s against 36 s to produce those edges, which is why the lock-free
CAS version the design mentions stays unbuilt.
