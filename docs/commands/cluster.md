# `cluster`

**Goal:** join the predictions into duplicate clusters with a union–find, write the partition,
and — given ground truth — score it.

For deduplication, the connected components at a threshold τ *are* the answer.

## Synopsis

```sh
cpplink cluster --schema <schema.json>
                --predictions <dir|file.csv|file.parquet>
                [--out <file.csv>]
                [--threshold BITS | --probability P] [--truth <file.csv>]
                [--min-size N] <file.parquet>...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required. Only `unique_id` is used |
| `--predictions <path>` | — | required; either the directory of `shard-*.bin` from [`predict`](predict.md) or the single `.csv` or `.parquet` file it merged them into |
| `--out <file.csv>` | none | write the partition. Without it the report is printed and nothing is saved |
| `--threshold BITS` | −∞ | keep only predictions at or above this weight. **Re-filters without re-scoring** |
| `--probability P` | — | the same threshold as a posterior in (0, 1) |
| `--truth <file.csv>` | none | known duplicate pairs; adds a quality section |
| `--min-size N` | 2 | smallest cluster written to the output file |
| *(positional)* | — | required; the parquet file |

!!! tip "Clustering is where you sweep the threshold"
    Every prediction carries its weight, so clustering above the threshold `predict` wrote at
    costs a re-read and **no re-scoring**. Write at a low threshold once, then sweep here
    in seconds.

## Shards or one file

Both inputs are the same predictions and close into the same partition; what differs is how a row is
named. A binary shard carries **row indices**, which cost nothing to read and mean nothing
outside the run that wrote them. A merged file carries `unique_id`s, which is what makes it
worth anything to another tool and what costs an index to read back:

```text
Read 142,541 predictions from predictions.parquet in 0.350294 s
Resolved their ids against the record ids in 0.21232 s
```

That index is one `uint32` per record kept in the order of the id it names, so a lookup is a
binary search and the structure is 4 bytes a record beside the union-find's 5. A hash map over
20M ids would cost an order of magnitude more than the thing it feeds. On the 1.8M-row sample
it builds in 0.21 s and the whole pass takes 0.35 s, against 0.014 s to read the same predictions as
shards. Clustering the merged file of a run gives a **byte-identical** partition to clustering
that run's shards; `tests/cluster_test.cpp` asserts it both ways.

A prediction naming an id no loaded record has is counted and skipped rather than failing the
run,
and the report warns:

```text
WARNING: 12 predictions name a record this data file does not hold (0.01%); they were skipped
```

That is the wrong data file, the wrong schema, or predictions from another run. It is loud
because
it is silent damage otherwise: those pairs simply do not join anything.

`cluster` loads only the id column — it clears `schema.columns` before handing the schema to
the loader, so no dictionary is built and no comparison column is interned.

## Example

```sh
cpplink cluster --schema examples/sample_schema.json --predictions predictions/ \
                --out clusters.csv --truth examples/sample.truth.csv \
                examples/sample.parquet
```

```text
Read 169,026 predictions from 8 shards in 0.0170545 s
Weights 30.017 to 115.841 bits
143,062 of them merged two components (84.64%)

122,171 clusters of two or more, covering 265,233 records (14.74%)
1,534,767 records stayed alone
Largest cluster 12 records; the partition asserts 169,922 duplicate pairs
265,233 rows written

Size            Clusters          Records
-------------------------------------------
1 (singleton)   1,534,767        1,534,767
2                 105,662          211,324
3                  13,234           39,702
4                   2,481            9,924
5                     580            2,900
6-10                  210            1,337
11-100                  4               46
-------------------------------------------

Against the known duplicates, with both sides closed transitively:
  listed     143,728 pairs in the truth file
  true       170,801 pairs across 122,675 clusters (largest 12)
  recovered  169,922
  asserted   169,922
  precision  1.0000
  recall     0.9949
  f1          0.9974

Wrote clusters.csv
```

## Reading the output

### The prediction pass

Clustering reads the predictions as the edges of a graph, so this section counts them as edges:
a prediction that joins two components is an edge that did something.

```text
Read 169,026 predictions from 8 shards in 0.0170545 s
Weights 30.017 to 115.841 bits
143,062 of them merged two components (84.64%)
```

| Line | Meaning |
| --- | --- |
| `Read ... predictions` | predictions above the clustering threshold, and the shards or the file they came from |
| `Weights` | the range of weights actually used. The minimum tells you whether `--threshold` bit |
| `... merged two components` | predictions that changed the partition. The rest were redundant — both endpoints were already connected |

**The merge rate is a structural read on the edge set.** 84.64% means the edges are close to a
forest: the duplicate groups are small and there is not much redundancy for union–find to
absorb. A low merge rate means many edges inside already-connected groups, which is what a
dense cluster looks like — and also what a runaway chain looks like just before it swallows
the file.

### The partition

| Line | Meaning |
| --- | --- |
| `N clusters of two or more, covering R records` | the answer, and its coverage of the file |
| `records stayed alone` | singletons — records with no above-threshold prediction |
| `Largest cluster` | **watch this.** Union–find is transitive, so one bad edge between two correct clusters merges both entirely. A largest cluster in the thousands means a chain ran away |
| `asserts N duplicate pairs` | \(\sum \binom{\text{size}}{2}\) over clusters — the pairs the partition *claims*, which is more than the pairs scored |
| `rows written` | rows in the output file, subject to `--min-size` |

Here 14.74% of records land in a cluster of two or more against a planted duplicate rate of 8%
of records, and the largest cluster is 12 — nothing chained.

### The size histogram

Bucketed cluster sizes with the record counts they account for. Its job is to make a runaway
component obvious: a healthy dedup run is dominated by clusters of 2 and 3, and any bucket at
the far right holding records is worth investigating with [`explain`](explain.md) on a pair
inside it.

### The quality section

Given `--truth`, pairwise precision and recall of the partition — **over the transitive
closure, not the predictions**:

| Field | Meaning |
| --- | --- |
| `listed` | pairs written in the truth file, as given |
| `true` | pairs after the truth side is **closed transitively**, and the clusters that closure implies |
| `recovered` | true duplicate pairs whose rows share a cluster |
| `asserted` | pairs the partition claims, i.e. \(\sum \binom{\text{size}}{2}\) |
| `precision` | `recovered / asserted` |
| `recall` | `recovered / true` |
| `f1` | their harmonic mean |

!!! danger "Both sides are closed, and scoring against the raw list is a bug"
    A truth file is a **list of planted pairs**, not a partition. If a–b and b–c were both
    planted, a–c is a genuine duplicate that no line of the file names. A partition is
    transitive and asserts a–c anyway, so comparing the two directly counts recovered
    duplicates as false positives. On this sample the raw list holds 143,728 pairs and its
    closure holds **170,801**, so the difference is 19% of the answer.

    This was a real defect, not a hypothetical one: it was one of the two measurement bugs
    that had every quality number in these pages reading about eight points low. `cluster`
    now closes the truth side before comparing and prints `listed` beside `true` so the gap
    stays visible.

!!! warning "Cluster precision is stricter than the precision `predict` reports"
    A chain a–b–c asserts a–c whether or not that pair was ever scored, so the partition
    claims pairs `predict` never wrote a prediction for. Here it asserts 169,922 pairs against
    169,026 predictions.

    **That gap is the number to watch when blocking widens.** One wrong prediction between two
    correct clusters turns into every cross pair between them, and it is completely invisible
    in the pair-level numbers `predict` reports.

## Sweeping the threshold

Because re-clustering costs a re-read, sweeping is cheap. On a 1M-row sample over a fixed set
of predictions:

| Threshold (bits) | Predictions kept | Clusters | Asserted pairs | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 93,816 | 67,802 | 94,315 | 1.0000 | 0.9953 | **0.9976** |
| 20 | 93,816 | 67,802 | 94,315 | 1.0000 | 0.9953 | **0.9976** |
| 40 | 93,796 | 67,801 | 94,307 | 1.0000 | 0.9952 | 0.9976 |
| 60 | 92,435 | 67,375 | 93,513 | 1.0000 | 0.9868 | 0.9934 |
| 80 | 74,830 | 58,318 | 77,347 | 1.0000 | 0.8162 | 0.8988 |
| 100 | 29,546 | 26,334 | 29,745 | 1.0000 | 0.3139 | 0.4778 |

Precision is already 1.0000 at 0 bits and recall does not move up to 40; above 60 recall
collapses for nothing, because precision has no room left to gain. **There is no threshold at
which this pipeline is materially better than it is at 20 bits**, and the ceiling is the
0.9925 blocking recall measured by [`recall`](recall.md).

That the whole 0-to-40-bit range selects the same partition is the same fact
[the model](../model.md#choosing-a-threshold) states about `--probability`: matching pairs
here score 100+ bits, so the threshold has a wide range over which it changes nothing.

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

The union–find is `uint32` parent plus `uint8` rank — **5 bytes a record**, 100 MB at 20M rows —
and the predictions are streamed past it with **none retained**, from a shard directory or from
a merged file alike (a merged file adds the id index, 4 bytes a record). The pass over 142k
predictions and 1.8M records took 0.016 s against 36 s to produce them, which is why the
lock-free
CAS version the design mentions stays unbuilt.
