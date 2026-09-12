# `gen-sample`

**Goal:** write a synthetic parquet file with the column mix cpplink is aimed at, planted
duplicates, and a sidecar recording exactly which pairs are duplicates.

It exists because the memory and throughput claims in the design are only worth making if they
are measured, and measuring needs data at scale with known answers. The same file doubles as
ground truth for [`recall`](recall.md) and [`cluster --truth`](cluster.md).

## Synopsis

```sh
cpplink gen-sample --out <file.parquet> [--rows N] [--seed N]
                   [--duplicate-rate F] [--truth <file.csv>]
                   [--out-b <file.parquet>]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--out <file>` | — | required; the parquet file to write |
| `--out-b <file>` | — | a second file; originals go to `--out` and every planted duplicate here, so each recorded pair crosses the two — the fixture for [linking](../linking.md) |
| `--rows N` | 1,000,000 | rows to generate |
| `--seed N` | 1 | RNG seed; the whole file is deterministic in it |
| `--duplicate-rate F` | 0.08 | fraction of rows that are corrupted copies of earlier rows |
| `--truth <file.csv>` | none | write the planted duplicate pairs as `id_a,id_b` |

## Example

```sh
cpplink gen-sample --out examples/sample.parquet --rows 1800000 \
                   --truth examples/sample.truth.csv
```

```text
Wrote 1800000 rows to examples/sample.parquet
Planted duplicate pairs listed in examples/sample.truth.csv
```

## What it writes

Ten columns, matching [`examples/sample_schema.json`](../reference/schema.md), which also
declares an `email_username` column derived from `email` at load:

| Column | Type | Shape |
| --- | --- | --- |
| `id` | string | `r0`, `r1`, … — the `unique_id` |
| `first_name`, `last_name` | string | Zipf-distributed, so the frequency skew is realistic |
| `gender` | string | two values close to even, missing on 3% of rows |
| `dob` | date | low cardinality, ~21k distinct values |
| `email`, `phone` | string | near-unique, ~1.65M distinct over 1.8M rows |
| `postcode` | string | 9,000 distinct |
| `latitude`, `longitude` | double | a coordinate pair, compared as one `geo_within` comparison |
| `address_tokens` | string_list | ~4.5 elements a row |

That mix is the point: it exercises the near-unique dictionaries that dominate memory, the
low-cardinality columns that break rare-value blocking, a boolean, a list column, a date, and
a double pair that carries no term frequencies.

An address is a username drawn from the names and a four-digit suffix under one of six
domains, skewed so the first takes about half the rows the way one provider does: 933k
distinct addresses over 920k distinct usernames at 1M rows, so usernames do collide across
domains and the username level has a `u` above the floor.

**Duplicates are corrupted copies of earlier rows.** The corruption model drops `email` 35% of
the time and `phone` 25%, independently, perturbs coordinates, and introduces typos into names
— which is why the [measured blocking findings](../blocking.md) lean the way they do, and why
they carry a caveat: real data with weaker identifiers would shift the balance back toward
fuzzy sources. Of the addresses it keeps, one in four keeps its username under a different
domain, which is the pair the email comparison's username level exists for. `gender` is
dropped on 5% of duplicates and flipped on 2%.

!!! note "Records are generated from their row index"
    Deterministically, so a duplicate can reproduce its original exactly without either being
    held in memory. Generating 20M rows is a streaming operation, not a 4 GB one: peak
    resident memory while writing them is 237 MB.

## The truth file

```csv
id_a,id_b
r5,r9
r17,r19
```

Every planted duplicate pair, as `unique_id` values. This is the format
[`recall --truth`](recall.md) and [`cluster --truth`](cluster.md) read; a header line beginning
`id_a` is optional and skipped.

Note that the truth file lists the pairs that were *planted*, so a group of three duplicates
appears as the pairs that were actually copied, not the full closure of the group.

## Scale

At 20M rows the file is about 1.6 GB of parquet and takes ~43 s to write. The 1.8M-row sample
used throughout this documentation is 142 MB and takes a few seconds.
