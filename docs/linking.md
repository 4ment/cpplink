# Linking two files

Deduplication asks which rows of one file are the same person. **Linking** asks which rows of
one file are the same person as rows of another — a customer list against a supplier list, this
year's extract against last year's. It is the same model, the same comparisons and the same
scoring; only the set of pairs changes.

```sh
cpplink predict --schema schema.json --model model.json --out edges/ \
                left.parquet right.parquet
```

Two files with nothing else said means linking. There is no separate mode to remember for the
common case.

## What a second file changes

Files are read **in order into one store**, and each becomes a *dataset*. Everything downstream
sees a single store of 1,000,000 rows; only the pair mode knows there were two files.

| Mode | Pairs scored | Written as |
| --- | --- | --- |
| Deduplication | every pair of one file | one file, no `--mode` |
| Linking | only pairs that cross the two files | two files, or `--mode link` |
| Link-and-dedup | every pair of both files read as one | `--mode dedup` or `--mode link-and-dedup` |

`--mode` is accepted by [`explain-blocking`](commands/explain-blocking.md),
[`recall`](commands/recall.md), [`estimate`](commands/estimate.md),
[`predict`](commands/predict.md) and [`rescore`](commands/rescore.md).
[`inspect`](commands/inspect.md) and [`cluster`](commands/cluster.md) take several files too,
but enumerate no pairs, so they need no mode.

## Linking is cheaper than deduplicating the same rows

The two files' own internal pairs are never generated — not filtered out afterwards, never
generated. On a 1M-row fixture split 750,405 / 249,595:

```text
Records          1,000,000
Inputs           2  (750,405 + 249,595 rows)
Mode             cross-dataset pairs
Pairs unblocked  187,297,335,975
...
Union, deduplicated                               15,969,937
```

| | Linking | Deduplicating the same rows |
| --- | ---: | ---: |
| Pair space | 187,297,335,975 | 499,999,500,000 |
| Candidates after blocking | **15,969,937** | 44,985,940 |
| Blocking recall | 99.55% | — |
| Cluster F1 | 0.9965 | — |

**2.82× less work for the same quality.** The saving is structural: a blocking group is already
sorted by row and the two files occupy contiguous row ranges, so within a group the partners of
a row are a single contiguous range and a cursor walks straight to it.

!!! note "Sorted neighbourhood is the exception"

    A sorted-neighbourhood window is ordered by *value*, so the two files interleave inside it
    and there is no contiguous range to jump to. That source pays one comparison per candidate
    in link mode. It is the only one that pays anything.

## What linking changes about the model

Two things move, and both are about *u* — the probability a level fires on a pair drawn at
random.

**`u` is estimated from cross-file pairs.** `u` is what a random **admissible** pair does, and
when linking, the admissible pairs are the cross-product. Estimating it over the whole triangle
would put the wrong denominator under every weight. In the cross-product the closed forms stop
being approximations at all: there are exactly *N₀·N₁* ordered cross draws and not one of them
is a row against itself, so the closed form for an exact level *equals* the cross-file
agreement rate.

**λ is put back on the cross-product**, not on the triangle, for the same reason.

**Term-frequency tables stay pooled** across the two files. The adjustment asks how rare a value
is in the population being linked, and the pooled tables answer that. The exact per-file form
would want *p<sub>v</sub>* of one file times *p<sub>v</sub>* of the other — a second set of
tables resident at scoring time, to correct an approximation that is exact when the two files
are drawn from the same population. That is the linkage assumption in the first place.

## Ground truth for a link run

[`gen-sample --out-b`](commands/gen-sample.md) writes the fixture: originals to the first file,
every planted duplicate to the second, so each recorded pair crosses the two by construction.

```sh
cpplink gen-sample --rows 1000000 --duplicate-rate 0.25 \
                   --out left.parquet --out-b right.parquet --truth pairs.csv
cpplink recall --schema schema.json --truth pairs.csv left.parquet right.parquet
```

```text
Known pairs  249595 resolved
Mode         cross-dataset pairs over 2 inputs
...
Union                               248483    99.55%
```

A truth pair that lies *inside* one file is not a miss when linking — it is out of scope, and
`recall` says how many there are rather than counting them against the sources.

## Clustering a link run

[`cluster`](commands/cluster.md) is unchanged. Connected components over cross-file edges give
the linked groups, and because the partition is transitive it will assert pairs *within* a file
that no edge scored — two rows of the right-hand file both linked to the same left-hand row are
the same person as each other. That is a correct inference, and `cluster --truth` scores it
against a transitively closed truth side so it is judged as one.

## Limits

More than two files works — they become datasets 0, 1, 2 — with one caveat: the `u` sampler
draws a uniform row and then a uniform partner from outside that row's file, which is exactly
uniform over cross pairs for **two** files and slightly favours the smaller files for more. The
number of inputs is reported beside `u` so the case is visible.
