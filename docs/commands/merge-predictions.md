# `merge-predictions`

**Goal:** turn the directory of per-thread prediction shards into one file another tool can
open, as csv or as parquet.

[`predict`](predict.md) writes one shard per thread because nothing else keeps the threads
from contending on a writer, and [`cluster`](cluster.md) reads that directory back without
minding how many files are in it.
Everything else does mind.
A run told to `--out predictions.parquet` merges its own shards when it finishes, so this
command is for the shard directories that already exist: an older run, a `--format csv` one, a
directory you want as both formats.
`merge-predictions` streams the shards past a single writer and keeps nothing: a prediction is
read, filtered, written and forgotten, so the merge costs what the output costs and not what
the run did.

## Synopsis

```sh
cpplink merge-predictions --shards <dir> --out <file.csv|file.parquet>
                          [--format csv|parquet] [--from bin|csv]
                          [--threshold BITS | --probability P]
                          [--schema <schema.json>] [<file.parquet>...]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--shards <dir>` | — | required; the directory of `shard-*` files from [`predict`](predict.md) or [`rescore`](rescore.md) |
| `--out <file>` | — | required; the single file to write |
| `--format csv\|parquet` | from the extension | `.parquet` or `.pq` means parquet, anything else csv |
| `--from bin\|csv` | binary if there is any | which shard format to read when the directory holds both |
| `--threshold BITS` | −∞ | keep only predictions at or above this weight |
| `--probability P` | — | the same threshold as a posterior in (0, 1) |
| `--schema <file>` | none | needed only to turn the row indices in binary shards into `unique_id`s. Only `unique_id` is used |
| *(positional)* | none | the parquet file the shards were scored from, required with `--schema` |

## Example

```sh
cpplink merge-predictions --schema examples/sample_schema.json --shards predictions/ \
                          --out predictions.parquet examples/sample.parquet
```

```text
Read 142,541 predictions from 8 bin shards in 0.123 s
Wrote 142,541 predictions to predictions.parquet (parquet)
```

The merged file holds the same five columns a csv shard does, whichever format it is written
in:

```csv
id_a,id_b,gamma,match_weight,match_probability
r0,r20,174729,106.288416,1.000000000
r1,r119514,174665,108.510385,1.000000000
```

## Which shards get read

A directory that has been written twice, once as `--format bin` for `cluster` and once as
`--format csv` to look at, holds **one run in two forms, not two runs**.
Reading both would double everything, so `merge-predictions` takes one format and says which:

```text
Read 142,541 predictions from 8 bin shards in 0.123 s
Ignored 8 csv shards holding the same run; --from picks the other side
```

Binary is preferred because it is what `cluster` reads and what `predict` writes by default.
`--from csv` takes the other side.

## Ids, and why `--schema` is optional

Binary shards carry **row indices**, not `unique_id`s, for the same reason `cluster` takes the
data file as well as the shard directory: a row index is only meaningful against the same
parquet file loaded through the same schema.
Given `--schema` and that file, `merge-predictions` loads the id column alone, with no dictionary
and no comparison column, and writes ids.
Without it, the merge still runs and the output names rows by index:

```text
Rows are named by index: binary shards carry no ids, so pass --schema and
the parquet input to write the record ids instead
```

Csv shards already carry their ids, so `--from csv` needs neither the schema nor the data.

## Filtering while merging

Edges carry their weight, so the merge can drop what a higher threshold would not have
written, exactly as [`cluster --threshold`](cluster.md#sweeping-the-threshold) does:

```sh
cpplink merge-predictions --shards predictions/ --out predictions.csv --threshold 60 \
                          --schema examples/sample_schema.json examples/sample.parquet
```

```text
Read 142,541 predictions from 8 bin shards in 0.099 s
Kept 140,796 above the threshold
Wrote 140,796 predictions to predictions.csv (csv)
```

Lowering the threshold is not available here for the reason it is not available in `cluster`:
a shard holds what one `predict` run kept, and what it dropped was never written down.
That is what [`predict --spill`](predict.md) and [`rescore`](rescore.md) are for.

## csv or parquet

| | csv | parquet |
| --- | --- | --- |
| Weight | rounded to six decimals, as the shards are | the full `f64` the scorer produced |
| `gamma` | decimal text | `uint32` |
| Size, 142,541 predictions | 6.2 MB | 3.6 MB, Snappy |
| Opens in | anything | anything that reads parquet |

Parquet is the one to pick when the predictions are going into a database or a dataframe: it
is
typed, it is smaller, and it keeps the weight exactly.
Csv is the one to pick when the next thing to look at the file is a person.

!!! note "The merged file is not an input to anything"
    Nothing in cpplink reads it back: `cluster` reads the shard directory, and `rescore`
    reads a spill.
    `merge-predictions` is the door out of the pipeline.

## Cost

One pass over the shards and one writer, with nothing retained: 142,541 predictions merge in
0.1 s
against the 11.7 s that produced them.
The store, when `--schema` is given, is the id column and nothing else.
