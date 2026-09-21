# The cluster viewer

`cpplink-viewer` serves a page over the clusters a run produced: a list of clusters on the left, and for the one picked, its members side by side with every disagreeing cell highlighted, and every prediction touching it with the one picked drawn as the ledger the scorer produced.
It reads what the pipeline wrote and runs none of it: the clusters and the predictions come from the files, the waterfalls from `explain --predictions`, and nothing in the viewer recomputes a bit of a weight.

```sh
cpplink predict --schema schema.json --model model.json --out predictions.parquet sample.parquet
cpplink cluster --schema schema.json --predictions predictions.parquet --out clusters.csv sample.parquet
cpplink explain --schema schema.json --model model.json --predictions predictions.parquet --out waterfalls.parquet sample.parquet
cpplink-viewer --schema schema.json --clusters clusters.csv --predictions predictions.parquet \
    --waterfalls waterfalls.parquet --model model.json --truth sample.truth.csv --open sample.parquet
```

The page is served at `http://127.0.0.1:8770/` (`--port` moves it, `--open` opens a browser on it).

## Install

The viewer is the `cpplink_viewer` package, shipped in the same wheel as the binding, with DuckDB as its one extra dependency:

```sh
pip install "cpplink[viewer]"
```

It never imports the compiled module, so it also runs from a checkout with nothing built, against a run the command line produced:

```sh
pip install duckdb pyarrow numpy
PYTHONPATH=python python -m cpplink_viewer --schema schema.json --clusters clusters.csv sample.parquet
```

## What it reads

| Option | File | Written by |
|---|---|---|
| `data` | the parquet file(s) the run read, in order | you |
| `--schema` | the schema, for the id column and the columns to show | `init` |
| `--clusters` | `unique_id, cluster_id, cluster_size`, csv or parquet | `cluster --out` |
| `--predictions` | one csv or parquet file, or the shard directory | `predict --out` |
| `--waterfalls` | one wide row per prediction, csv or parquet | `explain --predictions --out` |
| `--model` | the level labels and rates the waterfall shows | `estimate --out` |
| `--truth` | known pairs, to colour members by true entity | you, or `gen-sample --truth` |

Only `--schema` and the data are required, plus one of `--clusters` or `--threshold`.
Without `--predictions` the page shows the members and nothing about why they are together; without `--waterfalls` and `--model` it shows each prediction's weight but not the ledger behind it.
A shard directory names records by row, so it costs one pass over the id column that a merged file does not.

## The cache

The first run builds a DuckDB file (`--cache`, `cpplink_viewer.duckdb` by default) holding only what the page can ever show: the clustered records, never the singletons, their predictions, the waterfalls of those predictions, and one row of statistics per cluster.
At 20M records with 2.9M of them clustered the parquet is read exactly once and every later request touches a table seven times smaller than the file.
The cache is keyed on every input's path, size and modification time and on the options that shape it, so a changed input rebuilds it and an unchanged one is reused; `--rebuild` forces it.
`--memory` and `--threads` are DuckDB's limits while building.

## Thresholds and rejected predictions

`--threshold` keeps only the predictions at or above it.
With `--clusters`, the file is taken as what `cluster --threshold` wrote at that threshold and read as it stands.
Without one, the predictions are clustered here by the same union-find in the same order, so the cache holds exactly the partition that command would write, named by the same representatives; the test suite holds the two to the same file.

A prediction whose two records clustering put in different clusters, above the write threshold and below the clustering one, is the prediction most worth reading, so it is kept and listed under both of its clusters as `rejected`, each naming the other cluster its second record went to.
`--min-size` and `--max-size` bound the clusters listed; `--max-rows` bounds the members shown per cluster, the rest being counted only.

## The list

The list is sorted by disagreement (columns whose values differ within the cluster), size, weakest prediction or id, and filtered to every cluster, the split ones, the ones held together only transitively (fewer predictions than pairs), or the ones mixing entities against the truth file.
The search box reads any value, one column, the cluster id, or a comma-separated list of record ids matched whole, in which case it says which of them named a record.
The `network` checkbox draws the selected cluster's predictions as a graph, which is where a chain shows itself as a chain.
