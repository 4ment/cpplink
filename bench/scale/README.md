# A scale benchmark, in the shape of the splink one

[`bench/`](../README.md) answers the question that has to come first: does cpplink
produce the same quality of answer as splink on the same model?
It runs at 1,000 to 50,578 records, where both tools finish in seconds and quality is the
only thing worth measuring.

This directory answers the second question.
Splink's own published benchmark deduplicates 7M records over 1bn comparisons in just over
two minutes on a large EC2 instance, and completes the same work on an 8 vCPU / 16 GiB
machine in about 25 minutes.
That benchmark reports runtime only, and it is the number cpplink exists to beat.
So this is the same shape of benchmark on the same class of machine, with two things added
that the original does not report: the resident memory and scratch disk each tool needs,
and the same run for the other tool on the same data.

## What is measured

The pipeline is the one the splink benchmark times, end to end and including clustering:

| Stage | cpplink | splink |
|---|---|---|
| Parameter estimation | `estimate` (u, then per-column EM sessions) | `estimate_u_using_random_sampling`, then one EM round per blocking column |
| Scoring | `predict` | `inference.predict` |
| Clustering | `cluster` | `clustering.cluster_pairwise_predictions_at_threshold` |

Loading is inside the measured region for both, because both have to read the parquet
before they can do anything.

## What is held fixed

Both tools compile their configuration from
[`parity_schema.json`](make_schema.py), so neither is given a different model:

| Held fixed | How |
|---|---|
| **Input** | One parquet file, read by both. |
| **Comparisons** | Six columns, each a `null -> exact -> fuzzy… -> else` ladder with the same thresholds in the same order. |
| **Level vocabulary** | Only levels both libraries implement identically: exact match, Levenshtein at a distance, Jaro-Winkler at a similarity. |
| **Term frequency** | On, for the same columns, in both. |
| **Blocking** | The same five single-column exact-agreement rules. |
| **λ** | cpplink's learned value, passed to splink as `probability_two_random_records_match`. |
| **u sampling** | 10⁶ random pairs in both. |
| **Threshold** | Match probability 0.99, for both prediction and clustering, in both. |
| **Machine** | Apple M1, 8 cores, 16 GB, macOS. All cores available to both. |

The level vocabulary restriction is the same one the parity benchmark makes and costs the
same thing: no `date_within`, no `geo_within`, no list levels, so `dob` is compared for
exact agreement only.
That makes the model slightly worse in both tools, equally.

## What is deliberately not held fixed

The two tools estimate `m` differently and that is the point of the comparison rather than
a flaw in it.
cpplink runs one EM session per blocking column and holds out the comparisons that read a
tied column; splink runs one EM round per blocking column and averages.
Both are given the same λ and the same u sample, so the difference is confined to the
`m` estimator.
Quality is reported for both so that a runtime difference bought by a worse answer would
be visible.

## Reproducing

```bash
W=/tmp/scale && mkdir -p $W
python3 bench/scale/make_schema.py $W/parity_schema.json
for n in 1 2 4; do
  ./build/cpplink gen-sample --out $W/s${n}m.parquet --rows ${n}000000 \
      --truth $W/s${n}m.truth.csv --seed 7
done

# cpplink, one size
./bench/scale/run_cpplink_scale.sh $W s4m

# splink, same data and same schema; lambda comes from cpplink's model
python bench/scale/run_splink_scale.py --parquet $W/s1m.parquet \
    --schema $W/parity_schema.json --lam <lambda from s1m.model.json> \
    --out $W/splink_s1m --temp $W/duckdb_tmp

python3 bench/scale/summarise.py $W --out $W/scale.json
```

`gen-sample` plants duplicates against the chain's base record and writes the truth pairs,
so recall and precision are measured against a known answer rather than asserted.

## Results

Apple M1, 8 cores, 16 GB, macOS, all cores available to both tools.
cpplink timings are the minimum of two repeats, taken on a quiesced machine.
The first sweep was discarded because the operating system was reindexing and the load
average was above 20, which cost about 50% and is worth stating: at this scale the
benchmark is sensitive to what else the machine is doing.

### cpplink

| Records | Candidate pairs | `estimate` | `predict` | `cluster` | Wall | CPU | Peak RSS | Scratch | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1M | 89,846,415 | 15.2 s | 4.4 s | 0.4 s | **20.0 s** | 111 s | 700 MB | none | 0.9974 |
| 2M | 359,425,964 | 15.8 s | 15.6 s | 0.5 s | **32.0 s** | 184 s | 907 MB | none | 0.9975 |
| 4M | 1,438,216,166 | 21.0 s | 66.5 s | 1.7 s | **89.2 s** | 555 s | 1.16 GB | none | 0.9977 |

Candidate counts are exact and come from the term-frequency tables, not from enumeration.
Scoring holds a rate of 20 to 23M candidates a second across a 16x growth in pairs, because
`Scorer::Ceiling` drops 99.67% of candidates at 1M and 99.92% at 4M before any string metric
runs.
Peak resident memory grows 1.7x while candidate pairs grow 16x, which is the invariant
doing its job: memory tracks records, not pairs.

### splink, on the same data and the same schema

| Records | Configuration | Wall | EM | `predict` | Peak RSS | Peak spill | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1M | defaults | 184.8 s | 109.2 s | 53.2 s | 4.66 GB | 10.59 GB | 0.9961 |
| 1M | `estimate_without_term_frequencies` | 90.0 s | 40.0 s | 41.9 s | 3.95 GB | 9.29 GB | 0.9961 |
| 1M | that, plus `memory_limit=13GB` | **87.8 s** | 41.6 s | 40.6 s | 6.24 GB | 7.13 GB | 0.9961 |
| 2M | defaults | did not finish | | | 6.69 GB | >13.0 GiB | |
| 2M | `estimate_without_term_frequencies` | did not finish | | | 7.19 GB | >12.1 GiB | |

Both 2M runs ended in `Out of Memory Error: failed to offload data block ... This limit was
set by the 'max_temp_directory_size' setting`, on the EM session for a mid-cardinality
column.
That is a disk-capacity failure and not a crash: with more scratch than this machine had,
they would presumably have completed.
What is measured is the requirement, which is more than 13 GiB of scratch at 2M records.

### The comparison

At 1M records, against splink's best configuration here:

| | cpplink | splink | Ratio |
|---|---:|---:|---:|
| Wall, end to end | 20.0 s | 87.8 s | **4.4x** |
| Scoring stage alone | 4.4 s | 40.6 s | **9.2x** |
| Peak resident memory | 700 MB | 6.24 GB | **8.9x** |
| Scratch disk | none | 7.13 GB | n/a |
| Edges above threshold | 94,339 | 93,760 | 1.006x |
| F1 | 0.9974 | 0.9961 | |

Against splink's defaults rather than its best configuration the wall ratio is 9.2x.

Quality is equal to within a rounding of the third decimal, which is the point of reporting
it: neither tool is buying speed with a worse answer.
cpplink has slightly higher recall and splink slightly higher precision, and the two edge
sets are 0.6% apart in size.
Quality is scored by [`score_scale.py`](score_scale.py) for both tools from the cluster
files, so neither tool's own accounting is taken on trust; on cpplink's clusters it
reproduces what `cluster --truth` reports to four decimal places.

## Three findings

**Splink 4 already has this project's central idea, and it is off by default.**
`estimate_without_term_frequencies=True` makes splink fold the comparison vectors into
`__splink__agreement_pattern_counts` with a `GROUP BY` and iterate EM over the counts,
which is the same sufficient statistic the pattern histogram is.
Turning it on cut EM from 109.2 s to 40.0 s and the whole pipeline from 184.8 s to 90.0 s,
with a byte-identical edge count.
It is not the default, and the published splink benchmark did not use it.
The remaining difference is that splink has to materialise the comparison-vector table
before it can group it, whereas the fold here happens per thread during enumeration and no
pair is ever written down.

**Scratch disk, not memory, is what stops splink here.**
Peak resident memory never exceeded 7.2 GB in any splink run, comfortably inside a 16 GiB
machine.
What ran out was temp space: 7.13 GB at best at 1M records for 90M candidate pairs, which
is about 79 bytes of scratch per candidate pair, and more than 13 GiB at 2M.
Applying that rate to the published benchmark's 1bn comparisons suggests something on the
order of 80 GB of scratch, which that benchmark does not report and which its instance
types would have had.
The rate is measured under this blocking plan and should not be pushed far, but the
direction is the finding: the constraint is disk, and it is invisible in a runtime table.

**The article's headline is reachable on a laptop.**
The published benchmark does 7M records over 1bn comparisons in just over two minutes on a
large EC2 instance, and about 25 minutes on 8 vCPU / 16 GiB.
This benchmark does 4M records over 1.44bn comparisons in 89.2 seconds on 8 cores and
16 GB, with 1.16 GB resident and no scratch file.
That is a different dataset with different columns, so it is a comparison of the same shape
rather than the same thing, and the within-benchmark 4.4x at 1M is the defensible number.
