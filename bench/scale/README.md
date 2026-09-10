# A scale benchmark, in the shape of the splink one

[`bench/`](../README.md) answers the question that has to come first: does cpplink
produce the same quality of answer as splink on the same model?
It runs at 1,000 to 50,578 records, where both tools finish in seconds and quality is the
only thing worth measuring.

This directory answers the second question.
Splink's own published [benchmark](https://medium.com/data-science-collective/deduplicating-7-million-records-in-two-minutes-with-splink-4b1a87035a85) deduplicates 7M records over 1bn comparisons in just over
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

## The scaling sweep, one thread

The following benchmark was run on a MacBook Air (Apple M1, 16 GB RAM).

The parity benchmark below answers "against splink, on the same model, with all eight cores".
This sweep answers a different question: **what does the pipeline cost as the file grows**, up to and including the 20M-record design target.
Everything in it runs on one thread, so nothing in the curves is a story about how many cores happened to be free.

It has two tracks, because comparing the two tools means giving them the same model and splink cannot be given cpplink's.

The first track uses the project's own [`examples/sample_schema.json`](../../examples/sample_schema.json), the ten-column list-valued workload the design is aimed at, and runs to 20M records.
Splink has no bit-identical counterpart for its `geo_within`, `date_within` and list levels, and does not generate `rare_value` or `sorted_neighbourhood` rules at all, so it cannot run this track.

The second track uses the **shared schema** that [`make_schema.py`](make_schema.py) writes: the intersection of what both libraries express identically, which is six columns compared only by exact match, Levenshtein at a distance and Jaro-Winkler at a similarity, with five plain exact-value blocking rules.
Both tools run it, on the same rows, with the same thresholds, the same λ and the same u sample. 
It makes the model slightly worse in both, equally.

The two tracks block differently and so make different numbers of pairs out of the same rows, which is why the figures are drawn against candidate pairs rather than against records: that is the axis on which two blocking plans are comparable.

| Records | Candidate pairs | `estimate` | `predict` | `cluster` | Wall | Peak RSS | Scratch | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250k | 6,554,050 | 6.3 s | 1.9 s | 0.1 s | **8.3 s** | 0.29 GB | none | 0.9978 |
| 500k | 16,227,174 | 13.5 s | 4.3 s | 0.1 s | **18.0 s** | 0.42 GB | none | 0.9977 |
| 1M | 44,919,878 | 19.1 s | 11.1 s | 0.3 s | **30.4 s** | 0.52 GB | none | 0.9975 |
| 2M | 139,934,399 | 24.8 s | 31.7 s | 0.6 s | **57.1 s** | 0.83 GB | none | 0.9973 |
| 4M | 480,680,096 | 35.8 s | 98.7 s | 1.3 s | **135.8 s** | 1.58 GB | none | 0.9970 |
| 8M | 1,759,554,498 | 59.1 s | 357.2 s | 3.5 s | **419.9 s** | 2.38 GB | none | 0.9964 |
| **20M** | **10,090,976,646** | 143.9 s | 2,133.3 s | 10.2 s | **2,287.5 s** | 4.49 GB | none | 0.9946 |

And the shared schema, both tools on one thread:

| Records | Candidate pairs | cpplink | splink | cpplink RSS | splink RSS | splink scratch | cpplink F1 | splink F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250k | 5,645,873 | **5.5 s** | 21.1 s | 0.22 GB | 1.87 GB | none | 0.9981 | 0.9964 |
| 500k | 22,471,689 | **15.2 s** | 76.9 s | 0.34 GB | 2.17 GB | none | 0.9979 | 0.9961 |
| 1M | 89,846,415 | **32.3 s** | 343.5 s | 0.46 GB | 3.34 GB | 6.63 GB | 0.9978 | 0.9961 |
| 2M | 359,425,964 | **76.9 s** | did not finish | 0.69 GB | 5.53 GB | out of scratch | 0.9978 | |
| 4M | 1,438,216,166 | **249.8 s** | not run | 1.24 GB | | | 0.9977 | |

Both tools are timed by the same `/usr/bin/time -l` wrapper and scored by the same [`score_scale.py`](score_scale.py), which reproduces what cpplink's own `cluster --truth` reports to four decimal places, so neither tool's accounting is taken on trust.
cpplink's candidate counts here reproduce the eight-thread table below to the pair, which is the cross-check that the two sweeps are running the same plan.

![Wall time against candidate pairs, one thread](../../docs/img/scale-time.svg)

![Peak memory against candidate pairs, one thread](../../docs/img/scale-memory.svg)

Five things the two curves say:

- **Time is linear in candidate pairs, not in records.**
  Scoring runs at 4.4M to 5.3M candidates a second across three orders of magnitude of pairs.
  The small runs sit above the dashed guide because load and estimation are a fixed cost; by 20M `estimate` is 6% of the run.
  Records enter only through how many pairs the blocking plan makes of them, which here is quadratic.
- **Memory tracks records, not pairs.**
  Pairs grow 1,540x across the sweep and peak resident memory grows 15x.
  That is the invariant doing its job: no stage holds a row per pair, so the store and the pattern histogram are the whole footprint.
- **splink's memory is the other shape, and its scratch disk is the binding constraint.**
  Its resident memory rises 1.87 to 3.34 GB over a 16x growth in pairs while cpplink's rises 0.22 to 0.46 GB over the same range, and at 1M it also writes **6.63 GB of scratch** where cpplink writes none.
  At 2M it does not finish: `Out of Memory Error: failed to offload data block`, which is a disk-capacity failure rather than a crash.
  The temp directory was capped at 8 GB here to protect the volume; the eight-core run in the Results section below hit the same wall with a 16 GB cap, needing more than 13 GiB.
- **`estimate` flattens.**
  It goes 6.3 s to 143.9 s while pairs grow 1,540x, because a session Bernoulli-samples its stream: the `dob` session at 20M enumerated 9.61bn pairs and compared 10.0M of them, 0.10%.
- **The ceiling is what makes the scoring pass affordable, and it gets better with scale.**
  `Scorer::Ceiling` skips 99.57% of candidates at 250k and **99.96% at 20M** before any string metric runs, so the pass costs 207 ns a candidate rather than the ~4,090 ns a full fuzzy evaluation takes.

F1 falls from 0.9978 to 0.9946 across the ten-column sweep, entirely in recall (precision is 1.0000 at every size). That is blocking, not scoring: pair completeness at 20M is 98.58%, and [`recall --why`](../../docs/commands/recall.md) classifies all 22,708 unreachable pairs as null on `email` and `phone`, disagreeing on `dob`, and outside the sorted-neighbourhood window.
That is a plan gap rather than a threshold set too high.

At 1M, where both tools finish, cpplink is **10.6x** faster end to end and holds **7.2x** less memory, for the same quality to within a rounding of the third decimal.
**That ratio is not the one to quote.**
It is single-threaded, and duckdb parallelizes better than cpplink's per-column EM sessions do: on eight cores splink goes 343.5 s to 90.0 s (3.8x) while cpplink goes 32.3 s to 20.0 s (1.6x), which is how the same comparison collapses to the **4.4x** the Results section below reports against splink's fastest configuration.
What survives the thread count is the shape: cpplink's memory is set by the records and splink's by the pairs, and only one of them needs a scratch file.

For scale rather than for parity, the published splink benchmark is the other reference point: 1bn comparisons in about 25 minutes on 8 vCPU / 16 GiB.
The 20M row above is **10.1bn comparisons in 38 minutes on one core** of a laptop.
That is a different dataset with different columns and a different model, so it is a comparison of shape and not of like for like.

Reproduce it with:

```bash
W=/tmp/scale && mkdir -p $W
S=examples/sample_schema.json
TAGS="s250k:250000 s500k:500000 s1m:1000000 s2m:2000000 s4m:4000000 s8m:8000000 s20m:20000000"
for pair in $TAGS; do
  tag=${pair%%:*}; rows=${pair##*:}
  ./build/cpplink gen-sample --out $W/$tag.parquet --rows $rows \
      --truth $W/$tag.truth.csv --seed 7
  ./bench/scale/run_cpplink_scale.sh $W $tag 0.99 1 $S      # 0.99 threshold, 1 thread
done
# the shared-schema track: the same rows under the model splink can be given identically
P=$W/shared && mkdir -p $P
python3 bench/scale/make_schema.py $P/parity_schema.json    # the shared schema
for tag in s250k s500k s1m s2m s4m; do
  ln -sf $W/$tag.parquet $P/$tag.parquet; ln -sf $W/$tag.truth.csv $P/$tag.truth.csv
  ./bench/scale/run_cpplink_scale.sh $P $tag 0.99 1 $P/parity_schema.json
  python bench/scale/run_splink_scale.py --parquet $P/$tag.parquet \
      --schema $P/parity_schema.json --lam <lambda from $P/$tag.model.json> \
      --out $P/splink_$tag --temp $P/duckdb_tmp --threads 1 --pattern-counts
done

python3 bench/scale/plot_scale.py $W --tags s250k s500k s1m s2m s4m s8m s20m \
    --parity-dir $P --parity-tags s250k s500k s1m s2m s4m \
    --json bench/scale/single_thread.json --outdir docs/img
```

The figures are drawn with matplotlib where it is installed and by a hand-written SVG writer where it is not, so the sweep runs in an environment carrying no plotting library; `--backend matplotlib` or `--backend svg` forces one.
`--from-json bench/scale/single_thread.json` redraws both figures from the collected sweep alone, which is how a figure is restyled once the run logs are gone.

splink is given `estimate_without_term_frequencies=True` (`--pattern-counts`), which is its own best configuration here and the one that folds comparison vectors into agreement-pattern counts, so the comparison is against splink at its fastest rather than at its default.

The 20M step needs about 1.6 GB of disk for the parquet and takes 38 minutes on the machine below; every smaller step finishes in under eight.

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
