# cpplink vs splink: a deduplication benchmark

A like-for-like comparison of [splink](https://moj-analytical-services.github.io/splink/)
and cpplink on public deduplication datasets, at a scale small enough that both tools
finish in seconds and the whole thing is reproducible on a laptop.

The point is *not* to show cpplink winning. cpplink's design claim is about **18M records**,
where splink's pair table does not fit; at 50k records splink's pair table fits easily and
the claim is untestable. What a benchmark at this scale can establish is the thing that
must be true before the scale claim is worth making:

> **cpplink produces the same quality of answer as splink on the same model.**

If it does not, the scale story is irrelevant. So the design below spends its effort on
making the two tools genuinely comparable, and reports cost as a secondary measurement
with an explicit note about what it does and does not extrapolate.

## What is held fixed

Every configurable thing that could favour one tool is compiled from a single neutral
description in [`datasets.py`](datasets.py):

| Held fixed | How |
|---|---|
| **Input** | One parquet file per dataset, read by both. Every column is a string, and the label column is not in it, so neither tool can see the answer. |
| **Comparisons** | The same columns in the same order, each a `null → exact → fuzzy… → else` ladder with the same thresholds. |
| **Level vocabulary** | Only levels both libraries implement identically: exact match, Levenshtein at a distance, Jaro-Winkler at a similarity. |
| **Term frequency** | On, for the same columns, in both. |
| **Blocking (matched track)** | The same single-column exact-agreement rules, in the same order, with the same "not produced by an earlier rule" exclusion. |
| **λ** | One constant per dataset, derived once offline and passed to both (`--lambda` / `probability_two_random_records_match`). |
| **u sampling** | 10⁶ random pairs, same seed, in both. |
| **Threshold** | Swept on *match probability*, which both tools accept directly. |
| **Metric** | One scorer, [`score.py`](score.py), reading both tools' cluster files. |

The level vocabulary restriction is the one real cost of this design. It forgoes cpplink's
`date_within`, `geo_within` and list levels, and splink's `DateOfBirthComparison` and
`PostcodeComparison`, because those have no bit-identical counterpart across the two. Dates
are therefore compared as strings under Levenshtein. That makes the model slightly worse in
both tools, equally.

### Why the matched track really is matched

`bench.py --verify` prices every blocking rule in both tools and requires exact agreement.
cpplink's count is closed form from its term-frequency tables and splink's is a SQL
`group by`, so agreeing to the pair is evidence, not tautology:

```
historical_50k:
  dob                  cpplink    1,549,081   splink    1,549,081   ok
  surname              cpplink      733,085   splink      733,085   ok
  postcode_fake        cpplink      112,172   splink      112,172   ok
  first_name           cpplink   16,372,982   splink   16,372,982   ok
```

Both tools also drop nulls from blocking keys for the same reason — cpplink returns
`kNoKey`, SQL says `NULL = NULL` is false — so the candidate sets agree there too.

## What is left to differ

Everything downstream of the candidate pairs, which is what the benchmark measures:

- **How `m` is estimated.** splink runs one EM pass per blocking rule, holding out the
  comparisons that rule blocks on. cpplink runs one session per *column* an EM-safe source
  conditions on, holds out every comparison reading that column, and refuses a session with
  fewer than three free comparisons. Same idea, different bookkeeping.
- **How `u` is estimated.** Both sample 10⁶ random pairs, but cpplink computes `u` in
  closed form from term frequencies for null and exact levels and samples only the rest.
- **Scoring.** cpplink brackets each pattern's TF-adjusted score (`Δ_max`/`Δ_min`) and skips
  the TF tables when the bracket does not straddle the threshold. This is a pruning
  optimisation that is supposed to be exactly equivalent; if the two tools' edge sets differ
  materially, that is a finding.
- **Everything about execution.** duckdb SQL over materialised relations versus a threaded
  streaming fold.

## Tracks

**`matched`** — identical candidate sets. Differences in output are the model code, not the
blocking. This is the parity test.

**`native`** — cpplink additionally turns on its automatic sources (`rare_value` and
`sorted_neighbourhood`) on top of the same exact-value rules. splink 4 *prices* blocking
rules for you — `count_comparisons_from_blocking_rule` and `n_largest_blocks`, which is the
same job `explain-blocking` does — but it does not generate them: the rules are written by
hand, and splink 3's `find_blocking_rules_below_threshold_comparison_count` has no public
counterpart in 4.0.16. So splink's configuration is unchanged here; its two rows are
identical runs and serve as a repeatability check. This track measures what cpplink's
automatic sources buy in recall and what they cost.

## Datasets

All from `splink_datasets`, deduplication only, all well under 1M records.

| Dataset | Records | Entities | Truth pairs | Character |
|---|---:|---:|---:|---|
| `fake_1000` | 1,000 | 251 | 2,031 | Smoke test. Small enough to enumerate every pair. |
| `febrl3` | 5,000 | 2,000 | 6,538 | The FEBRL synthetic dedup set: one original per entity plus up to five corrupted copies, so the error model is typographic and known. |
| `historical_50k` | 50,578 | 5,156 | 303,961 | The main benchmark. Real, messy variation — nicknames, 22% missing `dob`, 9% missing `surname`, 50% missing `occupation`. |

Ground truth is the `cluster` column (`fake_1000`, `historical_50k`) or the entity encoded
in `rec_id` (`febrl3`, where `rec-1496-org` and `rec-1496-dup-3` are one person).

## The metric

Pairwise precision and recall over the **transitive closure** of the partition — a chain
a–b–c asserts a–c whether or not that pair was ever scored. That is stricter than edge
precision and is the number that matters, because the closure is what a user of the output
actually gets.

It is computed from contingency sums rather than by enumerating pairs, so a catastrophic
merge is priced correctly without materialising its pairs. With `n_e` records of true
entity `e`, `n_c` in predicted cluster `c` and `n_ec` in both:

```
truth pairs    = Σ_e  C(n_e, 2)
asserted pairs = Σ_c  C(n_c, 2)
true positives = Σ_ec C(n_ec, 2)
```

Records that neither tool clustered are singletons, and are counted as distinct singletons.
cpplink reports this same number itself via `cluster --truth`; the benchmark recomputes it
from the cluster files so that neither tool's own accounting is taken on trust.

Recall is capped by blocking. `bench.py` records cpplink's `recall` report per dataset, and
because the matched track's candidate sets are provably identical, that cap applies to both
tools in that track.

## Cost, and what it does not mean

Wall clock is the median of `--repeat` runs, split by stage. Peak resident set is
`getrusage`: `RUSAGE_CHILDREN` for cpplink (whose driver only spawns `cpplink`) and
`RUSAGE_SELF` for splink (duckdb runs in-process).

Four honest caveats:

1. **Memory extrapolates only from the largest dataset here, and only for splink.** At
   `fake_1000` and `febrl3` both tools' peak resident sets are dominated by fixed costs —
   the Python interpreter and duckdb for splink, the binary and Arrow's shared-library graph
   for cpplink — and a ratio between them measures startup, not the model. `historical_50k`
   is the one row where the variable cost dominates: at the 222 B/pair marginal rate fitted
   between it and `febrl3`, its 18.3M candidate pairs account for 4.07 GB of splink's 4.49 GB
   peak — 91%, leaving a ~0.4 GB fixed cost that matches `febrl3`'s 448 MB total. There the
   pair table *is* the memory. That consistency is what makes the extrapolation in [the scale
   section](#the-scale-claim-and-where-it-is-still-unmeasured) worth writing down, and it is
   still a line through two points.
2. **Neither does time.** duckdb has a fixed startup cost that dominates at 1,000 records
   and is invisible at 18M. Per-stage timings are reported so the fixed and variable parts
   can be told apart, but a single speedup ratio from this benchmark would be meaningless.
3. **The threshold grid sets the predict cost.** Both tools score everything above the
   grid's lowest threshold, so a wider grid is a more expensive run for both.
4. **splink is not being run the way its authors would run it.** It is constrained to the
   comparison levels cpplink also has, denied its tuned `NameComparison` and
   `DateOfBirthComparison` templates, and given a fixed λ. That is the price of parity, and
   it is a reason not to read the quality table as "which tool is better" — it is
   "do these two implementations of the same model agree".

## Reading the output

Three things to check before believing a row.

**Is the best threshold at the edge of the grid?** If it is, the grid is too narrow and both
tools are understated. `bench.py` prints a warning when this happens. The default grid runs
from 0.1 to 0.99999 because `historical_50k` wants a very high threshold — its optimum is
past 0.999, which a conventional 0.5–0.99 sweep would miss entirely.

**Is the dataset actually discriminating?** `febrl3` includes `soc_sec_id`, which is very
nearly a unique key: blocking on it alone reaches 85.7% of truth pairs and produces almost
nothing else. Both tools score above 0.999 F1 there and the run says little beyond "neither
is broken". `historical_50k` is the dataset that separates them.

**Is the recall ceiling binding?** Blocking recall bounds pipeline recall. cpplink's
`recall` report is captured per run into `timings.json`; if pipeline recall is at that
ceiling, the model is not the limiting factor and a quality difference means nothing.

The predict stage is run at `min(--thresholds)`, so widening the grid downward increases
what both tools score. Both get the same minimum, so the cost comparison stays fair, but
cost numbers from different grids are not comparable to each other.

## Results

Measured 2026-09-04 on darwin/arm64, both tools multithreaded, median of 3 runs, thresholds
swept from 0.1 to 0.99999. Reproduce with `python bench/bench.py --repeat 3`; the full
numbers, including every stage timing, are in `results/summary.json`.

### Quality at each tool's best threshold

| dataset | track | tool | thr | precision | recall | F1 |
|---|---|---|---:|---:|---:|---:|
| fake_1000 | matched | cpplink | 0.1 | 0.9574 | 0.8419 | 0.8960 |
| fake_1000 | matched | splink | 0.5 | 1.0000 | 0.8208 | **0.9016** |
| fake_1000 | native | cpplink | 0.5 | 1.0000 | 0.8621 | **0.9260** |
| febrl3 | matched | cpplink | 0.9 | 1.0000 | 0.9982 | 0.9991 |
| febrl3 | matched | splink | 0.9 | 1.0000 | 0.9982 | 0.9991 |
| febrl3 | native | cpplink | 0.1 | 0.9992 | 0.9992 | 0.9992 |
| historical_50k | matched | cpplink | 0.999 | 0.9212 | 0.8179 | **0.8664** |
| historical_50k | matched | splink | 0.999 | 0.8819 | 0.8209 | 0.8503 |
| historical_50k | native | cpplink | 0.999 | 0.9166 | 0.8096 | 0.8598 |

**The parity test passes.** On identical candidate sets the two tools land within 1.6 points
of F1 everywhere, and on `febrl3` they agree to the fourth decimal. Neither implementation
of the model is broken relative to the other.

**On the dataset that discriminates, the difference is precision, not recall.** On
`historical_50k`/matched both tools recall ~0.82 against a blocking ceiling of 0.8449 —
recall is capped by blocking, not by either model, so it is not where the tools can differ.
Precision is: 0.921 versus 0.882 on the same pairs. That gap is the estimation and scoring
code, and it is the one result here worth investigating further.

### What the ceiling explains

| dataset | track | candidate pairs | blocking recall |
|---|---|---:|---:|
| fake_1000 | matched | 2,538 | 0.7351 |
| fake_1000 | native | 9,981 | 0.8400 |
| febrl3 | matched | 76,509 | 0.9956 |
| febrl3 | native | 110,570 | 0.9983 |
| historical_50k | matched | 18,340,573 | 0.8449 |
| historical_50k | native | 18,436,900 | 0.8499 |

Two things fall out of this table.

**Closure recall can exceed blocking recall, and does.** On `fake_1000`/matched only 73.5%
of truth pairs are ever generated as candidates, yet cpplink recovers 84.2% of them. That is
not a bookkeeping error: a–b and b–c both being scored puts a and c together whether or not
a–c was ever a candidate. Blocking recall bounds *edge* recall; the closure gets the rest
for free. This is why the ceiling is a diagnostic and not a hard cap.

**cpplink's automatic blocking pays where the declared rules are weak, and not otherwise.**
On `fake_1000` it lifts the ceiling from 0.735 to 0.840 and F1 from 0.896 to 0.926 — the
largest quality difference in the whole benchmark. On `historical_50k` it adds 96k candidates
and 0.5 points of ceiling, and F1 goes *down* slightly (0.8664 → 0.8598): the extra
candidates are mostly not matches, so at a fixed threshold they cost precision. Automatic
blocking is worth switching on when the hand-written rules leave recall on the table, which
is a thing this harness can now measure rather than assume.

`febrl3` is a weak discriminator and should not be read as a tie-break: `soc_sec_id` is
nearly a unique key there, blocking on it alone reaches 85.7% of truth pairs, and both tools
score above 0.999.

### Cost

| dataset | track | tool | pipeline s | peak RSS |
|---|---|---|---:|---:|
| fake_1000 | matched | cpplink | 0.28 | 34 MB |
| fake_1000 | matched | splink | 1.54 | 229 MB |
| febrl3 | matched | cpplink | 0.43 | 255 MB |
| febrl3 | matched | splink | 1.93 | 427 MB |
| historical_50k | matched | cpplink | 3.11 | 277 MB |
| historical_50k | matched | splink | 37.51 | 4.2 GB |

Read this table with the three caveats above, and one more that the numbers make concrete:
at 1,000 records splink's 1.54 s is almost entirely duckdb and interpreter startup, which is
a fixed cost that would be invisible at 18M rows. The `historical_50k` row is the only one
where the variable cost dominates, and even there 18.3M candidate pairs is 0.4% of the 5×10⁹
the design targets. **Do not quote a speedup ratio from this benchmark.** What it shows is
that the shapes are as designed — cpplink's memory is flat in the number of pairs and
splink's is not — not by how much that will matter at scale.

## Comparing blocking methods

The benchmark above holds blocking fixed and measures the model. This section does the
opposite: it holds the matcher out entirely and asks which *candidate generator* is worth
running, which is the question a default configuration turns on.

It has to be asked as a curve, not a number. Every method here can buy pair completeness
with candidates — a sorted-neighbourhood window can be widened until it enumerates the file,
a rare-value cap raised until it is standard blocking — so a single operating point per
method compares nothing. `bench/sweep_blocking.py` sweeps each method over its own knob and
reports the **frontier**: at each candidate budget, the method reaching the most known pairs.

```sh
python bench/sweep_blocking.py --datasets historical_50k
```

Each point is one `recall --json` run over a plan holding exactly **one** source. Measuring a
source inside a plan credits it only with what earlier sources left over, which is the right
number for "should I add this" and the wrong one for "which method is better"; both are
wanted, so the sweep runs isolated curves and then a marginal pass against the declared plan.

### Methods

| Family | Knob swept | Prior art |
|---|---|---|
| `exact_value` | the column (one point each) | standard blocking; the baseline every survey uses |
| `sorted_neighbourhood` | window, 2 → 128 | Hernández & Stolfo 1995 |
| `rare_value` | max frequency, 2 → 500 | IDF-weighted canopies, McCallum, Nigam & Ungar 2000 |
| `minhash` | bands, 2 → 32, at 4 rows per band | Broder 1997; Indyk & Motwani 1998 |

None of these is new, which is the point of measuring rather than arguing about them.

### Which methods hold the frontier

168 configurations per dataset, of which the undominated ones are:

| method | historical_50k | febrl3 |
|---|---:|---:|
| `exact_value` | 2 of 8 | 2 of 8 |
| `rare_value` | 29 of 64 | 15 of 64 |
| `sorted_neighbourhood` | 10 of 56 | 6 of 56 |
| `minhash` | 0 of 40 | 1 of 40 |

**MinHash over a single column holds no point of the `historical_50k` frontier and one of
febrl3's, across 40 configurations each.** That is the same result DESIGN.md records from the
recall harness, now measured against every alternative at every budget rather than at one
operating point: bands over one column are correlated in a bimodal population, so the bands
buy candidates without buying independent chances to match. MinHash is not the wrong idea in
general — it is the wrong idea *per column*, and it is what an ANN source over a whole record
would have to beat.

**`rare_value` holds most of the frontier below a mid-sized budget** — 29 of 41 points on
`historical_50k` — and holds pair quality above 0.75 throughout that range. Rare-value
agreement is the cheapest recall in this comparison, which is the argument for automatic
blocking made as a measurement rather than as a claim.

**Sorted neighbourhood owns the top of the frontier and nothing else.** It is the only method
that keeps climbing past PC 0.6 on `historical_50k`, and it pays for it: pair quality falls
from 0.73 at window 2 to 0.037 at window 128.

### The `historical_50k` frontier

Abbreviated to one row per doubling of the budget; the full 41 points are in
`results/blocking/historical_50k.json`.

| candidates | PC | PQ | method | column | knob |
|---:|---:|---:|---|---|---:|
| 1 | 0.0000 | 1.0000 | `rare_value` | gender | 2 |
| 7 | 0.0000 | 1.0000 | `rare_value` | gender | 5 |
| 43 | 0.0001 | 0.9070 | `rare_value` | occupation | 2 |
| 118 | 0.0004 | 1.0000 | `rare_value` | gender | 20 |
| 298 | 0.0009 | 0.9195 | `rare_value` | birth_place | 2 |
| 1,107 | 0.0023 | 0.6242 | `rare_value` | dob | 2 |
| 2,626 | 0.0066 | 0.7593 | `rare_value` | occupation | 10 |
| 6,577 | 0.0197 | 0.9104 | `rare_value` | surname | 5 |
| 14,481 | 0.0415 | 0.8707 | `rare_value` | birth_place | 10 |
| 34,023 | 0.0975 | 0.8713 | `rare_value` | surname | 10 |
| 69,070 | 0.2144 | 0.9433 | `rare_value` | first_and_surname | 10 |
| 174,594 | 0.3611 | 0.6286 | `sorted_neighbourhood` | birth_place | 4 |
| 368,468 | 0.5483 | 0.4523 | `sorted_neighbourhood` | surname | 8 |
| 1,473,488 | 0.6684 | 0.1379 | `sorted_neighbourhood` | surname | 32 |
| 5,887,808 | 0.7071 | 0.0365 | `sorted_neighbourhood` | surname | 128 |

No single source exceeds PC 0.71, while the declared plan's union reaches 0.845 — the sources
are complementary, which is the case for a union rather than for picking a winner.

### What the declared plan was missing

The marginal pass found something the quality benchmark above could not: the `matched` track's
blocking rules leave out `birth_place`, and adding almost any source on it moves pair
completeness from 0.845 to over 0.92.

`recall --why` reaches the same conclusion independently, and says *which kind* of source to
use — `birth_place` agrees **exactly** on 49.7% of the missed pairs, not fuzzily, so this is a
column the plan simply omitted rather than one needing a similarity measure:

```text
Of those 47154, what still agrees (a column agreeing here and not blocked on is free recall)
Comparison                   exact     share       fuzzy     share    blocked?
------------------------------------------------------------------------------
birth_place                  23441    49.71%         300     0.64%          no
occupation                    8620    18.28%           0     0.00%          no
gender                       19815    42.02%           0     0.00%          no
```

Three ways to block on it, added to the declared plan:

| added source | PC | candidates | added |
|---|---:|---:|---:|
| `exact_value birth_place` | 0.9220 | 23,691,110 | +4,923,790 |
| `rare_value birth_place`, cap 100 | 0.8909 | 19,299,435 | +532,115 |
| `sorted_neighbourhood birth_place`, window 128 | 0.9233 | 24,346,392 | +5,579,072 |
| *(declared plan alone)* | 0.8449 | 18,767,320 | — |

Sweeping the rare-value cap prices the trade directly:

| max frequency | PC | added candidates | known pairs gained | gained per 1k candidates |
|---:|---:|---:|---:|---:|
| 20 | 0.8601 | 46,776 | 4,624 | 98.9 |
| 50 | 0.8710 | 158,269 | 7,949 | 50.2 |
| 100 | 0.8909 | 532,115 | 14,007 | 26.3 |
| 200 | 0.9071 | 1,078,853 | 18,917 | 17.5 |
| 400 | 0.9136 | 1,498,634 | 20,904 | 13.9 |
| 800 | 0.9187 | 2,229,109 | 22,436 | 10.1 |
| 3200 | 0.9220 | 4,923,790 | 23,441 | 4.8 |

**A cap of 800 reaches 99.6% of what standard blocking on the same column reaches, for 45% of
its candidates.** That is the concrete case for rare-value blocking over standard blocking on
a mid-cardinality column, and it is a number rather than an intuition.

### What this does not settle

The comparison is between methods this repository implements, on two datasets under 51k
records. It does not include the learned blocking schemes (Michelson & Knoblock 2006; the
`dedupe` library), schema-agnostic token blocking and meta-blocking (Papadakis et al.), or
embedding-plus-ANN blocking (DeepBlocker, Sparkly), any of which could hold parts of this
frontier. It also cannot speak to the ANN source phase 6 plans, whose whole-record signature
is exactly the case single-column MinHash fails at here — and which, per DESIGN.md stage 2,
may feed prediction but not estimation.

## Layout

```
bench/
  datasets.py      the neutral description; both configurations compile from it
  prepare.py       splink_datasets -> parquet + ground truth + cpplink schemas
  run_cpplink.py   drives the cpplink binary, stage by stage
  run_splink.py    drives splink, stage by stage
  score.py         the one scorer, used for both
  bench.py         orchestration, repeats, the parity check, the report
  sweep_blocking.py  the blocking-method frontier: one curve per method per knob
  schemas/         generated cpplink schemas, one per dataset per track
  data/            generated inputs (gitignored)
  results/         generated outputs (gitignored)
```

## Running it

Needs cpplink built at `build/cpplink`, and a Python environment with splink 4, duckdb,
pandas and pyarrow:

```sh
conda create -n splink -c conda-forge python splink duckdb pandas pyarrow
conda activate splink

python bench/prepare.py                   # download, convert, write ground truth
python bench/bench.py --verify            # assert the candidate sets match
python bench/bench.py --repeat 3          # the benchmark
python bench/sweep_blocking.py           # the blocking-method frontier
```

`sweep_blocking.py` needs only the cpplink binary and the prepared parquet — no splink.

Useful subsets:

```sh
python bench/bench.py --datasets fake_1000 --tracks matched --repeat 1
python bench/bench.py --threads 1         # both tools single-threaded
python bench/run_splink.py historical_50k --out /tmp/s --estimate-lambda
python bench/score.py bench/data/febrl3.entities.csv <clusters.csv>
```

`bench.py` writes `results/summary.json` with every stage timing, every threshold and every
quality number, and prints three markdown tables: quality at each tool's best threshold,
cost, and the F1 sweep across thresholds.

## The scale claim, and where it is still unmeasured

Phase 6 has landed — the spill path, link mode and the signature filter are all in — so the
harness this section used to defer is no longer blocked on the pipeline. It is blocked on
hardware: the machine these numbers were taken on has 16 GB of RAM and 2.4 GB of free disk,
and an experiment that runs splink until it fails needs room for splink to fail *in*. Running
duckdb to exhaustion against a nearly full filesystem risks the machine, not just the run.

What can be said without that experiment is more than the caveats above admit, because the
three datasets already bracket the shape. Peak resident set per candidate pair:

| dataset | candidate pairs | cpplink | splink |
|---|---:|---:|---:|
| `fake_1000` | 2,538 | 14,086 B/pair | 94,463 B/pair |
| `febrl3` | 76,509 | 3,490 B/pair | 5,858 B/pair |
| `historical_50k` | 18,340,573 | **16 B/pair** | **245 B/pair** |

**cpplink's cost per pair falls by three orders of magnitude across this range and splink's
converges to a constant.** That is the design claim, visible in data already collected: for
cpplink these are not per-pair costs at all — the memory is the record store and the pattern
histogram, both set by the number of *records*, so dividing by pairs just measures how many
pairs the same store produced. For splink the memory *is* the pair table, so the ratio
converges to the width of a row in it.

Fitting splink's marginal cost between the two larger datasets gives **222 bytes per
candidate pair**, which turns the scale claim into an arithmetic prediction rather than an
assertion:

| candidate pairs | predicted splink peak RSS |
|---:|---:|
| 100M | 22 GB |
| 1×10⁹ | 222 GB |
| 5×10⁹ *(the 18M-record design target)* | ~1.1 TB |

On this schema `historical_50k`'s 50,578 records yield 18.3M candidates, so 16 GB is
exhausted at roughly 72M candidates — **somewhere near 100k records**. That is a falsifiable
prediction on ordinary hardware, and it is the experiment to run first, on a machine with
disk to spare: sweep `cpplink gen-sample` upward, run both tools at each size, and record the
size at which splink stops finishing. Predicting where a tool breaks is not the same as
watching it break, and the second is what this section will hold when the disk exists.

Two limits stay honest about what even that would establish. The extrapolation is linear in
candidates from two points and assumes duckdb's per-row width does not change with scale or
spill strategy — splink spilling to disk does not fail, it slows, and "fails" would need
defining as a wall-clock budget rather than an OOM. And the 18M-record claim is only
partly measured: DESIGN.md's phase 2 priced and enumerated blocking at 18M synthetic rows
(63.5×10⁹ candidates, 8.25×10⁹ enumerated in 142 s single-threaded), but **nothing from
phase 6 has been validated at that size** — the signature filter, the spill path and link
mode were all measured at 1–1.8M — and no end-to-end run at 18M has been done at all.
