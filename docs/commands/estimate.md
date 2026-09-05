# `estimate`

**Goal:** learn the model — `m`, `u` and `λ` — from the data itself, and write it to
`model.json`.

The concepts behind this command are on [Estimation and EM](../em.md); this page is the
interface and how to read what it prints.

## Synopsis

```sh
cpplink estimate --schema <schema.json> [--out <model.json>]
                 [--u-sample N] [--session-pairs N] [--threads N]
                 [--iterations N] [--lambda F] [--seed N] [--mode MODE]
                 <file.parquet>...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required; must declare both `comparisons` and `blocking` |
| `--out <file>` | none | write the model as JSON. Without it the model is printed and discarded |
| `--u-sample N` | 1,000,000 | uniformly random pairs drawn to estimate `u` for levels the closed form cannot reach |
| `--session-pairs N` | 10,000,000 | per-session cap on pairs actually *compared*. Above it, the fold becomes a Bernoulli sample |
| `--threads N` | hardware | threads for the histogram fold |
| `--iterations N` | 500 | EM iteration cap. Never approached in practice |
| `--lambda F` | derived | override the prior match rate with a count you trust |
| `--seed N` | 20260903 | seed for `u` sampling and session subsampling |
| `--fuzzy-u` | off | compute `u` for the fuzzy levels exactly, by self-joining each column's dictionary, instead of sampling and rescaling |
| `--ball-budget N` | 4e10 | value pairs the self-join may look at for one column |
| *(positional)* | — | required; the parquet file |
| `--mode dedup\|link\|link-and-dedup` | link with more than one file, else dedup | which pairs to enumerate; see [linking](../linking.md) |

The two knobs that matter are `--u-sample` and `--session-pairs`. Raising `--u-sample` buys
precision on fuzzy-level `u` values only — exact and null levels are closed form and unaffected
by it. Raising `--session-pairs` buys very little: 10⁷ pairs already saturates the pattern
counts, and estimation needs *counts*, not pairs.

## Example

```sh
cpplink estimate --schema examples/sample_schema.json \
                 --out model.json examples/sample.parquet
```

```text
u from 999,999 random pairs in 0.5 s, plus 14 levels in closed form from the term frequencies

Session               Enumerated      Compared  Patterns  Iters Match rate  Seconds
-----------------------------------------------------------------------------------
email                     90,600        90,600       242      3   0.999934      0.0
phone                    121,195       121,195       274      2   0.870795      0.1
dob                   77,931,552    10,002,206       270      4   0.001535      3.5
last_name             39,059,914     7,413,527       553     10   0.003205      3.2
-----------------------------------------------------------------------------------

Session dob
  sources    dob exact_value
  held out   dob
  sampled    1.28e-01 of 77,931,552 enumerated pairs
  em         4 iterations, last change 4.5e-11, converged
  implies    119630 matching pairs among the 77,931,552 it reached

Session last_name
  sources    last_name rare_value, last_name sorted_neighbourhood
  held out   last_name
  sampled    1.90e-01 of 39,059,914 enumerated pairs
  em         10 iterations, last change 3.9e-07, converged
  implies    125201 matching pairs among the 39,059,914 it reached

warning: lambda is a lower bound: it counts only the matches a session's blocking reached,
and blocking recall is below one. Pass --lambda to set it from a count you trust.

lambda       7.73e-08  (lower bound: session last_name implies 125,201 matches over 1,619,999,100,000 pairs)
prior weight -23.625 bits

Comparison        Level                                m             u    weight  note
--------------------------------------------------------------------------------------------
last_name         null                          1.00e-12      1.00e-12      0.00  unreachable: no pair can land here
                  exact                         0.609559      2.38e-05     14.64  u exact
                  jaro_winkler >= 0.92          0.279753      6.00e-06     15.51  u from only 6 random pairs
                  jaro_winkler >= 0.85          0.078072      0.000209      8.55
                  else                          0.032616      0.999761     -4.94
email             null                          0.335455      0.055215      2.60  u exact
                  exact                         0.577950      5.96e-07     19.89  u exact
                  jaro_winkler >= 0.93          0.086592      5.00e-07     17.40  u below the sampling floor
                  else                          3.46e-06      0.944784    -18.06  m at the floor: no matching pair reached this level
--------------------------------------------------------------------------------------------
Weight is log2(m/u): the bits of evidence agreeing at that level carries.

Wrote model.json
```

## Reading the output

### The `u` line

```text
u from 999,999 random pairs in 0.5 s, plus 14 levels in closed form from the term frequencies
```

Two sources of truth, deliberately. **14 levels needed no sampling at all**: every `exact`
level on a single interned column, and every leading `null` level, is computed exactly as
\(\sum_v p_v^2\) or as a null count. The sampled levels take what the exact ones leave, so the
two cannot disagree about the total. See [`u` in closed form](../em.md#u-in-closed-form).

### The session table

One row per column that an EM-safe blocking source conditions on. Each session holds that
column out and estimates every other comparison's `m` from it.

| Column | Meaning |
| --- | --- |
| `Enumerated` | candidate pairs the session's sources produced |
| `Compared` | pairs actually evaluated. Below `Enumerated` means Bernoulli sampling kicked in at `--session-pairs` |
| `Patterns` | **distinct** γ values in the histogram — the entire input to EM |
| `Iters` | EM iterations to `max|Δ| < 1e-6` |
| `Match rate` | the session's own λ, i.e. the posterior match mass among *its* pairs |
| `Seconds` | wall time for enumeration, comparison and EM together |

Three things to notice:

- **`Patterns` is tiny.** 242 to 553 distinct patterns is the whole reason EM is free here.
  The cost of EM is `patterns × comparisons × iterations` and has nothing to do with how many
  pairs were folded.
- **`Compared` ≪ `Enumerated` is fine and intended.** The `dob` session sampled 12.8% of its
  78M pairs and still converged in 4 iterations. Since `m`, `u` and `λ` are all *ratios* of
  pattern counts, a Bernoulli sample needs no reweighting.
- **`Match rate` differs wildly by session and that is expected.** The `email` session is
  99.99% matches — agreeing on an exact email is nearly conclusive — while `dob` is 0.15%.
  A session at ~1.0 gets a note saying its `m` is close to a direct count over blocked pairs
  rather than a mixture split.

### The per-session detail

```text
Session last_name
  sources    last_name rare_value, last_name sorted_neighbourhood
  held out   last_name
  sampled    1.90e-01 of 39,059,914 enumerated pairs
  em         10 iterations, last change 3.9e-07, converged
  implies    125201 matching pairs among the 39,059,914 it reached
```

`held out` is the identifiability mechanism: every comparison reading that column is excluded
from this session's EM, because blocking selected on it and its `m` is degenerate. `implies`
is what feeds λ.

Warnings that can appear here, and what each means:

| Warning | Meaning |
| --- | --- |
| `only N comparison(s) are free in this session` | Fewer than three free comparisons does not identify the mixture — a 2×2 table has two exact roots and EM returns whichever it walked to. **The session is refused, not merged.** Give the schema more comparisons that do not read that column |
| `every comparison reads this column` | The session can learn nothing and is skipped |
| `this session's m estimates were not merged into the model` | Follows any of the above, or the label-swap check firing |
| `comparison "x" was held out of every session` | Its `m` is a starting value, not an estimate. Add a blocking source on a different column |

### λ and the prior weight

```text
lambda       7.73e-08  (lower bound: session last_name implies 125,201 matches over 1,619,999,100,000 pairs)
prior weight -23.625 bits
```

λ is put back on the **full-data** pair count, never a session's own match rate, which
blocking biases upward by exactly the thing blocking is for. It is a **lower bound** twice
over — blocking recall is below one, and EM's posterior mass is itself conservative. `prior
weight` is \(\log_2(\lambda/(1-\lambda))\): the −23.6 bit hurdle every pair starts behind.

Pass `--lambda` if you have a count you trust; it is the single parameter most worth
overriding, since it shifts every score by a constant.

### The parameter table

The model itself, one block per comparison, one row per level.

| Column | Meaning |
| --- | --- |
| `m` | \(P(\text{level} \mid \text{match})\) — how reliable the field is |
| `u` | \(P(\text{level} \mid \text{non-match})\) — how discriminating it is |
| `weight` | \(\log_2(m/u)\), the bits of evidence this level carries |
| `note` | provenance and floors, **never hidden** |

The `note` column is the part to read carefully. Four kinds appear:

| Note | Meaning |
| --- | --- |
| `u exact` | closed form from term frequencies. Trust it |
| `u from only N random pairs` | sampled, and thinly. The weight is real but noisy |
| `u below the sampling floor` | the sample never hit this level; `u` is half a sampled pair. **These are the least trustworthy numbers in the model** |
| `m at the floor: no matching pair reached this level` | no matching pair landed here; `m` is `0.5 / matching mass` — half a pair, the resolution of the evidence |
| `unreachable: no pair can land here` | closed-form `u` is exactly zero, e.g. the null level of a column with no missing values. Weight is set to **exactly 0** rather than floored |

In the run above, `location within 1.00 km` at 20.83 bits and `address exact` at 20.08 bits
both rest on `u` below the sampling floor. They are the weights to distrust, and raising
`--u-sample` is the way to firm them up — geo radii and list-set equality are precisely what
the closed form cannot reach.

## Output

With `--out`, a `model.json` carrying per-level `m`, `u`, whether `u` was exact, how many
sampled pairs supported it, the EM support mass behind `m`, plus `lambda`, `lambda_basis` and
`records`. See [Output formats](../reference/output-formats.md#modeljson).

Because the model is a small JSON file, it can be hand-edited: overriding a suspect `u`, or a
weight you have external knowledge about, needs no re-run.

## Exact `u` for the fuzzy levels

`u` is closed form for a leading null level and for a single-column exact level, and *sampled*
for everything else, then rescaled to fill what the exact levels leave.
A sample of a million random pairs sees a fuzzy level that fires on one pair in 300,000 about
three times, which is not an estimate of anything.

`--fuzzy-u` computes it instead. `u` for a level is the chance two random records land on it,
which is a sum over pairs of *values*:

```
u_l = sum over ordered value pairs landing on l of p_v * p_w
```

and the dictionary self-join enumerates exactly that. Measured on the 1M sample:

| Comparison | Level | Sampled `u` | Exact `u` |
| --- | --- | ---: | ---: |
| `first_name` | levenshtein ≤ 1 | 1.320e-03 | 1.306e-03 |
| `first_name` | jaro_winkler ≥ 0.88 | 6.140e-04 | 6.072e-04 |
| `last_name` | jaro_winkler ≥ 0.85 | 1.810e-04 | 1.985e-04 |
| `last_name` | jaro_winkler ≥ 0.92 | 3.000e-06 | **6.389e-06** |

Where the sample has support it was already within 1%. Where it does not — the last row, which
the sample hit about three times — it is **wrong by a factor of 2.1**, which is 1.1 bits on
every pair that lands there.

The cost is quadratic in the number of distinct values: 11.2 billion value pairs for the 1M
sample's 149k surnames took 140 s on eight threads, and 16k first names took 0.6 s.
A near-unique column is refused by `--ball-budget` rather than approximated — 919k distinct
emails is 423 billion value pairs, and no signature filter makes that affordable.
