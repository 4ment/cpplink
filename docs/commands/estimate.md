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
                 [--interactions] [--max-interactions N] [--interaction-bits F]
                 <file.parquet>...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required; must declare both `comparisons` and `blocking` |
| `--out <file>` | none | write the model as JSON. Without it the model is printed and discarded |
| `--u-sample N` | 1,000,000 | uniformly random pairs drawn to estimate `u` for levels the closed form cannot reach |
| `--session-pairs N` | 10,000,000 | per-session cap on pairs actually *compared*. Above it, the fold becomes a Bernoulli sample |
| `--threads N` | hardware | threads for the histogram fold |
| `--iterations N` | 500 | EM iteration cap. Reaching it is a symptom, not a setting to raise: a session that will not converge is usually one whose blocking conditions on a column it is trying to estimate |
| `--lambda F` | derived | override the prior match rate with a count you trust |
| `--seed N` | 20260903 | seed for `u` sampling and session subsampling |
| `--fuzzy-u` | off | compute `u` for the fuzzy levels exactly, by self-joining each column's dictionary, instead of sampling and rescaling |
| `--ball-budget N` | 4e10 | value pairs the self-join may look at for one column |
| `--no-tie-holdout` | off | stop holding out comparisons tied to the column a session blocks on. Off means the hold-out is on; this is for measuring what it is worth |
| `--tied-bits F` | 0.25 | `u`-side overlap in bits past which two columns count as tied |
| `--tie-sample-rows N` | 500,000 | rows the tie pass reads. `0` reads every row |
| `--interactions` | off | fit two-way corrections for the comparison pairs that are not conditionally independent. See [below](#relaxing-conditional-independence) |
| `--max-interactions N` | 2 | how many terms may enter the model. Implies `--interactions` |
| `--interaction-bits F` | 0.25 | bits a term must move an average matching pair by, in every session that can see it, to be admitted. Implies `--interactions` |
| `--min-interaction-sessions N` | 2 | sessions that must be able to see a term before it is identified |
| `--min-pairs-per-parameter F` | 0 | refuse a term with fewer matching pairs per free parameter than this. Reported either way |
| `--interaction-clamp F` | 6.0 | the most any single cell of a term may move the weight, in bits |
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
\(\sum_v c_v(c_v-1) \,/\, N(N-1)\) or as a null count. The sampled levels take what the exact
ones leave, so the two cannot disagree about the total. See
[`u` in closed form](../em.md#u-in-closed-form), which explains why the numerator is
\(c_v(c_v-1)\) and not \(c_v^2\).

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

`also tied` is the same discipline one step wider, and it is not optional:

```text
Session first_name
  sources    first_name exact_value
  held out   first_name
  also tied  first_and_surname   (blocking on this column conditions on theirs)
```

Blocking on a column conditions on **everything that column decides**, not only on the column
itself.
`first_and_surname` contains `first_name`, so among the pairs this session blocks on, two rows
named "john smith" and "john brown" clear `jaro_winkler >= 0.80` on the strength of the shared
first token alone.
Their agreement rate on `first_and_surname` inside the session is nothing like the `u` the
model holds for it over the whole file, and EM has no way to read the difference as anything
but evidence that the pairs are matches.

Measured on `historical_50k`: **the `first_name` session's match rate reads 0.9996 where the
truth is 0.0080**, a factor of 125, and it reads 0.018 once `first_and_surname` is held out
too.
End to end that is worth **0.8536 to 0.8676 F1**.
Ties are found by one pairwise pass over the rows, the same one [`profile`](profile.md) runs,
which reads no candidate pair and no model; two columns count as tied when one occurs inside
the other on 90% of rows or their `u`-side overlap is worth `--tied-bits`, 0.25 by default.
`--no-tie-holdout` turns it off, which is how the two numbers above were measured against each
other.

A column tied to the blocked column in *every* session ends up with no estimate at all, and the
run says so with the `held out of every session` warning below.
That is the cost of the discipline and it is the right price: an `m` from a session that
conditioned on the answer is worse than no `m`.

Warnings that can appear here, and what each means:

| Warning | Meaning |
| --- | --- |
| `only N comparison(s) are free in this session` | Fewer than three free comparisons does not identify the mixture — a 2×2 table has two exact roots and EM returns whichever it walked to. **The session is refused, not merged.** Give the schema more comparisons that do not read that column |
| `every comparison reads this column` | The session can learn nothing and is skipped |
| `this session's m estimates were not merged into the model` | Follows any of the above, or the label-swap check firing |
| `comparison "x" was held out of every session` | Its `m` is a starting value, not an estimate. Add a blocking source on a different column, or break the tie that keeps holding it out |

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

## Relaxing conditional independence

Fellegi–Sunter multiplies `m` and `u` across comparisons, which is correct only where agreement
is independent *given the class*.
On real data it is not: where one column contains another, the model counts one signal twice
and every matching pair is scored too high.
`--interactions` fits a small set of two-way corrections and writes them into the model.

The weight becomes

\[
w(\gamma) \;=\; \log_2\frac{\lambda}{1-\lambda}
\;+\; \sum_c \log_2 \frac{m_c(\gamma_c)}{u_c(\gamma_c)}
\;+\; \sum_{(c,d) \in S} \delta_{cd}(\gamma_c, \gamma_d)
\]

\[
\delta_{cd}(i,j) \;=\;
\log_2 \frac{M_{cd}(i,j)}{m_c(i)\,m_d(j)}
\;-\;
\log_2 \frac{U_{cd}(i,j)}{u_c(i)\,u_d(j)}
\]

**Both halves are needed.** Two columns that agree together among matches are double-counted
only to the extent that they do not also agree together among non-matches, which is what the
main-effect `u` already prices.

Nothing about scoring changes: \(\delta\) is a function of γ like everything else, so below 22
bits of γ it is folded into the tabulated base weight once at bind time and the hot loop never
sees it.

### Where each half comes from

| Half | Source | Why not somewhere else |
| --- | --- | --- |
| M | the session histograms, weighted by the responsibility EM already computed, averaged over the sessions that left both comparisons free | it is the fit that already exists; nothing new is enumerated |
| U | the uniformly random pairs `u` is drawn from | **never candidates.** A source selecting on a column induces agreement correlation involving that column, so dependence measured on the candidate stream measures the blocking plan |

Both are then fitted to the model's own margins by iterative proportional fitting, which moves
the margins and leaves every odds ratio alone.
That is what makes the term a *pure* correction: adding it cannot shift `m` or `u`, so the main
effects keep carrying the margins and the interaction carries only the association.

### Reading the report

```sh
cpplink estimate --schema historical_50k.json --interactions historical_50k.parquet
```

```text
Two-way interactions, ranked by what they move a match by. 0.0 s.
The u side had 2.71e-04 of its pairs subtracted as this file's own duplicates.

Comparison          Comparison              effect  per match  weakest    dup u  pairs/par         G^2         p  verdict
------------------------------------------------------------------------------------------------------------------------
first_name          first_and_surname         4.26      -2.87    -2.86      31%       6164    264053.2   0.0e+00  fitted
surname             first_and_surname         3.58      -3.14    -3.11      43%       6164     63821.2   0.0e+00  fitted
first_name          surname                   1.80      -1.63    -1.58      43%       6164     16386.6   0.0e+00  past --max-interactions
dob                 postcode_fake             0.61      -0.19    -0.04      26%      17742    102502.3   0.0e+00  one session moves a match by only -0.04 bits
first_and_surname   dob                       0.52       0.16     0.16      49%       6508      5722.7   0.0e+00  only 1 session can see it, so nothing can disagree
------------------------------------------------------------------------------------------------------------------------
2 of 28 candidate pairs entered the model; 4 cell(s) hit the clamp.
```

| Column | Meaning |
| --- | --- |
| `effect` | the mean **absolute** correction an average matching pair gets, in bits. This is what the ranking is on |
| `per match` | the same thing signed. Negative means the plain model was double-counting that many bits on every match |
| `weakest` | the least any one session says on its own, zero where two disagree about the sign. A term has to clear `--interaction-bits` here, not just on the average |
| `dup u` | how much of the `u` side was this file's own duplicates before they were subtracted. See below |
| `pairs/par` | matching pairs per free parameter of the term |
| `G^2`, `p` | the deviance against independence, printed as a diagnostic and used for nothing |

The two terms admitted here are exactly the two pairs
[`profile`](profile.md) names from the rows alone — `first_and_surname` contains both name
columns — reached from the histograms with no notion of containment at all.

!!! warning "`G^2` cannot choose the number of terms, for the same reason it could not in `levels` or `simplify`"
    Its power is the size of the run. All 28 candidate pairs here read `p = 0` at 18M
    candidates. The effect size decides: `--interaction-bits` sets the bar and
    `--max-interactions` caps the count.

### What it is worth

Measured end to end at the shipping defaults, against truth the estimator never sees:

| Dataset | Terms | Plain best F1 | Corrected best F1 |
| --- | ---: | ---: | ---: |
| `historical_50k` | 2 | 0.8676 at 12 bits | **0.9107 at 6 bits** |
| `febrl3` | 0 | 0.9992 at −4 bits | 0.9992 at −4 bits |
| `fake_1000` | 2 | 0.9386 at −1 bits | **0.8724** at −3 bits |

On `historical_50k` precision *and* recall both rise (0.9259 → 0.9542, 0.8161 → 0.8709) and the
corrected model dominates the plain one at every recall on the frontier.
`febrl3` fits nothing and the model is byte-identical, which is the right answer rather than a
null result: its corruption is applied field by field on independent coins, so there is no
dependence to find.

!!! danger "Thresholds move, so comparing at a fixed one compares nothing"
    Each admitted term takes two to three bits off every matching pair, so the whole weight
    scale shifts down and the operating point moves with it — 12 bits becomes 6. Read at a
    *fixed* 12 bits the correction on `historical_50k` looks like −0.10 F1. Compare
    best-against-best, or compare the precision-recall frontiers.

The number of terms is capped at two because that is what measured best:

| Terms | 1 | **2** | 3 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `historical_50k` best F1 | 0.8878 | **0.9135** | 0.9091 | 0.9035 | 0.8847 |

Past the real dependencies there is nothing left to correct and the extra terms fit noise.

### The cost, which is the ceiling and not the scoring

A correction is not separable across comparisons, so the pair's own cell is unknown until its
levels are, and [`predict`](predict.md)'s pair-global ceiling has to add each term's *largest*
cell to stay admissible. It gets looser:

| | Candidates skipped before any metric | `predict` |
| --- | ---: | ---: |
| plain, 12 bits | 37.61% | 1.8 s |
| corrected, 6 bits | 4.06% | 2.2 s |

At the same threshold the skip rate falls 29.09% to 4.06%, so this is the loosening and not the
lower operating point. The tabulated weight itself costs nothing at all.

`predict --no-interactions` scores the plain model out of the same file, which is how the two
were measured against each other.

### Two traps, one of them the standing limit

!!! danger "The `u` side of a joint is the file's own duplicates"
    A uniformly random pair is a match with probability λ, so a random draw shows
    \((1-\lambda)u + \lambda m\). For one column that is a small inflation. For the *joint* of
    two high-cardinality columns it is the entire measurement: on `febrl3` two independent
    draws land on the `date_of_birth` × `soc_sec_id` exact-agreement cell 0.21 times per
    million, and the 452 matching pairs in a million random draws land there 335 times — **1,565×
    the independent rate, all of it duplicates**.

    The first working version therefore put a −4 bit correction on *every one* of that dataset's
    28 pairs. λ·m is now subtracted back out, and where the subtraction leaves nothing the pair
    is refused rather than guessed. That is the `dup u` column, and it silences `febrl3`
    entirely.

!!! warning "The correction is only as good as λ, and λ is a lower bound"
    A session-derived λ counts only the matches blocking reached. On `fake_1000` it reads
    1.77e-3 against a true 4.07e-3, low by 2.3×, so the subtraction removes less than half the
    duplicates it should and the residue pushes every correction negative. Given the true λ on
    the command line, that file's two largest terms **flip sign** (−1.90, −1.93 → +1.26, +1.06)
    and best F1 goes 0.8724 → 0.9225.

    The report warns whenever λ is a bound. There are two ways out: pass `--lambda` from a count
    you trust, or divide the reported one by the pair completeness
    [`completeness`](completeness.md) estimates with no truth file.

!!! note "Letting a correction run to its full size is worse, and the clamp is why"
    The arithmetic wants about −10 bits on the `surname` × `first_and_surname` exact-agreement
    cell; `--interaction-clamp` holds it at −6. Raising the clamp does not help:

    | clamp (bits) | 4 | **6** | 8 | 12 | 20 |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | `historical_50k` best F1 | 0.9073 | **0.9107** | 0.9061 | 0.9042 | 0.9042 |

    That cell is 43% duplicates before the subtraction, so its magnitude is the least reliable
    part of the fit even though its *sign* is certain. The clamp is a shrinkage on exactly the
    cells whose estimate is worst. The curve is shallow across the whole range, so the default
    is not load-bearing — but removing the clamp is a measurable loss.

Two guards are not optional. A cell the data barely reaches is a ratio of two floors — a column
duplicated under two names can never land on "one agrees and the other does not" — so each
correction is scaled by \(n/(n+5)\), the contingency table's own bar for an expected count; the
empty cell naming that impossible case was reading +3.7 bits before it. And a term must be
visible to two sessions, which is the interaction form of the `sessions == 0` refusal
[`completeness`](completeness.md) already makes.

!!! note "Cross-session *agreement* does not work as a guard, and it was the obvious thing to try"
    The hypothesis was that a spurious term would read differently in each session and a real
    one would not. On `fake_1000` the sessions agree to within 0.06 bits on corrections that
    have the wrong sign. The bias is systematic, so nothing that looks for noise can find it.
