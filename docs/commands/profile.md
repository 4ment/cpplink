# `profile`

**Goal:** find out what the columns can be worth, and which pairs of them are the same evidence
twice, before any model exists to be wrong about it.

Every other stage costs minutes or hours and each one assumes things about the columns that
nobody has checked.
This command checks two of them.
It needs no model, no blocking plan, no known pairs, and it enumerates no candidate pair, so it
runs in seconds and is the first thing to point at a new file.

## Synopsis

```sh
cpplink profile --schema <schema.json> [--sample-rows N] [--no-pairs]
                [--expected-matches N] [--threads N] [--json] [--mode MODE]
                <file.parquet>...
```

| Option | Meaning |
| --- | --- |
| `--schema <file>` | required; only the `columns` section is read, so a schema with no comparisons and no blocking plan is enough |
| `--sample-rows N` | rows the pairwise pass reads, default 2,000,000; `0` reads every row |
| `--no-pairs` | skip the pairwise pass and report the ledger and the per-column table only |
| `--expected-matches N` | matching pairs assumed present, which is what the prior odds are made of; default is one duplicate per record |
| `--threads N` | default is the hardware concurrency |
| `--json` | the same numbers as JSON |
| `--mode` | `dedup`, `link` or `link-and-dedup`; sets which pair space the prior odds are taken over |
| *(positional)* | required; one or more parquet files |

## The two questions

### One: can these columns separate the classes at all?

A comparison is worth `log2(m/u)` bits when it agrees.
For an exact-match level `u` is the rate two random rows carry the same value, which is
`sum_v p_v^2` over the term-frequency table the loader already built, and `m <= 1` always.
So the most a column can ever be worth is

```
bits(c) = -log2(sum_v p_v^2)
```

and its inverse `1 / sum_v p_v^2` is the **effective cardinality**: the number of equally likely
values the column behaves as if it had.
That is the number the distinct count in [`inspect`](inspect.md) is a bad proxy for.
900,000 distinct surnames with one value on 40% of the rows behaves as about six.

Weight each column's ceiling by the chance both rows carry a value, set the sum against the
prior odds that the pair space imposes, and the result is a ledger that ends in a margin.
A negative margin means no model and no blocking plan will ever separate these classes, which
is worth knowing in the first minute rather than the second day.

### Two: how much of that evidence is the same evidence twice?

Fellegi-Sunter assumes agreement is independent across comparisons conditional on match status.
The U half of that assumption is a property of the value distributions alone, so it needs no
matches and no sampling of pairs.
The pairwise version of the same closed form is

```
u_cd = sum over (v,w) of p_vw^2        phi_cd = u_cd / (u_c * u_d)
```

over the joint value table of two interned columns, and `log2(phi_cd)` is the number of bits the
score double-counts.
It is reported in bits rather than as a correlation because bits are the correction term that
would appear in the weight, which is what makes it actionable.

Two structural checks ride the same pass.
**Determination** is the share of rows kept by mapping each value of one column to its commonest
partner in the other, which is 1 exactly when the first column determines the second.
**Containment** is the share of rows on which one value occurs literally inside the other, which
is the detector for a column built out of another.

## Reading the output

```text
Records      50,578
Mode         all pairs
Rows read    50,578  (all)
Time         0.0 s  (8 threads)

Evidence ledger
  Pair space       1.279e+09 pairs     30.25 bits
  Match rate       3.954e-05                   (50,578 matching pairs assumed)
  Prior odds                          -14.63 bits
  Available                           +47.30 bits
  Redundant                            -1.29 bits   (pairwise, u side only)
  Margin                              +31.38 bits

Column            Type            Distinct     Null  Top val    Eff vals     Bits  Cov bits
-------------------------------------------------------------------------------------------
first_name        string             4,413    0.13%    5.50%       77.91     6.28      6.27
surname           string             6,195    8.93%    1.34%        1447    10.50      8.71
first_and_surname string            20,479    0.13%    1.07%        4862    12.25     12.21
dob               string             8,985   22.55%    1.52%       495.3     8.95      5.37
postcode_fake     string            12,363   22.59%    0.09%        6833    12.74      7.63
birth_place       string             2,373   13.70%    5.32%       193.5     7.60      5.66
occupation        string               453   50.00%   12.10%       25.43     4.67      1.17
gender            string                 7   22.20%   83.60%       1.378     0.46      0.28
```

`Bits` is the ceiling above, over the rows that carry a value.
`Cov bits` is that ceiling weighted by the chance both rows of a pair carry one, and it is what
the ledger adds up.
`gender` is the whole point of the table: seven distinct values, but 83.6% of the rows share
one, so it behaves as 1.378 values and can never contribute more than a quarter of a bit.
A column with no term frequencies, and a list column whose exact agreement is not a
single-value event, is listed and left unscored.

`Redundant` in the ledger is the sum of the **positive** overlaps found by the pairwise pass
below.
A pair whose joint collides less often than independence predicts reads a negative overlap, and
those are not credited back: an anti-correlation between two columns is not extra evidence to
spend.

The pairwise table follows, worst first:

```text
Column            Against                   Rows     L->R     R->L   Substr   Redund
------------------------------------------------------------------------------------
first_name        first_and_surname       50,511    0.234    1.000    1.000        -
surname           first_and_surname       46,063    0.514    1.000    1.000        -
postcode_fake     birth_place             34,706    0.921    0.298    0.000        -
...
first_name        occupation              25,270    0.336    0.149    0.001     0.53

Suspects
  first_and_surname determines first_name: drop first_name, or make the two one comparison
  first_and_surname determines surname: drop surname, or make the two one comparison
```

`first_and_surname` is the concatenation of the other two name columns, and the profile names it
from the rows alone in under a second: it determines both of them exactly (`R->L` of 1.000) and
both occur inside it on every row (`Substr` of 1.000).
That is the same dependence the [`completeness`](completeness.md) diagnostic finds from the
model, reached here with no model at all.

## Checking it against the model

`Bits` and the model's `u` are two readings of one quantity, and they agree exactly.
The profile takes its collision rate over the rows that carry a value; the model's `u` for an
exact level is over all pairs, including the ones a null makes disagree.
The two therefore differ by the coverage alone, and on `historical_50k` they do, for every
column:

| Column | `Bits` | `+ 2 log2(1/coverage)` | model `-log2(u)` | model weight |
| --- | ---: | ---: | ---: | ---: |
| `first_name` | 6.28 | 0.00 | 6.29 | 5.71 |
| `surname` | 10.50 | 0.27 | 10.77 | 9.83 |
| `first_and_surname` | 12.25 | 0.00 | 12.25 | 11.23 |
| `dob` | 8.95 | 0.74 | 9.69 | 7.55 |
| `postcode_fake` | 12.74 | 0.74 | 13.48 | 11.49 |
| `birth_place` | 7.60 | 0.42 | 8.02 | 6.87 |
| `occupation` | 4.67 | 2.00 | 6.67 | 4.77 |
| `gender` | 0.46 | 0.72 | 1.19 | 0.53 |

The last column is what the fitted model actually pays for exact agreement, and it sits below
the ceiling by `-log2(m)`.
`first_and_surname` gets 11.23 of its 12.25 available bits because `m` is 0.49: half the
matching pairs do not agree exactly on it.
**A positive margin is necessary and not sufficient**, and this is the gap it does not see.

## What it refuses, and why

Three things here look estimable and are not.
All three were found by measuring rather than by reasoning, and each one is now a guard.

### The plug-in collision rate reads `1/n` on singletons

`sum_v p_hat_v^2` over a joint table where every cell holds one row reads `1/n` when the truth
is near zero.
On `historical_50k` that put **9.4 spurious bits** of redundancy on a pair of independent
columns.
Every `u` here is the without-replacement collision form

```
sum_v c_v (c_v - 1) / (n (n - 1))
```

which reads exactly zero on a table of singletons, and which is what `u` means anyway: the rate
two *distinct* rows collide.

One visible consequence: on a near-unique column `Eff vals` runs above the distinct count.
That is correct rather than a bug.
Two distinct rows agreeing is rarer there than one value in however many the column holds.

### Mutual information is not estimable at this cardinality

The uncertainty coefficient read `U(L|R) = 0.920` for two independent high-cardinality columns,
because the two marginals and the joint all saturate at `log2(n)` together.
Sample size cannot fix it and the Miller-Madow correction is orders of magnitude too small to
repair it, so mutual information is not reported at all.
The determination share degrades far more gracefully and replaces it.
Its two failure modes are checked and refused rather than printed:

- a **near-unique determinant** determines everything and says nothing by doing so;
- a **target with one dominant value** is determined by everything, so a determination only
  counts when it beats the target's own commonest value.

A direction that fails either test is shown as `-`.

### On a file holding duplicates, the joint measures the duplicates

`u` is the rate two *non-matching* rows collide, and what is actually observed on a
deduplication file is `(1 - lambda) * u + lambda * m`.
For one column that is a small inflation.
For the joint of two high-cardinality columns the independent rate is far below `lambda` and the
measurement is nothing but the duplicates.
On `historical_50k` the joint of `first_and_surname` and `postcode_fake` holds **85,584
collisions where independence predicts 52**, and every one of the extras is a duplicate row
rather than a dependence between the columns.

So `Redund` is reported only where the independent rate stands clear of the match rate, and the
footer says how many pairs were refused on that ground.
It is refused on 22 of 28 pairs of that file.
Raising `--expected-matches` refuses more; lowering it refuses fewer.

The rule generalises, and it is worth carrying: **anything estimated per pair inherits the
duplicates, anything estimated per row does not**.
The determination and containment columns are per-row measures and are immune.
`Bits` is a per-pair measure, so a column whose collision rate is within reach of the match rate
is flagged and its ceiling read as a floor, which is the safe direction for a margin.

## Sampling

The pairwise pass is the only part that reads rows, and it is `O(columns^2)` per row.
`--sample-rows` bounds it.
Rows are selected by a hash of the row index, so the choice is deterministic, independent of
input order and spread evenly across the inputs.
Every quantity printed is a ratio of counts, so a sample needs no reweighting.

On the 1.8M-row synthetic sample, `--sample-rows 200000` takes **1.2 s against 10.7 s** for the
full pass and names the same suspects.
The per-column table is not affected by sampling at all: it comes from the term-frequency tables,
which cover every row.

## The M side is not estimated here

Everything above is the U half of the independence assumption.
Agreement among **matching** pairs is the larger correlation of the two and it needs matching
pairs to measure, which this command deliberately does not have.
[`completeness`](completeness.md) reports the M-side dependence between blocked comparisons once
a model exists, and its dependence table is the companion to the one here.

## Link mode

With two input files and no `--mode`, the pair space is the cross product rather than the
triangle, which changes the prior odds and the match rate the redundancy check is read against.
The per-column table is unchanged: term frequencies pool the inputs, which is the same
approximation the scorer makes.
See [linking two files](../linking.md).

## JSON

`--json` writes the whole report, including every column pair rather than the suspects alone:

```json
{
  "records": 50578,
  "mode": "all pairs",
  "pair_space": 1279041753.0,
  "expected_matches": 50578,
  "match_rate": 3.954366609328351e-05,
  "space_bits": 30.25241621426746,
  "prior_bits": -14.626193844911983,
  "available_bits": 47.29891935612495,
  "redundant_bits": -1.2880838361517108,
  "margin_bits": 31.384641675061257,
  "unresolved_pairs": 22,
  "walked": true,
  "sampled": false,
  "sampled_rows": 50578,
  "seconds": 0.036589833,
  "columns": [ ... ],
  "pairs": [ ... ]
}
```

Each entry of `columns` carries `name`, `type`, `scored`, `distinct`, `nulls`, `coverage`,
`top_share`, `collision`, `effective_values`, `bits`, `covered_bits` and `bits_floored`.
Each entry of `pairs` carries the two names, `rows`, `determines_left` and `determines_right`
with their `informative` flags and baselines, `containment`, the three `u` values,
`redundant_bits`, `joint_collisions`, `expected_collisions`, `resolved`, `suspect` and the
`verdict` string.
A pair with `resolved` false has a `redundant_bits` of zero that means *not readable* rather
than *not redundant*.

## What to do about a suspect

Cheapest first:

1. **Drop one of the columns.** If one determines the other, the second carries no information
   the first does not, and dropping it costs nothing and removes the double count exactly.
2. **Make the two one comparison.** Levels are ordered and the first hit wins, so a single
   comparison over both columns counts the evidence once by construction.
   This is what the `list_contains` level does for a name against an alias list.
3. **Subtract the bits.** `Redund` is the correction term, so a comparison whose columns overlap
   by 0.5 bits is worth 0.5 bits less than the model thinks.
   This is the only option that keeps both columns as they are, and it is approximate, because
   the correction is exact only on the U side.

## See also

- [`inspect`](inspect.md): the same file, from the loader's side: cardinality and memory
- [`completeness`](completeness.md): the M-side dependence diagnostic, once a model exists
- [`estimate`](estimate.md): where the `u` this predicts actually comes from
- [The model](../model.md): what a bit of match weight is
