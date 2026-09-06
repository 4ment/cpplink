# `profile`

**Goal:** find out what the columns can be worth, what a matching pair will actually score, and
which pairs of columns are the same evidence twice, before any model exists to be wrong about it.

Every other stage costs minutes or hours and each one assumes things about the columns that
nobody has checked.
This command checks three of them.
It needs no model, no blocking plan and no known pairs, and it enumerates no candidate pair over
the whole pair space, so it runs in seconds and is the first thing to point at a new file.

## Synopsis

```sh
cpplink profile --schema <schema.json> [--sample-rows N] [--no-pairs]
                [--expected-matches N] [--threads N] [--json] [--mode MODE]
                [--no-anchors] [--anchor-rows N] [--anchor-margin BITS]
                [--anchor-pairs N] <file.parquet>...
```

| Option | Meaning |
| --- | --- |
| `--schema <file>` | required; only the `columns` section is read, so a schema with no comparisons and no blocking plan is enough |
| `--sample-rows N` | rows the pairwise pass reads, default 2,000,000; `0` reads every row |
| `--no-pairs` | skip the pairwise pass and report the ledger and the per-column table only; this also turns off the M side, which depends on it |
| `--no-anchors` | skip the M side and report the ceiling alone |
| `--anchor-rows N` | rows the anchor pass reads, default 4,000,000; `0` reads every row |
| `--anchor-margin BITS` | posterior bits an anchor must clear before its pairs are read as matches, default 6 |
| `--anchor-pairs N` | anchor pairs one session walks before it stops, default 1,000,000 |
| `--expected-matches N` | matching pairs assumed present, which is what the prior odds are made of; default is one duplicate per record |
| `--threads N` | default is the hardware concurrency |
| `--json` | the same numbers as JSON |
| `--mode` | `dedup`, `link` or `link-and-dedup`; sets which pair space the prior odds are taken over |
| *(positional)* | required; one or more parquet files |

## The three questions

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

### Two: what will a matching pair actually score?

The ceiling takes `m = 1`, which no real column reaches, and on the three benchmark datasets it
overstates by a factor of two to three.
Closing that gap needs matching pairs, and matching pairs are what nobody has yet.

They come from **anchor pairs**: two rows agreeing exactly on a subset of columns whose combined
exact-agreement bits put the posterior far enough above even odds that agreement alone makes the
pair a match.
Hold the anchor out, and every other column's agreement rate over those pairs is `m`.
The anchors on the three benchmark datasets hold **99.6% true pairs or better**, so what is
measured is matches rather than collisions.
[Below](#the-m-side-from-anchor-pairs) is what the estimate is worth and where it is wrong.

### Three: how much of that evidence is the same evidence twice?

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

  Expected                            +33.05 bits   (m over 195,168 anchor pairs)
  Double counted                       -0.07 bits   (pairwise, both sides)
  Est. margin                         +18.35 bits

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

The ledger's second half replaces `m = 1` with `m` as measured.
`Expected` is what an average matching pair scores, summing each column's agreement weight when
it agrees and its disagreement penalty when it does not; `Double counted` is what the weight
over-counts once the M-side overlap is set against the U-side one; and `Est. margin` is the
prior plus both.
On this file the ceiling says a match has 31.38 bits of headroom and the estimate says 18.35,
which is closer to the truth of 6.51.

Then `m` per column, and the anchors it came from:

```text
Column             Sessions       Pairs     Cov       m       Spread   Weight  Exp bits
---------------------------------------------------------------------------------------
postcode_fake             1      30,387  78.75%   0.748            -    12.32      6.86
first_and_surname         1      42,994  99.91%   0.529            -    11.33      5.47
surname                   1      38,888  90.37%   0.821            -    10.21      7.17
dob                       2      84,543  74.45%   0.652  0.625-0.680     8.34      3.65
birth_place               4     166,523  85.32%   0.858  0.840-0.865     7.37      5.06
first_name                1      70,759 100.00%   0.595            -     5.53      2.77
occupation                4      95,591  48.98%   0.909  0.897-0.914     4.53      1.87
gender                    4     149,594  76.65%   0.951  0.942-0.957     0.39      0.19

Anchors, over 50,578 rows
  first_and_surname + dob                          21.20 bits    +6.57 post      38,586 pairs
  dob + postcode_fake                              21.69 bits    +7.06 post      43,031 pairs
  first_and_surname + postcode_fake                24.99 bits   +10.36 post      42,792 pairs
  surname + postcode_fake                          23.24 bits    +8.61 post      70,759 pairs
```

`Cov` here is the share of *anchor pairs* carrying the column on both rows, which is coverage
among matches and not the row coverage in the table above.
`Weight` is `log2(m/u)`, and it can never exceed `Bits`, because `m <= 1`.
`Spread` is the same `m` read off anchors of different strengths, and a wide one is the estimate
saying it depends on which matches it was shown.
`post` is how far above even odds a pair agreeing on that anchor sits, under `m <= 1`.
A column where every anchor pair agreed, or none did, has an `m` of exactly 1 or 0 and no finite
weight; it is floored half a pair off the edge and named in a footer, rather than floored
quietly.

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
**A positive margin is necessary and not sufficient**, and that gap is exactly what the
[M side](#the-m-side-from-anchor-pairs) below estimates from the rows alone.

## What it refuses, and why

Three things on the U side look estimable and are not; the M side's own refusals are
[further down](#what-refuses-it).
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

The anchor pass has its own budget and a much larger one, because anchor pairs are rare by
construction and a row sample keeps only the *square* of its own fraction of them: sampling a
tenth of the rows would leave a hundredth of the pairs.
`--anchor-rows` defaults to 4,000,000, so every file below that size is read whole, and
`--anchor-pairs` stops a session once it has walked a million pairs, which is a hundred times
what `m` to a hundredth needs.
Both samples are unbiased for `m`, because a pair survives a row sample only when both its rows
do, and that has nothing to do with whether the pair agrees.

## The M side, from anchor pairs

`m` is the probability two *matching* rows agree, so estimating it needs matching pairs, and the
whole point of this command is that nobody has any yet.
An **anchor** manufactures them.

### How an anchor is chosen

Agreeing exactly on a column is worth at most `-log2(u)` bits, so agreeing on a set of columns is
worth at most the sum, less the pairwise overlap between them that the U-side pass just measured.
Set that against the prior odds and the result is a bound on the posterior that a pair agreeing
on all of them is a match.
An anchor is admissible when the bound clears even odds by `--anchor-margin`, six bits by
default, which is odds of about 64 to 1.
On `historical_50k` `first_and_surname + postcode_fake` is worth 24.99 bits against prior odds of
-14.63, so a pair agreeing on both sits at +10.36 bits, and **99.92% of the pairs it selects are
in fact true matches**.
Across all the anchors the three benchmark datasets produce, the worst is 99.6%.

Anchors are built one per column held out, strongest columns first, and identical ones collapse,
so a file usually ends with three or four sessions of two or three columns each.

### What is held out, and why

A session may not be read for a column whose agreement its own anchor has already forced.
Three things force it, and all three are checked:

1. **The anchor's own columns.** They agree on every pair by construction, so `m` would read 1.
2. **Anything the anchor determines.** If `first_and_surname` is in the anchor then `surname`
   agrees whenever it does, and the determination and containment checks from the U-side pass are
   what detect that. This is the same held-out-column discipline the per-column EM sessions in
   [`estimate`](estimate.md) run.
3. **Anything whose agreement moves with an anchor column's.** This is determination's pairwise
   twin and the rows cannot see it: two columns can be independent value by value and still be
   corrupted by the same event, so that they agree together among matches far more often than
   agreeing apart would predict. It is measured on the sessions that anchor on neither, and any
   column coupled to an anchor column by more than a tenth of a bit is dropped from that session.

The third check was not in the design and was added because a test caught it: a fixture column
redrawn on exactly the duplicates another column was redrawn on read `m` of 0.897 against a
planted 0.701, because two of its three sessions anchored on its twin.

### What it is worth

Measured against the ground truth the benchmark datasets ship and the tools never see:

| dataset | mean abs error in `m` | worst column | ledger ceiling | `Est. margin` | truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| `febrl3` | **0.003** | 0.008 | +57.12 | **+31.42** | +31.63 |
| `fake_1000` | **0.016** | 0.024 | +18.74 | **-0.83** | -1.16 |
| `historical_50k` | **0.117** | 0.228 | +31.38 | **+18.35** | +6.51 |

The ceiling overstates a matching pair's score by a factor of two to three on all three.
The estimate closes 99% of that gap on `febrl3`, 94% on `fake_1000` and 54% on `historical_50k`.

`fake_1000` is the case that shows why the M side is worth having: the ceiling reads **+18.74
bits** of headroom, and the truth is that an average matching pair scores **below even odds**.
The anchor estimate gets the sign right.
That is the same file whose best F1 sits at the bottom of the threshold grid, and the ledger now
says why before anything has been run.

### Where it is wrong, and in which direction

Every `m` on `historical_50k` reads high:

| column | truth `m` | anchor `m` |
| --- | ---: | ---: |
| `first_and_surname` | 0.301 | 0.529 |
| `first_name` | 0.431 | 0.595 |
| `dob` | 0.509 | 0.652 |
| `postcode_fake` | 0.594 | 0.748 |
| `surname` | 0.713 | 0.821 |
| `birth_place` | 0.800 | 0.858 |
| `occupation` | 0.859 | 0.909 |
| `gender` | 0.916 | 0.951 |

The ranking is exact and every value is 0.03 to 0.23 too high, and neither of those is an
accident.
Anchors are the matches that happened to agree on a clean identifier, and on real data being
clean is a property of the *record* rather than of the field, so a pair that agrees on two
columns is more likely to agree on the rest.
Conditioning on the anchor therefore selects the easy matches, and every rate read off them
moves up.

On `febrl3` the same estimator is accurate to 0.003, because its corruption is applied field by
field on independent coins: agreement really is conditionally independent among matches there,
and conditioning on an anchor selects nothing.

So **the error is itself the reading**.
Where the anchor `m` is close to the truth, the independence Fellegi-Sunter assumes holds among
matches; where it is far, that assumption is failing, by about the amount it is far.
The size of the gap cannot be measured without truth, but its direction can be relied on, so
`Expected` is a second ceiling and a much tighter one, rather than an estimate to be trusted as
a point.

`Spread` is the one visible symptom: it is the same `m` read off anchors of different strengths,
and it widens exactly when the estimate depends on which matches it was shown.

### What was tried and does not work

`m` rises smoothly with the anchor's strength, by about 0.02 to 0.03 per bit on
`historical_50k`, which invites a regression back to no conditioning at all.
Fitted over all 45 two- and three-column anchors it halves the error on the worst columns and
**doubles it on the best**, overshooting `surname`, `birth_place` and `occupation` by 0.13 to
0.18.
The reason is structural: the low end of that fit is anchors whose pairs are mostly not matches,
so the correction is bought by violating the premise the estimate rests on.
It is not applied.

Raising `--anchor-margin` is the same trade in the other direction and costs more than it buys.
At 12 bits the mean error on `historical_50k` goes from 0.117 to 0.148 and on `fake_1000` the M
side is refused entirely, while anchor precision at the default is already 99.6% or better.
There is no precision left to buy.

### What refuses it

The report says `No m:` and gives the reason rather than guessing:

- `--no-pairs`, because the pairwise pass is what tells an anchor from a column it determines;
- no admissible column set, and the message says what the strongest one is worth against what a
  match needs;
- anchors that select fewer than 100 pairs;
- no column carrying 100 of those pairs on both rows, which is `fake_1000` at a high margin.

A single column shows `-` when every anchor built either contains it or has already spoken for
it, and the footer says how many, because those contribute nothing to `Expected` and the
estimated margin is then a floor.

[`completeness`](completeness.md) reports the M-side dependence between blocked *comparisons*
once a model exists, which is the same question one level up, over levels rather than columns.

## Link mode

With two input files and no `--mode`, the pair space is the cross product rather than the
triangle, which changes the prior odds and the match rate the redundancy check is read against.
The per-column table is unchanged: term frequencies pool the inputs, which is the same
approximation the scorer makes.
Anchor pairs are required to cross the inputs, so the M side estimates `m` for a linkage rather
than for a deduplication.
On a 200,000-row synthetic pair of files whose duplicates are planted with independent per-field
corruption, `m` comes back within **0.0011** of the planted rate on every column.
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
  "anchored": true,
  "anchor_rows": 50578,
  "anchor_pairs": 195168,
  "expected_bits": 33.04627275908069,
  "double_counted_bits": -0.0723902075663121,
  "estimated_margin_bits": 18.3476887066024,
  "estimated_columns": 8,
  "scored_columns": 8,
  "unresolved_pairs": 22,
  "walked": true,
  "sampled": false,
  "sampled_rows": 50578,
  "seconds": 0.057234084,
  "columns": [ ... ],
  "sessions": [ ... ],
  "matches": [ ... ],
  "pairs": [ ... ]
}
```

Each entry of `columns` carries `name`, `type`, `scored`, `distinct`, `nulls`, `coverage`,
`top_share`, `collision`, `effective_values`, `bits`, `covered_bits` and `bits_floored`.
Each entry of `pairs` carries the two names, `rows`, `determines_left` and `determines_right`
with their `informative` flags and baselines, `containment`, the three `u` values,
`redundant_bits`, `joint_collisions`, `expected_collisions`, `resolved`, `suspect` and the
`verdict` string, plus the M side: `m_pairs`, `m_left`, `m_right`, `m_joint`,
`m_redundant_bits`, `m_resolved` and `net_redundant_bits`.
A pair with `resolved` false has a `redundant_bits` of zero that means *not readable* rather
than *not redundant*, and the same holds for `m_resolved`.

The M side adds `anchored`, `anchor_refusal`, `anchor_rows`, `anchor_pairs`, `expected_bits`,
`double_counted_bits`, `estimated_margin_bits`, `estimated_columns` and `scored_columns` at the
top level.
Each entry of `matches` carries `name`, `estimated`, `sessions`, `pairs`, `coverage`, `m`,
`m_low`, `m_high`, `weight`, `expected_bits` and `floored`.
Each entry of `sessions` carries `anchor`, `learns`, `bits`, `posterior_bits`, `groups`,
`pairs`, `oversized_groups`, `capped` and `used`.

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
- [`completeness`](completeness.md): the M-side dependence between comparisons, once a model exists
- [`estimate`](estimate.md): where the `u` this predicts actually comes from
- [The model](../model.md): what a bit of match weight is
