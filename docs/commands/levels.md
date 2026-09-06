# `levels`

**Goal:** check the schema's fuzzy thresholds against the column they actually run on, and
propose better ones, with no model, no blocking plan and no known pairs.

`jaro_winkler >= 0.9` and `levenshtein <= 2` are guesses.
They get written by hand, copied from the last schema, and never checked, even though level
order **is** the model: the first level that fires wins, so where the cut points sit decides
what every pair is worth.
This command measures where they should sit.

## Synopsis

```sh
cpplink levels --schema <schema.json> [--out <schema.json>] [--json]
               [--truth <pairs.csv>]
               [--levels N] [--max-levels N] [--jaro-floor F]
               [--jaro-step F] [--edit-max N] [--min-match-pairs N]
               [--anchor-margin BITS] [--anchor-rows N]
               [--anchor-pairs N] [--ball-budget N] [--threads N]
               [--mode MODE] <file.parquet>...
```

| Option | Meaning |
| --- | --- |
| `--schema <file>` | required; the comparisons are what is being checked |
| `--out <file>` | write the whole schema back out with the proposal substituted, ready to run |
| `--truth <file>` | an `id_a,id_b` CSV; both partitions are re-measured against it, and nothing is fitted to it |
| `--levels N` | propose at `N` levels; the default is the count the schema already uses |
| `--max-levels N` | how far the level-count sweep runs, default 8 |
| `--jaro-floor F` | the similarity grid's floor, default 0.75; lowered automatically where a declared level is looser |
| `--jaro-step F` | grid resolution, default 0.01 |
| `--edit-max N` | the edit-distance grid's ceiling, default 4 |
| `--min-match-pairs N` | anchor pairs a comparison needs before a curve is read off it, default 500 |
| `--ball-budget N` | value pairs the self-join may look at for one comparison |
| `--json` | both curves, both partitions and the sweep, as JSON |

Anchor options (`--anchor-margin`, `--anchor-rows`, `--anchor-pairs`, `--expected-matches`)
mean what they mean in [`profile`](profile.md), which is where the matching pairs come from.

## The two curves

### `u(t)` is exact, and this pipeline already pays for it

Because values are interned, sweeping a similarity threshold over the **dictionary self-join**
gives

```
u(t) = P(sim(a, b) >= t) for two random records carrying both values
```

for the whole column, in one scan over the distinct values rather than over the rows.
That is 6,195 surnames rather than 50,578 rows, and it is the same self-join
[`predict --fuzzy-tf`](predict.md) already runs for the neighbourhood masses, accumulated one
bin instead of one level.
Almost no linkage system can afford this curve.
It is exact: not sampled, not bounded, not approximated.

The grid is the uniform one **plus every threshold the schema already declares**, so no bin
straddles a current level's boundary and one histogram answers both questions.
That is what makes the current and proposed partitions comparable rather than merely adjacent.

### `m(t)` comes from anchor pairs

Anchor pairs are the matching pairs [`profile`](profile.md#the-m-side-from-anchor-pairs)
manufactures: two rows agreeing exactly on a column set strong enough that agreement alone
makes them a match.
Bin the similarity of the held-out column over those pairs and that is `m(t)`.

It carries the anchor's upward bias, and that matters far less here than it does for `m`
itself: the bias lifts the whole curve, and a cut point is chosen from the curve's *shape*.
The measurement below bears that out.

## Where the cuts go

A comparison is worth

```
KL(m || u) = sum over levels of m_l * log2(m_l / u_l)
```

bits to a matching pair, which is the mutual information between the discretised similarity and
the latent match indicator, divided by lambda, in the limit lambda is actually in.
The best partition is a dynamic program over the bins, which is Fayyad-Irani minimal-entropy
partitioning against a latent target rather than an observed one.

**The number of levels is not chosen by this.**
Merging two bins can only lose bits, so the criterion always prefers more levels, and the
proposal is therefore made at the count the schema already uses.
That holds gamma's width fixed and asks the one question the curves answer well: given this many
cuts, where do they go.

## Reading the output

```text
first_name
  jaro_winkler over 4,413 values, 3,983,540 of 9,735,078 value pairs past the bound
  m from 70,759 anchor pairs on surname + postcode_fake

  Current                                  m           u   Weight     Bits
    exact                          0.5948      0.0128    +5.53    +3.29
    jaro_winkler >= 0.90           0.0968     0.00411    +4.56    +0.44
    jaro_winkler >= 0.80           0.1149     0.00655    +4.13    +0.47
    else                           0.1936       0.977    -2.33    -0.45
                                                                   3.76 bits

  Proposed, 4 levels
    exact                          0.5948      0.0128    +5.53    +3.29
    jaro_winkler >= 0.84           0.1969     0.00886    +4.47    +0.88
    jaro_winkler >= 0.75           0.0451     0.00724    +2.64    +0.12
    else                           0.1633       0.971    -2.57    -0.42
                                                                   3.87 bits

  Levels        2       3       4       5       6       7       8
  Bits       2.77    3.83    3.87    3.88    3.89    3.89    3.89
  y bits        1       2       2       3       3       3       3
                                *                             bic
  Against 303,264 known pairs, which the proposal never saw: current 2.83 bits,
  proposed 2.95 bits, +0.13
  The lowest cut sits on the grid floor of 0.75, so the partition wanted to go
  lower and --jaro-floor stopped it.
```

`Bits` is `m * log2(m/u)`, what a level contributes to a matching pair, and the column sums to
what the whole comparison is worth.
`y bits` is the gamma field width each level count needs, which is the cost BIC cannot see.
`*` marks the proposal, `bic` marks what BIC would take.

The `Against` line appears with `--truth` and is the point of the whole exercise: both
partitions re-measured against known pairs that nothing above the line ever read.

## What it found

On `historical_50k`, with `--truth` supplying the second reading:

| Comparison | Levels | Current | Proposed | Gain | Cur (T) | Prop (T) | Gain (T) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_name` | 4 | 3.76 | 3.87 | +0.12 | 2.83 | 2.95 | +0.13 |
| `surname` | 4 | 8.95 | 8.96 | +0.01 | 8.13 | 8.15 | +0.02 |
| `first_and_surname` | 4 | 7.42 | 7.80 | **+0.38** | 5.48 | 5.99 | **+0.51** |
| `dob` | 4 | 6.46 | 6.46 | +0.00 | 5.90 | 5.90 | +0.00 |
| `postcode_fake` | 3 | 9.44 | 9.67 | **+0.23** | 7.70 | 8.09 | **+0.38** |
| `birth_place` | 3 | 5.80 | 5.81 | +0.01 | 5.40 | 5.42 | +0.02 |
| **total** | | 41.82 | 42.57 | **+0.75** | 35.45 | 36.51 | **+1.05** |

and on `febrl3`:

| Comparison | Levels | Current | Proposed | Gain | Gain (T) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `soc_sec_id` | 3 | 9.26 | 9.61 | **+0.35** | +0.36 |
| `postcode` | 3 | 6.87 | 7.19 | **+0.32** | +0.31 |
| `address_1` | 4 | 8.16 | 8.42 | **+0.26** | +0.26 |
| `given_name` | 4 | 4.88 | 4.95 | +0.07 | +0.06 |
| others | | 22.90 | 22.94 | +0.04 | +0.02 |
| **total** | | 52.07 | 53.11 | **+1.04** | +1.01 |

The benchmark harness passes `--truth` on every run and prints both totals beside each other,
so these numbers are re-measured rather than remembered.

**The gains survive the move to the truth curve**, and on `historical_50k` they get larger,
which is the answer to the obvious objection: the cuts are placed on the anchor curve, so if
they were fitting the anchor's bias rather than the column, the truth reading would take them
back.

Two findings are worth stating on their own.
`postcode_fake` declares `levenshtein <= 1` and the data wants `levenshtein <= 2`: the level as
written reaches 6.9% of matching pairs where 11.2% are within two edits, and widening it is
worth 0.38 truth bits.
And `first_and_surname`, the concatenated name column, is the one the hand-written 0.90 and 0.80
serve worst, because the similarity distribution of a two-token field is not the distribution of
a one-token field and the same two numbers were written for both.

`fake_1000` is **refused outright** and the refusal is the right answer: 1,000 records leave
164 to 214 anchor pairs per column against the 500 a curve needs.

## What it does not buy

Extra bits per matching pair do not automatically become F1, and this is the honest limit of
the command.
Running the proposed schema end to end on `historical_50k`, over a fine threshold sweep in bits:

| Threshold (bits) | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.564 | 0.793 | **0.861** | 0.853 | 0.833 | 0.806 | 0.766 | 0.723 |
| proposed | 0.355 | 0.649 | 0.852 | **0.862** | 0.847 | 0.823 | 0.788 | 0.745 |

Best F1 moves 0.8606 to 0.8617, which is nothing, and **the threshold it sits at moves from 10
bits to 12**.
More evidence per matching pair shifts the whole score distribution, so a fixed threshold reads
the change as a loss: at the bench harness's own grid the proposed schema looks 0.007 worse
because its optimum has moved past a grid point.
On `febrl3` the proposal is equal or better at every threshold (0.9992 against 0.9991 at the
optimum, 0.947 against 0.935 at 30 bits), which is the more useful shape of the same result:
what the extra bits buy is robustness to where the threshold is put, not a higher peak.

So this is a diagnostic first.
It tells you a level fires on nothing, that an edit distance cannot reach the corruption in the
column, or that a comparison is worth half what you thought.
It is not a free quality win, and reporting it as one would be false.

## What it refuses, and why

- **A comparison with no fuzzy level.** There is no threshold to place; an exact level is not a
  cut point.
- **A comparison written in two metrics at once.** One similarity axis cannot measure both, and
  a partition of an axis that does not exist is not worth printing.
- **A column every anchor contains or has already spoken for.** This is the held-out discipline
  from [`profile`](profile.md#what-is-held-out-and-why), and without it `m` would be the
  anchor's own selection read back.
- **Too few anchor pairs**, which is `fake_1000` at any setting.
- **A dictionary over `--ball-budget`.** A near-unique column is a self-join no signature bound
  makes affordable, the same wall [`predict --fuzzy-tf`](predict.md) hits.

## Cost

One dictionary self-join per comparison, threaded, with the signature bound doing the pruning.
On `historical_50k` the whole command is **11.4 s at 8 threads**, most of it one comparison:
`first_and_surname` has 20,479 distinct values, so 210M value pairs, of which 186M survive the
bound because a floor of 0.75 is much looser than the 0.80 the schema declares.
Raising `--jaro-floor` is the lever if it matters; lowering it costs more and finds more, which
is what the "lowest cut sits on the grid floor" note is telling you.

The anchor pass is the same one [`profile`](profile.md#sampling) runs and takes its budget from
the same options.

## Writing the proposal out

`--out <file>` writes the whole schema back with the proposal substituted, so what comes out is
a file that runs:

```sh
cpplink levels --schema schema.json --out proposed.json data.parquet
cpplink estimate --schema proposed.json --out model.json data.parquet
```

Only the `levels` arrays of proposed comparisons are touched.
The null level is carried over rather than proposed, because whether a comparison has one is not
a threshold question, and everything else the file holds survives verbatim.

## JSON

`--json` writes both curves bin by bin, both partitions with their `m`, `u`, `weight` and
`bits`, the level-count sweep with its BIC scores and gamma widths, and where `--truth` was
given, the truth reading of all of it.
Each entry of `comparisons` carries `proposed` and, where it is false, `refusal`.

## See also

- [`profile`](profile.md): where the matching pairs come from, and what the bias in them is
- [`predict`](predict.md): `--fuzzy-tf`, which runs the same dictionary self-join
- [The schema](../reference/schema.md): what a level is and how order decides the model
- [The model](../model.md): what a bit of match weight is
