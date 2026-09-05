# `completeness`

**Goal:** estimate blocking recall on data with no known pairs at all.

[`recall`](recall.md) is the falsifiability check for automatic blocking, and it needs a truth
file.
On an 18M-row production file there is no truth file, which leaves the most important number in
the pipeline unmeasurable exactly where it matters most.
This command estimates it instead, from the model and the term frequencies, and reports what it
assumed to do so.

## Synopsis

```sh
cpplink completeness --schema <schema.json> --model <model.json>
                     [--truth <pairs.csv>] [--sample R] [--threads N]
                     [--value-weighting records|pairs] [--min-observed N]
                     [--bound-only] [--json] [--mode MODE]
                     <file.parquet>...
```

| Option | Meaning |
| --- | --- |
| `--schema <file>` | required; must declare `blocking` and `comparisons` |
| `--model <file>` | required; the model [`estimate`](estimate.md) wrote |
| `--truth <file.csv>` | optional; known pairs, used only to score the estimator against the thing it replaces |
| `--sample R` | Bernoulli-sample the pair walk; everything read off it is a ratio, so a sample needs no reweighting |
| `--value-weighting` | how a matching pair's shared value is drawn: like a record's (default) or like a random agreeing pair's |
| `--min-observed N` | pairs a pattern needs before its own observed capture rate is used instead of the pooled one |
| `--bound-only` | skip the pair walk entirely and report the product model alone |
| `--json` | the same numbers as JSON |
| *(positional)* | required; the parquet file |

## How it works

Pair completeness is a sum over agreement patterns:

```
PC = sum over gamma of  m(gamma) * P(some source fires | gamma)
```

Two things make every term of it available.

**The firing probability is analytic, not estimated.** An exact-value source on a column fires
exactly when the pair agrees on that column, which γ states outright.
A rare-value source with a cap fires when the shared value is under the cap, and the term
frequency table answers that exactly.
Sorted neighbourhood and MinHash are not analytic in γ; they are estimated from the capture
overlap, and leaving them out entirely can only move mass from *reached* to *dark*.

**The dark patterns are reachable through the model.** `m` is defined for every pattern,
including the ones where no source fires and no pair was ever enumerated.
That is the whole trick: EM extrapolates into the region blocking never shows you.

### The table, and why the product is not enough

Taking `m(gamma)` as the product of the learned parameters assumes agreement is conditionally
independent across columns.
On real data it is not — agreement is positively correlated among matches, so the product
overstates the chance that at least one blocked column agrees.
Measured on `historical_50k` the product model reads **95.3%** where the truth is **85.0%**.

So the product is used only as a reference line. The estimate comes from a table whose cells are
combinations of levels on the blocked comparisons:

- where an **exact-value source** fires, the candidate stream holds *every* pair of the file in
  that cell, so the cell is observed whole and its matching pairs are counted rather than
  modelled (in expectation, through the model's posterior);
- where only a **capped source** fires, the cell is observed with a known probability and is
  inflated by it;
- where **nothing** fires, the cell is dark, and only there does a model have to say anything.

Filling the dark cells of an incomplete contingency table is the classical multiple-systems
problem, and the classical answer applies: fit a log-linear model to the observed cells and let
it predict the missing ones.
Independence and all two-way interactions are both fitted, and BIC picks between them.
On `historical_50k` that takes the error from **+10.3 points to +1.9**.

## Reading the output

```
Pair completeness                           Estimate
----------------------------------------------------
  product model, best source per pattern      95.188
  product model, sources combined             95.188
  product model, with the correction          95.295
  table, independent dark cells               90.823
  table, two-way interactions                 86.898
----------------------------------------------------
  ESTIMATE (interaction)                      86.898
  measured against known pairs                84.988
```

The dependence table that follows names which pairs of blocked comparisons the independence fit
cannot explain, worst first.
On `historical_50k` the worst is `surname` against `first_and_surname`, which is a column
literally containing the other — a dependence the diagnostic finds without being told.

## When it refuses

Three conditions are checked, and all of them are readable off the plan and the model without
any known pairs:

- **No analytic source.** A plan of nothing but windows has no firing probability to compute.
- **`m` was never learned.** EM holds out every comparison reading the column a session
  conditions on, so a column that *every* source blocks on has no session left to learn its `m`.
  The estimate is a sum weighted by exactly those parameters, so the report says `REFUSE` and
  prints the model's default rather than dressing it up as a measurement.
  Measured: every single-blocked-column plan is wrong by up to 30 points, and every one of them
  is flagged.
- **No residual degrees of freedom.** A model can only fill the dark cells if the observed cells
  outnumber its free parameters. With two blocked columns and one-way margins the counts are
  equal, the fit is saturated, and the dark cell is whatever the model says — the two-list
  capture-recapture problem, and the answer here is the same as there: fall back rather than
  report.

## Measured accuracy

Against `recall` on the same data, with the plan that ships with each dataset:

| Data | Estimate | Measured | Error |
| --- | ---: | ---: | ---: |
| `febrl3` | 99.81% | 99.83% | −0.02 |
| 1M synthetic sample | 98.48% | 99.18% | −0.70 |
| `historical_50k` | 86.90% | 84.99% | +1.91 |
| `fake_1000` | 90.75% | 84.00% | +6.75 |

`fake_1000` is 1,000 records and about 2,000 matching pairs, and BIC cannot see the dependence
in it: the interaction model is available and is not selected.
That is the honest limit of the method — detecting dependence takes data, and where there is not
enough of it the estimate inherits the independence bias.

## See also

- [`recall`](recall.md) — the same number, measured, when known pairs exist
- [`estimate`](estimate.md) — where `m` comes from, and why it can be missing
