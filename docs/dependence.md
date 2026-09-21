# Dependence between comparisons

Fellegi-Sunter adds one weight per comparison, and that sum is only the log-odds of a match if the comparisons are **conditionally independent given the class**.
Real columns are not: a surname and a full name agree together, a record with a missing postcode is likelier to be missing its gender, and a duplicate that kept one field clean tends to have kept the others.
Every place the tool measures that assumption, every number it prints about it, and every remedy it offers is on this page, so that a column header in a report can be traced to the formula behind it.

The assumption, written out, is

\[
P(\gamma \mid M) = \prod_c m_c(\gamma_c) \qquad P(\gamma \mid U) = \prod_c u_c(\gamma_c)
\]

and the weight [the model](model.md) scores a pair with is the sum that follows from it.
Where the assumption fails the true joint is not the product, and the difference is a correction in bits that the sum leaves out.
Everything below is a way of reading that correction, from the rows, from manufactured matches, from the known pairs, or from the fitted model.

## The measures

| Measure | Printed as | Command | Needs | Formula |
| --- | --- | --- | --- | --- |
| U-side overlap | `Redund`, `U redu` | [`profile`](commands/profile.md) | rows only | [below](#u-side-overlap) |
| determination share | `L->R`, `R->L` | `profile` | rows only | [below](#determination-and-containment) |
| containment | `Substr` | `profile` | rows only | [below](#determination-and-containment) |
| M-side overlap | `M redu`, `Net` | `profile` | anchor pairs | [below](#m-side-overlap) |
| M-side overlap, known pairs | `True redu` | `profile --truth` | a truth file | [below](#m-side-overlap) |
| tie hold-out | `also tied` | [`estimate`](commands/estimate.md) | the profile's pass | [below](#the-tie-hold-out) |
| two-way correction | `per match`, `effect` | `estimate --interactions` | the session histograms | [below](#the-two-way-correction) |
| fit residual | `fit`, `worst pair`, `residuals` | `estimate`, `--report` for every pair | the session histograms | [below](#the-fit-residual) |
| capture dependence | `Dependence` | [`completeness`](commands/completeness.md) | a model | [below](#the-capture-table) |

Two things they have in common.
Every overlap, correction and residual is in **bits**, because a bit is the unit the weight is in, and a dependence of half a bit is a weight wrong by half a bit; the two structural checks, determination and containment, are shares of rows, because they ask a yes-or-no question about the columns rather than pricing one.
And none of them is an entropy or a correlation coefficient: mutual information is [not estimable](commands/profile.md#mutual-information-is-not-estimable-at-this-cardinality) at the cardinalities record data has, and a correlation coefficient is not a quantity the weight can be corrected by.

## U-side overlap

The `u` of an exact level is the collision rate of the column, and two columns have a joint collision rate the same way.
Over the joint value table of two interned columns, with \(c_{vw}\) the number of rows holding value \(v\) in the first and \(w\) in the second,

\[
u_{cd} = \frac{\sum_{v,w} c_{vw}(c_{vw}-1)}{n(n-1)}
\qquad
\phi_{cd} = \frac{u_{cd}}{u_c\,u_d}
\qquad
\text{Redund} = \log_2 \phi_{cd}
\]

`Redund` is the number of bits the weight over-counts when both columns agree, if the dependence is the same under both classes.
The collision form \(c(c-1)\) rather than \(c^2\) is not cosmetic: a joint table of singletons reads \(1/n\) under the plug-in form and exactly zero under this one, and on `historical_50k` the plug-in put 9.4 spurious bits on an independent pair.

**The refusal.** On a file holding duplicates the joint of two columns measures the duplicates: a random pair is a match with probability \(\lambda\), so what is observed is \((1-\lambda)u_{cd} + \lambda m_{cd}\), and for two high-cardinality columns the second term is the whole of it.
`Redund` is reported only where \(u_c u_d\), the rate independence predicts, stands clear of \(\lambda\), and shown as `-` otherwise.
On `historical_50k` that refuses 22 of 28 pairs.

## Determination and containment

Both are per-row measures, so neither inherits the file's duplicates.

**Determination** is the complement of the \(g_3\) error the functional-dependence literature measures an approximate dependency by: map every value of the left column to its commonest partner in the right, and count the rows that mapping keeps.

\[
\text{L->R} = \frac{1}{n}\sum_v \max_w c_{vw}
\]

It is 1 exactly when the left column determines the right.
Two shapes read 1 for saying nothing and are refused, shown as `-`: a near-unique determinant, which determines everything, and a target with one dominant value, which is determined by everything.
The second is why a determination only counts when it beats the target's own commonest value, its `baseline` in the JSON.

**Containment** (`Substr`) is the share of rows on which one value occurs literally inside the other, and the direction is reported.
It is the detector for a column built out of another, which is `first_and_surname` against either name on `historical_50k`: `Substr` 1.000, `R->L` 1.000.

## M-side overlap

The same \(\phi\) among matches is the larger of the two correlations and the one the rows cannot show.
Over a set of matching pairs, with \(a_c\) the share agreeing on column \(c\) and \(a_{cd}\) the share agreeing on both,

\[
\text{M redu} = \log_2 \frac{a_{cd}}{a_c\,a_d}
\]

`M joint` in the table is \(a_{cd}\).
`profile` reads it off [anchor pairs](commands/profile.md#the-m-side-from-anchor-pairs), pairs agreeing exactly on a column set strong enough that agreement alone makes them a match, over the sessions whose anchor contains neither column.
With `--truth` the same rate is read a second time off the known pairs, printed as `True redu`, and `Err` is the anchor reading less the true one.
On `historical_50k` the anchor reading is within **0.02 bits** of the truth on average, and 0.20 bits low on `surname` against `first_and_surname`, its worst pair.

**What the weight actually gets wrong** is the difference of the two sides, because the score adds \(\log_2(m/u)\) for each column and only the part of the M-side association that the U side does not share is counted twice:

\[
\text{Net} = \text{M redu} - \text{U redu}
\]

printed only where both sides resolved.
The ledger's `Double counted` line is \(-\sum a_{cd} \max(\text{Net}, 0)\) over the pairs, the bits an average match is over-scored by; an anti-correlation is not credited back.

## The tie hold-out

Blocking on a column conditions on **everything that column decides**.
A session blocking on `first_name` sees, among its non-matching pairs, an agreement rate on `first_and_surname` that is nothing like the file-wide `u`, and EM reads the gap as evidence of matching: that session's match rate read 0.9996 against a truth of 0.0080 before the hold-out was widened.
So [`estimate`](commands/estimate.md#the-per-session-detail) holds out of a session every comparison reading a column *tied* to the blocked one, where tied is containment on 90% of rows or `Redund` at or above `--tied-bits` (0.25), read from the same pairwise pass `profile` runs.
A declared derivation (`derive`, `derived_from`) is a tie both hold-outs are told about without the pass having to find it.
Determination is deliberately **not** the tie test: a column with a dominant value is determined by everything, and the version that used it scored worse.

## The two-way correction

Where a dependence is real and neither column can go, [`estimate --interactions`](commands/estimate.md#relaxing-conditional-independence) fits it into the weight.
For a pair of comparisons \((c,d)\) the weight gains

\[
\delta_{cd}(i,j) = \log_2 \frac{M_{cd}(i,j)}{m_c(i)\,m_d(j)} - \log_2 \frac{U_{cd}(i,j)}{u_c(i)\,u_d(j)}
\]

which is `Net` above, level by level instead of once for exact agreement.
\(M_{cd}\) is the session histograms weighted by the responsibility EM converged on; \(U_{cd}\) is the uniformly random pairs `u` is drawn from, with \(\lambda m\) subtracted back out because the random pairs hold the file's own duplicates.
Both are fitted to the model's own margins by iterative proportional fitting, which moves the margins and leaves every odds ratio alone, so the term carries the association and nothing else.
`per match` is what the term moves an average matching pair by, and it is the number to read.

## The fit residual

Every measure above asks about one pair of columns.
The fit residual asks whether the fitted mixture, with whatever terms it carries, explains the histogram it was fitted to, and it is the one reading that needs no notion of which columns are related.

A session's histogram is \(O[\gamma]\), the count of every pattern over the comparisons the session left free, and the fitted model predicts

\[
E[\gamma] = N\Big(\lambda \prod_c m_c(\gamma_c) + (1-\lambda)\prod_c u_c(\gamma_c)\Big)
\]

with the session's own \(\lambda\) and \(m\).
The deviance

\[
G^2 = 2\sum_\gamma O[\gamma]\,\ln\frac{O[\gamma]}{E[\gamma]}
\]

is the likelihood-ratio statistic of the saturated model against the fitted one, and divided by \(2N\ln 2\) it is the Kullback-Leibler divergence of the observed table from the fitted one **in bits per pair**: the average number of bits the model is wrong by on one of the session's pairs.
The same deviance over one pair of comparisons, with \(O\) and \(E\) summed to their two-way table, is the bivariate residual of the latent class literature, and it names the pair the independence fails on.

Three things about reading it.

- **Bits, not `p`.** \(G^2\) grows with \(N\) at a fixed misfit, so every pair on a run of millions of pairs reads \(p = 0\) and none on a run of thousands does; the same lesson [`levels`](commands/levels.md), [`simplify`](commands/simplify.md) and the interaction fit each learnt.
  The p-value is printed as the diagnostic it is.
- **The sampling floor.** A model that was exactly right would still show a deviance, because \(N\) pairs over \(K\) patterns are a sample: about \((K-1)/(2N\ln 2)\) bits, the chi-square expectation on the same scale.
  On `febrl3`'s 5,601-pair session that floor is 0.125 bits of the 0.533 read; on `historical_50k`'s 1.5M-pair session it is 0.0015 of 0.402.
  Misfit is what sits above it.
- **A residual is not a correction.** It measures the marginal fit over both classes, and a dependence that is the same among matches and non-matches shows here and costs the weight nothing.
  The interaction table's `per match` is what it costs; the residual is where to look.

`with terms` and `after` repeat the numbers with the admitted interactions in the model.
A term is applied to a session by refitting its joint to that session's own margins by IPF, so it asserts its odds ratios and nothing else.
A pair the terms do not touch keeps its residual exactly; a pair sharing one comparison with a term moves only through the normalisation two terms on one comparison need, which is small.
So the pair a term names is fitted by construction, and what the whole-table deviance keeps is what the terms did not reach.

## The capture table

[`completeness`](commands/completeness.md) meets the same assumption from the other side: it needs \(m(\gamma)\) for the patterns no blocking source produced, and the product of the learned parameters reads 95.3% pair completeness on `historical_50k` where the truth is 85.0%.
Its `Dependence` table is \(G^2\) between pairs of blocked comparisons among matches over the cells the capture table observed, and BIC chooses between the independence fit and the one with all two-way interactions.
It is the M-side overlap one level up, over levels rather than exact agreement, and confined to the comparisons blocking reaches.

## Which number answers which question

| Question | Read |
| --- | --- |
| Before any model: are two columns the same evidence twice? | `profile`: `Substr`, `L->R`, then `Redund` and `Net` |
| Which session is safe to read a comparison from? | `estimate`: `held out`, `also tied` |
| Does the independent mixture describe the pairs at all? | `estimate`: `fit`, above its sampling floor |
| On which pair does it fail? | `estimate`: `worst pair`, and `--report <file>` for every pair, worst first |
| What does that failure cost a match? | `estimate --interactions`: `per match` |
| Were the terms enough? | `estimate --interactions`: `with terms` against `fit` |
| Was the anchor reading of the overlap right? | `profile --truth`: `True redu`, `Err` |
| Are the blocked comparisons dependent among the matches blocking missed? | `completeness`: `Dependence` |

## What to do about it

Cheapest first, and the suspects table in `profile` prints the one that applies under each finding.

1. **Drop a column that another determines.** It carries nothing the other does not, and the double count goes with it.
2. **Make a column and the column it contains one comparison.** Levels are ordered and the first hit wins, so one comparison over both counts the evidence once; `first_and_surname` with `first_name` is the shape.
   Declaring the derivation (`derive`, `derived_from`) tells both hold-outs the same thing.
3. **Fit the correction.** `estimate --interactions` for an association that is statistical rather than structural, priced level by level and guarded as [described there](commands/estimate.md#two-traps-one-of-them-the-standing-limit).

What is measured and not yet repaired: correlated **missingness**.
The largest residuals on `historical_50k` are not the name columns but the null levels of `postcode_fake`, `occupation` and `gender`, because a record missing one is 1.3 to 1.6 times as likely to be missing another and the mixture multiplies two independent null rates.
A two-way term can carry it, but the interaction fit ranks on what a term moves a *match* by and those pairs move a match by under a tenth of a bit, so they are not admitted.
That is the residual doing its job: it names a misfit the correction has no reason to price.

## See also

- [The model](model.md): the weight the assumption is made for
- [`profile`](commands/profile.md): the row-level measures and the anchor pairs
- [`estimate`](commands/estimate.md): the hold-outs, the corrections and the fit
- [`completeness`](commands/completeness.md): the same assumption on the blocking side
