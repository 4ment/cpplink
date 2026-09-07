# `simplify`

**Goal:** merge the adjacent comparison levels a run of this size cannot tell apart, and narrow the packed pattern in the process.

[`levels`](levels.md) asks where the cut points should sit.
This command asks a different question: whether a cut point is earning its place at all.
A level costs a code in the packed γ, a row of parameters for EM to fit, and a share of the histogram, and it repays that only if pairs landing on it score differently from pairs landing on its neighbour.

## Synopsis

```sh
cpplink simplify --schema <schema.json> --model <model.json>
                 [--out <schema.json>] [--alpha F] [--min-gap BITS]
                 [--pair-cap N] [--threads N] [--seed N] [--json]
                 [--mode MODE] <file.parquet>...
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema <file>` | — | required; the comparisons whose levels are being judged |
| `--model <file>` | — | required; `m` and `u` are the rates the test compares |
| `--out <file>` | none | write the whole schema back out with the merges applied |
| `--alpha F` | 0.01 | significance level; a pair merges when \(p \ge \alpha\) |
| `--min-gap BITS` | 0.0 | also merge any pair whose weights differ by less than this, however significant |
| `--pair-cap N` | none | Bernoulli-sample the candidate stream to this many pairs |
| `--threads N` | all | threads for the histogram pass |
| `--seed N` | 1 | sampling seed |
| `--json` | — | the levels, the tests and the merges as JSON |

## What the test actually is

Two adjacent levels differ in exactly one thing: \(\log_2(m/u)\).
So the pairs landing on either of them form a 2×2 table of level against class, and "these two levels carry the same weight" is precisely "the rows and columns of that table are independent":

\[
G^2 \;=\; 2 \sum O \log \frac{O}{E}
\]

which is chi-square on one degree of freedom, and `p` is its tail, \(\mathrm{erfc}(\sqrt{G^2/2})\).

**The rates come from the model and only the scale comes from the run.** `m` and `u` are the model's own, which is what \(\log_2(m/u)\) is made of; the histogram supplies how many pairs actually landed where, split by each pattern's responsibility. So what the test asks is not "are these levels alike" but "can a run this size tell their weights apart".

## Two constraints on top of the test

**A merge has to be expressible.**
Merging is implemented by deleting the stronger level and letting the weaker one absorb its pairs, which reproduces the intended partition only where the stronger predicate *implies* the weaker.
Descending Jaro thresholds nest, so `jaro_winkler >= 0.90` merges into `jaro_winkler >= 0.80`.
`levenshtein <= 1` against `jaro_winkler >= 0.88` does not nest either way, and is refused.

**The null level never merges, including into `else`.**
Missing is not a degree of agreement, and the report says so on the line rather than printing a p-value.
That `else` absorbs whatever sits above it, and null sometimes sat above it, is a case a test caught rather than a case anyone reasoned about.

## Reading the output

```text
Records                 5,000
Candidates            110,570
Matches                 6,512  expected, by the model
Packed gamma               20 bits -> 18

surname  5 levels -> 4, 3 bits -> 2
  level                              pairs         m           u    weight    stream
  null                               1,861    0.0301    0.031353     -0.06      0.80
  - exact                           37,255    0.5641    0.002981      7.56      0.77
  jaro_winkler >= 0.90               4,208    0.2286    0.001097      7.70      3.06
  jaro_winkler >= 0.80               4,429    0.0188    0.001879      3.32     -1.23
  else                              62,817    0.1584    0.962689     -2.60     -1.71

  adjacent pair                                      G^2         p     gap  verdict
  null / jaro_winkler >= 0.90                          -         -    7.66  missing is not a degree of agreement
  exact / jaro_winkler >= 0.90                       0.7    0.3946    0.14  merged, drops the exact level
  jaro_winkler >= 0.90 / jaro_winkler >= .         539.7   <0.0001    4.28  kept
  jaro_winkler >= 0.80 / else                      689.5   <0.0001    5.93  kept
```

| Column | Meaning |
| --- | --- |
| `pairs` | candidate pairs of this run that landed on the level |
| `m`, `u` | the model's rates, unchanged by this command |
| `weight` | \(\log_2(m/u)\), the bits the level is worth |
| `stream` | the weight **this run's own counts** imply |

A `-` in the left margin marks a level the merge deletes.

!!! note "`stream` is not `weight`, and where they differ blocking is the reason"
    `weight` is what the model says a level is worth against a random pair. `stream` is what the run's counts say, and the run was chosen by blocking. Where `stream` has collapsed towards zero on a level, blocking has **already spent that column's evidence**: every candidate agrees there, so agreeing there no longer distinguishes anything within the stream. Above, `surname exact` is worth 7.56 bits to the model and 0.77 to this run, because `surname` is a blocking column.

## What it found, and why the stated criterion does not work

The likelihood-ratio test was supposed to decide the level count. It cannot, and the reason is the same one that defeated BIC in [`levels`](levels.md#what-it-does-not-buy): **its power is the size of the run.**

| Dataset | Candidates | Levels merged at `--alpha 0.01` |
| --- | ---: | ---: |
| `historical_50k` | 18,436,900 | **0** |
| `febrl3` | 110,570 | **4** |
| `fake_1000` | 9,981 | 3 |

On `historical_50k` the closest call is \(G^2 = 604\) on a gap of 1.75 bits, and nothing merges.
The same code at the same alpha merges four levels on `febrl3`.
So the answer is decided by how much data there is rather than by how alike the levels are, and on a file this project is aimed at nothing will ever merge.

**What decides instead is the effect size.**
The gap in bits is printed beside every p-value for exactly that reason, and `--min-gap` merges on it: a level a tenth of a bit from its neighbour is not earning a γ code, however many pairs prove the tenth of a bit real.

## Measured

`--min-gap 1.0` on `historical_50k` merges two levels — `first_name`'s `jaro >= 0.90` into `>= 0.80` at a gap of 0.46 bits, and `dob`'s `levenshtein <= 2` into `else` at 0.72:

| | Before | After |
| --- | ---: | ---: |
| Packed γ | 20 bits | **18 bits** |
| Tabulated patterns | 90,000 | 57,600 |
| `predict` throughput | 8.37M candidates/s | **8.93M** (+6.7%) |
| F1, any threshold 8 to 16 bits | — | moves by at most **0.0007** |

On `febrl3` the four merges the test proposed by itself take γ from 20 bits to 18 and move F1 **up** at every threshold, 0.9350 to 0.9441 at 30 bits.
That is the case worth understanding: a level a run cannot separate is a parameter fitted to too little data, so deleting it is not a compromise but a correction.

## What it warns about

A merge that deletes an **exact** level from a comparison asking for term-frequency adjustment is called out, because the adjustment then has nowhere to attach.
It moves to the fuzzy level that absorbed it, which needs the neighbourhood masses that [`predict --fuzzy-tf`](predict.md#term-frequency-on-the-fuzzy-levels) builds; without them, that comparison silently loses its adjustment.

## Writing the proposal out

```sh
cpplink simplify --schema examples/sample_schema.json --model model.json \
                 --min-gap 1.0 --out simpler.json examples/sample.parquet
```

`--out` writes the whole schema, not a fragment, so it can be run directly.

!!! warning "Re-run `estimate` against the rewritten schema"
    The merged parameters are **not** the pooled ones, and γ packs differently, so a model
    learned against the old schema does not describe the new one. The command says this on
    every run that merges anything.

## See also

- [`levels`](levels.md) — where the cut points should sit, before asking whether they are worth having
- [`estimate`](estimate.md) — must be re-run after a merge
- [`predict`](predict.md) — what a narrower γ buys
- [the model](../model.md) — why level order is the model
