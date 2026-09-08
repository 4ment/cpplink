# Estimation and EM

The parameters \(m\), \(u\) and \(\lambda\) defined in [the model](model.md) have to come
from the data itself — nobody labels 18 million records. This page is what
[`cpplink estimate`](commands/estimate.md) does, and why each of the three parameters comes
from a different place.

| Parameter | Depends on the candidate set? | Where it comes from |
| --- | --- | --- |
| \(u\) | **No** | exact and null levels in closed form from term frequencies; the rest sampled from uniformly random pairs |
| \(\lambda\) | biased upward by any blocking | full-data pair counts, never a blocked session's match rate |
| \(m\) | **Yes** | EM over the pattern histogram, restricted to EM-safe sources |

## The pattern histogram is the interface

EM never sees a pair. The blocking iterator produces candidate pairs, each is reduced to a
packed γ, and γ is folded into a counter:

```text
count[γ]        the only thing EM reads
```

Because γ is a sufficient statistic for the Fellegi–Sunter likelihood, this loses nothing.
The histogram is bounded by the number of *distinct patterns*, which is tiny: the design
allowed for ~10⁵ and the measurement is **242 to 553** per session. Eight comparisons with
three to five levels can only reach ~10⁵ in principle, and real data occupies a vanishing
corner of that space.

Implementation, in `histogram.*`:

- **Below 22 bits of γ, a flat array** (4M slots × 8 B = 34 MB per thread, no hashing at
  all); above that, open addressing `uint32 → uint64`.
- **Per-thread, merged at join.** Nothing is contended during the fold.
- **Work is handed out from an atomic counter over groups**, and an oversized group is cut
  into row ranges of its own triangle — otherwise one thread is left holding "Smith" alone.
- **A `pair_cap` turns the fold into a Bernoulli sample of the stream**, which needs no
  reweighting: \(m\), \(u\) and \(\lambda\) are all *ratios* of pattern counts. Sampling
  buys the thing that costs (the comparison) rather than the thing that does not
  (enumeration).

Once the histogram exists, re-fitting the model is instantaneous — which makes parameter work
interactive in a way it cannot be when every iteration re-reads a 130 GB table.

## `u` in closed form

Splink estimates \(u\) by sampling random pairs. For **exact-match levels that is
unnecessary**: \(u\) is precisely the probability that two randomly drawn records carry the
same value, which the term-frequency table gives exactly.

\[
u_{c,\text{exact}} \;=\; \frac{\sum_v c_v (c_v - 1)}{N (N - 1)}
\qquad \text{exact, } O(\text{distinct values}), \text{ zero sampling error}
\]

where \(c_v\) is the number of records holding value \(v\) and \(N\) the number of records.

!!! warning "It is \(c_v(c_v-1)\), not \(c_v^2\), and the difference is not cosmetic"
    A pair is **two distinct records**, so a record agreeing with itself is not one. The
    plug-in form \(\sum_v p_v^2\) counts all \(N\) of those self-agreements, which inflates
    \(u\) by \(1/N\). That is negligible while \(u\) is large and *dominant* once \(u\)
    approaches \(1/N\) — which is exactly where the strongest columns live. On the 1M sample
    the 919k-value `email` column read **3.5 bits low** under the plug-in form, and at 18M
    records it reads 1.5 bits low. The random-pair sampler had always skipped `a == b`, so
    the two halves of one estimator disagreed about what a pair is until this was corrected.

A **null level placed first** fires exactly when either side is missing, which is also a count
over the data, so it is exact for *any* comparison — including multi-column ones.

This matters enormously at high cardinality. On the sample, `email exact` comes out at
\(u = 5.96\times10^{-8}\), exactly. Sampling 20M random pairs would have expected **1.2 hits**
for that level, which is not an estimate.

What is left — geo radii, list-set equality, Jaccard, fuzzy string levels — is sampled from
uniformly random pairs. The two sources are then reconciled: **the sampled levels take what
the exact ones leave**, rescaled so the level probabilities sum to one and the two sources of
truth cannot disagree about the total. A level the sample never hit gets **half a count**,
which is a floor with a meaning rather than a guess, and the report says how many pairs it
actually saw:

```text
last_name         jaro_winkler >= 0.92          0.279753      6.00e-06     15.51  u from only 6 random pairs
location          within 1.00 km                0.933802      5.00e-07     20.83  u below the sampling floor
```

Those are the weights to distrust, and they are named on the line rather than hidden.

!!! note "`u` never sees a candidate pair"
    This is the payoff of the closed form, and it is why blocking choices cannot bias \(u\).
    Splink, which estimates \(u\) by sampling within blocking, does not get this.

## EM over the histogram

With \(u\) fixed, EM alternates two steps. The E step assigns each *pattern* a responsibility
— the posterior probability that a pair with that pattern is a match:

\[
r[\gamma] \;=\; \frac{1}{1 + 2^{-W(\gamma)}}
\]

The M step re-estimates \(m\) and the session's match rate as count-weighted averages:

\[
m_{c\ell} \;=\; \frac{\sum_{\gamma:\,\ell_c(\gamma)=\ell} \text{count}[\gamma]\, r[\gamma]}
                     {\sum_\gamma \text{count}[\gamma]\, r[\gamma]}
\qquad
\lambda \;=\; \frac{\sum_\gamma \text{count}[\gamma]\, r[\gamma]}{\sum_\gamma \text{count}[\gamma]}
\]

The cost is `patterns × comparisons × iterations` and has **nothing to do with how many pairs
were folded** to build the histogram. That is the whole reason the histogram is the interface.

### Numerics

- **Never form the Bayes-factor product.** With \(u\) near \(10^{-7}\) across ten comparisons
  it reaches \(2^{\pm200}\). Everything stays in log2 and the posterior is recovered only
  through the logistic form, which cannot overflow.
- Pattern counts reach \(10^{10}\), comfortably below \(2^{53}\), so `count × r` is exact in
  double. The *accumulation* is not — responsibilities span many orders of magnitude — so it
  uses **Neumaier compensated summation**.
- The level each pattern assigns each comparison is unpacked once before the loop, since the
  inner loop runs `iterations × patterns × comparisons` times and should not re-shift.

### Convergence

Iteration count tracks **how well separated the mixture is, not how much data there is**. On
the real sample every session reached `max|Δ| < 1e-6` in **2 to 10 iterations**; on a toy
fixture with four comparisons and eight distinct patterns the same tolerance took 148 and 334.
Weak evidence, not volume, is what makes EM crawl. The iteration cap is 500 and is never
approached in practice.

## Sessions: why there is more than one EM run

EM cannot be run over the blocked pairs naively, because blocking is not a random sample of
pairs — it is a sample selected on agreement.

Blocked EM is valid because of conditional independence: if the selection event is exactly
`{level_j = exact}`, then for every other comparison

\[
P(\ell_c \mid M, \ell_j = \text{exact}) = P(\ell_c \mid M) \qquad \text{for all } c \neq j
\]

so \(m_j\) is degenerate and held fixed while every other \(m_c\) is estimated without bias.

cpplink therefore runs **one EM session per column that an EM-safe source conditions on**, and
in each session **holds out every comparison that reads that column**:

```text
Session               Enumerated      Compared  Patterns  Iters Match rate  Seconds
-----------------------------------------------------------------------------------
email                     90,600        90,600       242      3   0.999934      0.0
phone                    121,195       121,195       274      2   0.870795      0.1
dob                   77,931,552    10,002,206       270      4   0.001535      3.5
last_name             39,059,914     7,413,527       553     10   0.003205      3.2
```

The final \(m_{c\ell}\) is the **average over the sessions that could learn it** — a
comparison held out of every session gets a starting value, not an estimate, and the report
says so out loud.

### EM safety

A source may only feed EM if its **selection event factors as a condition on an excludable
column subset**:

| Source | Selection event | Safe for `m`? |
| --- | --- | --- |
| Exact value on column *j* | `level_j = exact` | ✅ exclude *j* |
| Rare value on column *j* | `level_j = exact ∧ n_v ≤ cap` | ✅ exclude *j*, with a caveat |
| MinHash LSH **per column** *j* | function of column *j* only | ✅ exclude *j* |
| Sorted neighbourhood on a key from *j* | mostly *j*; window membership also depends on other records | ⚠️ approximate |
| MinHash on the concatenated record | function of all columns | ❌ |
| ANN over a record embedding | function of all columns | ❌ |

A source whose selection depends on all columns jointly does not factor, conditional
independence buys nothing, and **every** \(m_c\) comes out biased with no column to exclude to
repair it. So estimation and prediction deliberately run over **different unions of sources**:
estimation over the EM-safe subset (perhaps \(10^7\) pairs), prediction over everything.

!!! warning "The rare-value caveat"
    A rare-value source conditions on the specific *value*, not only the level, so validity
    needs \(\ell_c \perp (v_j, \ell_j) \mid M\) — slightly stronger than plain conditional
    independence. It is the same assumption TF adjustment already relies on, so it adds no new
    exposure, but it is an assumption rather than a consequence.

### A session needs at least three free comparisons

With two free comparisons the histogram is a 2×2 table with three degrees of freedom while
the mixture has three free parameters, so the fit is **saturated**: it has two exact solutions
and the likelihood cannot prefer either. EM returns whichever root it walked to, silently.

A session with fewer than three free comparisons after the held-out column is **refused with a
warning, not merged**. This is why holding out a column is not free — it has to leave enough
behind.

### The label-swap check

The likelihood is symmetric under swapping the match and non-match classes, so a session can
converge to the inverted labelling with every parameter backwards, and the likelihood cannot
tell. It is caught from the *meaning* instead: matches must not disagree more often than
random pairs do. A session that fails this is disqualified and its \(m\) is not merged.

## λ is a lower bound

Each session implies `λ_s × pairs_s` matching pairs; cpplink takes the **tightest** (largest)
of those and puts it back over the full-data pair count, because a blocked session's match
rate is biased upward by exactly the thing blocking is for:

```text
lambda       7.73e-08  (lower bound: session last_name implies 125,201 matches over 1,619,999,100,000 pairs)
prior weight -23.625 bits
```

It is a bound twice over — blocking recall is below one, and EM's posterior mass is itself
conservative (a match agreeing only on a 10-bit signal against a 23-bit prior is not called a
match). Pass `--lambda` to set it from a count you trust.

!!! warning "λ being a bound is harmless for the weights and is *not* harmless for the interactions"
    The prior shifts every pair by the same amount, so a λ that is low by a factor of two moves
    the threshold and nothing else. The two-way corrections are different: λ is the rate at
    which the file's own duplicates contaminate the `u` side of a joint, and it is subtracted
    there. On a 1,000-record fixture whose λ reads 2.3× low, the two largest fitted terms come
    out with the **wrong sign**. See
    [`estimate --interactions`](commands/estimate.md#two-traps-one-of-them-the-standing-limit).
    `cpplink completeness` estimates the pair completeness this bound is missing, without a
    truth file.

## The three floors, which are three different things

They look alike in the output and they are not the same:

| Case | Floor | Why |
| --- | --- | --- |
| A level whose closed-form \(u\) is exactly zero | weight exactly **0** | It is *unreachable* — the null level of a column with no missing value. Flooring it at 1e-12 while `m` floored higher once made "both values present" the strongest signal in the model, which is how this was found. |
| A level no matching pair reached | \(m = 0.5 / \text{matching mass}\) | Half a pair is the resolution of the evidence. A fixed 1e-12 floor put −40 bits on one comparison, enough for a single disagreement to veto every score. |
| A level no random pair reached | \(u = \) half a sampled pair | And the report says how many pairs it did see, because a weight resting on one observation is worth knowing about. |

Every floored parameter is **reported rather than hidden**:

```text
email             else                          3.46e-06      0.944784    -18.06  m at the floor: no matching pair reached this level
last_name         null                          1.00e-12      1.00e-12      0.00  unreachable: no pair can land here
address           exact                         0.554006      5.00e-07     20.08  u below the sampling floor
```

A level with no support is a modelling bug, not a number to clamp.

## Measured cost

At 18M rows, four sessions, eight comparisons, 20M random pairs for `u`:

| Stage | Enumerated | Compared | Patterns | EM iterations | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `u` sampling | — | 20,000,000 | — | — | 15.1 |
| session `email` | 909,997 | 909,997 | 386 | 3 | 0.8 |
| session `phone` | 2,613,571 | 2,613,571 | 450 | 3 | 4.3 |
| session `dob` | **7,782,017,109** | 10,001,483 | 233 | 4 | **17.5** |
| session `last_name` | 467,529,352 | 8,616,883 | 497 | 10 | 6.4 |

**81 s wall, 289 s CPU, 4.19 GB peak RSS**, load included. Two things worth keeping:

- **Enumeration parallelises cleanly.** The `dob` session walked 7.78 billion candidate pairs
  in 17.5 s — about 2.2 ns per pair wall against 17 ns single-threaded, so the atomic counter
  over groups delivers roughly the core count with no visible contention.
- **Bernoulli sampling is what makes a broad session affordable.** `dob` compared 10M of its
  7.78 billion pairs — 0.13% — and still produced 233 distinct patterns and a converged fit in
  4 iterations. The remaining 7.77 billion comparisons would have bought nothing and cost
  about nine hours.
