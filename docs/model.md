# The model

cpplink implements the Fellegi–Sunter model of record linkage. This page presents the model
itself: what it assumes, what its parameters mean, how a pair is turned into a score, and
what term-frequency adjustment does to that score. How the parameters are *learned* is
[the next page](em.md).

## The setup

Take the set of all pairs of records drawn from the data. Each pair is either a **match**
(both records describe the same entity, \(M\)) or a **non-match** (\(U\)). We never observe
which; we observe only how the two records *compare*.

Each declared **comparison** — surname, date of birth, a coordinate pair — reduces one pair
to a small integer: the index of the first comparison **level** that fires.

```json
{"name": "last_name", "term_frequency": true, "levels": [
  {"type": "null"},
  {"type": "exact"},
  {"type": "jaro_winkler", "threshold": 0.92},
  {"type": "jaro_winkler", "threshold": 0.85},
  {"type": "else"}]}
```

Levels are evaluated top-down and the first hit wins, so **level order in the config *is* the
model**: put the strongest evidence first. Every comparison must end in an `else` level, which
is checked at parse time before a file is opened.

The vector of level indices across all comparisons is the pair's **agreement pattern** γ:

```text
Comparison        Level                     Index   Values
----------------------------------------------------------------------------------
last_name         exact                         1   vipiest  |  vipiest
first_name        exact                         1   sheanwean  |  sheanwean
dob               exact                         1   8605d  |  8605d
email             null                          0   sheanwean.vipiest7140@examp.  |  <null>
phone             null                          0   0428000809  |  <null>
postcode          exact                         1   8582  |  8582
location          within 1.00 km                1   -24.1911, 126.2921  |  -24.1873, 126.2908
address           exact                         1   {priengvieck quiem ret pick.  |  {priengvieck quiem ret pick.

gamma  0x0002a049   0010 1010 0000 0100 1001   (20 bits)
```

That is the output of [`cpplink explain`](commands/explain.md), which exists to show exactly
this reduction for one pair.

### γ packs into a `uint32`

Each comparison occupies \(\lceil \log_2 L_c \rceil\) bits, laid out contiguously:

```text
Comparison            Levels   Bits    Shift   Columns
------------------------------------------------------------------
last_name                  5      3        0   last_name
first_name                 5      3        3   first_name
dob                        5      3        6   dob
email                      4      2        9   email
phone                      4      2       11   phone
postcode                   3      2       13   postcode
location                   4      2       15   latitude, longitude
address                    5      3       17   address_tokens
------------------------------------------------------------------
Packed width 20 bits of 32.
```

A configuration whose packed γ overflows 32 bits is rejected at parse time.

!!! note "The packed space is larger than the reachable space"
    A comparison with five levels occupies three bits, so three of the eight codes in that
    field name no level. The reachable space for the schema above is
    5·5·5·4·4·3·4·5 = 120,000 of the 1,048,576 packed values. Tabulating over the unreachable
    ones indexes past the end of the weight vector, so `Scorer::Reachable` excludes them.

## The parameters

For each comparison \(c\) and level \(\ell\):

| Parameter | Meaning |
| --- | --- |
| \(m_{c\ell}\) | \(P(\text{level } \ell \mid \text{the pair is a match})\) |
| \(u_{c\ell}\) | \(P(\text{level } \ell \mid \text{the pair is a non-match})\) |
| \(\lambda\) | the prior share of pairs that are matches |

Two mnemonics: \(m\) measures how *reliable* a field is (a surname that survives typos has
high \(m\) on `exact`), and \(u\) measures how *discriminating* it is (an email address
agreeing by chance has \(u \approx 6\times10^{-7}\), a postcode \(u \approx 10^{-4}\)).

## The match weight

Fellegi–Sunter assumes the comparisons are **conditionally independent given match status**.
Under that assumption the log-odds of a pair being a match decomposes into a sum:

\[
W(\gamma) \;=\; \log_2\frac{\lambda}{1-\lambda}
\;+\; \sum_c \log_2 \frac{m_{c,\ell_c(\gamma)}}{u_{c,\ell_c(\gamma)}}
\]

Everything is in **bits of evidence**, and everything stays in log space — with \(u\) near
\(10^{-7}\) across ten comparisons the raw Bayes-factor product reaches \(2^{\pm 200}\), so
it is never formed. The posterior is recovered only through the logistic form, which cannot
overflow:

\[
P(M \mid \gamma) \;=\; \frac{1}{1 + 2^{-W(\gamma)}}
\]

The per-level weights \(\log_2(m/u)\) are what [`estimate`](commands/estimate.md) prints in
its last column:

```text
Comparison        Level                                m             u    weight  note
--------------------------------------------------------------------------------------------
last_name         exact                         0.609559      2.38e-05     14.64  u exact
                  jaro_winkler >= 0.92          0.279753      6.00e-06     15.51  u from only 6 random pairs
                  jaro_winkler >= 0.85          0.078072      0.000209      8.55
                  else                          0.032616      0.999761     -4.94
email             exact                         0.577950      5.96e-07     19.89  u exact
                  else                          3.46e-06      0.944784    -18.06  m at the floor
```

Read that as: two records sharing a surname exactly is worth **+14.64 bits**; two records
whose emails differ is worth **−18.06 bits**. The prior contributes
\(\log_2(\lambda/(1-\lambda)) = -23.6\) bits at \(\lambda = 7.7\times10^{-8}\), which is the
hurdle every pair starts behind.

!!! warning "Conditional independence is doing real work"
    Record data violates it — forename and sex are correlated, postcode and address more so.
    Splink carries the same exposure. Correlated comparisons double-count evidence and push
    weights apart, which is one reason the posterior saturates so aggressively below.

## Term-frequency adjustment

A shared surname of "Zolnerowich" is far stronger evidence than a shared "Smith", but both
land on the same level and therefore the same γ. Term-frequency adjustment restores that
distinction by making the weight depend on the *value*, not just the level:

\[
\Delta_c(v) \;=\; w_c \cdot \log_2 \frac{u_{c,\text{exact}}}{\max(p_v,\; p_{\min,c})}
\qquad
W(\gamma, \text{values}) \;=\; W(\gamma) + \sum_{c \in \text{TF},\; \ell_c = \text{exact}} \Delta_c(v_c)
\]

where \(p_v\) is the value's relative frequency, \(p_{\min,c}\) is the singleton floor, and
\(w_c\) is a damping factor (`--tf-damping`, 1.0 by default). The adjustment applies only on
**exact** levels of comparisons declared `"term_frequency": true`, and only where both sides
agree — so the value can be read from either record.

TF adjustment is the one thing that breaks γ's sufficiency, which is why cpplink keeps it
**out of EM and applies it at scoring**. Estimation stays exact on the pattern histogram, and
the learned \(m\) and \(u\) keep their usual averaged-over-values meaning.

### The admissible bracket

The cost of TF is that emission can no longer be decided from γ alone: a pattern below
threshold on average can clear it for a rare enough value. cpplink brackets it instead. Once
the term-frequency tables exist, the extreme adjustments per comparison are known constants —
\(\Delta_{\max}\) from the clamp floor (the rarest value the column can hold) and
\(\Delta_{\min}\) from the column's most common value. That brackets **every** pair a pattern
can produce:

| Zone | Test | Action |
| --- | --- | --- |
| **Drop** | \(W(\gamma) + \sum \Delta_{\max} < \tau\) | skip the whole pattern, no TF lookup |
| **Check** | the bracket straddles \(\tau\) | compute the exact TF-adjusted score |
| **Emit** | \(W(\gamma) + \sum \Delta_{\min} \ge \tau\) | emit directly, no TF lookup |

Both bounds are **admissible**, so dropping a pattern on its bound emits *exactly* the edges
scoring every pair would have. That is not an assertion:
[`tests/predict_test.cpp`](https://github.com/4ment/cpplink/blob/main/tests/predict_test.cpp)
runs both ways at five thresholds and compares the edge sets, and `predict --no-bounds`
reproduces it at scale. Measured on the 1.8M-row sample at 20 bits:

```text
Zone           Candidate pairs       Share        Patterns
----------------------------------------------------------
drop               116,793,569      99.88%          60,973
check                      434       0.00%          14,358
emit                   142,541       0.12%          44,669
```

99.88% of candidate pairs never touch a term-frequency table.

!!! info "What the bracket is and isn't worth"
    The design originally billed this as "the piece most worth getting right". Measured, the
    bounded and exhaustive runs differ by about 0.5% of wall time at 1M rows: the bracket
    removes the TF lookup, and the TF lookup was never the expensive part — **the comparison
    is**, at roughly 3.5 µs of CPU per pair. The bracket stays because it is free, it is
    exact, and its value grows with the TF tables, which reach 124 MB at 18M rows where every
    lookup is a cache miss.

## Choosing a threshold

`predict` and `cluster` both take `--threshold` in bits or `--probability` in (0, 1). Use
bits. Conditional independence over eight comparisons puts matching pairs at 100+ bits, where
the posterior is 1 to more decimal places than a double carries, so probability is a nearly
useless knob — a posterior of 0.5 and one of 0.999999 select the same edges.

Swept on a 1M-row sample over a fixed edge set (edge-level quality):

| Threshold (bits) | Edges | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 79,413 | 0.9203 | 0.9161 | 0.9182 |
| 20 | 79,195 | 0.9228 | 0.9161 | **0.9194** |
| 60 | 77,995 | 0.9271 | 0.9063 | 0.9166 |
| 80 | 62,994 | 0.9490 | 0.7493 | 0.8374 |
| 100 | 24,253 | 0.9751 | 0.2964 | 0.4547 |

Everything from 0 to 40 bits is the same answer. Recall tops out at 0.916 against a blocking
recall of 0.911 — **scoring recovers essentially everything blocking reaches, so recall is a
blocking problem, not a model problem.**

## Comparison levels available

| Level type | Fires when | Reads |
| --- | --- | --- |
| `null` | any of the comparison's columns is missing on either side | any |
| `exact` | interned id or value equality | any |
| `levenshtein` | edit distance ≤ `threshold` | string |
| `jaro_winkler` | similarity ≥ `threshold` | string |
| `date_within` | \|difference\| ≤ `threshold` days | date |
| `numeric_within` | \|difference\| ≤ `threshold` | double |
| `geo_within` | great-circle distance ≤ `threshold` km | two doubles |
| `list_overlap` | intersection size ≥ `threshold` | string_list |
| `list_jaccard` | Jaccard similarity ≥ `threshold` | string_list |
| `else` | always; must be last | — |

A configuration that applies a level to a column type it cannot read is rejected at parse
time. See [the schema reference](reference/schema.md) for the full file format.

## Why the comparison is fast

Values are interned to dense `uint32` ids per column at load, so an `exact` level is an
integer equality — one or two nanoseconds, no string touched. Fuzzy levels are the expensive
path, and they run on **every non-matching pair**, which is nearly all of them.

Two mitigations ship, both exact:

- **Per-value signatures.** Beside each interned id sits the value's length and a 64-bit
  character-presence mask. A character present in `a` but absent from `b` cannot match, so
  \(\text{matches} \le |a| - \text{popcount}(\text{mask}_a \wedge \neg\text{mask}_b)\), which
  bounds Jaro from above and Levenshtein distance from below. Two loads and two popcounts
  reject a pair before a character is read. Measured on the blocked distribution: **−24% of
  comparison CPU**.
- **Myers' bit-vector Levenshtein**, dispatched when the shorter value fits a machine word.
  3–8× faster than the banded DP in isolation.

Both bounds are admissible, so neither may change a single γ — `predict --no-signatures` and
`ComparisonSet::Bind(..., use_signatures=false)` exist so the same data can be run both ways
and compared pair for pair.
