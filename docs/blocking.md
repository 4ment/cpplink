# Blocking

Scoring every pair of 1.8M records means 1.6 trillion comparisons; at 18M records it is
1.6×10¹⁴. Blocking is what makes the problem finite: it generates a *candidate set* that
contains nearly all the true matches and a vanishing fraction of everything else.

```text
Records          1,800,000
Pairs unblocked  1,619,999,100,000
...
Union, deduplicated                               116,939,057
Blocking keeps 8.08e-05 of all possible pairs.
```

Unlike splink, cpplink's blocking is **automatic** rather than rule-written. At 18M records
hand-tuning rules is the slowest part of the work, and the term-frequency tables the model
already needs turn out to be enough to generate good candidates on their own.

## A blocking source is an iterator, not a join

Blocking is not a rule language. A **pair source** is anything that yields `(row_a, row_b)` on
demand, and the rest of the pipeline neither knows nor cares which source produced a pair.
Several sources run in union, in the order they are declared.

```json
"blocking": [
  {"type": "exact_value", "column": "email"},
  {"type": "exact_value", "column": "phone"},
  {"type": "exact_value", "column": "dob"},
  {"type": "rare_value", "column": "last_name", "max_frequency": 100},
  {"type": "sorted_neighbourhood", "column": "last_name", "window": 20}
]
```

### Union without a global DISTINCT

For source *k*, emit a pair only if no **earlier** source would have produced it:

```text
emit(a,b) from source k  ⟺  produces_k(a,b)  ∧  ∀ j<k : ¬produces_j(a,b)
```

The predicate is cheap for every source type, and there is **no global deduplication state of
any kind**:

| Source | `produces_j(a,b)` test | Cost |
| --- | --- | --- |
| Exact value / rare value | `key_j(a) == key_j(b)` | O(1) |
| MinHash LSH band | band hash equality | O(1) |
| Sorted neighbourhood | ranks within the window in pass *j* | O(1) |

Order sources cheapest-predicate-first. On the plan above the union is 116.9M pairs against a
summed 130.8M — the predicate removed 10.6% of the work without materializing anything.

!!! note "This much is shared with splink"
    splink applies the same rule: `splink/internals/blocking.py` emits
    `AND NOT (previous_rules)` for every blocking rule after the first, during blocked-pair
    generation, so duplicates never reach its comparison-vector table either. The difference
    is what happens to the pairs that *survive* the predicate — splink materializes them as a
    relation, cpplink folds each one into a histogram and forgets it. The saving is the table,
    not the deduplication.

!!! warning "Nulls must never block together"
    `KeyOf` returns `kNoKey` for a missing value, and two rows carrying `kNoKey` never become
    a candidate pair. This is easy to get wrong when keys are interned ids and null has an id.

## The four sources

### `exact_value` — every pair sharing a value

```json
{"type": "exact_value", "column": "email"}
```

The plain equi-join. For each distinct value, emit all pairs sharing it. Cost is
\(\sum_v n_v(n_v-1)/2\) over the column's term-frequency table.

Strongest on high-cardinality identifiers (email, phone), but **also the best use of
low-cardinality columns**: `dob` at 20,821 distinct values over 1.8M rows is the single
strongest contributor in this plan at 82.88% recall, for 78M candidates.

### `rare_value` — the primary automatic source

```json
{"type": "rare_value", "column": "last_name", "max_frequency": 100}
```

Block only on values seen at most `max_frequency` times:

\[
\text{pairs}(j, \text{cap}) \;=\; \sum_{v\,:\,n_v \le \text{cap}} \frac{n_v(n_v-1)}{2}
\]

Three properties fall out, and the third is the interesting one:

- **The cost is exactly computable from the TF table alone**, in O(distinct values) — no sort,
  no enumeration, no sampling.
- **It is automatic.** Nobody writes rules; you set a frequency cap and it adapts to the data.
- **It selects on the same quantity the model scores.** Agreement on a rare value *is* the
  high-match-weight event that [term-frequency adjustment](model.md#term-frequency-adjustment)
  encodes. The "Smith" problem dissolves, because common values are skipped exactly because
  they blow up quadratically *and* carry little evidence. **Cost and information content
  coincide** — no other source here has that alignment.

!!! note "The cap cannot be one global constant"
    At 18M rows with a cap of 100, `dob` yields 40 pairs and `postcode` yields zero, because
    18M records over 21k dates puts every value far above the cap. Rare-value blocking is
    useless on low-cardinality columns; those belong in `exact_value` instead. `inspect`
    prints a "Rare pairs" column so this is visible before a run.

### `minhash` — MinHash LSH over character n-grams

```json
{"type": "minhash", "column": "last_name", "bands": 10, "rows_per_band": 4, "ngram": 3}
```

A band hash *is* a blocking key, so `b` bands are `b` more key-sorted sources and everything
above applies untouched. Its distinctive advantage over ANN is an *analytic* recall curve:

\[
P(\text{pair becomes a candidate}) \;=\; 1 - (1 - s^r)^b
\]

with \(s\) the Jaccard similarity, \(r\) rows per band and \(b\) bands — so recall at a chosen
similarity threshold is computed before the run rather than measured after.

!!! danger "Measured: MinHash over a single column is strictly dominated on this data"
    Two EM-safe configurations at 18M rows, priced by `explain-blocking` and scored against
    1.44M planted duplicate pairs:

    | Configuration | Candidate pairs | Recall |
    | --- | ---: | ---: |
    | email + phone + rare(last_name) + SN + **10 MinHash bands** | 63,463,910,811 | 89.46% |
    | email + phone + **dob exact** + rare(last_name) + wider SN | 8,328,758,176 | **91.12%** |

    **7.6× the cost for less recall.** The ten bands account for 63.2 billion of the 63.5
    billion candidates and add 4.3% recall between them — roughly a million candidate pairs
    per additional true pair found. Worse, nine of the ten are near-pure waste: band 0
    contributed 55,545 pairs no earlier source reached, bands 1–9 contributed 533 to 867 each.

    The reason is that **bands over the same column are strongly correlated in the population
    that matters**, even though the S-curve treats them as independent given Jaccard
    similarity. A duplicate's surname is bimodal — either it survived corruption intact and
    every band agrees, or it was mangled and they all miss. There is little probability mass
    in between for extra bands to catch. The S-curve is not wrong; it is being averaged over
    the wrong distribution. A band also degenerated: one key covered 95,808 records and
    produced 14.2 billion pairs by itself.

### `sorted_neighbourhood` — the cheap complement

```json
{"type": "sorted_neighbourhood", "column": "last_name", "window": 20}
```

Sort by a normalized key, slide a window of `w`, emit within-window pairs. Exactly \(N \cdot w\)
pairs, \(O(N \log N)\), near-zero memory. Unglamorous, nearly free, and it fails in different
directions than the others — it catches values that sort adjacent but share no rare token.

### `ann` — deferred, and prediction-only

Approximate nearest neighbours over record embeddings reach matches no lexical method can:
nicknames, transliterations, reordered name parts. It is not built yet, and when it is it will
be **prediction-only**, because a whole-record embedding does not satisfy the
[EM-safety criterion](em.md#em-safety). Three further cautions apply:
the embedding is the hard
part rather than the index, distance and match weight are *different orderings* (linkage often
turns on one rare field agreeing while everything else differs — far away in cosine space),
and kNN is asymmetric, so emitting each pair exactly once needs all neighbour lists resident.

## Cost before enumeration

At scale a careless source is catastrophic — blocking on surname alone puts every "Smith" in
one group, and a single 100k group emits 5×10⁹ pairs by itself. Every source above can price
itself **without enumerating anything**, which is what
[`explain-blocking`](commands/explain-blocking.md) reports:

```text
Source                        EM-safe         Candidate pairs   Largest group
-----------------------------------------------------------------------------
email exact_value             yes                     104,702               9
phone exact_value             yes                     138,575              11
dob exact_value               yes                  77,957,842             129
last_name rare_value          yes                  16,659,085             100
last_name sorted_neighbourho. yes                  35,999,790              21
-----------------------------------------------------------------------------
Sum over sources                                  130,859,994
Union, deduplicated                               116,939,057
```

`CountPairs` is exact and reads only term frequencies. If it ever disagreed with enumeration
the report would be lying, so **that equivalence is a test**. `Largest group` is the number to
watch: one huge group is quadratic and is the usual reason a run never finishes.

!!! note "Also largely shared with splink"
    splink's `count_comparisons_from_blocking_rule` and `n_largest_blocks` answer the same
    question the same way — by aggregating block sizes rather than enumerating pairs. What is
    particular here is that the counts cost nothing extra, because they read the
    term-frequency tables the model already holds; that the report carries the `EM-safe`
    column deciding where a source may be used; and that the
    [recall harness](#recall-has-to-be-measured) sits beside it.

## Recall has to be measured

With hand-written rules you can reason about what they miss. **With automatic sources you
cannot, so recall has to be measured or the whole approach is unfalsifiable.**
[`cpplink recall`](commands/recall.md) takes known-match pairs and reports what each source
retrieves, individually and unioned:

```text
Source                           Found      PC      Candidates        PQ    First to   Marg PQ
----------------------------------------------------------------------------------------------
email exact_value               90,681  63.09%         104,702     86.6%      90,681     86.6%
phone exact_value              105,715  73.55%         138,575     76.3%      38,719     27.9%
dob exact_value                119,118  82.88%      77,957,842    0.153%      11,643   0.0149%
last_name rare_value            81,589  56.77%      16,659,085     0.49%       1,414  0.00849%
last_name sorted_neighbo.       88,361  61.48%      35,999,790    0.245%         200 0.000556%
----------------------------------------------------------------------------------------------
Union                          142,657  99.25%     130,859,994    0.109%
```

The **`First to`** column is the one that decides whether a source stays, and `Marg PQ`
prices it. `last_name sorted_neighbourhood` finds 61% of known pairs on its own, but only 200
of them that no earlier source reached — for 36M candidate pairs. **Do not add a blocking
source without running `recall` to see its marginal contribution.**

The 0.75% of known pairs that no source reaches is a hard ceiling on the edges: no amount of
scoring recovers them, because they are never generated as candidates. Measured end to end,
`cluster --truth` reports 0.9949 recall at a 20-bit threshold against this blocking recall of
0.9925 — **recall is a blocking problem, not a model problem**. (The partition can sit
slightly above the ceiling because it is transitive; see
[`recall`](commands/recall.md#the-union-line-and-the-ceiling).)

## Enumeration is not the bottleneck

Walking the deduplicated union of 8.25 billion pairs at 18M records, including the
earlier-source predicate, took **142 seconds single-threaded** — about 17 ns per pair. Against
the ~4,090 ns it takes to *evaluate* a pair, candidate generation is **240× cheaper than the
comparison it feeds**.

Effort spent making blocking faster is effort wasted. The place to spend it is either the
comparison (see [the model](model.md#why-the-comparison-is-fast)) or the plan itself: a
frequency cap or band count yielding 10¹¹ candidates instead of 5×10⁹ turns a two-minute run
into a day, and no amount of engineering below it helps. `explain-blocking` exists to make
that visible in seconds rather than hours.

## Honest caveat on the measurements

These findings come from synthetic data whose corruption model drops email 35% of the time and
phone 25%, independently. Real data with weaker identifiers would shift the balance back
toward fuzzy sources. **What generalizes is the method** — price every source, measure its
*marginal* recall, and delete the ones that do not earn their candidates.
