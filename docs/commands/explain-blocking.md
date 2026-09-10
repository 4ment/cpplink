# `explain-blocking`

**Goal:** price every blocking source in the plan — exactly, from the term-frequency tables,
without enumerating a single candidate pair.

This is the command that stops a run from taking a day. At scale a careless source is
catastrophic: blocking on surname alone puts every "Smith" in one group, and a single 100k
group emits 5×10⁹ pairs by itself. `explain-blocking` makes that visible in seconds.

## Synopsis

```sh
cpplink explain-blocking --schema <schema.json> [--count] [--mode MODE]
                         <file.parquet>...
```

| Option | Meaning |
| --- | --- |
| `--schema <file>` | required; must declare `blocking` |
| `--count` | also compute the exact deduplicated **union** size, not just the per-source sum |
| *(positional)* | required; the parquet file |
| `--mode dedup\|link\|link-and-dedup` | which pairs to enumerate: link when more than one file is given, otherwise dedup; see [linking](../linking.md) |
| `--all-pairs` | ignore the schema's sources and price the plan that does no blocking: every pair the mode admits. See [no blocking at all](../blocking.md#no-blocking-at-all) |

Without `--count` the report stops at the per-source table and the sum. `--count` costs an
extra pass over the plan's keys but still enumerates nothing.

## Example

```sh
cpplink explain-blocking --schema examples/sample_schema.json --count examples/sample.parquet
```

```text
Records          1,800,000
Pairs unblocked  1,619,999,100,000

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
                                             89.4% of the sum

Counts are exact and come from the term frequencies, with no pairs enumerated.
The sum bounds the deduplicated union from above.
Blocking keeps 8.08e-05 of all possible pairs.
```

## Reading the output

### The header

| Field | Meaning |
| --- | --- |
| `Records` | rows loaded |
| `Pairs unblocked` | \(N(N-1)/2\) — the work blocking is avoiding |

### The source table

One row per bound source, **in plan order**. A `minhash` spec expands into one row per band,
because each band independently makes a pair a candidate.

| Column | Meaning |
| --- | --- |
| `EM-safe` | whether this source may feed [parameter estimation](../em.md#em-safety). `yes` means its selection event factors as a condition on one column, so that column can be held out and every other `m` stays identifiable |
| `Candidate pairs` | \(\sum_v n_v(n_v-1)/2\) over the source's groups (or \(N \cdot w\) for sorted neighbourhood) — **exact**, from term frequencies, nothing enumerated |
| `Largest group` | the biggest group the source would enumerate, in rows |

**`Largest group` is the number that kills runs.** Cost is quadratic in group size, so one
group of 100k rows is 5×10⁹ pairs by itself regardless of how well the rest of the plan
behaves. Anything in the thousands deserves a second look; anything in the tens of thousands
is a bug in the plan. Here the worst is `dob` at 127 rows, which is fine.

### The totals

| Line | Meaning |
| --- | --- |
| `Sum over sources` | what the plan would cost with no deduplication. **An upper bound on the union** |
| `Union, deduplicated` | what will actually be enumerated, after the "not produced by an earlier source" predicate |
| `% of the sum` | how much the sources overlap. 89.4% means the predicate removes 10.6% of the work |
| `Blocking keeps ...` | the union as a fraction of all possible pairs |

A union close to 100% of the sum means the sources are nearly disjoint, which is what you want
— each is finding different pairs. A union far below it means sources are duplicating each
other's work, and [`recall`](recall.md) will show whether the duplicated source contributes
anything unique.

!!! note "Exactness is a test, not a claim"
    `CountPairs` reads only term frequencies. If it ever disagreed with actual enumeration the
    report would be lying, so that equivalence is asserted in the test suite.

## Using it

The workflow is: change the plan, re-run this, look at `Candidate pairs` and `Largest group`,
then run [`recall`](recall.md) to see what the change bought. The two commands answer the two
halves of the same question — cost here, benefit there — and neither is interpretable alone.

Rough scale for what a number means, given comparison throughput of roughly 3.2M
candidates/second on eight threads:

| Union size | `predict` wall time |
| ---: | --- |
| 10⁸ | ~30 s |
| 10⁹ | ~5 min |
| 10¹⁰ | ~50 min |
| 10¹¹ | most of a day — reconsider the plan |
