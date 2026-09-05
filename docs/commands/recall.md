# `recall`

**Goal:** measure what fraction of *known* duplicate pairs the blocking plan actually reaches
— per source, and as a union.

This is the falsifiability check for automatic blocking. With hand-written rules you can reason
about what they miss; with a rare-value cap or an LSH band count you cannot. **If recall is not
measured, a source that silently misses a third of the matches looks identical to one that
works.** That is the most likely way for this pipeline to produce confidently wrong output.

## Synopsis

```sh
cpplink recall --schema <schema.json> --truth <truth.csv> [--why]
               [--show-misses N] [--count] [--json] [--mode MODE]
               <file.parquet>...
```

| Option | Meaning |
| --- | --- |
| `--schema <file>` | required; must declare `blocking`, and the schema needs a `unique_id` |
| `--truth <file.csv>` | required; known matching pairs |
| `--why` | additionally report *why* each source missed the pairs it missed |
| `--show-misses N` | print `N` missed pairs in full; implies `--why` |
| `--count` | price the union by enumerating it, instead of by the sum over sources |
| `--json` | write the same numbers as JSON, for a harness that sweeps a knob |
| *(positional)* | required; the parquet file |
| `--mode dedup\|link\|link-and-dedup` | which pairs to enumerate: link when more than one file is given, otherwise dedup; see [linking](../linking.md) |

### The truth file

Two columns of `unique_id` values, with an optional header line beginning `id_a`:

```csv
id_a,id_b
r5,r9
r17,r19
```

Ids that do not resolve to a loaded record are counted and reported as `unresolved` rather
than failing the run. [`gen-sample --truth`](gen-sample.md) writes this format; in production
it comes from a high-precision deterministic rule, or from labels where they exist.

## Example

```sh
cpplink recall --schema examples/sample_schema.json \
               --truth examples/sample.truth.csv examples/sample.parquet
```

```text
Known pairs  143730 resolved
Pairs unblocked  1,619,999,100,000

Source                           Found      PC      Candidates        PQ    First to   Marg PQ
----------------------------------------------------------------------------------------------
email exact_value               85,844  59.73%          90,600     94.8%      85,844     94.8%
phone exact_value               99,181  69.01%         121,195     81.8%      34,908     28.8%
dob exact_value                110,863  77.13%      77,931,552    0.142%       9,654   0.0124%
last_name rare_value            77,395  53.85%      16,675,804    0.464%       1,080  0.00648%
last_name sorted_neighbo.       83,427  58.04%      35,999,790    0.232%         137 0.000381%
----------------------------------------------------------------------------------------------
Union                          131,623  91.58%     130,818,941    0.101%
Reduction ratio               0.999919

...

12107 known pairs are reachable by no source. No amount of scoring recovers
them: they are never generated as candidates.
```

## Reading the output

| Column | Meaning |
| --- | --- |
| `Found` | known pairs this source produces **on its own**, ignoring the plan order |
| `PC` | pair completeness: `Found` as a share of resolved known pairs |
| `Candidates` | pairs the source generates — exact, from the term frequencies |
| `PQ` | pair quality: `Found / Candidates` |
| `First to` | known pairs this source is the **first in plan order** to produce — its *marginal* contribution |
| `Marg PQ` | `First to / Candidates`: what the source alone reaches, per pair it costs |

`PC`, `PQ` and the reduction ratio under the union line are the three numbers the
blocking literature has scored indexing schemes with since Christen's 2012 survey, so a
plan here can be compared against a published one. They are reported together because
none of them means anything alone: **every source can buy pair completeness with
candidates**, so a source quoted at its recall and not its cost has not been evaluated.

### `First to` is the column that decides

`Found` measures a source in isolation and is almost always flattering, because the sources
overlap heavily on easy pairs. `First to` measures what the source adds to the plan it is
actually in.

Read the table above that way:

- **`email exact_value` is the plan's foundation.** It is first, so `First to` equals `Found`.
- **`dob exact_value` earns its place.** 9,654 pairs no earlier source reached, for 78M
  candidates — the strongest single contributor after the near-unique identifiers, and it is a
  low-cardinality column that a rare-value cap would have excluded entirely.
- **`last_name sorted_neighbourhood` does not.** It finds 58% of known pairs on its own, but
  only **137** that no earlier source reached, at 36M candidate pairs — 31% of the whole
  plan's union for 0.1% of its recall.

`Marg PQ` is that argument as one number, and the spread across this plan is five orders of
magnitude: `email` reaches one known pair for every candidate it generates, `last_name
sorted_neighbourhood` one for every 262,000. **That ratio, not recall, is what a source is
dropped on.** Both denominators are the source's whole candidate count rather than its
marginal one, so `Marg PQ` understates a source that sits late in the plan — it is a lower
bound, and a source it condemns is condemned.

**Do not add a blocking source without running this to see its marginal contribution.** The
same measurement is what showed MinHash to be strictly dominated on this data: ten bands, 63.2
billion candidates, and between 533 and 867 marginal pairs each. See
[Blocking](../blocking.md#minhash-minhash-lsh-over-character-n-grams).

### Comparing methods rather than sources

The table ranks the sources of *one* plan. To rank the methods themselves — is a rare-value
cap or a sorted-neighbourhood window the better way to spend a candidate budget — each has to
be swept over its own knob, because a single operating point per method compares nothing.
`bench/sweep_blocking.py` drives `--json` over those grids and reports the frontier: at each
candidate budget, the method that reaches the most known pairs. See
[bench/README.md](https://github.com/4ment/cpplink/blob/main/bench/README.md).

### The union line, and the ceiling

```text
Union                          131,623  91.58%     130,818,941    0.101%
Reduction ratio               0.999919

12107 known pairs are reachable by no source.
```

**This is a hard ceiling on the entire pipeline.** A pair that is never generated as a
candidate is never compared, never scored, and never clustered — no threshold, no model
improvement and no amount of CPU recovers it.

Measured end to end on this sample, `cluster --truth` reports 0.9158 recall at 20 bits —
against this blocking recall of 0.9158. Scoring is recovering essentially everything blocking
reaches, so **recall is a blocking problem, not a model problem**. If you want better
recall, the lever is here, not in the model.

## How it is computed

`recall` does not enumerate the candidate set. For each known pair it asks
`BlockingPlan::ProducedByAny(a, b)`, which costs O(sources) key comparisons per pair rather
than a walk over billions of candidates — which is why this runs in seconds even when the plan
it is measuring would take an hour to enumerate.
