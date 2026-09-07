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
Known pairs  143728 resolved
Pairs unblocked  1,619,999,100,000

Source                           Found      PC      Candidates        PQ    First to   Marg PQ
----------------------------------------------------------------------------------------------
email exact_value               90,681  63.09%         104,702     86.6%      90,681     86.6%
phone exact_value              105,715  73.55%         138,575     76.3%      38,719     27.9%
dob exact_value                119,118  82.88%      77,957,842    0.153%      11,643   0.0149%
last_name rare_value            81,589  56.77%      16,659,085     0.49%       1,414  0.00849%
last_name sorted_neighbo.       88,361  61.48%      35,999,790    0.245%         200 0.000556%
----------------------------------------------------------------------------------------------
Union                          142,657  99.25%     130,859,994    0.109%
Reduction ratio               0.999919

...

1071 known pairs are reachable by no source. No amount of scoring recovers
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
- **`dob exact_value` earns its place.** 11,643 pairs no earlier source reached, for 78M
  candidates — the strongest single contributor after the near-unique identifiers, and it is a
  low-cardinality column that a rare-value cap would have excluded entirely.
- **`last_name sorted_neighbourhood` does not.** It finds 61% of known pairs on its own, but
  only **200** that no earlier source reached, at 36M candidate pairs — 28% of the whole
  plan's union for 0.14% of its recall.

`Marg PQ` is that argument as one number, and the spread across this plan is five orders of
magnitude: `email` reaches one known pair for every 1.2 candidates it generates, `last_name
sorted_neighbourhood` one for every 180,000. **That ratio, not recall, is what a source is
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
Union                          142,657  99.25%     130,859,994    0.109%
Reduction ratio               0.999919

1071 known pairs are reachable by no source.
```

**This is a hard ceiling on the entire pipeline.** A pair that is never generated as a
candidate is never compared, never scored, and never clustered — no threshold, no model
improvement and no amount of CPU recovers it.

Measured end to end on this sample, `cluster --truth` reports 0.9949 recall at 20 bits against
this blocking recall of 0.9925. Scoring is recovering essentially everything blocking reaches,
so **recall is a blocking problem, not a model problem**. If you want better recall, the lever
is here, not in the model.

!!! note "Why end-to-end recall can sit slightly *above* the blocking ceiling"
    The ceiling binds pair by pair, and clustering does not work pair by pair. `recall` scores
    the truth file as listed, while `cluster` scores it closed transitively; a pair blocking
    never produced is still asserted if the partition connects its two rows through a third
    record. So union–find recovers a little of what blocking missed, and the ceiling is a
    ceiling on the **edges**, not on the partition built from them.

## How it is computed

`recall` does not enumerate the candidate set. For each known pair it asks
`BlockingPlan::ProducedByAny(a, b)`, which costs O(sources) key comparisons per pair rather
than a walk over billions of candidates — which is why this runs in seconds even when the plan
it is measuring would take an hour to enumerate.
