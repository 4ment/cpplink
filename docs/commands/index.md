# Commands

```text
usage: cpplink <command> [options]

commands:
  inspect     load a parquet file and report cardinality and memory
  profile     what the columns can be worth, what a matching pair will
              score, and which pairs of them are the same evidence twice
  levels      check the schema's fuzzy thresholds against the column they
              run on, and propose better ones
  explain     show the levels a single pair lands on, and with a model
              the waterfall of bits behind its score
  explain-blocking  price every blocking source without enumerating
  recall      measure what fraction of known pairs blocking reaches,
              and with --why, diagnose the ones it does not
  estimate    learn m, u and lambda and write the model
  completeness  estimate blocking recall with no known pairs at all
  predict     score the candidate pairs and write the edges above a threshold
  cluster     join the scored edges into duplicate clusters
  rescore     re-score a spilled run under a new model, without comparing again
  gen-sample  write a sample parquet file with planted duplicates

options:
  -h, --help       show this message and exit
  -v, --version    show the version and exit
```

Every command that reads data takes **one or more** parquet files. They are read in order into
one store and each becomes a dataset, so two files mean linking rather than deduplicating — see
[linking two files](../linking.md).

## What each one is for

| Command | Answers | Reads | Writes |
| --- | --- | --- | --- |
| [`inspect`](inspect.md) | Did the data load as I expected, and what does it cost in memory? | parquet + schema | stdout |
| [`profile`](profile.md) | What can these columns be worth, what will a matching pair score, and which pairs of them are the same evidence twice? | parquet + schema | stdout |
| [`levels`](levels.md) | Are my fuzzy thresholds anywhere near right for this column? | parquet + schema | stdout, optionally a schema |
| [`explain`](explain.md) | Why did *this* pair get *that* pattern, and with `--model`, that score? | parquet + schema (+ model) | stdout |
| [`explain-blocking`](explain-blocking.md) | How many candidate pairs will this plan cost me? | parquet + schema | stdout |
| [`recall`](recall.md) | What fraction of true matches does blocking even reach? | parquet + schema + truth | stdout |
| [`estimate`](estimate.md) | What are `m`, `u` and `λ`? | parquet + schema | `model.json` |
| [`completeness`](completeness.md) | What fraction of true matches does blocking reach, with no truth file? | parquet + schema + model | stdout |
| [`predict`](predict.md) | Which pairs score above the threshold? | parquet + schema + model | edge shards |
| [`rescore`](rescore.md) | What would a different model have scored? | parquet + schema + model + spill | edge shards |
| [`cluster`](cluster.md) | Which records are the same entity? | parquet + schema + edges | `clusters.csv` |
| [`gen-sample`](gen-sample.md) | Give me realistic data with known answers. | — | parquet + truth csv |

## The order you actually run them

```text
gen-sample ──▶ inspect ──▶ profile ──▶ levels ──▶ explain-blocking ──▶ recall ──▶ estimate ──▶ predict ──▶ cluster
   (or your      sanity      is there    are the   is the plan          does the    model.json    edges/     clusters.csv
    own data)     check      anything   thresholds affordable?          plan reach
                             to find?     right?                         the matches?
```

Tuning is a loop back through the middle of that line, not a rerun of it. `predict --spill`
keeps each pair's agreement pattern, and [`rescore`](rescore.md) replays it under a new
model in a fraction of the time the first pass took — 0.02 s against 14 s on the 1M sample,
because every string metric has already been paid for.

The first six are diagnostics and cost seconds. They exist because the two decisions that
determine the quality of the output — the comparison levels and the blocking plan — are both
made *before* any expensive work runs, and both are hard to reason about without measurement.
[`profile`](profile.md) comes first among them because it needs neither: no model, no blocking
plan and no known pairs, just the columns. [`levels`](levels.md) is the one that answers the
first of those two decisions, and it needs nothing more than `profile` does.

## Conventions shared by every command

- **`--schema <file.json>` is required** by everything except `gen-sample`. It declares
  columns, comparisons and blocking sources; see [the schema reference](../reference/schema.md).
- **The parquet file is a positional argument**, always last in the examples here.
- Errors go to stderr prefixed with `cpplink:` or `cpplink <command>:`, and the exit code is
  `1`. A configuration error — a level applied to a column type it cannot read, a comparison
  with no trailing `else`, a γ that overflows 32 bits — is reported at parse time, **before a
  data file is opened**.
- Thresholds are given either in **bits** of match weight (`--threshold`) or as a **posterior
  probability** (`--probability`). Prefer bits; see
  [choosing a threshold](../model.md#choosing-a-threshold).
- `cpplink::Run(args, out, err)` takes explicit streams rather than using `std::cout`, so
  every output shown in these pages is captured verbatim in the test suite.
