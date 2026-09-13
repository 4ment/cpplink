# cpplink studio

The page for the stage before the pipeline runs.
It opens a csv or parquet, profiles every column, guesses what each holds, drafts the schema with the types and roles as dropdowns, and prices the blocking plan before a single pair is compared.

```sh
conda activate cpplink
pip install -e studio
cpplink-studio data.parquet
```

Headless, the same engine writes the default schema and the typed parquet the binary reads:

```sh
cpplink-studio data.csv --draft schema.json --typed data.studio.parquet
```

## What it answers

**What is in the file.**
Per column: the type read or sniffed, the nulls, the distinct values, the effective cardinality `1 / P(two random rows agree)` and its bits (the most an exact agreement on the column can be worth), the top value's share, and the values that look like missing markers or hold far more rows than a value of that column typically does.
The collision rate is the pair form `sum c(c-1) / (N(N-1))`, the same estimator [`profile`](commands/profile.md) and `u` use.

**What two columns share.**
Containment, determination against the baseline, null co-occurrence, and the U-side overlap in bits.
The overlap is inflated by the file's own duplicates wherever two near-unique columns are involved, so the page bounds the duplicate pairs from below by the largest excess of observed over expected joint collisions and trusts an overlap only where independence alone would produce five times that many collisions.
For the whole-file ledger, a button runs [`profile`](commands/profile.md).

**What the schema should say.**
A role per column, guessed from the name and from the values independently with the reason shown, turned into the comparison, derived columns and blocking source a column of that role gets.
Everything is editable, and the binary's parser is the validator: a button runs [`inspect`](commands/inspect.md) on the draft.

**What a blocking plan would do.**
Every `exact_value`, `rare_value` and `sorted_neighbourhood` source is priced live from the value counts with the closed forms [`explain-blocking`](commands/explain-blocking.md) uses, so a cap moves on a slider and the candidate count moves with it.
Per source: candidates, share of the plan, the rows the source can see, the largest group; per plan: the sum, candidates per row, the reduction ratio and a wall-time estimate from the measured cost per candidate.
Graphs draw candidates per source against the budget, candidates and coverage against the cap, and the pairs each group size contributes.
A budget slider drafts a plan under it.
MinHash, the deduplicated union and any derived column whose transform is not reproduced in SQL are priced by the binary, through `explain-blocking --json` and `--count`; with a truth file, [`recall`](commands/recall.md) reports what the plan reaches.

## Why it reads parquet and writes parquet

cpplink reads parquet only, and its loader wants the Arrow types exactly: `string` (not `large_string`, not an integer), `date32`/`date64`/`timestamp`, `double`/`float`, `bool`, `list<string>`, and a string `unique_id`.
The Data tab rewrites any csv, or any parquet holding an integer id or postcode, with the types picked in the Columns tab, blanks and the chosen sentinels as missing, a row-number id when there is none, and counts what each cast would lose before writing it.

The tests in `studio/tests/` hold the page's closed forms to `explain-blocking`'s numbers source by source.
