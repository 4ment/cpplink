# cpplink studio

A page for the stage before any of the pipeline runs: open a file, see what its columns hold, draft the schema, and price the blocking plan before a single pair is compared.

```sh
conda activate cpplink
pip install -e studio          # once; gives the cpplink-studio command
cpplink-studio data.parquet    # opens the page in a browser
cpplink-studio data.csv --draft schema.json --typed data.studio.parquet   # headless
```

The page reads the file through DuckDB and needs the `cpplink` binary only for the numbers it alone can produce.
It is found in `$CPPLINK`, then `build/cpplink`, then on `PATH`.

## What the tabs do

**Data.**
Opens a csv or parquet and shows the types DuckDB read or sniffed, mapped to cpplink's five (`string`, `string_list`, `date`, `double`, `boolean`).
cpplink reads parquet only, and its loader is strict: a string must be Arrow `string`, a date `date32`/`date64`/`timestamp`, a boolean `bool`, a list `list<string>`, and the `unique_id` a string.
So a csv, or a parquet holding an integer id or postcode, is rewritten with the types picked in the Columns tab, blanks and the chosen sentinel values as missing, and a row-number id when the file has none.
Every cast is a `TRY_CAST`, and a button counts the values a cast would lose before it is written.

**Columns.**
One row per column with the type and role as dropdowns, beside the statistics: nulls, distinct values, the effective cardinality `1 / P(two random rows agree)` and its bits, the top value's share, and flags for overrepresented values and missing markers such as `UNKNOWN` or `9999`.
The collision rate is `sum c(c-1) / (N(N-1))`, the same estimator the binary's `u` uses, not the plug-in `sum p^2`.
The role is guessed from the name (a synonym table reduced to letters, so `date_of_birth`, `DateBirth` and `dob` agree) and from the values (regexes on a sample) independently, and the reason is shown so the guess can be argued with.
The detail view draws the top values and the pairs each group size contributes, which is where a rare-value cap is read off.

**Correlation.**
On a sample of up to a million rows, for every pair of discrete columns: containment (one column's text inside the other's, the `first_name` inside `first_and_surname` shape, which `estimate` holds out as a tie), determination against the baseline share of the commonest value, null co-occurrence, and the U-side overlap in bits, `log2(u_ab / (u_a u_b))`.
The overlap is the trap: on a file holding duplicates, two near-unique columns collide together almost only on the duplicate pairs, so the joint reads as strong dependence between independent columns.
The largest excess of observed over expected joint collisions across all pairs is a lower bound on the duplicate pairs in the file, and an overlap is trusted only where independence alone would produce five times that many collisions; the rest are marked `dup`.
A button runs `cpplink profile --json` for the ledger over the whole file.

**Comparisons.**
The derived columns and the comparisons the roles proposed, each editable: columns, term frequency, and the levels as a table with type, threshold, label and, for a comparison over several string columns, the column a level reads.
Levels are evaluated top down, so the order is the model; every comparison starts at `null` and ends at `else`.
Templates follow `examples/sample_schema.json` and are data in `templates.py`.

**Blocking.**
The plan as an ordered table of sources, and every source priced live from the value counts: the candidate count of an exact-value or rare-value source is `sum c(c-1)/2` over the values under the cap and a window is closed form in the non-null rows, the same formulas `BlockingPlan::CountPairs` uses.
`tests/test_blocking.py` holds the two implementations to the same numbers, source by source, including a derived `email_username` reproduced in SQL.
What is shown per source: candidates, share of the plan, rows the source can see (non-null and under the cap), largest group.
For the plan: the sum (which bounds the deduplicated union from above), candidates per row, the reduction ratio, and the wall time at the measured 207 ns per candidate with the score ceiling on and 9 ns per pair to enumerate.
Graphs: candidates per source against the budget line; for the selected source, candidates and coverage against the cap (or the window), and pairs by group size.
A budget slider and a button draft a plan under it: exact agreement on the identifier-like columns whose count fits, rare-value caps on the names solved from what is left, one sorted neighbourhood on the surname, an exact-value source on the email username, never MinHash.
Three buttons reach the binary: `explain-blocking --json` for the exact counts (seconds), `--count` for the deduplicated union (enumerates every pair), and `recall --json` against a truth file for pair completeness, pair quality and the pairs each source is the first to reach.

**Schema.**
The JSON, editor-level problems (undeclared columns, a comparison not starting at `null`, blocking on a list or a double), save and download, and a button that runs `cpplink inspect` on it, since the binary's parser is the validator.

## Layout

| File | Holds |
|---|---|
| `cpplink_studio/data.py` | opening a file through DuckDB, the type mapping, the typed parquet export |
| `cpplink_studio/stats.py` | value counts, column statistics, content probes, what two columns share |
| `cpplink_studio/guess.py` | the role guesser: synonym table plus content probes |
| `cpplink_studio/templates.py` | role to comparison, derived columns and blocking source |
| `cpplink_studio/blocking.py` | the closed-form pricing, the knob curves, the drafted plan |
| `cpplink_studio/schema.py` | the draft, the schema JSON in and out, editor-level checks |
| `cpplink_studio/binary.py` | finding and running `cpplink`, parsed JSON back |
| `cpplink_studio/cli.py` | the entry point, and the headless `--draft` |
| `cpplink_studio/app.py` | the Streamlit page, one function per tab |
| `tests/` | pytest; the sample is written by the binary once per session |

The engine modules import no UI, so every number the page shows is reproducible from Python and tested without it.

```sh
cd studio && python -m pytest
```
