# Schema file

One JSON file tells cpplink three things: **what the columns are**, **how to compare them**,
and **how to generate candidate pairs**. Every command except `gen-sample` takes it as
`--schema`.

```json
{
  "unique_id": "id",
  "columns":     [ ... ],
  "comparisons": [ ... ],
  "blocking":    [ ... ]
}
```

`comparisons` and `blocking` are optional — `inspect` needs neither, `explain` needs only
`comparisons`, `explain-blocking` and `recall` need only `blocking`, and `estimate` and
`predict` need both.

The complete working example is
[`examples/sample_schema.json`](https://github.com/4ment/cpplink/blob/main/examples/sample_schema.json).

## `unique_id`

```json
"unique_id": "id"
```

The parquet column holding a stable record identifier, used to resolve `--pair`, truth files,
and cluster output. Optional: without it the row index is the id, but then `recall` and
`cluster --truth` cannot resolve their pairs.

!!! warning
    The `unique_id` column **must not also appear in `columns`**. It is loaded into its own
    arena, is never interned or compared, and declaring it twice is an error.

## `columns`

```json
"columns": [
  {"name": "last_name",      "type": "string"},
  {"name": "dob",            "type": "date"},
  {"name": "latitude",       "type": "double"},
  {"name": "address_tokens", "type": "string_list"}
]
```

| `type` | Stored as | Term frequencies? |
| --- | --- | --- |
| `string` *(default)* | interned to a dense `uint32` id | yes |
| `string_list` | CSR of interned ids, sorted and deduplicated per row | yes |
| `date` | `int32` days since 1970-01-01 | yes, dense over the observed range |
| `double` | as-is | **no** |

Interning is not a memory trick — it is why the hot loop is fast. An `exact` level becomes a
`uint32` equality, one or two nanoseconds, no string touched. Sorting list cells at load time
makes intersection, Jaccard and any-overlap a linear merge with no hashing and no allocation.

**A `double` column carries no term frequencies**, because two doubles agreeing to the last bit
is not a discrete event worth counting. That has one consequence worth knowing: **a `double`
column cannot drive any blocking source.** Compare coordinates with `geo_within` and block on
something else.

Names must be unique, and every column named by a comparison or blocking source must be
declared here.

## `comparisons`

A comparison is one logical field, which may span more than one column — a coordinate pair is
**one** comparison, not two.

```json
{
  "name": "location",
  "columns": ["latitude", "longitude"],
  "levels": [
    {"type": "null"},
    {"type": "geo_within", "threshold": 1},
    {"type": "geo_within", "threshold": 25},
    {"type": "else"}
  ]
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `columns` | yes | the store columns this comparison reads |
| `levels` | yes | ordered, non-empty; see below |
| `name` | no | defaults to the first column's name; used in reports |
| `term_frequency` | no | `true` enables [TF adjustment](../model.md#term-frequency-adjustment) on this comparison's `exact` level |

### Levels

**Levels are evaluated top-down and the first hit wins, so their order *is* the model.** Put
the strongest evidence first.

| `type` | `threshold` | Columns | Fires when |
| --- | --- | --- | --- |
| `null` | — | any | any of the comparison's columns is missing on either side |
| `exact` | — | any | interned id or value equality |
| `levenshtein` | required | 1 string | edit distance ≤ threshold |
| `jaro_winkler` | required | 1 string | similarity ≥ threshold |
| `date_within` | required | 1 date | \|difference\| ≤ threshold days |
| `numeric_within` | required | 1 double | \|difference\| ≤ threshold |
| `geo_within` | required | 2 doubles | great-circle distance ≤ threshold km |
| `list_overlap` | required | 1 string_list | intersection size ≥ threshold |
| `list_jaccard` | required | 1 string_list | Jaccard similarity ≥ threshold |
| `else` | — | any | always; **must be last** |

An optional `"label"` overrides the generated description in reports:

```json
{"type": "date_within", "threshold": 370, "label": "within a year"}
```

### Two conventions worth following

- **Put `null` first.** It then fires whenever either side is missing, which means (a) a
  missing value is never scored as a disagreement, and (b) its `u` is computable in closed form
  as a null count, for *any* comparison including multi-column ones.
- **Put `exact` second**, before any fuzzy level. Exact is an integer compare; every level
  above a fuzzy one that fires saves a string metric.

### Validation, all at parse time

Before a data file is opened, cpplink rejects a schema that:

- applies a level to a column type it cannot read (`"comparison "x" applies level "geo_within"
  to a string column, which it cannot read"`);
- gives a level the wrong number of columns (`geo_within` reads exactly two);
- omits a required `threshold`;
- does not end in `else`, or places an `else` before the end (levels after it can never fire);
- needs more than 32 bits for the packed γ.

The packed width is \(\sum_c \lceil \log_2 L_c \rceil\); [`explain`](../commands/explain.md)
prints the resulting layout.

## `blocking`

Sources are unioned **in the order declared**, and a pair is emitted by the first source that
produces it. Order them cheapest-predicate-first.

```json
"blocking": [
  {"type": "exact_value", "column": "email"},
  {"type": "rare_value", "column": "last_name", "max_frequency": 100},
  {"type": "minhash", "column": "last_name", "bands": 10, "rows_per_band": 4},
  {"type": "sorted_neighbourhood", "column": "last_name", "window": 20}
]
```

| Field | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `type` | — | all | `exact_value`, `rare_value`, `minhash`, `sorted_neighbourhood` |
| `column` | — | all | required; must not be a `double` column |
| `name` | `"<column> <type>"` | all | label used in reports |
| `max_frequency` | 100 | `rare_value` | block only on values seen at most this many times |
| `window` | 8 | `sorted_neighbourhood` | window width; emits exactly \(N \cdot w\) pairs |
| `bands` | 12 | `minhash` | LSH bands; each becomes its own source |
| `rows_per_band` | 4 | `minhash` | rows per band |
| `ngram` | 3 | `minhash` | character n-gram size for shingling |
| `seed` | 1 | `minhash` | hash seed |

`minhash` additionally requires a `string` column, and non-zero `bands`, `rows_per_band` and
`ngram`. `sorted_neighbourhood` requires a non-zero `window`.

See [Blocking](../blocking.md) for what each source is good at, and for the measured finding
that single-column MinHash was strictly dominated on this data.

!!! warning "Every source here selects on a single column"
    That is what makes them usable for [estimating `m`](../em.md#em-safety): the selection
    event factors as a condition on that column, so the column can be held out and every other
    comparison stays identifiable. A whole-record source (an embedding, a concatenated-record
    signature) would not, and would have to be prediction-only.

## A complete minimal schema

```json
{
  "unique_id": "id",
  "columns": [
    {"name": "first_name", "type": "string"},
    {"name": "last_name",  "type": "string"},
    {"name": "dob",        "type": "date"},
    {"name": "email",      "type": "string"}
  ],
  "comparisons": [
    {"name": "last_name", "columns": ["last_name"], "term_frequency": true, "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "jaro_winkler", "threshold": 0.92},
      {"type": "else"}]},
    {"name": "first_name", "columns": ["first_name"], "term_frequency": true, "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "levenshtein", "threshold": 1},
      {"type": "else"}]},
    {"name": "dob", "columns": ["dob"], "term_frequency": true, "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "date_within", "threshold": 2},
      {"type": "else"}]},
    {"name": "email", "columns": ["email"], "term_frequency": true, "levels": [
      {"type": "null"},
      {"type": "exact"},
      {"type": "else"}]}
  ],
  "blocking": [
    {"type": "exact_value", "column": "email"},
    {"type": "exact_value", "column": "dob"},
    {"type": "rare_value", "column": "last_name", "max_frequency": 100}
  ]
}
```

Four comparisons is the practical minimum, because an EM session holding one column out needs
**at least three free comparisons** to identify the mixture. See
[a session needs at least three free comparisons](../em.md#a-session-needs-at-least-three-free-comparisons).
