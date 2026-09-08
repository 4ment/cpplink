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

### Derived columns

A column can be **computed from another** instead of being read from the file:

```json
"columns": [
  {"name": "surname",     "type": "string"},
  {"name": "surname_key", "derive": {"from": "surname", "transform": "soundex"}},
  {"name": "dob",         "type": "date"},
  {"name": "birth_year",  "derive": {"from": "dob", "transform": "year"}},
  {"name": "name_key",    "derive": {"from": "full_name",
                                     "transform": ["normalize", "sorted_tokens"]}}
]
```

`transform` is one name or a list applied in order. `type` is not given: it comes from the
transform.

| Transform | Reads | Produces | Does | `"Smith-Jones, John"` → |
| --- | --- | --- | --- | --- |
| `normalize` | `string` | `string` | lowercases ASCII; every byte that is neither a letter nor a digit becomes one space, with runs collapsed and the ends trimmed | `smith jones john` |
| `sorted_tokens` | `string` | `string` | splits on whitespace, sorts the tokens, rejoins with single spaces | `John Smith-Jones,` |
| `soundex` | `string` | `string` | the Soundex phonetic key | `S532` |
| `year` | `date` | `string` | the year | `1974` |
| `month` | `date` | `string` | the month, zero-padded | `03` |
| `day` | `date` | `string` | the day of the month, zero-padded | `09` |
| `year_month` | `date` | `string` | `YYYY-MM` | `1974-03` |

A separator becomes a space under `normalize` rather than vanishing, which is what lets
`["normalize", "sorted_tokens"]` chain and still see tokens: that pair takes the same input to
`john jones smith`, where `sorted_tokens` alone leaves the punctuation attached and sorts on
it. Bytes above ASCII are passed
through unchanged: lowercasing them needs a locale this project does not carry, and dropping
them would erase most of a name rather than normalize it.

Every transform produces a `string`, which is the point of them: an `exact` level on an
interned key is an integer equality. A derived column is interned, counted, blocked on,
compared and profiled exactly like a read one, and nothing downstream knows it was derived.

**The transform runs once per distinct value of the source dictionary**, not once per row.
A phonetic key over 18M records is one Soundex per distinct surname plus a gather.

!!! note "An empty result is a missing value, not an empty one"
    If the transform yields nothing — a name that was entirely punctuation, say — the derived
    cell is **null**. Treating it as the empty string would put every such row in one blocking
    group, which is exactly the runaway a phonetic key is supposed to avoid.

!!! warning "A derivation is a declared tie, and estimation is told about it"
    A derived column and its source are not independent evidence, so
    [`estimate`](../commands/estimate.md) holds the source out of any session the derived
    column conditions on, and vice versa. It has to be **declared** rather than detected:
    containment does not hold between `smith` and `S530`, and the pairwise dependence checks
    read the pair as *unresolved*. Blocking on a phonetic key without that hold-out is the
    `first_and_surname` runaway again.

Type errors are caught at parse time: `{"from": "surname", "transform": "year"}` is rejected
because `year` reads a date, and so is a chain whose stages do not fit together.

!!! tip "Measure before adding one"
    On `historical_50k` a Soundex key on `surname` takes blocking completeness from 84.99% to
    88.28% for 12% more candidates, and end-to-end F1 from 0.8676 **down** to 0.8658: the
    extra pairs disagree on surname, so they do not clear the threshold, while the extra
    candidates cost a few false positives that do. The mechanism is cheap and correct, and on
    that data it reaches more pairs without reaching better ones. Run
    [`recall`](../commands/recall.md) rather than assuming.

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
| `list_contains` | — | 1 string + 1 string_list | either row's value is an element of the other row's list |
| `contains_levenshtein` | required | 1 string + 1 string_list | either row's value is within threshold edits of an element of the other's list |
| `contains_jaro_winkler` | required | 1 string + 1 string_list | either row's value is that similar to an element of the other's list |
| `list_levenshtein` | required | 1 string_list | some element pair is within threshold edits |
| `list_jaro_winkler` | required | 1 string_list | some element pair is at least this similar |
| `else` | — | any | always; **must be last** |

An optional `"label"` overrides the generated description in reports:

```json
{"type": "date_within", "threshold": 370, "label": "within a year"}
```

### The closest pair of two lists: `list_levenshtein` and `list_jaro_winkler`

`list_overlap` and `list_jaccard` read the elements two rows **share**.
The pairwise levels read the closest pair of elements over the **cross product** of the two
rows' lists, so a pair of rows whose lists intersect in nothing can still agree.

```json
{"name": "emails", "columns": ["emails"], "levels": [
  {"type": "null"},
  {"type": "list_overlap", "threshold": 1},
  {"type": "list_levenshtein", "threshold": 1},
  {"type": "list_jaro_winkler", "threshold": 0.9},
  {"type": "else"}]}
```

`list_levenshtein` fires when *some* element of one list is within `threshold` edits of
*some* element of the other; `list_jaro_winkler` when some element pair is at least
`threshold` similar.
Both read one `string_list` column, the same as the set-valued levels.

**Put the set-valued levels above them.**
A shared element is a pair at distance zero and similarity one, which no other pair of the
two lists can beat, so an intersection level is the special case of a pairwise one and
belongs higher in the ladder.
Ordering them the other way makes the intersection level unreachable.

Two things to know before using one.

- **The cost is the cross product.**
  A pairwise level runs |a| × |b| metric evaluations where a scalar `jaro_winkler` runs one,
  and the lists are per row, so it is the only level whose per-pair cost is set by the data
  rather than by the schema.
  Two cheap tests stand in front of it: a shared element settles the level outright by a
  linear merge over the two sorted cells, and the per-value signature bound rejects an
  element pair for two loads and two popcounts without reading a character.
  Neither helps a column holding lists of hundreds of elements, and that column should be
  compared some other way.
- **No term-frequency adjustment, of either kind.**
  A list column's term frequencies count values rather than sets, the same reason a list
  `exact` level gets none, and the neighbourhood mass `--fuzzy-tf` and `--fuzzy-u` need is
  defined over two values of one column rather than two cells of one.

The levels are also accepted inside the `list_contains` shape below, where they read the
list column, but think before using them there: a pairwise level over two **alias** lists is
the fuzzy version of intersecting them, and it agrees on the pair the next section explains
why membership refuses.

### A value against a list: `list_contains` and its fuzzy half

Every other level compares a column against itself.
The three membership levels are the exception: they read a scalar string column against a
**different** column holding a list.
`list_contains` fires when either row's value is an element of the other row's list;
`contains_levenshtein` and `contains_jaro_winkler` fire when either row's value is within a
threshold of an *element* of the other row's list.

The case it exists for is nicknames.

```json
{"name": "forename", "columns": ["first_name", "nicknames"], "levels": [
  {"type": "null"},
  {"type": "exact"},
  {"type": "list_contains"},
  {"type": "jaro_winkler", "threshold": 0.9},
  {"type": "else"}]}
```

The scalar column comes first and the list second, and the order matters: "is this name one
of those aliases" is not the same question with the columns the other way round.

**It fires in both directions**, so a row named `bill` matches one whose `nicknames` hold
`bill`, whichever of the two was drawn first.
That is deliberately not the same as intersecting the two alias lists with `list_overlap`.
Two rows named `bill` and `robert` can both carry `will` in their alias lists without either
being the other's name; membership says no there, and intersection says yes.

Note the shape of the comparison above: it names two columns, but `exact` and `jaro_winkler`
inside it read only one.
That is the one place a level may read fewer columns than the comparison names, and it is the
reason to want the shape at all.
Levels are ordered evidence, so the alias bridge belongs **in the same comparison as the name
levels it ranks against**, below an exact match and above a fuzzy one.
Splitting it into a second comparison would work, but the model would then count "the names
agree" and "one is the other's alias" as two independent pieces of evidence about the same
field, which they are not.
Inside the shape, a one-column level reads whichever of the two columns has the type it
understands: `exact`, `levenshtein` and `jaro_winkler` the scalar one, `list_overlap`,
`list_jaccard` and the two pairwise levels the list one.
The three membership levels are the ones that read both.

Two consequences worth knowing.

- **`null` means the row holds neither part.**
  A row with no name and an empty list can produce no match with any partner, so it is null.
  A row with a name but no aliases is *not* null, because the other row's list can still
  contain its name, and marking it null would let the null level pre-empt a level that fires.
  The cost is that two rows which both have names and both have empty lists land on `else`,
  read as a disagreement where "no alias data" is the truer statement.
  Keeping a fuzzy level in the same comparison is what makes that harmless: the pair is then
  charged for the names disagreeing, which they do.
- **No term-frequency adjustment.**
  A membership comparison reads two columns and the TF machinery is defined over one, so
  it is skipped rather than approximated, the same way it is for a list column's `exact`
  level.
  That applies to the [fuzzy-level TF](../model.md#term-frequency-adjustment) too, which is
  the price of keeping the fuzzy levels inside this comparison: the neighbourhood mass comes
  from self-joining one column's dictionary, and a level a row's *list* can pre-empt is not
  answerable from two values of the other column.
  A name column carrying an alias comparison keeps `--fuzzy-tf` only if its fuzzy levels sit
  in a separate single-column comparison, and that is the trade against the dependence
  argument above.

#### The fuzzy half

An alias list is written by someone, so it carries typos like every other field.
`contains_levenshtein` and `contains_jaro_winkler` are `list_contains` with a metric where it
has an equality, and everything above holds for them unchanged: the same two columns in the
same order, the same both-directions rule, the same null rule, and no term frequency.

```json
{"name": "forename", "columns": ["first_name", "nicknames"], "levels": [
  {"type": "null"},
  {"type": "exact"},
  {"type": "list_contains"},
  {"type": "contains_levenshtein", "threshold": 1},
  {"type": "contains_jaro_winkler", "threshold": 0.92},
  {"type": "else"}]}
```

**Rank them below `list_contains`.**
Exact membership is the value at distance zero from an element, so it implies both fuzzy
levels and they can never fire above it — [`simplify`](../commands/simplify.md) knows that
implication, and will merge a fuzzy level into the exact one above it where a run cannot tell
them apart.
It is also the stronger evidence.
Being 0.91 similar to something in an alias list is not being in it: on the project's own
fixture, `william` scores 0.91 against `will`, which sits in an unrelated `robert`'s alias
list, so the fuzzy level agrees on a pair membership refuses.
That is a weaker claim, not a wrong one, and the model prices it as such — which is exactly
what having it as its own level buys.

The cost is |a| + |b| metric evaluations rather than the pairwise levels' |a| × |b|, because
only one side of each comparison is a list.
An exact hit short-circuits it through the alias map below, and the signature bound rejects an
element for two popcounts.

Dense ids are per column, so the two dictionaries are aligned once at bind time into a table
from scalar value id to list value id.
That is one `uint32` per distinct scalar value, built by one pass over each dictionary, and it
is what keeps the pair path integer-only: no string is touched, and the level costs two
lookups and a walk over a cell holding a handful of ids.

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
- gives a level the wrong number of columns (`geo_within` reads exactly two, the membership
  levels a scalar column then a list one);
- names columns of different types, unless they are the string-then-string_list pair the
  membership levels read;
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
