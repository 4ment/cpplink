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
  {"name": "last_name"},
  {"name": "dob"},
  {"name": "latitude"},
  {"name": "address_tokens"}
]
```

A parquet file already knows what its columns are, so `type` is optional: a column that says nothing takes its type from the file.

| Arrow type in the file | `type` | Stored as | Term frequencies? |
| --- | --- | --- | --- |
| `string`, `large_string`, any integer | `string` | interned to a dense `uint32` id | yes |
| `list` or `large_list` of the same | `string_list` | CSR of interned ids, sorted and deduplicated per row | yes |
| `date32`, `date64`, `timestamp` | `date` | `int32` days since 1970-01-01 | yes, dense over the observed range |
| `bool` | `boolean` | `int8` | yes |
| `float`, `double` | `double` | as-is | **no** |

The first column is wider than the second because a file written from Python rarely holds the type its author had in mind.
pandas and polars write text as `large_string` and lists as `large_list`, and a column that was numeric in the frame arrives as an integer: a postcode, a year, a phone number, or the `unique_id` itself.
An integer is text: it is read as its decimal digits, so `2000` in one file and `"2000"` in another intern to the same id and compare equal, and a writer who wanted a double wrote one.
A floating point column is a `double`, never text, so an integer column that pandas promoted to `float64` over a missing value (`2000.0`) is refused where a string is wanted; cast it to a nullable integer (`Int64`) or to text before writing.

Declaring `type` still works, and what it says is checked against the file rather than used instead of it: `{"name": "dob", "type": "string"}` over a `date32` column is an error at load.
The one thing a declaration buys is an earlier error.
The checks a type decides, which levels a comparison may apply, which transforms a derivation may chain, which columns a source may block on, run when the schema is parsed for the columns it types and when the first file's footer is read for the rest, so either way they fail before a row is read.

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
| `email_username` | `string` | `string` | the part before the first `@`, or the whole value when there is none | `Smith-Jones, John` |
| `email_domain` | `string` | `string` | the part after the last `@`; missing when there is none | *missing* |

`email_username` and `email_domain` exist for the [email comparison](#a-comparison-over-several-string-columns) below, where the username is compared as its own level of the address's comparison rather than extracted per pair.

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
A phonetic key over 20M records is one Soundex per distinct surname plus a gather.

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

### Keys the file already holds: `derived_from`

A file often arrives with a key its preparer computed from other columns: a memo prefix, an amount rounded beside its week, a phonetic code written by another tool.
cpplink reads such a column from the file like any other, and `derived_from` declares where it came from:

```json
"columns": [
  {"name": "memo",       "type": "string"},
  {"name": "amount",     "type": "double"},
  {"name": "memo_key",   "type": "string", "derived_from": "memo"},
  {"name": "amount_key", "type": "string", "derived_from": ["amount", "date"]}
]
```

The value is one column name or a list, each of which must be a declared column other than the one it is on.
It changes nothing about how the column is loaded, interned or compared.
What it carries is the dependency: an [`estimate`](../commands/estimate.md) session blocked on `memo_key` holds the comparisons on `memo` out, exactly as it holds a `derive`d column's source out, and [`profile`](../commands/profile.md) reads the pair as tied rather than trying to detect it.
A column may declare `derive` or `derived_from`, not both, since a column cpplink computes already knows its source.

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
| `term_frequency` | no | `true` enables [TF adjustment](../model.md#term-frequency-adjustment) on each of this comparison's `exact` levels, over the column that level reads |

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
| `percentage_within` | required | 1 double | \|difference\| / max(\|a\|, \|b\|) < threshold, a fraction in (0, 1] |
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

`percentage_within` is splink's percentage difference: the gap over the larger of the two magnitudes, strictly below the threshold, for an amount whose tolerance scales with its size.
`exact` on a `double` is equality to the last digit.

Two more optional fields apply to one level type each.

#### `direction` on `date_within`

```json
{"type": "date_within", "threshold": 3, "direction": "forward"}
```

`"forward"` fires when the **later input's** date is on or after the earlier input's, within the threshold: a payment arrives after it is sent, and a window either side would admit every near-date pair with the dates the wrong way round.
The level orients by which file each row came from, not by which side of the pair it sits on, because the sampler and the enumerator do not agree about pair order.
It therefore needs two inputs and is refused on one; two rows of the same input under `--mode link-and-dedup` have no earlier side and read the window either way.
The default, `"either"`, is the plain window.
See [Linking two files](../linking.md).

#### `column` on a level of a comparison over several string columns

A level may say which of the comparison's columns it reads, which is how a field and a key derived from it are ranked inside one comparison rather than counted as two pieces of evidence.
The [next section](#a-comparison-over-several-string-columns) is the worked example.

### A comparison over several string columns

This is how splink's email comparison is written here: exact on the address, then exact on the username, then Jaro-Winkler on either.
The username is a [derived column](#derived-columns), split once at load, and the comparison names both:

```json
{"name": "email_username", "derive": {"from": "email", "transform": "email_username"}}
```

```json
{
  "name": "email",
  "columns": ["email", "email_username"],
  "term_frequency": true,
  "levels": [
    {"type": "null"},
    {"type": "exact"},
    {"type": "exact", "column": "email_username"},
    {"type": "jaro_winkler", "threshold": 0.88},
    {"type": "jaro_winkler", "threshold": 0.88, "column": "email_username"},
    {"type": "else"}
  ]
}
```

The rules, all checked at parse time:

- `column` must be one of the comparison's `columns`, every column of the comparison must be a `string`, and only a level that reads one column may name one.
- A level without a `column` reads the first one.
- The comparison is **null wherever any of its columns is**, so an address with nothing before the `@` compares as missing rather than falling through to `else`.
- With `term_frequency`, each `exact` level gets its own adjustment over the column it read, and the first one's `u` is still closed form: an address is the column whose collision rate sits near `1/N`, where a sampled `u` sees nothing.
- A level that names a column is labelled `exact on email_username` in reports unless it carries its own `label`, so two exact levels do not print alike.

**Why a declared column rather than an expression the comparison evaluates per pair**, which is what splink's `EmailComparison` does with `regexp_extract`: per pair, the split would turn an integer equality into a `find('@')`, a substring and a string compare on every candidate.
Declared as a column, it runs once per distinct address, is interned and counted like any other column, and that is what gives its exact level a term-frequency adjustment, a closed-form `u`, a signature table for the fuzzy bound, and the option of blocking on it.
The cost is one `uint32` a row plus the username dictionary, and a column that shows in `inspect` and `profile` under the name you gave it.

What the shape does not get: `--fuzzy-tf`, `--fuzzy-u` and [`levels`](../commands/levels.md) read a single string column and refuse it, and [`simplify`](../commands/simplify.md) never merges two levels reading different columns.

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
| `type` | — | all | `exact_value`, `rare_value`, `minhash`, `sorted_neighbourhood`, `all_pairs` |
| `column` | — | all but `all_pairs` | required; must not be a `double` column; rejected on `all_pairs`, which names none |
| `name` | `"<column> <type>"` | all | label used in reports |
| `max_frequency` | 100 | `rare_value` | block only on values seen at most this many times |
| `window` | 8 | `sorted_neighbourhood` | window width; emits exactly \(N \cdot w\) pairs |
| `bands` | 12 | `minhash` | LSH bands; each becomes its own source |
| `rows_per_band` | 4 | `minhash` | rows per band |
| `ngram` | 3 | `minhash` | character n-gram size for shingling |
| `seed` | 1 | `minhash` | hash seed |
| `use` | `both` | all | `both`, `estimate` or `predict`: which of the two unions the source joins |

`minhash` additionally requires a `string` column, and non-zero `bands`, `rows_per_band` and
`ngram`. `sorted_neighbourhood` requires a non-zero `window`.

### What a source is for: `use`

Estimation and prediction run over different unions of sources, and `use` says which one a source joins.
`"use": "estimate"` conditions an EM session and produces no candidate; `"use": "predict"` produces candidates and conditions no session; the default does both.

```json
"blocking": [
  {"type": "exact_value", "column": "email"},
  {"type": "rare_value", "column": "last_name", "max_frequency": 100},
  {"type": "exact_value", "column": "phone", "use": "estimate"}
]
```

That is splink's split between the rule passed to its EM call and its `blocking_rules_to_generate_predictions`, and it is how a session can be blocked on a column the run has no reason to score on.
Every plan is built through one of the two filtered views of the schema, so a training source never produces a candidate and a scoring rule never conditions a session.
See [Blocking](../blocking.md) and [Estimation and EM](../em.md#em-safety).

`all_pairs` is the degenerate source: it produces every pair the mode admits, which is the
whole triangle when deduplicating and the cross product when linking.
It takes no options, and a plan holding one has no use for any other source, since every pair
a second source could produce was produced already.
The whole `blocking` section can be left out and replaced at the command line with
`--all-pairs`, which the five plan-building commands accept.
See [Blocking](../blocking.md#no-blocking-at-all) for when that is the right thing to do and
for what it costs.

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
