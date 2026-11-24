# Comparisons

A **comparison** is one field of evidence about a pair.
It reads one logical field of both records, walks an ordered ladder of **levels**, and returns
the index of the first one that fires.
That index is one field of the pair's agreement pattern γ, and γ is the only thing the model
ever sees, so everything a comparison decides is decided here and nowhere later.

This page is about the ladder: what each level reads, what it costs, what it may not do, and
how to choose one.
What happens to γ afterwards is [the model](model.md), and the JSON that declares all of it is
[the schema reference](reference/schema.md#comparisons).

## One field, one comparison

```json
{
  "name": "last_name",
  "term_frequency": true,
  "columns": ["last_name"],
  "levels": [
    {"type": "null"},
    {"type": "exact"},
    {"type": "jaro_winkler", "threshold": 0.92},
    {"type": "jaro_winkler", "threshold": 0.85},
    {"type": "else"}
  ]
}
```

Five levels, so the comparison occupies \(\lceil \log_2 5 \rceil = 3\) bits of γ, and a pair
lands on exactly one of them.

**Keep one field in one comparison.**
Fellegi-Sunter multiplies the comparisons together as independent evidence, so two comparisons
reading the same field are counted twice: the model sees "the names agree" and "the names
nearly agree" as two separate observations and adds both weights.
This is why the alias levels below live *inside* the name comparison rather than beside it,
and why a coordinate pair is one comparison over two columns rather than a latitude comparison
and a longitude one.
Where fields genuinely are correlated and cannot be merged,
[`estimate --interactions`](commands/estimate.md#relaxing-conditional-independence) prices the
correlation instead.

## The ladder is the model

Levels are evaluated top-down and **the first hit wins**.
Nothing below a level that fires is consulted, so the order in the file is not a formatting
choice.
It decides which of two true statements about a pair becomes the one the model is told.

Put the strongest evidence first, and mind these orderings in particular.

| Put this above | this | because |
| --- | --- | --- |
| `null` | everything | a missing value must not be scored as a disagreement |
| `exact` | any fuzzy level | it is the strongest statement and the cheapest test |
| a tight threshold | a loose one of the same metric | `jaro_winkler >= 0.85` above `>= 0.92` makes the tighter level unreachable |
| `list_overlap` | `list_levenshtein`, `list_jaro_winkler` | a shared element is a pair at distance zero, which no other pair beats |
| `list_contains` | `contains_levenshtein`, `contains_jaro_winkler` | membership is the value at distance zero from an element |

Only two things are checked when the file is parsed: the ladder must end in `else`, and no
level may follow it.
A ladder that is merely *wrong* parses cleanly.
`jaro_winkler >= 0.85` above `jaro_winkler >= 0.92` is a valid schema in which the second
level never fires, costs a field of γ, and gets a parameter fitted to nothing.
[`simplify`](commands/simplify.md) prints the pairs each level actually took, which is where a
dead level shows up as a row of zeros, and [`explain`](commands/explain.md) shows one pair's
walk down the ladder.

!!! tip "The `else` level is a real level"
    It is where every pair that agreed on nothing lands, so its weight is the penalty for
    disagreement, and it is usually the largest single number in the model.
    A comparison with no fuzzy levels charges every typo that full penalty.

## What each level reads

Sixteen level types, in six groups.
The `threshold` column says whether one is required; the schema reference has
[the same table as syntax](reference/schema.md#levels).

### Structural

| Level | Fires when |
| --- | --- |
| `null` | the comparison has nothing to read on either row |
| `else` | always, and it must be last |

### One string against one string

| Level | Threshold | Fires when |
| --- | --- | --- |
| `exact` | — | the two interned ids are equal |
| `levenshtein` | edits, an integer | the edit distance is at most the threshold |
| `jaro_winkler` | similarity in (0, 1] | the similarity is at least the threshold |

`exact` is an integer equality because values are interned to dense `uint32` ids at load, not a
string compare.
`levenshtein` never computes a distance beyond its threshold: the algorithm abandons as soon
as the bound is exceeded, since the only question asked is "within k".
`jaro_winkler` uses the standard prefix bonus, scale 0.1 over at most four leading characters,
applied only above a Jaro of 0.7.

### One ordered quantity against another

| Level | Threshold | Reads |
| --- | --- | --- |
| `date_within` | days | one date column |
| `numeric_within` | absolute difference | one double column |
| `geo_within` | kilometres | two double columns, latitude then longitude |

Dates are stored as days since 1970-01-01, so `date_within` is an integer subtraction.
`geo_within` is the great-circle distance, and is the ordinary reason a comparison names two
columns.
`exact` on a double is rejected at parse time: two floats agreeing to the last bit says
nothing useful.

### One list against another

Lists are sorted and deduplicated per row at load, so everything here is a linear merge with
no hashing and no allocation in the hot path.

| Level | Threshold | Fires when |
| --- | --- | --- |
| `exact` | — | the two rows hold the same set |
| `list_overlap` | a count | the intersection holds at least that many elements |
| `list_jaccard` | in (0, 1] | \(\lvert A \cap B\rvert / \lvert A \cup B\rvert\) is at least the threshold |
| `list_levenshtein` | edits | *some* element of one is within that many edits of *some* element of the other |
| `list_jaro_winkler` | similarity | some element pair is at least that similar |

The first three read the elements two rows **share**, so a typo inside an element is no
agreement at all.
The last two read the closest pair of elements over the **cross product** of the two cells,
which is what makes two lists that intersect in nothing still able to agree.
They are the one place a level's per-pair cost is set by the data rather than by the schema:
\(\lvert a \rvert \times \lvert b \rvert\) metric evaluations, with the lists being per row.
A shared element settles them by the same linear merge before that walk starts, and a column
holding lists of hundreds of elements should be compared some other way.

### One value against the other row's list

| Level | Threshold | Fires when |
| --- | --- | --- |
| `list_contains` | — | either row's value is an element of the other row's list |
| `contains_levenshtein` | edits | either row's value is within that many edits of an element of the other's list |
| `contains_jaro_winkler` | similarity | either row's value is at least that similar to an element of the other's list |

These are the levels for an alias column: a `first_name` beside the `nicknames` a person may
be listed under.
They read two columns of *different* types, the scalar first, and they test membership in both
directions rather than intersecting the two alias lists.
The distinction matters.
Two rows named `bill` and `robert` can both carry `will` among their aliases without either
being the other's name, and an intersection agrees there where membership does not.

The fuzzy two cost \(\lvert a \rvert + \lvert b \rvert\) metric evaluations rather than the
pairwise levels' product, because only one side of each comparison is a list.
They are weaker evidence than exact membership and belong below it: on the project's own
fixture `william` scores 0.914 against `will`, which sits in an unrelated `robert`'s alias
list, so the fuzzy level agrees on a pair membership refuses.
That is a weaker claim, and having it as its own level is what lets the model price it as one.

[The schema reference](reference/schema.md#a-value-against-a-list-list_contains-and-its-fuzzy-half)
has the full shape, including what it costs to keep the fuzzy name levels inside this
comparison.

## Missing values

Every comparison has its own notion of "nothing to read": a null interned id, a null date, a
NaN double, an empty list.
The `null` level fires when **either** row is missing, which is why it belongs at the top.

There are two reasons to write one even though nothing requires it.
A pair that is missing a value has not disagreed, and without a `null` level it lands on
`else` and is charged the full disagreement penalty.
And the null level's \(u\) is exact rather than sampled, because it is a count over the data
rather than a property of pairs, which is worth having for free.
See [Estimation and EM](em.md#u-in-closed-form).

### A null level as evidence or as nothing

By default a `null` level is a level like any other: its \(u\) is the closed-form rate at which a random pair has a field missing, its \(m\) is learned by EM, and it scores \(\log_2(m_\text{null}/u_\text{null})\) bits.
That weight is the answer to one question: is a pair with this field missing more or less likely to be a match than a random pair?
The answer depends on how the data lost the value, and it can go either way.

| How values go missing | \(m_\text{null}\) against \(u_\text{null}\) | Weight |
| --- | --- | --- |
| Independently on each record, at the same rate for matches and non-matches | equal | 0 bits; the level says nothing and keeping it costs nothing |
| A property of the *entity*: a person whose occupation is unknown is unknown in every record of them | lower, since the null level fires when **either** side is missing, and a random pair has two chances where a matching pair has nearly one | negative, approaching −1 bit for a rare gap |
| A property of the *pair*: a duplicate was keyed from a worse source and lost fields the original kept | higher | positive |

Both signs occur in practice.
On the three bench datasets every null level is negative: on `historical_50k` from −0.36 bits (`postcode_fake`) to −2.10 (`first_name`), on `febrl3` from −0.06 to −0.42, and on `fake_1000` within 0.05 of zero.
On `gen-sample`, which deletes fields on the duplicates it plants, `email`, `phone` and `postcode` are worth +2.8 bits each for being missing.

There are two reasons to distrust that weight even where it is estimated well.
Missingness is usually a property of how the records were collected, and that need not be the same in the file the model was trained on and the file it scores.
And the model multiplies the null levels as if they were independent, when a record missing one field is often missing others: on `historical_50k` a record missing one of `postcode_fake`, `occupation` and `gender` is 1.3 to 1.6 times as likely to be missing another, which is the worst-fitting part of the model ([dependence](dependence.md)).

splink takes the other reading by declaration.
A level declared with `is_null_level=True` has no \(m\) or \(u\) at all, its Bayes factor is fixed at 1 (0 bits), and splink's own comparison templates, such as `NameComparison`, `EmailComparison` and `PostcodeComparison`, open with one.
So in splink a missing value is uninformative unless you build the comparison yourself and declare the null test as an ordinary level.
splink also leaves the null pairs out when it estimates the other levels of that comparison, in EM and in the random-pair \(u\) alike, so those levels' \(m\) and \(u\) are conditional on the field being present.

cpplink can score the same model file the splink way, per comparison, and there are three ways to ask.
The schema declares it on the level, `{"type": "null", "informative": false}`, which is where splink puts `is_null_level` and where it belongs when it is a fact about the column.
A run names comparisons with `--uninformative-null NAME`, repeated, or all of them with `--uninformative-nulls`, on [`predict`](commands/predict.md), [`explain`](commands/explain.md), [`rescore`](commands/rescore.md) and [`search`](commands/search.md); in [Python](python.md) that is `uninformative_nulls=["email", "phone"]` or `uninformative_nulls=True`.
The run can add comparisons to what the schema declares but never take one away, and a name that matches no comparison is refused rather than ignored.
Such a `null` level scores 0 bits, and every other level of that comparison is read given the field is present, which is not the same as zeroing the one cell.
The levels' \(m\) and \(u\) are unconditional rates that already lost the mass the null level took, so each is rescaled by its own denominator:

\[
w_\ell \;=\; \log_2 \frac{m_\ell / (1 - m_\text{null})}{u_\ell / (1 - u_\text{null})}
\;=\; \log_2 \frac{m_\ell}{u_\ell} \;+\; \log_2 \frac{1 - u_\text{null}}{1 - m_\text{null}}
\]

The second term is one constant per comparison: +0.61 bits on `gen-sample`'s `email`, −0.75 on `historical_50k`'s `occupation`.
A term-frequency move needs no change, being a ratio of two \(u\) on the same level that the conditioning scales alike.
A fitted [interaction](commands/estimate.md#relaxing-conditional-independence) scores 0 on every cell where either of its comparisons is null, since a comparison that is not evaluated takes part in no association; the observed cells keep their bits.
The [waterfall](commands/explain.md) shows the null step at exactly 0.0 and still sums to the score, and the shortcuts `predict` takes stay exact.

Measured with the threshold swept from −10 to 40 bits, since the operating point moves:

| Dataset | Model | Best F1, null as evidence | Best F1, uninformative |
| --- | --- | ---: | ---: |
| `fake_1000` | plain | **0.9361** | 0.9344 |
| `fake_1000` | `--interactions` | **0.8828** | 0.8741 |
| `febrl3` | plain | **0.9993** | 0.9992 |
| `historical_50k` | plain | **0.8684** at 9.5 bits | 0.8670 at 11.5 bits |
| `historical_50k` | `--interactions` | **0.9111** at 3.0 bits | 0.9069 at 4.5 bits |
| `gen-sample`, 200k | plain | 0.9965 | 0.9965 |

**Keeping the null evidence is better everywhere it differs, by 0.0001 to 0.009.**
On `historical_50k` dropping it buys precision (0.9158 to 0.9482) and costs more recall (0.8257 to 0.7987): a matching pair with a field missing loses the renormalising shift on every observed level and drifts below the threshold.
On `gen-sample` the best F1 is the same because blocking, not scoring, sets it there, but the curve holds 0.9965 up to 42 bits with the null evidence and only to 36 without, because the bonus was carrying the duplicates that lost their contact fields.
So the default stays: the estimated null weight is a parameter like any other, and on these files it is fitted well enough to be worth its bits.

One comparison at a time on `historical_50k` does not find a column to single out either:

| Null held uninformative | Plain model | `--interactions` |
| --- | ---: | ---: |
| none | 0.8684 | 0.9111 |
| `first_name` | 0.8685 | 0.9111 |
| `surname` | **0.8703** | 0.9057 |
| `first_and_surname` | 0.8685 | 0.9111 |
| `dob` | 0.8675 | **0.9119** |
| `postcode_fake` | 0.8677 | 0.9114 |
| `birth_place` | 0.8672 | 0.9117 |
| `occupation` | 0.8672 | 0.9117 |
| `gender` | 0.8690 | 0.9114 |

Every move but one is within 0.002 and the best one changes with the model.
`first_name` and `first_and_surname` are missing on 0.3% of random pairs, so their null levels barely fire.
`surname` is the one large move and it changes sign: +0.0019 on the plain model and −0.0054 with `--interactions`, whose `surname x first_and_surname` term was fitted to the null cells the option zeroes.
Picking the best of eight against the same truth that scores them is selection, not a finding, so none of these is a reason to change a schema.
The per-comparison switch is for saying what a column's missingness means, not for tuning.

Reach for `--uninformative-nulls` when the weight was learned under one way of losing values and is applied under another.
Linking a source that never records a field to one that usually does is that case.
So is a [`search`](commands/search.md) query, where a column is missing because nobody typed it rather than because the data lost it; the order of the hits does not change, but their totals and posteriors stop carrying bits for what the query left out.

One limit.
It acts at scoring only: EM still reads a null as evidence while it estimates, which splink does not, so the conditional rates are recovered from the model's unconditional ones rather than estimated directly.

The membership shape is the exception, and deliberately so.
There a row is null only when it holds **neither** a value nor a list, because a row with a
name and no aliases of its own is still reachable through the other row's list, and marking it
null would let the null level pre-empt a level that fires.

## Comparisons that span two columns

Two shapes name more than one column.

**A coordinate pair** is one comparison over latitude and longitude, in that order.
Splitting it into two `numeric_within` comparisons would multiply two halves of one fact.

**A scalar beside a list** is the alias shape above.
It is the only place where the two columns have different types, and the only place where a
level may read *fewer* columns than its comparison names: `exact` and `jaro_winkler` in that
comparison read the scalar column, `list_overlap` reads the list one, and the membership
levels read both.
That is the reason to want the shape rather than two comparisons, since the alias bridge has
to rank between an exact name match and a fuzzy one inside a single field of γ.

### A date with unknown parts

| Level | `match` | Fires when |
| --- | --- | --- |
| `partial_date` | `full` | both dates are complete and identical |
| `partial_date` | `typo` | both are complete and one keying error apart |
| `partial_date` | `compatible_month` | one is partial, the parts known on both agree, and both know the month |
| `partial_date` | `compatible_year` | one is partial and the parts known on both agree |

This is a date of birth written `DDMMYYYY` with `00` for an unknown day or month, one per record in a `string` column or several per record in a `string_list`.
A `date` column cannot say that its month is unknown, and splitting the date into a year, a month and a day compared separately would let a record's aliases agree part by part on a date neither record holds.
Each level instead reads one date of each record whole, and the ladder lands on the best pair.
Two partial dates are never a full match, so two records that both say only `1978` are priced as sharing a year.
[The schema reference](reference/schema.md#dates-with-unknown-parts-partial_date) has the full definition and what fails the load.

## What a level costs

The comparison is the expensive part of the pipeline, by two orders of magnitude.
Enumerating a candidate pair costs about **17 ns**; evaluating one cost **4,090 ns** when it
was first measured, and 97% of that was string metrics running on pairs that end at `else`,
which is nearly every pair.
Blocking is not where to optimize.
The ladder is.

| Level | Cost |
| --- | --- |
| `exact`, `date_within`, `numeric_within` | one or two nanoseconds, integer or double arithmetic |
| `list_overlap`, `list_jaccard`, `exact` on a list | a linear merge over two sorted cells |
| `list_contains` | two integer lookups and a walk over a sorted cell |
| `partial_date` | two lookups of a date parsed at bind time per date pair, \(\lvert a \rvert \times \lvert b \rvert\) of them over lists |
| `geo_within` | four trigonometric calls |
| `levenshtein`, `jaro_winkler` | the string metric, unless a bound rejects the pair first |
| `contains_levenshtein`, `contains_jaro_winkler` | \(\lvert a \rvert + \lvert b \rvert\) of those |
| `list_levenshtein`, `list_jaro_winkler` | \(\lvert a \rvert \times \lvert b \rvert\) of those |

Three mechanisms keep the fuzzy levels affordable, and all three are exact.

**Per-value signatures.** Every column a fuzzy level reads gets a table holding, per interned
value, its length and a 64-bit character-presence mask.
A character present in `a` and absent from `b` cannot be matched, which bounds Jaro-Winkler
from above and edit distance from below, so two loads and two popcounts reject a pair before a
character is read.
That is twelve bytes per distinct value, built once per dictionary and shared by every
comparison reading it, and it is worth **24% of comparison CPU** on a blocked stream.
It is worth more than that inside the list levels, where the same two popcounts are paid back
once per element pair rather than once per row pair.

**Myers' bit-vector edit distance**, dispatched when the shorter value fits a machine word.
Between 3 and 8 times faster than the banded dynamic program in isolation, and invisible end
to end on a schema that leans on Jaro rather than Levenshtein.

**The pair-global ceiling**, which is the one that matters most.
A filter that rejects one level of one comparison still walks the pair comparison by
comparison.
Instead, every comparison is granted the best level its cheap bounds still admit, those
weights are summed, and if that ceiling is under the threshold the pair is dropped before a
single metric runs.
Measured at 20 bits on a 1M-row sample: **90% of candidates never compared, 11.2 s against
36.3 s, and the predictions byte-identical**.
The saving grows with the threshold, which is the right direction.

!!! note "Why branch-and-bound on the running weight does not work"
    The obvious version prunes nothing here.
    A disagreement is worth a few negative bits while an agreement is worth fifteen or more
    positive, so after two disagreements the optimistic remainder is still enormous.
    All of the pruning power comes from the signature bound capping the *agreement* side.

## The admissibility contract

Every one of those filters is subject to one rule: **a level's cheap bound may never be false
where the level itself would be true.**
That is what lets the ceiling drop a pair without knowing what its γ would have been, and it
is why the filters change the run's cost and not its output.

This is what each level answers when asked the cheap question.

| Level | The cheap answer |
| --- | --- |
| `null`, `else`, `exact`, `date_within`, `numeric_within`, `list_contains`, `partial_date` | the level itself, which is already cheaper than any bound |
| `levenshtein`, `jaro_winkler` | the signature bounds |
| `geo_within` | the latitude arc, a subtraction where the haversine is four trigonometric calls |
| `list_overlap` | \(\min(\lvert A \rvert, \lvert B \rvert)\), two subtractions off the offset array |
| `list_jaccard` | \(\min / \max\) of the two set sizes |
| the four fuzzy list levels | the same walk over the same element pairs, with the metric left off |

None of this is asserted and left there.
The test suite runs the same data with the filters on and off and compares the packed patterns
pair for pair, over randomly generated populations of near misses, and both signature bounds
are checked against reference implementations over tens of thousands of random pairs.
At scale, `predict --no-signatures`, `--no-ceiling` and `--no-bounds` do the same thing on real
files.
A bound that is wrong once is a silent correctness bug, not a slow path.

## Term frequency, level by level

[Term-frequency adjustment](model.md#term-frequency-adjustment) makes a weight depend on the
*value* rather than only the level, so a shared "Zolnerowich" outscores a shared "Smith".
It is opt-in per comparison with `"term_frequency": true`, and it does not reach every level.

| Shape | Adjustment |
| --- | --- |
| `exact` on a string or date column | yes, from the term-frequency table |
| `levenshtein`, `jaro_winkler` on a string column | with [`--fuzzy-tf`](commands/predict.md), from the mass of the value's neighbourhood |
| any level of a list column | no |
| any level of the membership shape | no |
| `partial_date` | no |
| `exact` on a double | no such level exists |

The fuzzy case is the interesting one.
The adjustment there uses the mass of the value's *neighbourhood* under the level's predicate
instead of the value's own frequency, which is the same formula as the exact case when the
neighbourhood is the value itself.
It is affordable only because values are interned: the neighbourhood masses come from
self-joining a column's dictionary, 149k distinct surnames rather than 1M rows, once per
column and amortised over every pair scored.
Measured on `historical_50k`, it takes F1 from 0.7716 to 0.8007 at 20 bits, all of it in
recall at unchanged precision.
A near-unique column is refused by a budget rather than approximated.

The list and membership shapes get nothing because a list column's frequencies count values
rather than sets, and the neighbourhood scan is defined over two values of one column rather
than over cells.
Moving the fuzzy levels of an alias comparison into a separate single-column comparison buys
the adjustment back and pays the double-counting described above.
That trade is real and unmeasured.

## Letting the data choose the levels

Three commands answer questions about a ladder that guessing answers badly.

[`profile`](commands/profile.md) runs before any of this, needs no model and costs seconds.
It says what each column can be worth at best, and which pairs of columns are the same evidence
twice, which is the question behind "should these be one comparison".

[`levels`](commands/levels.md) places the fuzzy thresholds from the data.
It builds the exact \(u(t)\) curve from the dictionary self-join and an \(m(t)\) curve from
anchor pairs, then cuts the grid to maximise the divergence between them, holding the level
count fixed at whatever the schema already declares.
Measured against truth it never reads, the proposals are worth about **+1 bit** of evidence per
comparison, out of sample, on both bench datasets that carry enough anchor pairs to fit a
curve.

[`simplify`](commands/simplify.md) goes the other way and merges the levels a run cannot tell
apart, which is how a dead or redundant level is found after the fact.
Merging two levels on `historical_50k` took γ from 20 bits to 18 and `predict` 6.7% faster,
with F1 moving by at most 0.0007.

!!! warning "More levels is not more quality"
    Neither BIC nor a likelihood-ratio test can choose the number of levels here, and both were
    tried.
    Their power is the size of the run, so the same code and the same significance level merge nothing on `historical_50k`'s 18.4M candidate pairs and three levels on `fake_1000`'s 10,000, which is a statement about how much data there is rather than about how alike the levels are.
    What decides is the effect size, printed in bits beside every proposal.
    And a better-placed threshold buys robustness to *where* the score threshold sits more than
    it buys a higher peak F1, which is the honest reading of the measurement above.

## Validation

Everything below is checked when the schema is parsed, before a data file is opened.

- A level is applied to a column type it can read.
- A level gets the right number of columns: `geo_within` exactly two, the membership levels a
  scalar then a list, everything else one.
- The columns of one comparison share a type, unless they are the scalar-then-list pair.
- A level that needs a `threshold` has one.
- The ladder ends in `else`, and nothing follows it.
- The packed γ fits 32 bits.

[`explain`](commands/explain.md) prints the resulting layout, one row per comparison, with the
bits and shift each occupies.

## See also

- [The model](model.md) for what happens to γ once a comparison has produced it.
- [The schema file](reference/schema.md#comparisons) for the JSON, field by field.
- [`explain`](commands/explain.md) to watch one pair walk the ladder.
- [`simplify`](commands/simplify.md) and [`levels`](commands/levels.md) to change a ladder on
  evidence rather than on taste.
