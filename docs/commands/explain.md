# `explain`

**Goal:** show, for one specific pair of records, which level every comparison assigned, what
values it saw, and the packed γ that results.

This is how a comparison configuration gets debugged. When a pair that should obviously match
scores badly, `explain` shows which comparison disagreed and on what values — usually a level
threshold that is too tight, or a level ordered after one that swallows it.

## Synopsis

```sh
cpplink explain --schema <schema.json> --pair <id_a>,<id_b> <file.parquet>
cpplink explain --schema <schema.json> --rows <i>,<j> <file.parquet>
cpplink explain --schema <schema.json> --pair <id_a>,<id_b> \
                --model <model.json> [--threshold BITS] <file.parquet>
cpplink explain --schema <schema.json> --model <model.json> --json \
                --pairs <file|-> <file.parquet>
cpplink explain --schema <schema.json> --model <model.json> \
                --predictions <predictions.parquet> --out <waterfalls.parquet> <file.parquet>
```

| Option | Meaning |
| --- | --- |
| `--schema <file>` | required |
| `--pair <id_a>,<id_b>` | the two records by their `unique_id` values |
| `--rows <i>,<j>` | the two records by zero-based row index |
| `--pairs <file>` | many pairs, one `<id_a>,<id_b>` per line; `-` reads them from stdin |
| `--row-pairs <file>` | the same, with `<i>,<j>` row indices per line |
| `--predictions <file>` | every prediction in the csv or parquet `predict --out` wrote; needs `--out` |
| `--out <file>` | where to write the waterfalls, one wide row per prediction; `.csv` or `.parquet` |
| `--model <file>` | optional; adds the score waterfall below the level table |
| `--threshold BITS` | default 0; only affects the zone and the emit/drop line |
| `--tf-damping F` | default 1.0; scale the term-frequency move, as in [`predict`](predict.md) |
| `--fuzzy-tf`, `--ball-budget N` | adjust fuzzy levels by neighbourhood mass, as in [`predict`](predict.md) |
| `--no-interactions` | score the plain model out of a file that carries two-way corrections |
| `--json` | one JSON object per pair instead of the text report; needs `--model` |
| *(positional)* | required; the parquet file |

Give **exactly one** of `--pair`, `--rows`, `--pairs`, `--row-pairs` or `--predictions`. The
schema must declare `comparisons`; `blocking` is not needed. Without `--model` there is no
weight to explain, and the waterfall is simply not printed.
Give `explain` the same scoring options the run used, or it explains a different weight than the one the run wrote.

## Example

```sh
cpplink explain --schema examples/sample_schema.json --pair r5,r9 examples/sample.parquet
```

```text
Comparison            Levels   Bits    Shift   Columns
------------------------------------------------------------------------------
last_name                  5      3        0   last_name
first_name                 5      3        3   first_name
dob                        5      3        6   dob
email                      4      2        9   email
phone                      4      2       11   phone
postcode                   3      2       13   postcode
location                   4      2       15   latitude, longitude
address                    5      3       17   address_tokens
------------------------------------------------------------------------------
Packed width 20 bits of 32.

Pair  r5  /  r9

Comparison        Level                     Index   Values
----------------------------------------------------------------------------------------------------
last_name         exact                         1   vipiest  |  vipiest
first_name        exact                         1   sheanwean  |  sheanwean
dob               exact                         1   8605d  |  8605d
email             null                          0   sheanwean.vipiest7140@examp.  |  <null>
phone             null                          0   0428000809  |  <null>
postcode          exact                         1   8582  |  8582
location          within 1.00 km                1   -24.1911, 126.2921  |  -24.1873, 126.2908
address           exact                         1   {priengvieck quiem ret pick.  |  {priengvieck quiem ret pick.

gamma  0x0002a049   0010 1010 0000 0100 1001   (20 bits)
```

## Reading the output

### The γ layout table

Printed first, and independent of the pair — it is a property of the schema.

| Column | Meaning |
| --- | --- |
| `Levels` | how many levels the comparison declares, including `null` and `else` |
| `Bits` | \(\lceil \log_2 \text{Levels} \rceil\) — the field width in the packed pattern |
| `Shift` | the field's offset within γ, so `level = (gamma >> shift) & ((1 << bits) - 1)` |
| `Columns` | the store columns the comparison reads. A comparison may span several — a coordinate pair is one comparison, not two |

`Packed width 20 bits of 32` is the constraint check. If the sum exceeded 32 the schema would
have been rejected at parse time.

!!! note "Five levels in three bits"
    A comparison with five levels occupies three bits, so three of the eight codes in that
    field name no level. The **reachable** pattern space here is 5·5·5·4·4·3·4·5 = 120,000 of
    1,048,576 packed values, which is why `predict` reports a pattern space smaller than
    \(2^{20}\).

### The pair table

One row per comparison, in schema order:

| Column | Meaning |
| --- | --- |
| `Level` | the label of the **first** level that fired — level order is the model, first hit wins |
| `Index` | that level's index, which is what goes into γ |
| `Values` | the two records' actual values, truncated. `<null>` where the value is missing |

The pair above is a planted duplicate: the surname, forename, date, postcode and address token
set all agree exactly, the coordinates are 400 m apart, and `email` and `phone` are missing on
one side — which is exactly what the corruption model in
[`gen-sample`](gen-sample.md) does. Note that `null` fires *before* `exact` in every comparison
here, so a missing value never gets counted as a disagreement; it gets its own level with its
own weight.

## The waterfall — why the pair scored what it did

γ says which levels fired. It does not say why the pair scored what it scored, and the gap
between those two is the term-frequency adjustment: **two exact surname matches carry the same
γ and can differ by ten bits, because one of them is "Smith".** Pass `--model` to see the
arithmetic.

```sh
cpplink explain --schema examples/sample_schema.json --model model.json \
                --pair r5,r9 --threshold 20 examples/sample.parquet
```

```text
Comparison        Level                       bits        tf     running
------------------------------------------------------------------------
(prior)           lambda                    -22.86                -22.86
last_name         exact                     +14.62     +1.59       -6.65
first_name        exact                      +9.33     +3.15       +5.83
dob               exact                     +13.96     +0.23      +20.01
email             null                       +2.65                +22.67
phone             null                       +2.66                +25.32
postcode          exact                     +12.84     -0.12      +38.05
location          within 1.00 km            +19.87                +57.92
address           exact                     +20.12                +78.04
------------------------------------------------------------------------
Match weight   78.036 bits    posterior 1.000000000

Term frequency, for the comparisons that agreed exactly:
  last_name         vipiest                            8 rows    8.00e-06     +1.59 bits
  first_name        sheanwean                        133 rows    1.33e-04     +3.15 bits
  dob               8605d                             42 rows    4.20e-05     +0.23 bits
  postcode          8582                             119 rows    1.19e-04     -0.12 bits

Pattern bracket  61.302 to 93.229 bits, against a threshold of 20.000
Zone             emit -- every pair with this pattern clears it, so no term-frequency table is consulted
This pair would be emitted.
```

| Column | Meaning |
| --- | --- |
| `bits` | \(\log_2(m/u)\) for the level that fired — the evidence that level carries, on average over values |
| `tf` | the term-frequency move for *this* value. Blank where the comparison has no term frequencies or did not agree exactly |
| `running` | the cumulative total, starting from the prior |

The first row is the prior, \(\log_2(\lambda/(1-\lambda))\) — the bits you start in debt by,
because almost no pair is a match. Everything after it is evidence paying that debt down.

### Reading the term-frequency block

Each line is one exact agreement and what it was worth:

- `first_name sheanwean` occurs in 133 of 1,000,000 records. Two random records collide on a
  forename with probability \(u = 0.00118\), so agreeing on *this* forename is
  \(\log_2(0.00118 / 0.000133) = +3.15\) bits **more** than an average forename agreement.
- `postcode 8582` runs the other way: it is slightly *more* common than the average postcode
  collision, so it gives back 0.12 bits.

That is the entire content of the term-frequency adjustment, and it is why the adjustment has
to be applied at scoring rather than folded into the model — see
[the model](../model.md#term-frequency-adjustment).

### The bracket and the zone

The last two lines connect the pair to what [`predict`](predict.md) would do with it. The
bracket is the range this *pattern* can score across all possible values, from the column's
commonest value to its rarest; the zone is which side of the threshold that range falls on.
A pattern in `drop` cannot clear the threshold whatever values it carries, and `predict`
skips it without touching a term-frequency table.

!!! tip "The waterfall must add up"
    The running total ends at exactly the number `predict` computes — that equality is
    asserted over every pair in a fixture, because a report that plausibly explains a
    *different* calculation from the one that runs would be worse than no report at all.

## Many pairs from one load

`--pairs` and `--row-pairs` answer one pair per input line, from a single load of the store.
That is the difference between a tool that can ask about a pair and one that cannot: at 20M records the load is the cost, and the pair is free.
With `-` the lines come from stdin and each answer is flushed as it is written, so a process can be held open and asked one pair at a time.
A line that names no record is reported on its own line and does not end the batch.

`--json` writes the waterfall as one object per line, with the same numbers as the text report: `prior`, one entry per comparison in `steps` (`name`, `level`, `label`, the two `values`, `m`, `u`, `bits`, `tf`, `frequency` where a move was made, and the `running` total), the `interactions`, then `weight`, `probability`, the pattern `bracket`, `threshold`, `zone` and `emitted`.

## Every prediction of a run, as a file

`--predictions <file> --out <file>` explains the whole prediction file `predict --out` wrote and writes one wide row per pair, csv or parquet by the extension:

```text
id_a, id_b, gamma, prior, match_weight, match_probability,
bracket_low, bracket_high, threshold, zone, emitted,
<comparison>_level, <comparison>_bits, <comparison>_tf, <comparison>_frequency,   one set per comparison
<left>_x_<right>_bits                                                             one per two-way correction
```

`prior` plus every `_bits` and `_tf` plus every interaction column is `match_weight`, to the last bit the run wrote.
A level's label, `m` and `u` are the model's rather than the pair's, so the file does not repeat them; the model file carries them, indexed by `<comparison>_level`.
Ids are resolved through a sorted index over the id column, 4 bytes a record, so a file of millions of predictions costs one load and one pass.
A prediction naming an id no record holds is skipped and counted, and so is one whose stored `gamma` the comparisons no longer produce, which means the schema changed after `predict` ran and the ledgers explain today's schema rather than the file's weights.

This is what the cluster viewers in `tools/` draw from: `tools/cluster_view.py --waterfalls` embeds a ledger per prediction, and `tools/cluster_server.py --waterfalls` loads the file beside the predictions so a click is one lookup.
Nothing in either recomputes a bit of the weight, and neither runs this binary.

## See also

- [`predict`](predict.md) — the same scoring, over every candidate pair
- [The model](../model.md) — where `m`, `u`, `λ` and the weights come from

### The packed pattern

```text
gamma  0x0002a049   0010 1010 0000 0100 1001   (20 bits)
```

Hex, binary and width. Reading the binary **right to left** in the field widths from the layout
table gives `001` `001` `001` `00` `00` `01` `01` `001` — that is, level 1 for last_name,
level 1 for first_name, level 1 for dob, level 0 for email, level 0 for phone, level 1 for
postcode, level 1 for location, level 1 for address, matching the table above.

This integer is the *entire* input the model gets for this pair. Everything downstream —
the histogram, EM, the base weight — sees only this. The only thing that escapes it is the
[term-frequency adjustment](../model.md#term-frequency-adjustment), which is applied at
scoring precisely because it does not fit in γ.

## Typical uses

- **A pair that should match doesn't.** Run `explain` on it, find the comparison sitting at
  `else`, and look at the values. If they are visibly similar, the threshold is too tight.
- **A level never fires.** `estimate` reports levels with no support; `explain` on a pair that
  should have hit it shows which earlier level swallowed it instead.
- **Checking a threshold change.** Levels are ordered and first-hit-wins, so moving a
  `jaro_winkler` threshold from 0.92 to 0.88 also removes pairs from the level *below* it.
