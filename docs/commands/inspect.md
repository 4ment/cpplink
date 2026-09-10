# `inspect`

**Goal:** load a parquet file through the schema and report what actually arrived — column
cardinality, null rates, the skew of each column, and what every structure costs in memory.

This is the sanity check before anything expensive. It answers three questions: did the schema
match the file, is a column skewed badly enough to wreck blocking, and does the store fit in
memory at this record count.

## Synopsis

```sh
cpplink inspect --schema <schema.json> <file.parquet>...
```

| Option | Meaning |
| --- | --- |
| `--schema <file>` | required; the [schema file](../reference/schema.md) |
| *(positional)* | required; the parquet file to load |

Only `columns` and `unique_id` are read from the schema. `comparisons` and `blocking` may be
absent.

## Example

```sh
cpplink inspect --schema examples/sample_schema.json examples/sample.parquet
```

```text
File         examples/sample.parquet
Records      1,800,000
Row groups   9
Load time    2.8 s  (648,979 rows/s)

Column            Type                Distinct      Null   Top val Mean len        Rare pairs
---------------------------------------------------------------------------------------------
first_name        string                23,060     0.00%     2.27%        -            52,729
last_name         string               173,347     0.00%     0.20%        -        16,675,804
dob               date                  20,821     0.00%     0.01%        -        68,631,954
email             string             1,663,384     2.80%     0.00%        -            90,600
phone             string             1,650,577     2.00%     0.00%        -           121,195
postcode          string                 9,000     1.20%     0.01%        -                 0
latitude          double                     -     0.00%         -        -                 -
longitude         double                     -     0.00%         -        -                 -
address_tokens    string_list           24,019     0.00%     0.71%     4.47                 0

"Rare pairs" is the candidate count a rare-value source would emit from that
column alone at a frequency cap of 100, from the term frequencies only.

Structure                                Bytes   Basis
---------------------------------------------------------------------------------------------
id (ids)                               32.0 MB   arena, no index
first_name dictionary                  1.50 MB   23060 values
first_name ids                         8.00 MB   u32 per record
first_name tf                          90.1 KB   u32 per value
last_name dictionary                   6.00 MB   173347 values
last_name ids                          8.00 MB   u32 per record
last_name tf                          677.1 KB   u32 per value
dob values                             8.00 MB   i32 days per record
dob tf                                 81.3 KB   dense over the observed range
email dictionary                       86.0 MB   1663384 values
email ids                              8.00 MB   u32 per record
email tf                               6.35 MB   u32 per value
...
address_tokens ids                     48.0 MB   8043400 entries, CSR
address_tokens offsets                 16.0 MB   u64 per record
---------------------------------------------------------------------------------------------
Resident total                        333.8 MB   194 bytes per record
```

## Reading the output

### The header

| Field | Meaning |
| --- | --- |
| `Records` | rows loaded |
| `Row groups` | parquet row groups; the loader converts one at a time, so this bounds transient Arrow memory |
| `Load time` | wall time and throughput of load-and-intern, which is single-threaded |

### The column table

| Column | Meaning |
| --- | --- |
| `Distinct` | distinct interned values. `-` for `double`, which is not interned |
| `Null` | share of rows where the value is missing. High null rates are not a problem — they are a level (`null` fires and gets its own weight) — but a column that is 90% null will not block |
| `Top val` | share of rows carrying the *most common* value. **This is the skew number.** A `Top val` of a few percent means the largest exact-value group is a few percent of the file, and a group that size is quadratic |
| `Mean len` | mean list length, `string_list` columns only |
| `Rare pairs` | how many candidate pairs a [`rare_value`](../blocking.md#rare_value-the-primary-automatic-source) source on this column would emit at a cap of 100, computed from the term-frequency table alone |

`Rare pairs` is the one to read before writing a blocking plan. `postcode` shows **0**: with
9,000 distinct postcodes over 1.8M records every value is far above a cap of 100, so a
rare-value source there produces nothing. That column belongs in `exact_value` if anywhere.
`dob` shows 68.6M — the cap does exclude most dates, but not all of them.

### The memory table

One row per allocated structure, with the basis it was computed from, ending in a resident
total and bytes per record. This is the number to extrapolate: at 194 bytes per record here,
20M records would be about 3.9 GB, and the measured figure at 20M rows on this schema is
229 bytes per record for 4.27 GB — the difference is dictionary growth, since email and phone
are near-unique and their dictionaries grow with the file while every id column does not.

Useful things it makes visible:

- **Near-unique string columns dominate.** `email` alone is 86 MB of dictionary at 1.66M
  distinct values, against 6 MB for `last_name` at 173k.
- **A `double` column costs 8 bytes a record and carries no term frequencies**, so it cannot
  drive rare-value blocking — two doubles agreeing to the last bit is not a discrete event
  worth counting.
- **List columns cost CSR ids plus `u64` offsets.** Narrowing the offset type would save a
  third of that, and has been deferred deliberately: it is 3% of resident and `u64` is the
  safe general choice.

!!! note "Accounting counts allocated capacity"
    The measured max RSS runs slightly *below* the accounted total, because the accounting
    sums `capacity()` rather than `size()`. Peak footprint *during* load runs above both,
    because Arrow's row-group buffers are alive until they are released.
