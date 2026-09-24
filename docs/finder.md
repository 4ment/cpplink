# The search page

`cpplink-finder` is a page for one question: who in this file is this person.

You fill in a few fields about somebody, and it returns the records scoring highest against
them under the model, best first.
The score is the match weight in bits, the same number [`predict`](commands/predict.md) puts on a pair, and the ledger behind every hit is one click away.

```bash
pip install -e ".[finder]"
cpplink-finder --schema schema.json --model model.json people.parquet
```

The three files are what a search needs and the page makes none of them.
The schema says what the columns are, [`estimate`](commands/estimate.md) wrote the model, and the parquet is the file being searched.
Given no arguments the page asks for all three in the sidebar.

## The form is the schema's

There is no list of field names anywhere in the page.
A schema already states what a search needs a form to know: `comparisons` is an ordered list of what the model scores, each naming its columns, and `columns` carries every type and every derivation.
So the boxes are what the model reads, in the order it reads them, and a schema with no gender column draws no gender box.

Three rules do all of it.

**A comparison's boxes are its columns resolved back to their non-derived source.**
The email comparison reads `email` and `email_username`, the second derived from the first, so it is one box: the core computes the derived value from the text that was typed, which is why asking for it again would be asking twice.
A location comparison reading latitude and longitude is two boxes under one heading.

**The widget is the column's type.**
A date is a date picker and the one form the core reads is `YYYY-MM-DD`.
A list column is split into its elements on spaces and commas.
A number is parsed and refused if it is not one.
A string is sent as it was typed.

**A string column the file holds few values of is a dropdown**, whose blank entry leaves the column out of the query rather than sending a value with it.
That is the one thing the schema cannot supply, because a schema does not record that a gender column holds two values; the data does, read off the file's first rows at load.
Gender is not a special case here: it is a dropdown because the file holds two values of it, and a column holding thousands is typed into.

The blank entry matters.
A column the query does not name is *missing*, which lands on the null level and costs nothing either way; a literal "unknown" would disagree with every record that holds a value, so not knowing would push the right answer down the list.

At least one box must be filled.
An empty form is turned away rather than run, because a query naming no column scores every record identically.

### Commas, and what the schema settles

Whether several values can go in one box is undecidable for free text.
Splitting an address on commas would be wrong; splitting a list of forenames would not.
So a list column is split and a string column is sent verbatim, and the box says which it is.

For a string column, whether `Michael, John` can match a record holding `John Michael` depends on the schema, and the schema says so: a derived column built with `normalize` turns punctuation into spaces before anything is compared, so the page reads the derivations and tells you in the box's help text.
Declaring one is a schema decision taken before the model is estimated, not something the page can add afterwards.

## The results

The table is the unique id and every column the file holds, after the rank, the match weight and the probability.
Derived columns are left out because they are not in the parquet, and a column in no comparison is still shown: not typing something in does not mean not wanting to see it.

**A cell is green where its comparison landed on the exact level, and amber where it landed on anything between exact and disagreeing** -- a Jaro-Winkler threshold, a date inside a window, an overlapping list, whatever the comparison happens to be written with.
`null` is the absence of a value and `else` is a disagreement, so neither is marked.

The colour comes from the level rather than from comparing the two texts, because a partial match has no textual definition, and because a date inside a window or a value agreeing only after normalisation is an exact level whose two sides do not read alike.
A comparison with no exact level, such as a geographic one, can only ever be amber.
Only a column you filled in is marked: a comparison reading a latitude and a longitude would otherwise paint the half you did not ask about.

The weight is the prior plus what each comparison is worth for agreeing or disagreeing, which is what the **Why these scores** panel lists per hit.

The prior is the one place a search differs from scoring a pair.
A model's own prior is λ, the match rate over the *pair* space, which is the question deduplication asks; a person asking who in this file is this person is asking a different one.
**Expected matches** in the sidebar states it as the number of records here you expect to be that person, and defaults to 1; zero asks for the model's own prior instead.
It shifts every hit by the same constant, so it moves the probability and the threshold and never the order.

**Minimum probability** is how "nobody here is this person" is expressed: records below it are not returned however few hits there are.

## The settings file

Everything the form draws comes from the schema, so what is saved beside it is only what a schema has no opinion about: how many hits, how many threads, what a score must clear, the prior, and the two cosmetic things a column name does not settle -- what a box is called, and whether to draw it at all.

Hiding a column takes it off the form and leaves it in the table, which is what `latitude` and `longitude` usually want: real columns of the model that nobody types into.

**Save as default** in the sidebar writes the file, named after the schema.

## What it costs

One store per process, loaded once, shared by every browser session.
At twenty million records that is gigabytes resident and most of a minute to load, so run one process and leave it up.

A search is serialised.
The query is installed as a row of the store for as long as it takes to answer it, so two at once would be two writers; the page holds a lock and the second search waits for the first.
**Threads** splits the work *within* one query, which is the lever that matters: even a one-field query spends most of a second walking the rows at twenty million, and the walk splits four to five ways over eight threads.

The caption under the results says where the time went.
`walk` is the pass over each queried column's distinct values, `gather` the pass over the records.
A slow query is almost always the walk, and the walk is almost always one near-unique column carrying a fuzzy level: an address, or an email.
Comparing on address tokens instead of the whole string is worth more than any amount of threading.

## What it does not do

It reads one input.
A query row would have to be a dataset of its own, so a store built from two files is refused with that reason.

It computes nothing.
The weight, the levels and the term-frequency moves are the core's, through [`Linker.search`](python.md).
A second implementation of any of them would be a second answer.
