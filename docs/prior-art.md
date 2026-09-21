# Prior art

cpplink is heavily inspired by [splink](https://moj-analytical-services.github.io/splink/), and reuses many of the ideas and design choices that make it such a useful record linkage tool: the Fellegi-Sunter model with EM estimation, comparison ladders, term-frequency adjustment, and the split between the blocking rules that train the model and the ones that generate predictions.
What differs is the execution model, which is described on [the home page](index.md) and in [Blocking](blocking.md).

The blocking methods here are drawn from the record linkage and entity resolution literature rather than invented for this tool, and it is worth being explicit about which is which.

| Source | Prior art |
| --- | --- |
| sorted neighbourhood | Hernández and Stolfo 1995 |
| rare-value inverted index | IDF-weighted canopies (McCallum, Nigam and Ungar 2000); rare-token ordering in prefix-filtering similarity joins (Bayardo, Ma and Srikant 2007; Xiao et al. 2008) |
| MinHash LSH | Broder 1997; Indyk and Motwani 1998; benchmarked for record linkage by Steorts, Ventura, Sadinle and Fienberg 2014 |
| ANN over embeddings | DeepER (2018), AutoBlock (2020), DeepBlocker (2021), Sparkly (2023) |
| automatic blocking as such | Michelson and Knoblock 2006; Bilenko, Kamath and Mooney 2006; Kejriwal and Miranker 2013; the `dedupe` library; token blocking and meta-blocking (Papadakis et al.) |

Christen's 2012 survey of indexing techniques covers most of these and is the reference for the pair-completeness, pair-quality and reduction-ratio metrics [`recall`](commands/recall.md) reports.

## What this project claims

The claims are narrower than the table, and none of them is a blocking method.

1. **The EM-safety criterion.** A pair source may feed estimation only if its selection event factors as a condition on an excludable column subset.
   A whole-record source biases *every* `m_c` with no column left to repair it, so estimation and prediction run over different unions of sources.
   See [EM safety](em.md#em-safety).
2. **Admissible per-pattern term-frequency brackets** that let most patterns be emitted or dropped without touching the TF tables, emitting exactly the predictions a full scoring pass would.
   See [the admissible bracket](model.md#the-admissible-bracket).
3. **Exact closed-form candidate pricing** from the term-frequency tables, which prices ten billion pairs without enumerating one.
   See [cost before enumeration](blocking.md#cost-before-enumeration).
4. **The streaming implementation.** γ's sufficiency is Fellegi and Sunter 1969; that a full Fellegi-Sunter pipeline with TF adjustment fits in memory at 20M records, because nothing in it holds a row per pair, is an engineering result rather than a statistical one.
