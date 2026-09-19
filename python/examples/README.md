# Examples

Notebooks that take the `cpplink` Python package through a whole job, with plots and tables.

| Notebook | What it does |
|---|---|
| [historical_50k.ipynb](historical_50k.ipynb) | Deduplicates 50,578 records of historical figures from [splink_datasets](https://github.com/moj-analytical-services/splink_datasets): fetch and prepare the data, draft a schema, profile the columns, price the blocking and measure its recall, fit the fuzzy thresholds, estimate the model with and without two-way interactions, score the candidates, explain a score, cluster, sweep the threshold against the truth, estimate blocking recall without the truth, simplify the levels, and rescore from a spill. |

## Running one

The notebooks need the package built in the project's conda environment, plus `pandas`, `matplotlib` and a Jupyter kernel:

```sh
conda activate cpplink
pip install -e . --no-build-isolation -Ccmake.define.CMAKE_PREFIX_PATH=$CONDA_PREFIX
pip install jupyter matplotlib pandas
jupyter notebook python/examples/historical_50k.ipynb
```

Each notebook downloads its data once into `python/examples/work/` and writes every file it produces there.
That directory is not tracked.
The notebooks are committed with their outputs, so they read without being run.
