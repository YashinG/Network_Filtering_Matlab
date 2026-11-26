# netfilter: Python port

A Python port of the MATLAB code in this repository. It covers MST and PMFG network filtering, DBHT clustering on a PMFG or TMFG, QIS and EWMA correlation estimation, market-mode removal, rolling-window analysis, bootstrap reliability and network metrics.

## Install

```bash
cd python
pip install -r requirements.txt
```

It needs Python 3.9 or later, with numpy, scipy, networkx, pandas and matplotlib. `tqdm` is optional; if it's installed, you get progress bars. No MATLAB toolboxes or compiled MatlabBGL files are needed.

## Quick start

```python
import netfilter as nf

data = nf.load_data()                     # reads ../Data/Data.mat
pmfg = nf.get_filtered_network(data["rets"], None, data["names"], "PMFG",
                               standardise=True, remove_market_mode=True,
                               dist_method="QIS_correlDistMetric", plot=True)
clusters = nf.get_clusters(data["rets"], None, data["names"],
                           "QIS_correlDistMetric", "DBHT_PMFG",
                           max_clusters=15, comp_partition=data["all_sector_num"])
print(clusters["tables"]["clusterIDsOrdered"])
```

Alternatively, run one of the scripts, which mirror the MATLAB `Scripts/` folder. Each script has its inputs at the top:

| MATLAB script | Python script |
|---|---|
| `scr_RunCorrelAnalysis.m` | `scripts/run_correl_analysis.py` |
| `scr_RunNetworkFilter.m` | `scripts/run_network_filter.py` |
| `scr_RunNetworkFilter_Dynamic.m` | `scripts/run_network_filter_dynamic.py` |
| `scr_RunClustering.m` | `scripts/run_clustering.py` |
| `scr_RunClustering_Dynamic.m` | `scripts/run_clustering_dynamic.py` |
| `scr_RunBootstrapNetwork.m` | `scripts/run_bootstrap_network.py` |
| `scr_RunBootstrapDBHT.m` | `scripts/run_bootstrap_dbht.py` |
| *(new)* multiplex analysis | `scripts/run_multiplex.py` |

```bash
python scripts/run_clustering.py
```

Unlike the MATLAB scripts, these can be run from any folder.

## Function map

| MATLAB | Python (`netfilter.`) |
|---|---|
| `EWMA_Wts` | `ewma_weights` |
| `covMatWtd` | `cov_weighted` |
| `QIS_Wtd` | `qis_weighted` |
| `preProcessRets` | `preprocess_returns` |
| `correlToDistMetric` | `corr_to_dist` |
| `corrcov` | `cov_to_corr` |
| `minspantree` | `mst` |
| `pmfg` (External Functions) | `pmfg` |
| `PMFG_T2s` (inside `DBHTs.m`) | `tmfg` |
| `getFilteredNetwork` / `_Dynamic` | `get_filtered_network` / `get_filtered_network_dynamic` |
| `networkMetrics` | `network_metrics` |
| `DBHTs` | `dbht` |
| `getClusters` / `_Dynamic` | `get_clusters` / `get_clusters_dynamic` |
| `RandIndex` | `adjusted_rand_index` |
| `correlationAnalysis` / `_Dynamic` | `correlation_analysis` / `correlation_analysis_dynamic` |
| `bootstrapNetwork` | `bootstrap_network` |
| `bootstrapDBHT` | `bootstrap_dbht` |
| `adj2pajek2` | `to_pajek` |
| *(new)* | `build_multiplex`, `Layer`, `multiplex_analysis`, `multiplex.*` |

Results come back as dictionaries with the same field names as the MATLAB output structs. For example, `out["metrics"]["node"]["hybrid"]["XpY"]` matches `stOut.metrics.node.Hybrid.XpY`, and `out["tables"]["ARI"]` is a pandas DataFrame.

## Multiplex analysis

`netfilter.multiplex` builds a **multiplex network**: several networks (layers) on the same assets. It then analyses the layers together, following the template in [ngozzi/multiplex](https://github.com/ngozzi/multiplex) (Battiston, Nicosia & Latora). By default the layers are Pearson, Spearman and Kendall PMFGs, but each layer can be almost anything.

The approach is inspired by *The multiplex dependency structure of financial markets*, N. Musumeci, V. Nicosia, T. Aste, T. Di Matteo, V. Latora (2016), https://arxiv.org/abs/1606.04872v1. The measures follow Battiston, Nicosia & Latora (2014, 2017); full references are in the [main README](../README.md#references).

```python
import netfilter as nf
from netfilter import Layer

data = nf.load_data()
sectors = dict(zip(data["names"], data["super_sector_names"]))

# default: Pearson, Spearman and Kendall PMFGs
mplex = nf.build_multiplex(data["rets"], None, data["names"])

# or choose your own layers
mplex = nf.build_multiplex(data["rets"], None, data["names"], layers=[
    "pearson",                                                  # built-in measure
    Layer("tail", "lower_tail", params={"q": 0.05}),            # built-in, with a parameter
    Layer("dcor", "distance_correlation", network_filter="MST"),  # its own filter
    Layer("beta", my_function, kind="similarity"),              # f(returns, weights) -> p x p
    Layer("analyst", my_matrix, kind="distance"),               # numpy array or labelled DataFrame
    Layer("supply_chain", my_graph),                            # networkx Graph, used as it is
])
res = nf.multiplex_analysis(mplex["layers"], sectors, plot=True)
```

Or run `python scripts/run_multiplex.py`. It prints the results and saves the tables, figures and layer edge lists to `results/multiplex/`. The script includes a worked example with custom layers (`EXAMPLE_CUSTOM_LAYERS`).

### Layers

Returns are pre-processed once, exactly as in `get_filtered_network` (standardising, market-mode removal and time weights). Each layer is a `Layer(name, source, kind, network_filter, raw_returns, params)`. A plain string is short for a built-in measure.

| `source` | What happens |
|---|---|
| a built-in name | the measure is computed from the processed returns |
| a function `f(returns, weights, **params)` | it gets the processed returns (raw returns if `raw_returns=True`) and must return a p × p matrix |
| a numpy array | a p × p matrix in the order of `names` |
| a pandas DataFrame | a matrix labelled by asset name, in any order |
| a networkx Graph | used as it is: nodes outside `names` are dropped and listed in `inputs["dropped_nodes"]`, and assets missing from the graph become unlinked nodes |

`kind` says how to read a matrix. It is required for functions and matrices; built-ins set it themselves.
- `"correlation"`: values in [-1, 1]. The distance is `sqrt(2(1 - x))` and the similarity `1 + x`.
- `"similarity"`: larger means more strongly linked.
- `"distance"`: smaller means more strongly linked.
- `"adjacency"`: the matrix is already a network, and its non-zero entries are the edges.

`network_filter` is `"PMFG"`, `"MST"` or `None` (no filtering) for each layer. If a layer doesn't set it, it uses the default from `build_multiplex`, and adjacency layers and graphs are never filtered. The MST and PMFG depend only on the order of the values, so a similarity and a distance that rank the pairs in the same order give the same network. A non-symmetric (directed) input keeps the stronger direction of each pair.

**Built-in measures** (`netfilter.multiplex.BUILTIN_MEASURES`). All of them accept time weights.

| Name | Kind | Measure |
|---|---|---|
| `pearson` | correlation | Pearson correlation, QIS-shrunk if `shrinkage=True`. This layer is the same network as `get_filtered_network`'s PMFG. |
| `spearman` | correlation | the same estimator applied to each asset's ranks |
| `kendall` | correlation | Kendall tau-b, with each pair of dates weighted by `w_s · w_t`. It matches `scipy.stats.kendalltau` with equal weights. |
| `partial` | correlation | the partial correlation of each pair given all the other assets, from the (QIS-shrunk) precision matrix |
| `distance_correlation` | similarity | distance correlation (Székely et al. 2007). It also picks up non-linear dependence, and it matches the `dcor` package with equal weights. |
| `lower_tail` / `upper_tail` | similarity | how often two assets have their worst (or best) `q` share of days together; `params={"q": 0.1}` by default |

**Measures** (`netfilter.multiplex`):

| Function | What it computes |
|---|---|
| `degree_table` | the degree in each layer, plus the overlapping degree `o_i = Σ_a k_i^a` |
| `top_nodes` | the most connected assets per layer and overall |
| `participation_coefficient` | the multiplex participation coefficient: 0 when all of an asset's links are in one layer, 1 when they are spread evenly |
| `cartography` | focused or multiplex × hub or leaf (`isolated` for assets with no links in any layer) |
| `layer_similarity` | the share of assets active in both layers (the template's measure; this is informative for supplied networks that don't link every asset) |
| `edge_overlap` | the Jaccard overlap of edge sets, and how many layers each edge is in |
| `sector_degrees` | the average degree per sector in each layer |
| `write_layers` / `read_layers` | save or load layers as edge lists |

The analysis functions take any `{name: networkx.Graph}` dict, so you can also use them on networks built elsewhere, such as the template's own edge lists.

**Differences from the template:**
- **Similarity:** PMFG and MST layers contain every asset, so the template's node-activity similarity is always 1. `edge_overlap` is added to compare the layers instead.
- **Hub threshold:** the template used a fixed overlapping degree of 60, which suited its 343-stock data. The default here is the mean plus 2 standard deviations (`z > 2`, as in Battiston et al. 2014), and you can pass any number. With only 20 assets, no stock in `Data.mat` reaches that default, so every stock is classed as a multiplex leaf.
- **Ties in `top_nodes`:** tied assets keep their table order. `tie_break="notebook"` reproduces the template's ordering exactly.

On the template's own data, the package reproduces the notebook's printed top-10 lists, its layer similarities and its cartography counts. `tests/test_multiplex.py` checks this when `MULTIPLEX_TEMPLATE_DATA` points to that repo's `data/` folder.

## Differences from the MATLAB version

- **Indexing.** Indices are 0-based. DBHT cluster labels (`T8`) are still 1..K. `dbht()` returns the linkage in SciPy format (`Z`) and in the original MATLAB format (`Z_matlab`).
- **DBHT linkage rows.** The rows of the SciPy linkage are sorted by height, as SciPy expects. The tree itself is unchanged.
- **Planarity test.** The PMFG uses `networkx.check_planarity` instead of the MatlabBGL Boyer–Myrvold test. Both give the same answer, so the PMFG is identical.
- **Leaf ordering.** `get_clusters` uses SciPy's `optimal_leaf_ordering`, which uses MATLAB's `'adjacent'` criterion rather than `'group'`. This only changes the plotted order and the numbering of the clusters in `clusterIDsOrdered`, not which stocks end up together. The adjusted Rand index is unaffected.
- **Bootstrap seeds.** The bootstrap functions take a `seed` argument so runs can be reproduced. The resamples will differ from MATLAB's `bootstrp`.
- **Bug fixes.** Three MATLAB bugs are fixed in the port:
  - The `(Time Wtd)` plot label was always shown, because `all(w) == 1/n` was always false.
  - For non-correlation distances such as `'euclidean'`, the similarity `S` was computed from a vector rather than a matrix. It is now `1 / (1 + D)`.
  - In `QIS_Wtd`, the effective sample size is rounded down where it is used as an index, so time-weighted QIS also works when there are more assets than effective observations.
- **Ties in centrality ranks.** Values equal to 12 significant figures count as tied. MATLAB's `tiedrank` can split an exact tie on floating-point noise. On the sample MST this happens for one pair of stocks (9 and 11), whose hybrid scores swap by 0.004.
- **Runtime.** A 10,000-sample PMFG bootstrap takes about 9 minutes for 20 stocks.

### Implementation notes

These choices are needed to match MATLAB exactly. Keep them when changing the code.
- **PageRank.** MATLAB's algorithm is reproduced: start from uniform values and stop when the largest change is at most 1e-4, with at most 100 iterations. A fully converged PageRank differs from MATLAB by about 1.5e-4.
- **PMFG edge order.** Edges are added in MATLAB's column-major `find` order with a stable sort, so ties are broken the same way.
- **DBHT linkage.** DBHT builds merges cluster by cluster, so `matlab_to_scipy_linkage` re-sorts the rows stably by height and relabels them. This is safe because each merge is strictly higher than the merges below it.
- **Distributions.** `ksdensity` uses MATLAB's default bandwidth, and `prctile` maps to numpy's `method="hazen"`.
- **Multiplex filters.** The MST and PMFG depend only on the order of the values, so layer similarities and distances are shifted to be strictly positive (zeros mean "no edge" in the filter code). For a non-symmetric input, each pair keeps the stronger direction.
- **Multiplex roles.** A node exactly on a threshold counts as hub or multiplex, and `on_boundary` flags it. A node with no links in any layer gets MPC = NaN and the role `isolated`.

## Verification against MATLAB

The port was run side by side with the original MATLAB functions (R2025b) on `Data/Data.mat`. These all match to within about 1e-14:
- the EWMA weights
- the pre-processed returns
- the QIS covariance and correlations
- the MST and PMFG edges
- every weighted centrality, including PageRank
- the hybrid scores (apart from the tie noted above)
- the DBHT clusters and full linkage, on both PMFG and TMFG
- the Ward and average-linkage clusters and merge heights
- the correlation summaries and densities
- every rolling-window output

The MATLAB reference values are saved in `tests/data/`, and `tests/test_matlab_reference.py` checks against them. You can re-run the comparison without MATLAB.

## Tests

```bash
cd python
pytest
```

`tests/test_matlab_reference.py` compares against MATLAB. The other tests check the defining properties of each method:
- The weighted covariance matches numpy.
- QIS preserves the trace and is positive definite.
- After market-mode removal, the residuals are orthogonal to the market factor.
- The MST is a tree, and the PMFG and TMFG are planar with 3(N − 2) edges. The PMFG contains the MST.
- Each DBHT linkage is valid and reproduces the DBHT clusters.
- The centralities match networkx.
- The adjusted Rand index matches a hand-worked example.

## Possible extensions

- A rolling-window multiplex: `build_multiplex` per window, tracking MPC, roles and edge overlap over time, following the `*_dynamic` pattern in `network.py`.
- Weighted multiplex measures, such as a strength-based overlapping degree or a weighted participation coefficient.
- Comparing DBHT clusters across layers, for example with the ARI between layers.
- A `pyproject.toml` so `netfilter` can be installed with pip. It is currently imported via `sys.path`.
