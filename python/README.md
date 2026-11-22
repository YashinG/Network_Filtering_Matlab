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
| *(new)* | `build_multiplex`, `multiplex_analysis`, `multiplex.*` |

Results come back as dictionaries with the same field names as the MATLAB output structs. For example, `out["metrics"]["node"]["hybrid"]["XpY"]` matches `stOut.metrics.node.Hybrid.XpY`, and `out["tables"]["ARI"]` is a pandas DataFrame.

## Multiplex analysis

`netfilter.multiplex` builds a **multiplex dependency network**: one filtered network (PMFG or MST) for each of the Pearson, Spearman and Kendall correlations of the same assets. It then analyses the layers together. The analysis follows the template in [ngozzi/multiplex](https://github.com/ngozzi/multiplex) (Battiston, Nicosia & Latora).

```python
import netfilter as nf

data = nf.load_data()
mplex = nf.build_multiplex(data["rets"], None, data["names"],
                           measures=("pearson", "spearman", "kendall"), network_filter="PMFG",
                           standardise=True, remove_market_mode=True, shrinkage=True)
res = nf.multiplex_analysis(mplex["layers"], sectors=dict(zip(data["names"], data["super_sector_names"])),
                            plot=True)
res["cartography"]            # MPC, overlapping degree and role for each asset
res["edge_overlap"]["jaccard"]
```

Or run `python scripts/run_multiplex.py`. It prints the results and saves the tables, figures and layer edge lists to `results/multiplex/`.

**Building the layers.** Returns are pre-processed once, exactly as in `get_filtered_network`. Each measure then gives a correlation matrix, which is turned into a distance and a similarity and filtered:
- **Pearson:** the weighted correlation, QIS-shrunk if `shrinkage=True`. This layer is the same network as `get_filtered_network`'s PMFG.
- **Spearman:** the same estimator applied to each asset's ranks.
- **Kendall:** a weighted Kendall tau-b, `kendall_tau_weighted`, in which each pair of dates gets weight `w_s · w_t`. With equal weights it equals `scipy.stats.kendalltau`, including with tied values. Shrinkage doesn't apply to Kendall.

All three layers accept EWMA time weights.

**Measures** (`netfilter.multiplex`):

| Function | What it computes |
|---|---|
| `degree_table` | the degree in each layer, plus the overlapping degree `o_i = Σ_a k_i^a` |
| `top_nodes` | the most connected assets per layer and overall |
| `participation_coefficient` | the multiplex participation coefficient: 0 when all of an asset's links are in one layer, 1 when they are spread evenly |
| `cartography` | focused or multiplex × hub or leaf |
| `layer_similarity` | the share of assets active in both layers (the template's measure) |
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
