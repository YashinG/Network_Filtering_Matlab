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

Results come back as dictionaries with the same field names as the MATLAB output structs. For example, `out["metrics"]["node"]["hybrid"]["XpY"]` matches `stOut.metrics.node.Hybrid.XpY`, and `out["tables"]["ARI"]` is a pandas DataFrame.

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
