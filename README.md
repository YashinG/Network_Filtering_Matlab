# MSc Network Filtering (MATLAB)

MATLAB code for building **filtered correlation networks** and **hierarchical clusters** from stock market return data. It was developed as part of an [MSc project](https://repository.up.ac.za/items/b176234c-1db9-4e0b-b31b-715cee1a56bf) at the University of Pretoria.

The toolkit covers:

- **Network filtering**: Minimum Spanning Tree (MST) and Planar Maximally Filtered Graph (PMFG).
- **Clustering**: Directed Bubble Hierarchical Tree (DBHT, on a PMFG or TMFG) plus the standard linkages (Ward, single, complete, average, centroid, weighted, median).
- **Correlation estimation**: sample correlations or Ledoit–Wolf **QIS** shrinkage, both with optional **exponential (EWMA) time weighting**.
- **Pre-processing**: z-score standardisation and removal of the **market mode** (first principal component).
- **Static and dynamic (rolling-window) analysis**: networks, clusters and correlation distributions through time.
- **Bootstrap reliability**: how stable network edges, network metrics and the DBHT cluster count are.
- **Network metrics**: degree, closeness, betweenness, eigenvector and PageRank centrality, eccentricity and tree length, plus the hybrid centrality score of Pozzi, Di Matteo & Aste (2013).

> **Python version:** a full Python port is in [`python/`](python/README.md). It gives the same results as this MATLAB code.

---

## Repository structure

```
Network_Filtering_Matlab/
├── Data/
│   └── Data.mat                 # Example dataset (struct stStockRets)
├── Scripts/                     # Entry points: set inputs and run
│   ├── scr_RunCorrelAnalysis.m
│   ├── scr_RunNetworkFilter.m
│   ├── scr_RunNetworkFilter_Dynamic.m
│   ├── scr_RunClustering.m
│   ├── scr_RunClustering_Dynamic.m
│   ├── scr_RunBootstrapNetwork.m
│   └── scr_RunBootstrapDBHT.m
├── Functions/                   # Core functions (author's code)
└── External Functions/          # Third-party code
    ├── PMFG/                    # PMFG implementation (Aste et al.)
    ├── matlab_bgl-4.0.1/        # MatlabBGL graph library (planarity tests used by PMFG)
    └── RandIndex.m              # Adjusted Rand index (D. Corney)
```

---

## Requirements

- MATLAB (R2016a or later recommended, because it uses `graph` objects and `centrality`).
- **Statistics and Machine Learning Toolbox**, for `pca`, `pdist`, `linkage`, `cluster`, `dendrogram`, `optimalleaforder` and `ksdensity`.
- MatlabBGL. It ships in `External Functions/`. Pre-compiled MEX files are included for common platforms. If the PMFG fails on your platform, you may need to recompile them.

---

## Quick start

1. Clone the repository.
2. In MATLAB, change directory to `Scripts/`. The scripts use relative paths (`../Functions`, `../Data`, and so on), so they must be run from that folder.
3. Open a script, edit the `stInputs` block at the top, and run it.

```matlab
cd Scripts
scr_RunNetworkFilter        % PMFG on the full sample, with a plot
```

Every script follows the same pattern:

1. `addpath` to the function folders.
2. `load('../Data/Data.mat')`.
3. Fill a struct `stInputs` with the data, pre-processing, method and plotting options.
4. Call one main function, which returns a results struct `stOut...`.

---

## Data format

`Data/Data.mat` contains a struct `stStockRets`. The example dataset has 20 anonymised stocks (`Stock_1` to `Stock_20`) across 8 ICB super-sectors, with 199 weekly returns from 14 Oct 2012 to 31 Jul 2016.

| Field | Size | Description |
|---|---|---|
| `rets` | nDates × pAssets | Return matrix (one column per stock) |
| `mlDates` | nDates × 1 | Observation dates |
| `aNames` | 1 × pAssets | Stock names or tickers |
| `SuperSectorNames` | 1 × pAssets | Sector names (used for plot labels) |
| `IndustryNum`, `SuperSectorNum`, `SectorNum`, `SubSectorNum` | 1 × pAssets | Numeric classification codes at each level |
| `allSectorNum` | 4 × pAssets | The four classification levels stacked. Used to compare clusters with the industry classification (Rand index). |
| `pajekStrData` | 1 × pAssets | Labels for Pajek export |

To use your own data, build a struct with at least `rets`, `mlDates` and `aNames`. Add the sector fields too if you want labelled plots or a cluster-versus-sector comparison.

---

## Scripts

| Script | Main function | What it does |
|---|---|---|
| `scr_RunCorrelAnalysis.m` | `correlationAnalysis`, `correlationAnalysis_Dynamic` | Distribution of pairwise correlations over the full sample and on a rolling window. Compares the results with and without market-mode removal, and with sample versus QIS correlations. |
| `scr_RunNetworkFilter.m` | `getFilteredNetwork` | Builds and plots an MST or PMFG on the full sample. |
| `scr_RunNetworkFilter_Dynamic.m` | `getFilteredNetwork_Dynamic` | Builds an MST or PMFG on every rolling window and tracks the network through time. |
| `scr_RunClustering.m` | `getClusters` | Hierarchical clustering (DBHT or a standard linkage), a dendrogram, and a comparison of the clusters with the sector classification. |
| `scr_RunClustering_Dynamic.m` | `getClusters_Dynamic` | Rolling-window clustering. |
| `scr_RunBootstrapNetwork.m` | `bootstrapNetwork` | Bootstrap resampling to estimate how reliable the edges and network metrics are. |
| `scr_RunBootstrapDBHT.m` | `bootstrapDBHT` | Bootstrap distribution of the number of clusters found by DBHT. |

### Main input options

| Input | Values | Meaning |
|---|---|---|
| `preProp.blnStdize` | `0` / `1` | Z-score each stock's returns (weighted) |
| `preProp.blnRemMktMode` | `0` / `1` | Remove the first PCA factor (the market mode) |
| `preProp.EWMA_Alpha` | e.g. `0`, `0.005` | Exponential decay factor (`0` gives equal weights) |
| `network.distanceMethod` | `'QIS_correlDistMetric'`, `'correlDistMetric'` | QIS-shrunk or sample correlation, turned into a distance |
| `network.filter` | `'MST'`, `'PMFG'` | Network filter |
| `cluster.linkage` | `'DBHT_PMFG'`, `'DBHT_TMFG'`, `'ward'`, `'single'`, `'complete'`, `'average'`, `'centroid'`, `'weighted'`, `'median'` | Clustering method |
| `cluster.maxClusters` | integer | Store cluster allocations for 2 to `maxClusters` clusters |
| `cluster.nClusters` | integer | Number of clusters to report (ignored by DBHT, which chooses its own) |
| `roll.window` | e.g. `52*3` | Rolling window length in observations (3 years of weekly data) |
| `roll.step` | e.g. `4` | Step between windows (about 1 month of weekly data) |
| `bootstrap.nSim` | integer | Number of bootstrap samples |

---

## Functions

| Function | Purpose |
|---|---|
| `preProcessRets` | Weighted standardisation and market-mode (PCA) removal |
| `EWMA_Wts` | Exponential weights `w_t ∝ exp(α(t − T))` and the effective sample size |
| `covMatWtd` | Weighted covariance matrix (Pozzi et al., 2012) |
| `QIS_Wtd` | Weighted covariance with Ledoit–Wolf Quadratic-Inverse Shrinkage |
| `correlToDistMetric` | Correlation to distance: `d_ij = sqrt(2(1 − ρ_ij))` (Mantegna, 1999) |
| `getFilteredNetwork` / `_Dynamic` | MST or PMFG construction, returned as a MATLAB `graph` plus filtered distance and similarity matrices |
| `networkMetrics` | Weighted and unweighted centralities, tree length and hybrid centrality ranks |
| `getClusters` / `_Dynamic` | Hierarchical clustering, dendrogram ordering, cluster tables, and Rand index against a reference partition |
| `DBHTs` | DBHT clustering on a PMFG or TMFG (Song, Di Matteo & Aste, 2012), adapted to allow either |
| `bootstrapNetwork` | Bootstrap edge reliability and metric distributions |
| `bootstrapDBHT` | Bootstrap of the DBHT cluster count |
| `correlationAnalysis` / `_Dynamic` | Correlation-distribution summaries (quartiles, mean, kernel density) |
| `adj2pajek2` | Export an adjacency matrix to Pajek `.net` / `.clu` files |

Each function has a full header comment describing its inputs and outputs. Run `help <functionName>` in MATLAB to read it.

---

## Method overview

1. **Returns** are optionally z-scored, and the market mode is removed by projecting out the first principal component.
2. **Correlations** are estimated with equal or exponential time weights, either as a sample estimate or with QIS shrinkage. QIS shrinkage gives a better-conditioned matrix when the number of assets is large relative to the number of observations.
3. **Distances** are computed as `d = sqrt(2(1 − ρ))`.
4. **Filtering** keeps only the strongest links:
   - The **MST** keeps N − 1 edges (a tree).
   - The **PMFG** keeps 3(N − 2) edges while staying planar, so it retains more information, including 3- and 4-cliques.
5. **Clustering**: DBHT uses the clique (bubble) structure of the PMFG or TMFG to find clusters and a hierarchy without a preset number of clusters. Standard linkages are provided for comparison.
6. **Validation**: clusters are compared with the industry classification using the adjusted Rand index, and stability is assessed by bootstrap.

---

## References

- Mantegna, R. N. (1999). Hierarchical structure in financial markets. *The European Physical Journal B*, 11(1), 193–197.
- Tumminello, M., Aste, T., Di Matteo, T. & Mantegna, R. N. (2005). A tool for filtering information in complex systems. *PNAS*, 102(30), 10421–10426.
- Song, W.-M., Di Matteo, T. & Aste, T. (2012). Hierarchical information clustering by means of topologically embedded graphs. *PLoS ONE*, 7(3), e31929.
- Pozzi, F., Di Matteo, T. & Aste, T. (2012). Exponential smoothing weighted correlations. *The European Physical Journal B*, 85(6), 175. https://doi.org/10.1140/epjb/e2012-20697-x
- Pozzi, F., Di Matteo, T. & Aste, T. (2013). Spread of risk across financial markets: better to invest in the peripheries. *Scientific Reports*, 3, 1665.
- Ledoit, O. & Wolf, M. (2022). Quadratic shrinkage for large covariance matrices. *Bernoulli*, 28(3), 1519–1547.
- Hubert, L. & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2, 193–218.

**MSc thesis**
- Gopi, Y. MSc dissertation, University of Pretoria. UPSpace: https://repository.up.ac.za/items/b176234c-1db9-4e0b-b31b-715cee1a56bf

**Multiplex analysis** (`python/netfilter/multiplex.py`)
- Musumeci, N., Nicosia, V., Aste, T., Di Matteo, T. & Latora, V. (2016). The multiplex dependency structure of financial markets. arXiv:1606.04872. https://arxiv.org/abs/1606.04872v1 (the inspiration for the multiplex code)
- Battiston, F., Nicosia, V. & Latora, V. (2014). Structural measures for multiplex networks. *Physical Review E*, 89, 032804.
- Battiston, F., Nicosia, V. & Latora, V. (2017). The new challenges of multiplex networks: Measures and models. *The European Physical Journal Special Topics*, 226, 401–416.
- ngozzi/multiplex: *Multiplex Dependency Structure of S&P500 Stocks* (the template notebooks). https://github.com/ngozzi/multiplex

## Third-party code

- **PMFG**: T. Aste and co-authors. See `External Functions/PMFG/license.txt`.
- **MatlabBGL 4.0.1**: David Gleich.
- **RandIndex.m**: David Corney (2000).
- **adj2pajek2.m**: Kevin Murphy (2007), based on code by Gergana Bounova, and adapted for this project.
- **QIS**: adapted from Ledoit & Wolf's `covShrinkage` on MATLAB File Exchange.

## Author

Yashin Gopi. MSc thesis: [UPSpace repository](https://repository.up.ac.za/items/b176234c-1db9-4e0b-b31b-715cee1a56bf)
