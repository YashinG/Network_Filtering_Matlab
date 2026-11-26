# Handoff: work done with Claude (22–26 Nov 2026)

This records a multi-session conversation in which the MATLAB repo `YashinG/Network_Filtering_Matlab` was:
- documented
- bug-fixed
- ported to Python (the `netfilter` package)
- verified number-for-number against MATLAB
- extended with a multiplex network analysis based on `ngozzi/multiplex`

It is written so that Claude Code (or a person) can pick up the work without the original chat.

Repo state at handoff: 4 commits on top of upstream `54dceab` (*Update path for matlab_bgl toolbox*). None of them have been pushed; the session had read-only access to GitHub.

```
64f606d Multiplex: user-defined layers (functions, matrices, graphs) and new built-in measures
c56383b Add multiplex analysis (Pearson/Spearman/Kendall PMFG layers) to netfilter
a2dcefe Add Python port (netfilter) verified against MATLAB output
f21b61c Expand README; fix variable name in eval lines and MatlabBGL path in scripts
54dceab (upstream) Update path for matlab_bgl toolbox
```

The test suite has 60 tests, and all pass. To run them: `cd python && pytest`. Two multiplex template tests are skipped unless `MULTIPLEX_TEMPLATE_DATA` is set (see section 4).

---

## 1. What the user asked for, in order

1. **"Can you see this GitHub folder?"** The repo was cloned and summarised.
2. **"Please update the README with more information."** It was rewritten using the actual code: structure, requirements, data format, script and option tables, function reference, method overview and references.
3. **"Fix please."** Three MATLAB bugs found while reading were fixed (section 2).
4. **"Can we port the repo over to Python?"** This became the full `python/netfilter` package, verified against MATLAB R2025b on the user's PC.
5. **"Can you see this: github.com/ngozzi/multiplex/tree/master/notebooks"** That repo was cloned and summarised.
6. **"Replicate the multiplex analysis in py files (not notebook)."** This produced a standalone `multiplex_py/` folder for the ngozzi repo. It was delivered as a zip and is not part of this repo (section 5).
7. **"Use this template for multiplex analysis and build the same functionality into my package."** This produced `netfilter/multiplex.py`.
8. **"Also allow other types of user-specified layers, not only correlation variants."** This produced the generic `Layer` spec and new built-in measures.
9. **"Export this conversation for Claude Code."** That is this file and `CLAUDE.md`.

---

## 2. MATLAB fixes (commit f21b61c)

| File | Bug | Fix |
|---|---|---|
| `Scripts/scr_RunNetworkFilter.m` | The last `eval` line used `stInputs.network.distance_method`, but the script sets `distanceMethod`, so it crashed at the end | Uses `distanceMethod` |
| `Scripts/scr_RunBootstrapNetwork.m` | The same variable-name mismatch, plus `.boostrap` was misspelled | Uses `distanceMethod` and `.bootstrap` |
| `Scripts/scr_RunNetworkFilter_Dynamic.m` | Used the field name `distance_method`, unlike every other script | Renamed to `distanceMethod` |
| All scripts | Only one script added `matlab_bgl-4.0.1/matlab_bgl` to the path; the others added `matlab_bgl-4.0.1`, where the functions aren't | All of them add `.../matlab_bgl` |

The README was expanded in the same commit.

Bugs that were noted but left in the MATLAB code (they are fixed in the Python port only):
- `all(inTimeWts) == 1/nDates` is always false, so plots always say "(Time Wtd)".
- For non-correlation distances (for example `'euclidean'`), `S = 1./(1+squareform(D))` turns the matrix into a vector, so `S` is wrong.
- In `QIS_Wtd`, when there are more assets than the effective number of observations, a non-integer effective sample size is used as an index.

---

## 3. Python port: `python/netfilter` (commit a2dcefe)

### Layout
```
python/
├── netfilter/
│   ├── estimation.py   EWMA weights, weighted cov, QIS, pre-processing, corr→dist, rolling windows
│   ├── filters.py      mst, pmfg (networkx planarity), tmfg (port of PMFG_T2s)
│   ├── metrics.py      network_metrics: MATLAB-compatible centralities + Pozzi hybrid score
│   ├── network.py      get_filtered_network(_dynamic)
│   ├── dbht.py         line-by-line port of DBHTs.m (PMFG or TMFG)
│   ├── clustering.py   get_clusters(_dynamic), adjusted_rand_index
│   ├── correlation.py  correlation_analysis(_dynamic), ksdensity
│   ├── bootstrap.py    bootstrap_network, bootstrap_dbht (seedable)
│   ├── multiplex.py    multiplex builder + analysis (section 4)
│   ├── plotting.py     matplotlib figures
│   └── io.py           load_data (Data.mat), to_pajek
├── scripts/            run_*.py — one per MATLAB script, inputs dict at the top, plus run_multiplex.py
├── tests/              test_netfilter.py, test_matlab_reference.py, test_multiplex.py, data/*.json
├── requirements.txt    numpy, scipy, networkx, pandas, matplotlib (tqdm optional)
└── README.md           install, function map, multiplex docs, differences, verification
```

### Conventions
- Function names follow the MATLAB ones in snake_case, and the positional arguments are in the same order as MATLAB's. The `python/README.md` function map lists them all.
- Results come back as plain dictionaries with the MATLAB struct field names, for example `out["metrics"]["node"]["hybrid"]["XpY"]` is MATLAB's `stOut.metrics.node.Hybrid.XpY`.
- Returns are `(n_dates, p_assets)` arrays. Time weights are `(n_dates,)` arrays, and `None` means equal weights.
- Indices are 0-based. The DBHT cluster labels `T8` are still 1..K. `dbht()` returns both `Z` (SciPy linkage) and `Z_matlab` (1-based, 3 columns).
- The scripts can be run from any folder: `scripts/_common.py` puts the package on `sys.path`.

### How it was verified against MATLAB
The original MATLAB functions were run on the user's machine (MATLAB R2025b, with the Statistics and Machine Learning Toolbox) on `Data/Data.mat`, and the results were compared with Python. These all match to within about 1e-14:
- EWMA weights
- pre-processed returns
- QIS covariance and correlations
- MST and PMFG edge sets
- all weighted centralities, including PageRank
- hybrid scores
- DBHT `T8` and the full linkage (PMFG and TMFG)
- Ward and average-linkage partitions and merge heights
- correlation summaries and densities
- every rolling-window output (normalised tree length, average similarity, hybrid centrality, correlation histograms, DBHT cluster counts and ARI, correlation densities)

MATLAB's numbers are stored in `tests/data/matlab_static.json` and `matlab_rolling.json`, and `tests/test_matlab_reference.py` checks against them. The MATLAB clone used for this was left at `C:\Users\ygopi\AppData\Local\Temp\netfilter_check` on the user's PC, and the user may delete it.

### Decisions made during the port (and why)
- **PageRank:** a converged PageRank differed from MATLAB by 1.5e-4. MATLAB's algorithm was reproduced instead: start from uniform values and stop when the largest change is ≤ 1e-4, with at most 100 iterations. That matches to 1e-16.
- **Ranking ties:** `tiedrank` in MATLAB can split an exact tie on floating-point noise. On the sample MST, stocks 9 and 11 have exactly equal weighted closeness, but MATLAB's two values differ at about 2e-18. Python ranks on values rounded to 12 significant figures, so they count as tied. Their hybrid scores differ from MATLAB's by ±0.0044, and their sum is unchanged; the tests allow for this.
- **DBHT linkage order:** DBHT builds merges cluster by cluster, so its rows aren't sorted by height. `matlab_to_scipy_linkage` re-sorts them stably by height and relabels them. This is safe because each merge is strictly higher than the merges below it.
- **PMFG planarity:** MatlabBGL's Boyer–Myrvold test is replaced by `networkx.check_planarity`. The edge order follows MATLAB's column-major `find` with a stable sort, so ties are broken the same way.
- **Leaf ordering:** SciPy's `optimal_leaf_ordering` uses the `'adjacent'` criterion, whereas MATLAB uses `'group'`. This only changes the plot order and the `clusterIDsOrdered` numbering, not the partitions.
- **Default distributions:** `ksdensity` uses MATLAB's default bandwidth, and `prctile` maps to numpy's `method="hazen"`.
- **Runtime:** a 10,000-sample PMFG bootstrap takes about 9 minutes for 20 assets.

---

## 4. Multiplex analysis: `netfilter/multiplex.py` (commits c56383b, 64f606d)

It is based on `ngozzi/multiplex` (`notebooks/multiplex_analysis.ipynb`), which uses the measures of Battiston, Nicosia & Latora. That repo has **no licence**, so none of its data was copied here.

### API
```python
import netfilter as nf
from netfilter import Layer

mplex = nf.build_multiplex(returns, weights, names,
                           layers=("pearson", "spearman", "kendall"),   # default
                           network_filter="PMFG", standardise=True,
                           remove_market_mode=True, shrinkage=True)
res = nf.multiplex_analysis(mplex["layers"], sectors=None, top=10,
                            mpc_threshold=0.5, overlap_threshold=None, plot=False)
```
`build_multiplex` returns:
- `layers`: `{name: nx.Graph}`. Every asset is a node, and each edge has a `weight` (the raw value) and, for filtered layers, a `distance`.
- `matrix`: the layer matrices, also available under the alias `corr`.
- `adjacency`: the 0/1 adjacency matrices.
- `specs`: the resolved `Layer` objects.
- `inputs`: the inputs, including `dropped_nodes`.

`multiplex_analysis` returns:
- `degrees`, `degree_distributions`, `top_nodes`
- `MPC`, `cartography`
- `similarity`, `edge_overlap`
- `sector_degrees`
- `figs` (when `plot=True`)

### Layer spec
`Layer(name, source, kind=None, network_filter="default", raw_returns=False, params={})`

`source` can be:
- **a built-in name**, which runs on the processed returns
- **a function** `f(returns, weights, **params) -> p×p`
- **a numpy array** in the order of `names`
- **a labelled DataFrame**, reindexed to `names`
- **a networkx Graph**, used exactly as given: nodes outside `names` are dropped and reported in `dropped_nodes`, and missing assets become isolated nodes

`kind` is one of these. Functions and matrices must set it; built-ins set their own.
- `correlation`: distance `sqrt(2(1-x))`, similarity `1+x`
- `similarity`
- `distance`
- `adjacency`: the non-zero entries are the edges, with no filtering

`network_filter` is `"PMFG"`, `"MST"` or `None`. With `"default"`, the layer uses the builder's filter, except adjacency layers, which use `None`.

The MST and PMFG depend only on the order of the values, so similarities and distances are shifted to be strictly positive. That matters because zeros mean "no edge" in the filter code. For a non-symmetric input, each pair keeps the stronger direction.

### Built-in measures (`BUILTIN_MEASURES`)
All of them accept time weights.

| Name | Kind | Checked against |
|---|---|---|
| `pearson` | correlation | Identical to `get_filtered_network`'s PMFG |
| `spearman` | correlation | `scipy.stats.spearmanr` (equal weights, no shrinkage) |
| `kendall` | correlation | `scipy.stats.kendalltau` (tau-b, including tied values). Each pair of dates is weighted `w_s·w_t`, and the computation is chunked |
| `partial` | correlation | Correlation of the regression residuals. Uses the pinv of the (QIS) covariance |
| `distance_correlation` | similarity | The `dcor` package (V-statistic). 0/1 weights give the same result as the subsample. Chunked |
| `lower_tail` / `upper_tail` | similarity | A manual count. Uses `q` (default 0.1); the upper tail is the lower tail of −x, so both tails use the same number of days |

### Analysis decisions (and why)
- **Edge overlap:** PMFG and MST layers contain every asset, so the template's node-activity similarity is always 1. `edge_overlap` was added: the Jaccard overlap of edge sets, the number of shared edges, and how many layers each edge is in.
- **Hub threshold:** the default cartography hub threshold is the mean + 2 SD of the overlapping degree (`z > 2`, as in Battiston et al. 2014). The template used a fixed 60, which only suits its 343 stocks. With the 20 stocks in `Data.mat`, nothing reaches the default, so all 20 are multiplex leaves.
- **Thresholds and boundary nodes:** a node exactly on a threshold counts as hub or multiplex, and the `on_boundary` column flags it. The template's strict inequalities silently dropped such nodes; EOG, with overlapping degree exactly 60, disappeared from its plot.
- **Unlinked nodes:** a node with no links in any layer gets MPC = NaN and the role `"isolated"`.
- **Top-node ties:** `top_nodes(tie_break="stable")` is the default. `"notebook"` reproduces the template's order exactly, which comes from re-sorting the same list repeatedly.
- **Template check:** on the template's own data, the package reproduces the notebook's printed top-10 lists, its layer similarities (0.8630, 0.8659, 0.9475) and its cartography counts (270 multiplex leaves, 56 multiplex hubs, 17 focused leaves at a threshold of 60). To run those tests, set `MULTIPLEX_TEMPLATE_DATA=/path/to/ngozzi/multiplex/data`.

### Results on `Data.mat` (default layers, equal weights)
- **Shared edges:** Spearman and Kendall share 80% of their edges. Pearson shares 71% with Spearman and 64% with Kendall. 41 of the 68 distinct edges appear in all three layers.
- **Custom-layer example:** the example in `scripts/run_multiplex.py` (`EXAMPLE_CUSTOM_LAYERS`: Pearson, lower tail, distance-correlation MST, beta similarity, and a same-sector graph) gives low overlaps. For example, distance correlation and Pearson share 0.106 (Jaccard).

---

## 5. Standalone deliverable outside this repo: `multiplex_py/`

This is a plain-Python version of the ngozzi notebook, for use **inside a clone of ngozzi/multiplex**:
- `run_multiplex_analysis.py`, which finds `../data` on its own
- the `multiplex_analysis/` package
- `tests/`, with 6 tests that reproduce the notebook's printed output

It was updated for current libraries:
- networkx ≥ 2.4 removed `connected_component_subgraphs`.
- The `seaborn-dark` style and `%pylab` no longer work.
- The original `requirements.txt` listed the standard-library modules `math` and `collections`.

It was delivered to the user as `multiplex_py.zip` and isn't needed for this repo, because `netfilter.multiplex` supersedes it.

Observation about that repo's data: its `data/edgeslist/*.txt` networks are **not** planar, and each layer splits into many components (the largest has about 100 of the 310–328 nodes). So they aren't PMFGs, even though its README says they are.

---

## 6. Open items and ideas (not started)
- **Push:** the 4 commits haven't been pushed. That needs the user's credentials or a push-enabled session.
- **Dynamic multiplex:** a rolling-window version (`build_multiplex` per window, tracking MPC, roles and edge overlap over time), following the `*_dynamic` pattern in `network.py`.
- **Weighted multiplex measures:** strength-based overlapping degree, or the weighted participation coefficient.
- **Mesoscale overlap:** comparing DBHT clusters across layers, for example ARI between layers.
- **Packaging:** a `pyproject.toml` to make `netfilter` pip-installable. It is currently imported via `sys.path`.
- **Test dependency:** `dcor` was only used to check `distance_correlation` and isn't a dependency; the tests use a naive reference implementation instead.
- **Deprecation warnings:** the tests are run with `-W ignore::DeprecationWarning`. These are networkx/scipy warnings, not package errors.
