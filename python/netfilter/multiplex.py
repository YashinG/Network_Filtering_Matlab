"""Multiplex dependency networks: one filtered network per correlation measure.

Builds a multiplex in which every layer is a filtered network (PMFG or MST)
of the same assets, estimated with a different dependence measure: Pearson,
Spearman or Kendall. It then analyses the layers together with the measures
used in the ngozzi/multiplex template (Battiston, Nicosia & Latora, 2014 and
2017):

* overlapping degree        ``o_i = sum_a k_i^a``
* participation coefficient ``P_i = M/(M-1) [1 - sum_a (k_i^a / o_i)^2]``
* multiplex cartography     (focused/multiplex × hub/leaf)
* node-activity similarity  ``Q[a, b] = (1/N) sum_i b_i^a b_i^b``
* edge overlap between layers (added here; see :func:`edge_overlap`)
* average degree per sector

The analysis functions take a ``{layer_name: networkx.Graph}`` dict, so they
also work on networks built elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .estimation import (_check_weights, cov_to_corr, cov_weighted, corr_to_dist,
                         is_equal_weighted, preprocess_returns, qis_weighted)
from .filters import mst, pmfg
from .network import adjacency_to_graph

MEASURES = ("pearson", "spearman", "kendall")
ROLES = ("focused leaf", "focused hub", "multiplex leaf", "multiplex hub")


# ------------------------------------------------------------ correlations
def _pearson(x, w, shrinkage):
    cov = qis_weighted(x, w) if shrinkage else cov_weighted(x, w)[0]
    return cov_to_corr(cov)


def kendall_tau_weighted(returns, weights=None, chunk=200_000):
    """Kendall's tau-b matrix, with optional time weights.

    Each pair of dates (s, t) gets weight ``w_s * w_t``:

        tau_ij = sum w_s w_t sgn(dx_i) sgn(dx_j)
                 / sqrt(sum w_s w_t [dx_i != 0] * sum w_s w_t [dx_j != 0])

    With equal weights this is exactly ``scipy.stats.kendalltau`` (tau-b,
    tie-corrected). Pairs of dates are processed in blocks of about ``chunk``
    pairs to limit memory use.
    """
    x = np.asarray(returns, dtype=float)
    n, p = x.shape
    w = _check_weights(weights, n)
    num = np.zeros((p, p))
    rows_per_block = max(1, chunk // n)
    for start in range(0, n - 1, rows_per_block):
        stop = min(n - 1, start + rows_per_block)
        blocks = []
        for s in range(start, stop):
            d = np.sign(x[s + 1:] - x[s])               # (n - s - 1, p)
            blocks.append(d * np.sqrt(w[s] * w[s + 1:])[:, None])
        A = np.vstack(blocks)
        num += A.T @ A
    den = np.sqrt(np.outer(np.diag(num), np.diag(num)))
    tau = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    np.fill_diagonal(tau, 1.0)
    return 0.5 * (tau + tau.T)


def dependence_matrix(processed_rets, weights=None, measure="pearson", shrinkage=True):
    """Correlation matrix for one layer of the multiplex.

    * ``'pearson'``  – weighted (QIS-shrunk if ``shrinkage``) Pearson correlation
    * ``'spearman'`` – the same estimator applied to each asset's ranks
    * ``'kendall'``  – weighted Kendall tau-b (:func:`kendall_tau_weighted`);
      ``shrinkage`` does not apply
    """
    x = np.asarray(processed_rets, dtype=float)
    w = _check_weights(weights, x.shape[0])
    if measure == "pearson":
        return _pearson(x, w, shrinkage)
    if measure == "spearman":
        return _pearson(np.apply_along_axis(rankdata, 0, x), w, shrinkage)
    if measure == "kendall":
        return kendall_tau_weighted(x, w)
    raise ValueError(f"measure must be one of {MEASURES}")


def build_multiplex(returns, weights, names, measures=MEASURES, network_filter="PMFG",
                    standardise=True, remove_market_mode=True, shrinkage=True):
    """Build one filtered network per dependence measure.

    Returns are pre-processed once (standardised and/or market mode removed),
    then each measure gives a correlation matrix, the distance
    ``sqrt(2 (1 - rho))`` and the similarity ``1 + rho``, and the MST (on
    distance) or PMFG (on similarity) filter.

    Returns
    -------
    dict with
      ``layers`` – ``{measure: networkx.Graph}`` with node labels ``names``
                   and edge attributes ``weight`` (= correlation) and ``distance``
      ``corr``   – ``{measure: (p, p) correlation matrix}``
      ``adjacency`` – ``{measure: (p, p) 0/1 adjacency matrix}``
      ``inputs`` – the inputs and the processed returns
    """
    returns = np.asarray(returns, dtype=float)
    names = list(names)
    w = _check_weights(weights, returns.shape[0])
    processed, *_ = preprocess_returns(returns, w, standardise, remove_market_mode)

    out = {"layers": {}, "corr": {}, "adjacency": {}}
    for m in measures:
        corr = dependence_matrix(processed, w, m, shrinkage)
        D, S = corr_to_dist(corr), 1.0 + corr
        np.fill_diagonal(D, 0.0)
        if network_filter == "PMFG":
            adj = pmfg(S) != 0
        elif network_filter == "MST":
            adj = mst(D) != 0
        else:
            raise ValueError("network_filter must be 'MST' or 'PMFG'")
        G = adjacency_to_graph(np.where(adj, corr, 0.0), names)
        for u, v in G.edges():
            G[u][v]["distance"] = float(D[names.index(u), names.index(v)])
        out["layers"][m], out["corr"][m], out["adjacency"][m] = G, corr, adj.astype(int)
    out["inputs"] = {"inReturns": returns, "processedRets": processed, "inNames": names,
                     "inTimeWts": w, "measures": tuple(measures), "network_filter": network_filter,
                     "standardise": standardise, "remove_market_mode": remove_market_mode,
                     "shrinkage": shrinkage, "time_weighted": not is_equal_weighted(w)}
    return out


# ------------------------------------------------------------ analysis
def degree_table(layers) -> pd.DataFrame:
    """Degree of every node in every layer (0 if absent) plus ``overlapping``.

    Rows are in order of first appearance, starting with the first layer.
    """
    names = list(layers)
    rows: dict = {}
    for name, G in layers.items():
        for node, k in G.degree():
            rows.setdefault(node, {n: 0 for n in names})[name] = k
    df = pd.DataFrame.from_dict(rows, orient="index", columns=names)
    df.index.name = "node"
    df["overlapping"] = df[names].sum(axis=1)
    return df


def _layer_cols(degrees):
    return [c for c in degrees.columns if c != "overlapping"]


def degree_distribution(G) -> pd.Series:
    """Number of nodes with each degree."""
    return pd.Series(dict(G.degree())).value_counts().sort_index()


def top_nodes(degrees: pd.DataFrame, n: int = 10, tie_break: str = "stable") -> pd.DataFrame:
    """The ``n`` highest-degree nodes per layer and by overlapping degree.

    ``tie_break``:
      * ``'stable'``   – tied nodes keep the row order of ``degrees``
      * ``'notebook'`` – reproduce the template notebook exactly. It sorts one
        list repeatedly (ascending, then reversed) by each layer in turn, so
        tied nodes come out in an order that depends on the earlier sorts.

    Returns a DataFrame with one column per layer plus ``overlapping``.
    """
    if tie_break == "stable":
        return pd.DataFrame({c: degrees[c].sort_values(ascending=False, kind="stable").index[:n]
                             for c in degrees.columns}, index=range(1, n + 1))
    if tie_break != "notebook":
        raise ValueError("tie_break must be 'stable' or 'notebook'")
    layers = _layer_cols(degrees)
    rows = [[node] + [int(degrees.at[node, c]) for c in layers] for node in degrees.index]
    rows.sort(key=lambda r: r[1])
    rows.reverse()
    out = {}
    for j, layer in enumerate(layers, start=1):
        rows.sort(key=lambda r, j=j: r[j])
        rows.reverse()
        out[layer] = [r[0] for r in rows[:n]]
    od = [[r[0], sum(r[1:])] for r in rows]
    od.sort(key=lambda r: r[1])
    od.reverse()
    out["overlapping"] = [r[0] for r in od[:n]]
    return pd.DataFrame(out, index=range(1, n + 1))


def participation_coefficient(degrees: pd.DataFrame) -> pd.Series:
    """Multiplex participation coefficient ``P_i = M/(M-1) [1 - sum_a (k_i^a/o_i)^2]``.

    It is 0 when all of a node's links are in one layer and 1 when they are
    spread evenly over the ``M`` layers.
    """
    layers = _layer_cols(degrees)
    M = len(layers)
    if M < 2:
        raise ValueError("Need at least two layers")
    o = degrees["overlapping"]
    return (M / (M - 1) * (1 - degrees[layers].pow(2).sum(axis=1) / o.pow(2))).rename("MPC")


def cartography(degrees: pd.DataFrame, mpc_threshold: float = 0.5, overlap_threshold=None):
    """Classify nodes as focused/multiplex (by MPC) and hub/leaf (by overlapping degree).

    ``overlap_threshold`` defaults to the mean plus two standard deviations of
    the overlapping degree. That is the hub rule of Battiston et al. (2014),
    ``z(o_i) > 2``, and adapts to the size and filter of the network. The
    template used a fixed 60 for its data. Nodes exactly on a threshold count
    as hub / multiplex and are flagged in ``on_boundary``.

    Returns a DataFrame with ``MPC``, ``overlapping``, ``z_overlapping``,
    ``role`` and ``on_boundary``. The threshold used is in ``.attrs``.
    """
    mpc = participation_coefficient(degrees)
    o = degrees["overlapping"].astype(float)
    sd = o.std(ddof=0)
    z = (o - o.mean()) / sd if sd > 0 else o * 0
    if overlap_threshold is None:
        overlap_threshold = o.mean() + 2 * sd
    hub, multi = o >= overlap_threshold, mpc >= mpc_threshold
    role = pd.Series("focused leaf", index=degrees.index)
    role[hub & ~multi] = "focused hub"
    role[~hub & multi] = "multiplex leaf"
    role[hub & multi] = "multiplex hub"
    out = pd.DataFrame({"MPC": mpc, "overlapping": degrees["overlapping"], "z_overlapping": z,
                        "role": role,
                        "on_boundary": np.isclose(mpc, mpc_threshold) | np.isclose(o, overlap_threshold)})
    out.attrs.update(mpc_threshold=mpc_threshold, overlap_threshold=float(overlap_threshold))
    return out


def layer_similarity(degrees: pd.DataFrame) -> pd.DataFrame:
    """Node-activity similarity ``Q[a, b] = (1/N) sum_i b_i^a b_i^b``.

    ``b_i^a = 1`` if node ``i`` has a link in layer ``a``. PMFG and MST layers
    contain every asset, so this is 1 for them. Use :func:`edge_overlap` to
    compare filtered layers instead.
    """
    layers = _layer_cols(degrees)
    active = (degrees[layers] != 0).astype(int)
    return (active.T @ active) / len(degrees)


def edge_overlap(layers) -> dict:
    """How many links the layers share.

    Returns
    -------
    dict with
      ``jaccard``  – DataFrame of ``|E_a ∩ E_b| / |E_a ∪ E_b|`` for each pair of layers
      ``shared``   – DataFrame of ``|E_a ∩ E_b|``
      ``multiplicity`` – Series: number of edges present in exactly 1, 2, …, M layers
      ``edge_layers``  – Series: for each edge, how many layers contain it
    """
    names = list(layers)
    E = {n: {frozenset(e) for e in G.edges()} for n, G in layers.items()}
    shared = pd.DataFrame([[len(E[a] & E[b]) for b in names] for a in names], index=names, columns=names)
    jac = pd.DataFrame([[len(E[a] & E[b]) / len(E[a] | E[b]) if E[a] | E[b] else np.nan
                         for b in names] for a in names], index=names, columns=names)
    counts: dict = {}
    for n in names:
        for e in E[n]:
            counts[e] = counts.get(e, 0) + 1
    edge_layers = pd.Series({tuple(sorted(e)): c for e, c in counts.items()}, name="n_layers")
    mult = edge_layers.value_counts().reindex(range(1, len(names) + 1), fill_value=0)
    mult.index.name = "layers"
    return {"jaccard": jac, "shared": shared, "multiplicity": mult, "edge_layers": edge_layers}


def sector_degrees(degrees: pd.DataFrame, sectors) -> pd.DataFrame:
    """Average degree per sector in each layer and overlapping.

    ``sectors`` is a mapping node -> sector or a sequence aligned with
    ``degrees.index``. Sectors are sorted by average overlapping degree.
    """
    sec = (pd.Series(sectors, index=degrees.index) if not isinstance(sectors, (dict, pd.Series))
           else pd.Series(sectors).reindex(degrees.index))
    out = degrees.groupby(sec).mean().sort_values("overlapping", ascending=False)
    out.index.name = "sector"
    return out


def giant_component(G) -> nx.Graph:
    """Largest connected component (as a copy)."""
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def write_layers(layers, out_dir, giant_only=False) -> dict:
    """Write each layer as ``<layer>_edgelist.csv`` (``node node weight`` per line)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, G in layers.items():
        paths[name] = out_dir / f"{name}{'_GCC' if giant_only else ''}_edgelist.csv"
        nx.write_weighted_edgelist(giant_component(G) if giant_only else G, paths[name])
    return paths


def read_layers(paths) -> dict:
    """Read weighted edge lists into a ``{name: Graph}`` dict (e.g. the template's data)."""
    return {name: nx.read_weighted_edgelist(p) for name, p in paths.items()}


def multiplex_analysis(layers, sectors=None, top=10, mpc_threshold=0.5, overlap_threshold=None,
                       plot=False, labels=None):
    """Run every analysis in the template on a ``{name: Graph}`` multiplex.

    Returns
    -------
    dict with ``degrees``, ``degree_distributions``, ``top_nodes``, ``MPC``,
    ``cartography``, ``similarity``, ``edge_overlap``, ``sector_degrees`` (if
    ``sectors`` given) and ``figs`` (if ``plot``).
    """
    from . import plotting

    deg = degree_table(layers)
    out = {"degrees": deg,
           "degree_distributions": {n: degree_distribution(G) for n, G in layers.items()},
           "top_nodes": top_nodes(deg, top),
           "MPC": participation_coefficient(deg),
           "cartography": cartography(deg, mpc_threshold, overlap_threshold),
           "similarity": layer_similarity(deg),
           "edge_overlap": edge_overlap(layers)}
    if sectors is not None:
        out["sector_degrees"] = sector_degrees(deg, sectors)
    if plot:
        out["figs"] = {
            "degree_distributions": plotting.plot_multiplex_degree_distributions(layers),
            "overlapping_degree": plotting.plot_overlapping_degree(deg),
            "participation_coefficient": plotting.plot_participation_coefficient(out["MPC"]),
            "cartography": plotting.plot_cartography(out["cartography"], labels=labels),
            "edge_overlap": plotting.plot_edge_overlap(out["edge_overlap"]),
        }
        if sectors is not None:
            out["figs"]["sector_degrees"] = plotting.plot_sector_degrees(out["sector_degrees"])
    return out
