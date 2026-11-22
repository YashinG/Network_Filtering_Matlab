"""Hierarchical clustering (DBHT and standard linkages). Ports of
getClusters.m, getClusters_Dynamic.m and RandIndex.m."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from .dbht import dbht
from .estimation import _check_weights, preprocess_returns, rolling_end_indices, similarity_and_distance
from .network import _progress, _subtitle
from . import plotting

LINKAGES = ("DBHT_PMFG", "DBHT_TMFG", "ward", "single", "complete", "average",
            "centroid", "weighted", "median")


def adjusted_rand_index(c1, c2):
    """Hubert & Arabie (1985) adjusted Rand index between two partitions."""
    c1, c2 = np.asarray(c1).ravel(), np.asarray(c2).ravel()
    _, a = np.unique(c1, return_inverse=True)
    _, b = np.unique(c2, return_inverse=True)
    C = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(C, (a, b), 1)
    n = C.sum()
    comb = lambda x: x * (x - 1) / 2
    sum_ij = comb(C).sum()
    sum_a, sum_b = comb(C.sum(axis=1)).sum(), comb(C.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb(n)
    max_idx = 0.5 * (sum_a + sum_b)
    if max_idx == expected:
        return 0.0
    return (sum_ij - expected) / (max_idx - expected)


def _renumber_in_leaf_order(T, leaf_order):
    """Renumber clusters 1, 2, ... in the order they appear along the dendrogram."""
    t = T[leaf_order]
    new = np.ones(len(t), dtype=int)
    for i in range(1, len(t)):
        new[i] = new[i - 1] + (t[i] != t[i - 1])
    out = np.empty_like(new)
    out[leaf_order] = new
    return out


def get_clusters(returns, weights, names, dist_method="QIS_correlDistMetric",
                 link_method="ward", standardise=True, remove_market_mode=True,
                 max_clusters=None, plot=False, n_clusters=None, comp_partition=None):
    """Hierarchical clustering of assets.

    Parameters
    ----------
    link_method : one of ``LINKAGES``. DBHT methods choose the number of
        clusters themselves (``n_clusters`` is then ignored).
    max_clusters : store allocations for 2..max_clusters clusters (default p).
    n_clusters : number of clusters to report (default ``max_clusters``).
    comp_partition : (n_part, p) array of reference partitions (e.g. ICB
        sectors) to compare against with the adjusted Rand index.

    Returns
    -------
    dict with ``Z`` (SciPy linkage), ``S``, ``D``, ``leafOrder``, ``labels``,
    ``labelsOrdered``, ``clusterIDs``, ``clusterIDsOrdered``, ``nClusters``,
    ``allClusterIDs``, ``allClusterIDsOrdered``, ``adjRandIndex``, ``tables``
    (pandas DataFrames), ``stOutDBHT`` (DBHT only) and ``inputs``.
    """
    returns = np.asarray(returns, dtype=float)
    names = list(names)
    n_dates, p = returns.shape
    w = _check_weights(weights, n_dates)
    if max_clusters is None:
        max_clusters = p
    if n_clusters is None:
        n_clusters = max_clusters

    processed, *_ = preprocess_returns(returns, w, standardise, remove_market_mode)
    corr, D, S = similarity_and_distance(processed, w, dist_method)
    Dc = squareform(D, checks=False)

    out_dbht = None
    if link_method in ("DBHT_PMFG", "DBHT_TMFG"):
        out_dbht = dbht(D, S, "TMFG" if link_method == "DBHT_TMFG" else "PMFG")
        Z = out_dbht["Z"]
        n_clusters = len(np.unique(out_dbht["T8"]))
    elif link_method in LINKAGES:
        Z = linkage(Dc, method=link_method)
    else:
        raise ValueError(f"Unknown linkage '{link_method}'")

    Z = optimal_leaf_ordering(Z, Dc)
    leaf_order = leaves_list(Z)

    req = list(range(2, max_clusters + 1))
    all_ids = np.zeros((p, len(req)), dtype=int)
    all_ids_ord = np.zeros((p, len(req)), dtype=int)
    for i, k in enumerate(req):
        all_ids[:, i] = fcluster(Z, k, criterion="maxclust")
        all_ids_ord[:, i] = _renumber_in_leaf_order(all_ids[:, i], leaf_order)

    T = fcluster(Z, n_clusters, criterion="maxclust")
    ordered_T = _renumber_in_leaf_order(T, leaf_order)
    heights = np.sort(Z[:, 2])
    m = Z.shape[0]
    th = heights[m + 1 - n_clusters] if n_clusters > 1 else heights[-1] + 0.001

    ari = None
    if comp_partition is not None and np.size(comp_partition):
        comp = np.atleast_2d(comp_partition)
        ari = np.array([[adjusted_rand_index(all_ids_ord[:, j], comp[i])
                         for j in range(len(req))] for i in range(comp.shape[0])])

    cols = [f"{k} Clusters" for k in req]
    tables = {
        "allClusterIDsOrdered": pd.DataFrame(all_ids_ord, index=names, columns=cols),
        "clusterIDsOrdered": pd.DataFrame(ordered_T, index=names, columns=[f"{n_clusters} Clusters"]),
    }
    if ari is not None:
        tables["ARI"] = pd.DataFrame(ari, index=[f"Partition_{i + 1}" for i in range(ari.shape[0])],
                                     columns=cols)

    out = {"Z": Z, "S": S, "D": D, "labels": names,
           "labelsOrdered": [names[i] for i in leaf_order], "leafOrder": leaf_order,
           "clusterIDs": T, "nClusters": n_clusters, "clusterIDsOrdered": ordered_T,
           "allClusterIDs": all_ids, "allClusterIDsOrdered": all_ids_ord,
           "adjRandIndex": ari, "tables": tables, "threshold": th,
           "inputs": {"inReturns": returns, "processedRets": processed, "corrMat": corr,
                      "inTimeWts": w, "standardise": standardise,
                      "remove_market_mode": remove_market_mode, "dist_method": dist_method,
                      "link_method": link_method, "n_clusters": n_clusters,
                      "max_clusters": max_clusters, "comp_partition": comp_partition}}
    if out_dbht is not None:
        out["stOutDBHT"] = out_dbht

    if plot:
        out["fig"] = plotting.plot_dendrogram(
            Z, [f"{nm} {c}" for nm, c in zip(names, ordered_T)], th,
            f"Hierarchical Clustering: {link_method}",
            _subtitle(dist_method, w, standardise, remove_market_mode, "Distance"))
    return out


def get_clusters_dynamic(returns, dates, weights, window, step, names,
                         dist_method="QIS_correlDistMetric", link_method="ward",
                         standardise=True, remove_market_mode=True, max_clusters=None,
                         plot=False, n_clusters=None, comp_partition=None, progress=True):
    """Run :func:`get_clusters` on rolling windows.

    Returns ``{'dynamicClusters': {...}, 'inputs': {...}}`` where
    ``dynamicClusters`` holds ``dates``, ``clusterIDs`` (p × T), ``nClusters``,
    ``adjRandIndex`` (n_part × T, if ``comp_partition`` given), ``correlDist``
    and the raw/processed returns per window.
    """
    returns = np.asarray(returns, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    n_dates, p = returns.shape
    w = _check_weights(weights, window)
    ends = rolling_end_indices(n_dates, window, step)
    T = len(ends)
    comp = None if comp_partition is None else np.atleast_2d(comp_partition)
    edges = np.round(np.arange(-1, 1.0001, 0.1), 10)
    tril = np.tril(np.ones((p, p), dtype=bool), -1)

    res = {"dates": [], "clusterIDs": np.zeros((p, T), dtype=int), "nClusters": np.zeros(T, dtype=int),
           "correlDist": np.zeros((len(edges) - 1, T)), "inReturns": [], "processedRets": []}
    if comp is not None:
        res["adjRandIndex"] = np.full((comp.shape[0], T), np.nan)

    for t, end in enumerate(_progress(ends, progress, "Running dynamic clustering")):
        tmp = get_clusters(returns[end - window:end], w, names, dist_method, link_method,
                           standardise, remove_market_mode, max_clusters, False, n_clusters, None)
        res["inReturns"].append(tmp["inputs"]["inReturns"])
        res["processedRets"].append(tmp["inputs"]["processedRets"])
        res["dates"].append(dates[end - 1])
        res["clusterIDs"][:, t] = tmp["clusterIDs"]
        res["nClusters"][t] = tmp["clusterIDs"].max()
        if comp is not None:
            res["adjRandIndex"][:, t] = [adjusted_rand_index(tmp["clusterIDs"], c) for c in comp]
        c = tmp["inputs"]["corrMat"][tril]
        res["correlDist"][:, t] = np.histogram(c, bins=edges)[0] / c.size

    res["dates"] = pd.DatetimeIndex(res["dates"])
    if plot:
        plotting.plot_dynamic_clusters(
            res["dates"], res["nClusters"], res.get("adjRandIndex"), link_method,
            _subtitle(dist_method, w, standardise, remove_market_mode, "Distance"))
    return {"dynamicClusters": res,
            "inputs": {"inReturns": returns, "labels": list(names), "inTimeWts": w,
                       "window": window, "step": step, "standardise": standardise,
                       "remove_market_mode": remove_market_mode, "dist_method": dist_method,
                       "link_method": link_method, "n_clusters": n_clusters,
                       "max_clusters": max_clusters, "comp_partition": comp_partition}}
