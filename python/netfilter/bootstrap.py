"""Bootstrap reliability estimates. Ports of bootstrapNetwork.m and bootstrapDBHT.m."""

from __future__ import annotations

import numpy as np

from .clustering import get_clusters
from .network import _progress, _subtitle, get_filtered_network
from . import plotting


def _bootstrap_samples(n_dates, n_sim, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_dates, size=(n_dates, n_sim))


def bootstrap_network(returns, names, standardise=True, remove_market_mode=True,
                      network_filter="PMFG", dist_method="QIS_correlDistMetric",
                      n_sim=1000, plot=False, seed=None, progress=True):
    """Edge reliability and hybrid-centrality stability by bootstrap resampling of dates.

    Returns
    -------
    dict with ``FullPeriod`` (full-sample network), ``bootsam`` (n_dates × n_sim
    resampled row indices), ``checkOverlap`` (edges × n_sim, True if the
    full-sample edge is present in the bootstrap network), ``edgeReliability``,
    ``XpY`` (p × n_sim), ``XpY_ave`` and ``XpY_stdev``.
    """
    returns = np.asarray(returns, dtype=float)
    n_dates, p = returns.shape
    w = np.full(n_dates, 1.0 / n_dates)
    full = get_filtered_network(returns, w, names, network_filter, standardise,
                                remove_market_mode, dist_method, False)
    full_edges = np.column_stack(np.nonzero(np.triu(full["filteredNetwork_D"], 1)))
    boots = _bootstrap_samples(n_dates, n_sim, seed)

    overlap = np.zeros((len(full_edges), n_sim), dtype=bool)
    xpy = np.zeros((p, n_sim))
    for s in _progress(range(n_sim), progress, "Running bootstrap resampling"):
        tmp = get_filtered_network(returns[boots[:, s]], w, names, network_filter, standardise,
                                   remove_market_mode, dist_method, False)
        fd = tmp["filteredNetwork_D"]
        xpy[:, s] = tmp["metrics"]["node"]["hybrid"]["XpY"]
        overlap[:, s] = fd[full_edges[:, 0], full_edges[:, 1]] != 0

    rel = overlap.mean(axis=1)
    out = {"FullPeriod": full, "bootsam": boots, "nSim": n_sim, "edges": full_edges,
           "checkOverlap": overlap, "edgeReliability": rel, "XpY": xpy,
           "XpY_ave": xpy.mean(axis=1), "XpY_stdev": xpy.std(axis=1, ddof=1)}
    if plot:
        plotting.plot_network_reliability(
            full["g"], [(names[i], names[j]) for i, j in full_edges], rel,
            f"{network_filter} Bootstrap",
            _subtitle(dist_method, w, standardise, remove_market_mode, "Distance"))
    return out


def bootstrap_dbht(returns, names, standardise=True, remove_market_mode=True,
                   link_method="DBHT_PMFG", dist_method="QIS_correlDistMetric",
                   max_clusters=20, n_sim=10, plot=False, seed=None, progress=True):
    """Bootstrap distribution of the number of clusters chosen by DBHT."""
    if link_method not in ("DBHT_PMFG", "DBHT_TMFG"):
        raise ValueError("This function only works for the DBHT linkage functions.")
    returns = np.asarray(returns, dtype=float)
    n_dates, p = returns.shape
    w = np.full(n_dates, 1.0 / n_dates)
    full = get_clusters(returns, w, names, dist_method, link_method, standardise,
                        remove_market_mode, max_clusters, False, None, None)
    boots = _bootstrap_samples(n_dates, n_sim, seed)
    n_cl = np.zeros(n_sim, dtype=int)
    for s in _progress(range(n_sim), progress, "Bootstrapping number of DBHT clusters"):
        tmp = get_clusters(returns[boots[:, s]], w, names, dist_method, link_method,
                           standardise, remove_market_mode, None, False, None, None)
        n_cl[s] = tmp["nClusters"]
    out = {"FullPeriod": full, "bootsam": boots, "nSim": n_sim, "nClusters": n_cl,
           "nClusters_ave": n_cl.mean(),
           "nClusters_stdev": n_cl.std(ddof=1) if n_sim > 1 else 0.0}
    if plot:
        plotting.plot_histogram(n_cl, "Bootstrap Estimate: Optimal Number of Clusters (DBHT)",
                                _subtitle(dist_method, w, standardise, remove_market_mode,
                                          "Distance"), "nClusters")
    return out
