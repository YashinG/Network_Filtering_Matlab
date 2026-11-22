"""Filtered correlation networks. Ports of getFilteredNetwork.m and
getFilteredNetwork_Dynamic.m."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .estimation import (_check_weights, is_equal_weighted, preprocess_returns,
                         rolling_end_indices, similarity_and_distance)
from .filters import mst, pmfg
from .metrics import network_metrics
from . import plotting


def _subtitle(method, weights, standardise, remove_market_mode, prefix="Method"):
    s = f"{prefix}: {method}"
    if not is_equal_weighted(weights):
        s += " (Time Wtd)"
    if standardise:
        s += ", Stdized"
    if remove_market_mode:
        s += ", Market Mode Removed"
    return s


def adjacency_to_graph(adj, names):
    """networkx Graph with string node labels and ``weight`` edge attributes."""
    G = nx.from_numpy_array(np.asarray(adj, dtype=float))
    return nx.relabel_nodes(G, dict(enumerate(names)))


def get_filtered_network(returns, weights, names, network_filter="PMFG",
                         standardise=True, remove_market_mode=True,
                         dist_method="QIS_correlDistMetric", plot=False):
    """Build an MST or PMFG from asset returns.

    Returns
    -------
    dict with
      ``S``, ``D``                – full similarity and distance matrices
      ``filterName``              – ``'MST'`` or ``'PMFG'``
      ``g``                       – networkx Graph (weights: distances for MST,
                                    similarities for PMFG, as in MATLAB)
      ``filteredNetwork``         – weighted adjacency of the filtered graph
      ``filteredNetwork_D/_S``    – D and S restricted to the filtered edges
      ``metrics``                 – output of :func:`network_metrics`
      ``inputs``                  – inputs and intermediate results
    """
    returns = np.asarray(returns, dtype=float)
    names = list(names)
    n_dates = returns.shape[0]
    w = _check_weights(weights, n_dates)

    processed, *_ = preprocess_returns(returns, w, standardise, remove_market_mode)
    corr, D, S = similarity_and_distance(processed, w, dist_method)

    if network_filter == "MST":
        adj = mst(D)
    elif network_filter == "PMFG":
        adj = pmfg(S)
    else:
        raise ValueError("network_filter must be 'MST' or 'PMFG'")

    mask = adj != 0
    out = {
        "S": S, "D": D, "filterName": network_filter,
        "filteredNetwork": adj,
        "filteredNetwork_D": np.where(mask, D, 0.0),
        "filteredNetwork_S": np.where(mask, S, 0.0),
        "g": adjacency_to_graph(adj, names),
    }
    out["metrics"] = network_metrics(out["filteredNetwork_D"], "Distance")
    out["inputs"] = {
        "inReturns": returns, "processedRets": processed, "corrMat": corr,
        "inNames": names, "inTimeWts": w, "standardise": standardise,
        "remove_market_mode": remove_market_mode, "network_filter": network_filter,
        "dist_method": dist_method,
    }

    if plot:
        plotting.plot_network(out["g"], network_filter,
                              _subtitle(dist_method, w, standardise, remove_market_mode))
    return out


def get_filtered_network_dynamic(returns, dates, weights, window, step, names,
                                 network_filter="MST", standardise=True,
                                 remove_market_mode=True,
                                 dist_method="QIS_correlDistMetric", plot=False,
                                 progress=True):
    """Run :func:`get_filtered_network` on rolling windows.

    Returns
    -------
    dict with ``dynamicNetwork`` holding, for each window: ``dates``,
    ``HybridCentrality`` (p × T), ``nTL`` (normalised tree length), ``aveS``
    (average off-diagonal similarity), ``correlDist`` (20-bin histogram of
    correlations), ``Data`` and ``DataFinal`` (raw and processed returns).
    """
    returns = np.asarray(returns, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    n_dates, p = returns.shape
    w = _check_weights(weights, window)
    ends = rolling_end_indices(n_dates, window, step)
    T = len(ends)
    edges = np.round(np.arange(-1, 1.0001, 0.1), 10)
    tril = np.tril(np.ones((p, p), dtype=bool), -1)

    res = {"dates": [], "HybridCentrality": np.full((p, T), np.nan),
           "nTL": np.full(T, np.nan), "aveS": np.full(T, np.nan),
           "correlDist": np.full((len(edges) - 1, T), np.nan),
           "Data": [], "DataFinal": []}
    for t, end in enumerate(_progress(ends, progress, "Running dynamic network filter")):
        tmp = get_filtered_network(returns[end - window:end], w, names, network_filter,
                                   standardise, remove_market_mode, dist_method, False)
        res["Data"].append(tmp["inputs"]["inReturns"])
        res["DataFinal"].append(tmp["inputs"]["processedRets"])
        res["dates"].append(dates[end - 1])
        res["HybridCentrality"][:, t] = tmp["metrics"]["node"]["hybrid"]["XpY"]
        res["nTL"][t] = tmp["metrics"]["tree"]["treeLenNorm"]
        res["aveS"][t] = tmp["S"][tril].mean()
        c = tmp["inputs"]["corrMat"][tril]
        res["correlDist"][:, t] = np.histogram(c, bins=edges)[0] / c.size

    res["dates"] = pd.DatetimeIndex(res["dates"])
    if plot:
        plotting.plot_time_series(res["dates"], res["nTL"], f"Dynamic {network_filter}",
                                  _subtitle(dist_method, w, standardise, remove_market_mode,
                                            "Distance"), "Normalised tree length")
    return {"dynamicNetwork": res,
            "inputs": {"inReturns": returns, "inNames": list(names), "inTimeWts": w,
                       "window": window, "step": step, "standardise": standardise,
                       "remove_market_mode": remove_market_mode,
                       "network_filter": network_filter, "dist_method": dist_method}}


def _progress(seq, enabled, desc):
    if not enabled:
        return seq
    try:
        from tqdm import tqdm
        return tqdm(seq, desc=desc)
    except ImportError:
        return seq
