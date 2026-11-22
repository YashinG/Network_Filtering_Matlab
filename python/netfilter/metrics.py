"""Network topology metrics. Port of networkMetrics.m.

Centrality definitions follow MATLAB's ``centrality`` function so results are
comparable with the MATLAB version:

* closeness  = (A / (N - 1))**2 / C, where A = number of reachable nodes and
  C = sum of distances to them (1 / C for a connected graph)
* betweenness is unnormalised, counting each unordered pair once
* eigenvector centrality is scaled to sum to one
* PageRank uses MATLAB's power iteration (damping 0.85, tolerance 1e-4)
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import rankdata


def _tiedrank_desc(x):
    """Average ranks of -x (rank 1 = largest), treating values equal to 12
    significant figures as ties so floating-point noise cannot break a tie."""
    x = np.asarray(x, dtype=float)
    scale = np.max(np.abs(x)) or 1.0
    return rankdata(-np.round(x / scale, 12), method="average")


def _closeness(dist):
    N = dist.shape[0]
    out = np.zeros(N)
    for i in range(N):
        d = dist[i]
        reach = np.isfinite(d)
        reach[i] = False
        A, C = reach.sum(), d[reach].sum()
        out[i] = 0.0 if C == 0 else (A / (N - 1)) ** 2 / C
    return out


def _eigenvector(adj):
    vals, vecs = np.linalg.eigh(adj)
    v = np.abs(vecs[:, np.argmax(vals)])
    return v / v.sum()


def _pagerank(adj, weighted, damping=0.85, tol=1e-4, max_iter=100):
    """PageRank by power iteration, using MATLAB's algorithm and defaults
    (uniform start, stop when the largest change is <= 1e-4)."""
    A = adj if weighted else (adj != 0).astype(float)
    n = A.shape[0]
    out_deg = A.sum(axis=0)
    dangling = out_deg == 0
    P = np.divide(A, out_deg, out=np.zeros_like(A), where=~dangling)
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        x_new = damping * (P @ x + x[dangling].sum() / n) + (1 - damping) / n
        done = np.max(np.abs(x_new - x)) <= tol
        x = x_new
        if done:
            break
    return x


def _betweenness(adj_dist, weighted):
    G = nx.from_numpy_array(adj_dist)
    b = nx.betweenness_centrality(G, weight="weight" if weighted else None, normalized=False)
    return np.array([b[i] for i in range(adj_dist.shape[0])])


def network_metrics(adj, weight_type="Distance"):
    """Weighted and unweighted network metrics plus the Pozzi et al. (2013) hybrid score.

    Parameters
    ----------
    adj : (N, N) weighted adjacency matrix (zeros = no edge)
    weight_type : ``'Distance'`` if the weights are distances, otherwise the
        weights are treated as correlations.

    Returns
    -------
    dict with keys ``info``, ``tree``, ``node`` (``uw``, ``wtd``, ``hybrid``).
    """
    adj = np.asarray(adj, dtype=float)
    N = adj.shape[0]
    iu, ju = np.nonzero(np.triu(adj, 1))
    wts = adj[iu, ju]
    if weight_type == "Distance":
        d_edge = wts
        s_edge = 2.0 - 0.5 * wts**2
    else:
        s_edge = 1.0 + wts
        d_edge = np.sqrt(2.0 * (1.0 - wts))

    A_D = np.zeros((N, N)); A_D[iu, ju] = d_edge; A_D += A_D.T
    A_S = np.zeros((N, N)); A_S[iu, ju] = s_edge; A_S += A_S.T
    A_U = (A_D != 0).astype(float)
    n_edges = len(iu)

    sp_uw = shortest_path(A_U, unweighted=True, directed=False)
    sp_w = shortest_path(A_D, method="D", directed=False)

    out = {
        "info": {"edges": np.column_stack([iu, ju]), "weight": wts, "D": d_edge, "S": s_edge},
        "tree": {"treeLen": d_edge.sum(), "treeLenNorm": d_edge.sum() / (n_edges - 1)},
    }

    off = ~np.tril(np.ones((N, N), dtype=bool))
    node = {}
    for key, weighted, sp in (("uw", False, sp_uw), ("wtd", True, sp_w)):
        m = {}
        if weighted:
            m["degree"] = A_S.sum(axis=1)
            m["degreeCentrality"] = m["degree"] / n_edges
            m["closeCentrality"] = _closeness(sp_w)
            m["betwCentrality"] = _betweenness(A_D, True)
            m["pagerankCentrality"] = _pagerank(A_S, True)
            m["EV_Centrality"] = _eigenvector(A_S)
        else:
            m["degree"] = A_U.sum(axis=1)
            m["degreeCentrality"] = m["degree"] / (n_edges - 1)
            m["closeCentrality"] = _closeness(sp_uw)
            m["betwCentrality"] = _betweenness(A_D, False)
            m["pagerankCentrality"] = _pagerank(A_S, False)
            m["EV_Centrality"] = _eigenvector(A_U)
        m["shortestPaths"] = sp
        m["eccentricity"] = 1.0 / sp.max(axis=0)
        m["ave"] = {k: float(np.mean(v)) for k, v in m.items() if k != "shortestPaths"}
        m["ave"]["shortestPaths"] = float(np.mean(sp[off]))
        m["ranks"] = {k: _tiedrank_desc(m[k])
                      for k in ("degree", "betwCentrality", "eccentricity",
                                "closeCentrality", "EV_Centrality")}
        node[key] = m

    uw, wt = node["uw"]["ranks"], node["wtd"]["ranks"]
    X = (uw["degree"] + wt["degree"] + uw["betwCentrality"] + wt["betwCentrality"] - 4) / (4 * (N - 1))
    Y = (uw["eccentricity"] + wt["eccentricity"] + uw["closeCentrality"] + wt["closeCentrality"]
         + uw["EV_Centrality"] + wt["EV_Centrality"] - 6) / (6 * (N - 1))
    node["hybrid"] = {"X": X, "Y": Y, "XpY": X + Y, "XmY": X - Y}
    out["node"] = node
    return out
