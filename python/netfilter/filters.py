"""Network filters: MST, PMFG and TMFG.

* ``mst``  – minimum spanning tree on a distance matrix.
* ``pmfg`` – Planar Maximally Filtered Graph (Tumminello et al., 2005), a port
  of ``External Functions/PMFG/pmfg.m``. The MatlabBGL Boyer–Myrvold
  planarity test is replaced by ``networkx.check_planarity``.
* ``tmfg`` – Triangulated Maximally Filtered Graph, a port of ``PMFG_T2s`` in
  ``DBHTs.m``.

All functions return a dense, symmetric, weighted adjacency matrix with zeros
for absent edges.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def mst(D):
    """Minimum spanning tree of distance matrix ``D`` (weights = distances)."""
    D = np.asarray(D, dtype=float)
    T = minimum_spanning_tree(D).toarray()
    return T + T.T


def pmfg(W, verbose=False):
    """Planar Maximally Filtered Graph of similarity matrix ``W``.

    Edges are added from strongest to weakest similarity, keeping each edge
    only if the graph stays planar, until ``3 (N - 2)`` edges are in place.
    Returns the adjacency matrix weighted by ``W``.
    """
    W = np.asarray(W, dtype=float)
    N = W.shape[0]
    if W.shape != (N, N):
        raise ValueError("W must be square")
    if not np.allclose(W, W.T):
        raise ValueError("W must be symmetric")
    W = 0.5 * (W + W.T)
    if N == 1:
        return np.ones((1, 1))

    # Edge list in MATLAB `find` (column-major) order, i < j, non-zero weights
    jj, ii = np.nonzero(np.triu(W, 1).T)
    w = W[ii, jj]
    keep = w != 0
    ii, jj, w = ii[keep], jj[keep], w[keep]
    order = np.argsort(-w, kind="stable")  # largest first; ties keep `find` order
    ii, jj, w = ii[order], jj[order], w[order]

    P = np.zeros((N, N))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    clu = np.arange(N)
    target = 3 * (N - 2)
    E = 0
    for k in range(len(w)):
        if E >= target:
            break
        a, b = ii[k], jj[k]
        if k < 6 or clu[a] != clu[b]:
            add = True  # the first 6 edges, or edges between components, are always planar
        else:
            G.add_edge(a, b)
            add = nx.check_planarity(G)[0]
            G.remove_edge(a, b)
        if add:
            G.add_edge(a, b)
            P[a, b] = P[b, a] = w[k]
            E += 1
            if clu[a] != clu[b]:
                m = clu.max() + 1
                clu[(clu == clu[a]) | (clu == clu[b])] = m
        if verbose and (k + 1) % 1000 == 0:
            print(f"Build PMFG: {k + 1}: {100 * E / target:.2f}% done")
    if E < target:
        raise RuntimeError("PMFG not found")
    return P


def tmfg(W):
    """Triangulated Maximally Filtered Graph of similarity matrix ``W``.

    Port of ``PMFG_T2s`` from the DBHT code (Song, Di Matteo & Aste). Returns
    the adjacency matrix weighted by ``W``.
    """
    W = np.asarray(W, dtype=float)
    N = W.shape[0]
    if N < 9:
        print("W Matrix too small")
    if np.any(W < 0):
        print("W Matrix has negative elements!")

    A = np.zeros((N, N))
    s = np.sum(W * (W > W.mean()), axis=1)
    j = np.argsort(-s, kind="stable")
    in_v = list(j[:4])
    ou_v = sorted(set(range(N)) - set(in_v))

    n_tri = 2 * N - 4
    tri = np.zeros((n_tri, 3), dtype=int)
    tri[0] = [in_v[0], in_v[1], in_v[2]]
    tri[1] = [in_v[1], in_v[2], in_v[3]]
    tri[2] = [in_v[0], in_v[1], in_v[3]]
    tri[3] = [in_v[0], in_v[2], in_v[3]]
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        A[in_v[a], in_v[b]] = 1

    gain = np.zeros((N, n_tri))
    ou = np.array(ou_v, dtype=int)
    for t in range(4):
        gain[ou, t] = W[np.ix_(ou, tri[t])].sum(axis=1)

    kk = 4
    for _ in range(4, N):
        ou = np.array(ou_v, dtype=int)
        if len(ou_v) == 1:
            vpos = 0
            tr = int(np.argmax(gain[ou_v[0], :]))
        else:
            G = gain[ou, :]
            v = np.argmax(G, axis=0)
            gij = G[v, np.arange(G.shape[1])]
            tr = int(np.argmax(gij))
            vpos = int(v[tr])
        ve = ou_v.pop(vpos)
        A[ve, tri[tr]] = 1
        old = tri[tr].copy()
        tri[kk] = [old[0], old[2], ve]
        tri[kk + 1] = [old[1], old[2], ve]
        tri[tr] = [old[0], old[1], ve]
        gain[ve, :] = 0
        ou = np.array(ou_v, dtype=int)
        if ou.size:
            for t in (tr, kk, kk + 1):
                gain[ou, t] = W[np.ix_(ou, tri[t])].sum(axis=1)
        kk += 2
    return W * ((A + A.T) == 1)
