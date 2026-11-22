"""Directed Bubble Hierarchical Tree (DBHT) clustering.

Line-by-line port of ``Functions/DBHTs.m`` (Song, Di Matteo & Aste, 2012,
"Hierarchical information clustering by means of topologically embedded
graphs", PLoS ONE 7(3): e31929), including the PMFG/TMFG option.

Indices are 0-based internally. Cluster labels in ``T8`` are 1..K as in
MATLAB. ``Z`` is returned both in MATLAB form (3 columns, 1-based) and as a
SciPy linkage matrix.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.sparse.csgraph import shortest_path

from .filters import pmfg, tmfg


# --------------------------------------------------------------------- helpers
def _reachable(adj, source):
    """Boolean mask of nodes reachable from ``source`` following adj[u, v] != 0."""
    adj = np.asarray(adj) != 0
    seen = np.zeros(adj.shape[0], dtype=bool)
    seen[source] = True
    q = deque([source])
    while q:
        u = q.popleft()
        for v in np.nonzero(adj[u])[0]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return seen


def _clique3(A):
    """All 3-cliques of adjacency ``A``, in the same order as the MATLAB code."""
    A = (np.asarray(A) != 0).copy()
    np.fill_diagonal(A, False)
    Ai = A.astype(np.int64)
    P = np.triu(((Ai @ Ai) != 0) & A)
    cols, rows = np.nonzero(P.T)  # MATLAB column-major `find`
    cliques, seen = [], set()
    for i, j in zip(rows, cols):
        for k in np.nonzero(A[i] & A[j])[0]:
            cand = tuple(sorted((int(i), int(j), int(k))))
            if cand not in seen:
                seen.add(cand)
                cliques.append(cand)
    cliques = np.array(cliques, dtype=int).reshape(-1, 3)
    return cliques[np.argsort(cliques[:, 0], kind="stable")]


def _find_disjoint(A, cliq):
    """Label vertices: 0 = in clique, 2 = connected to first other vertex, 1 = not."""
    N = A.shape[0]
    T = np.zeros(N, dtype=int)
    not_in = np.setdiff1d(np.arange(N), cliq)
    tmp = A.copy()
    tmp[cliq, :] = False
    tmp[:, cliq] = False
    reach = _reachable(tmp, not_in[0])
    T[~reach] = 1
    T[reach] = 2
    T[cliq] = 0
    return T


def _build_hierarchy(M):
    Nc = M.shape[1]
    pred = np.full(Nc, -1, dtype=int)
    for n in range(Nc):
        children = np.nonzero(M[:, n] == 1)[0]
        child_sum = M[children, :].sum(axis=0)
        parents = np.nonzero(child_sum == len(children))[0]
        parents = parents[parents != n]
        if parents.size:
            parent_sum = M[:, parents].sum(axis=0)
            a = np.nonzero(parent_sum == parent_sum.min())[0]
            if len(a) == 1:
                pred[n] = parents[a[0]]
            else:
                return None
    return pred


def _bubble_hierarchy(pred, Sb):
    Nc = len(pred)
    root = np.nonzero(pred == -1)[0]
    count = np.zeros(Nc, dtype=bool)
    count[root] = True
    cols = []
    if len(root) > 1:
        v = np.zeros(Nc); v[root] = 1; cols.append(v)
    while count.sum() < Nc:
        nxt = []
        for r in root:
            child = np.nonzero(pred == r)[0]
            v = np.zeros(Nc); v[np.r_[r, child]] = 1; cols.append(v)
            count[child] = True
            nxt.extend(int(c) for c in child if Sb[c] != 0)
        if not nxt and count.sum() < Nc:
            raise RuntimeError("DBHT: bubble hierarchy could not be completed")
        root = np.unique(nxt)
    Mb = np.column_stack(cols)
    Nb = Mb.shape[1]
    H = np.zeros((Nb, Nb))
    for n in range(Nb):
        idx = Mb[:, n] == 1
        if idx.sum() == 1:
            H[n, 0] = 1  # reproduces MATLAB behaviour for single-clique bubbles
        else:
            H[n, Mb[idx, :].sum(axis=0) >= 1] = 1
    H = H + H.T
    np.fill_diagonal(H, 0)
    return (H != 0).astype(float), Mb


def _cliq_hierarchy_tree(Rpm):
    A = np.asarray(Rpm) != 0
    N = A.shape[0]
    cliq_list = _clique3(A)
    Nc = len(cliq_list)
    M = np.zeros((N, Nc))
    Sb = np.zeros(Nc)
    for n in range(Nc):
        T = _find_disjoint(A, cliq_list[n])
        i1, i2, i0 = (np.nonzero(T == k)[0] for k in (1, 2, 0))
        idx_s = np.r_[i2, i0] if len(i1) > len(i2) else np.r_[i1, i0]
        Sb[n] = len(idx_s) - 3
        M[idx_s, n] = 1
    pred = _build_hierarchy(M)
    if pred is None:
        raise RuntimeError("DBHT: clique hierarchy is ambiguous")
    root = np.nonzero(pred == -1)[0]
    if len(root) > 1:  # 'uniqueroot'
        pred = np.r_[pred, -1]
        pred[root] = len(pred) - 1
    Hb, Mb = _bubble_hierarchy(pred, Sb)
    return Hb, Mb[:Nc, :], cliq_list


def _direct_hb(Rpm, Hb, Mb, Mv, cliq_list):
    Hb = Hb != 0
    cc, rr = np.nonzero(np.triu(Hb).T)
    Nb = Mv.shape[1]
    kb = Hb.sum(axis=0)
    Hc = np.zeros((Nb, Nb))
    for r, c in zip(rr, cc):
        shared = np.nonzero((Mb[:, r] != 0) & (Mb[:, c] != 0))[0]
        if shared.size == 0:
            raise RuntimeError("DBHT: adjacent bubbles share no clique")
        tmp = Hb.copy()
        tmp[r, c] = tmp[c, r] = False
        reach = _reachable(tmp, 0)
        vo = cliq_list[shared[0]]
        pair = np.array([r, c])
        bleft, bright = pair[reach[pair]], pair[~reach[pair]]
        vleft = np.setdiff1d(np.nonzero((Mv[:, reach] != 0).any(axis=1))[0], vo)
        vright = np.setdiff1d(np.nonzero((Mv[:, ~reach] != 0).any(axis=1))[0], vo)
        left = Rpm[np.ix_(vo, vleft)].sum()
        right = Rpm[np.ix_(vo, vright)].sum()
        if left > right:
            Hc[np.ix_(bright, bleft)] = left
        else:
            Hc[np.ix_(bleft, bright)] = right
    sep = (Hc.sum(axis=1) == 0).astype(float)
    sep[(Hc.sum(axis=0) == 0) & (kb > 1)] = 2
    return Hc, sep


def _bubble_cluster(Rpm, Dpm, Hb, Mb, Mv, cliq_list):
    Hc, sep = _direct_hb(Rpm, Hb, Mb, Mv, cliq_list)
    N = Rpm.shape[0]
    indx = np.nonzero(sep == 1)[0]  # converging bubbles
    if len(indx) <= 1:
        return np.zeros((N, 0)), np.ones(N, dtype=int)

    Adjv = np.zeros((N, len(indx)))
    for n, b in enumerate(indx):
        reach = _reachable(Hc.T, b)
        Adjv[(Mv[:, reach] != 0).any(axis=1), n] = 1

    Tc = np.zeros(N, dtype=int)
    Bubv = Mv[:, indx]
    rs = Bubv.sum(axis=1)
    cv, uv = np.nonzero(rs == 1)[0], np.nonzero(rs > 1)[0]
    Mdjv = np.zeros((N, len(indx)))
    Mdjv[cv, :] = Bubv[cv, :]
    all_cont = 3 * (Bubv.sum(axis=0) - 2)
    for v in uv:
        v_cont = (Rpm[:, v][:, None] * Bubv).sum(axis=0)
        Mdjv[v, int(np.argmax(v_cont / all_cont))] = 1
    v, ci = np.nonzero(Mdjv)
    Tc[v] = ci + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        Udjv = Dpm @ (Mdjv / (Mdjv != 0).sum(axis=0))
    Udjv[Adjv == 0] = np.inf
    Udjv = np.where(np.isnan(Udjv), np.inf, Udjv)
    unassigned = np.nonzero(Mdjv.sum(axis=1) == 0)[0]
    if unassigned.size:
        Tc[unassigned] = np.argmin(Udjv[unassigned, :], axis=1) + 1
    return Adjv, Tc


def _linkage_function(d, labelvec):
    lvec = np.unique(labelvec)
    best, pair = np.inf, None
    for r in range(len(lvec) - 1):
        vr = labelvec == lvec[r]
        for c in range(r + 1, len(lvec)):
            m = vr | (labelvec == lvec[c])
            dd = d[np.ix_(m, m)]
            val = dd[dd != 0].max()
            if pair is None or val < best:
                best, pair = val, (lvec[r], lvec[c])
    return pair, best


def _bubble_member(Rpm, Mv, Mc):
    Mvv = np.zeros_like(Mv)
    rs = Mc.sum(axis=1)
    single = rs == 1
    Mvv[single, :] = Mc[single, :]
    for x in np.nonzero(rs > 1)[0]:
        bub = np.nonzero(Mc[x, :] != 0)[0]
        vu_bub = (Rpm[:, x][:, None] * Mv[:, bub]).sum(axis=0)
        all_bub = np.diag(Mv[:, bub].T @ Rpm @ Mv[:, bub]) / 2
        Mvv[x, bub[int(np.argmax(vu_bub / all_bub))]] = 1
    return Mvv


def _dendro(Z, lv1, lv2, dist):
    changed = np.unique(lv1[lv1 != lv2])
    if len(changed) != 2:
        raise RuntimeError("DBHT: dendrogram construction failed")
    Z.append([changed[0], changed[1], dist])


def _hierarchy_construct(Rpm, Dpm, Tc, Mv):
    N = Dpm.shape[0]
    kvec = np.unique(Tc)
    lv1 = np.arange(1, N + 1)
    E = np.zeros((N, Tc.max()))
    E[np.arange(N), Tc - 1] = 1
    Z = []
    for k in kvec:
        Mc = E[:, k - 1][:, None] * Mv
        Mvv = _bubble_member(Rpm, Mv, Mc)
        bub = np.nonzero(Mvv.sum(axis=0) > 0)[0]
        nc = int((Tc == k).sum()) - 1
        for b in bub:  # complete linkage within each bubble
            V = np.nonzero(Mvv[:, b] != 0)[0]
            if len(V) > 1:
                dpm = Dpm[np.ix_(V, V)]
                lv = lv1[V].copy()
                lv2 = lv1.copy()
                for _ in range(len(V) - 1):
                    pair, _ = _linkage_function(dpm, lv)
                    lv[(lv == pair[0]) | (lv == pair[1])] = lv1.max() + 1
                    lv2[V] = lv
                    _dendro(Z, lv1, lv2, 1.0 / nc)
                    nc -= 1
                    lv1 = lv2.copy()
        V = np.nonzero(E[:, k - 1] != 0)[0]  # then merge the bubbles of the cluster
        dpm = Dpm[np.ix_(V, V)]
        lv = lv1[V].copy()
        lv2 = lv1.copy()
        for _ in range(len(bub) - 1):
            pair, _ = _linkage_function(dpm, lv)
            lv[(lv == pair[0]) | (lv == pair[1])] = lv1.max() + 1
            lv2[V] = lv
            _dendro(Z, lv1, lv2, 1.0 / nc)
            nc -= 1
            lv1 = lv2.copy()
    lv2 = lv1.copy()  # finally merge the clusters
    dcl = np.ones(N)
    for _ in range(len(kvec) - 1):
        pair, _ = _linkage_function(Dpm, lv1)
        m0, m1 = lv1 == pair[0], lv1 == pair[1]
        lv2[m0 | m1] = lv1.max() + 1
        dvu = np.unique(dcl[m0])[0] + np.unique(dcl[m1])[0]
        dcl[m0 | m1] = dvu
        _dendro(Z, lv1, lv2, dvu)
        lv1 = lv2.copy()
    if len(np.unique(lv2)) > 1:
        raise RuntimeError("DBHT: something went wrong in merging")
    return np.array(Z, dtype=float)


def matlab_to_scipy_linkage(Z):
    """Convert a 1-based, 3-column MATLAB linkage into a SciPy linkage matrix.

    DBHT builds merges cluster by cluster, so its rows are not in height
    order. Every merge is higher than the merges below it, so the rows are
    stably re-sorted by height (as SciPy expects) and relabelled.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0] + 1
    order = np.argsort(Z[:, 2], kind="stable")
    new_id = np.arange(2 * n - 1)
    new_id[n + order] = n + np.arange(n - 1)
    out = np.zeros((n - 1, 4))
    size = np.ones(2 * n - 1)
    for i, row in enumerate(order):
        a, b, h = Z[row]
        a, b = new_id[int(a) - 1], new_id[int(b) - 1]
        if a > b:
            a, b = b, a
        out[i] = [a, b, h, size[a] + size[b]]
        size[n + i] = size[a] + size[b]
    return out


def dbht(D, S, network_filter="PMFG"):
    """DBHT clustering of distance matrix ``D`` with similarity matrix ``S``.

    Parameters
    ----------
    D : (N, N) dissimilarity (e.g. ``sqrt(2 (1 - rho))``)
    S : (N, N) non-negative similarity (e.g. ``1 + rho``)
    network_filter : ``'PMFG'`` or ``'TMFG'``

    Returns
    -------
    dict with
      ``T8``   – (N,) cluster labels 1..K
      ``Rpm``  – (N, N) filtered-graph adjacency weighted by S
      ``Adjv`` – bubble cluster membership matrix
      ``Dpm``  – (N, N) shortest-path lengths on the filtered graph
      ``Mv``   – (N, Nb) bubble membership matrix
      ``Z_matlab`` – MATLAB-style linkage (1-based, 3 columns)
      ``Z``    – SciPy linkage matrix
    """
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)
    if network_filter == "PMFG":
        P = pmfg(S)
        Rpm = np.where(P != 0, S, 0.0)
    elif network_filter == "TMFG":
        Rpm = tmfg(S)
    else:
        raise ValueError("network_filter must be 'PMFG' or 'TMFG'")
    Apm = np.where(Rpm != 0, D, 0.0)
    Dpm = shortest_path(Apm, method="D", directed=False)

    Hb, Mb, cliq_list = _cliq_hierarchy_tree(Rpm)
    N = Rpm.shape[0]
    Mv = np.zeros((N, Mb.shape[1]))
    for n in range(Mb.shape[1]):
        Mv[np.unique(cliq_list[Mb[:, n] != 0].ravel()), n] = 1
    Adjv, T8 = _bubble_cluster(Rpm, Dpm, Hb, Mb, Mv, cliq_list)
    Zm = _hierarchy_construct(Rpm, Dpm, T8, Mv)
    return {"T8": T8, "Rpm": Rpm, "Adjv": Adjv, "Dpm": Dpm, "Mv": Mv,
            "Z_matlab": Zm, "Z": matlab_to_scipy_linkage(Zm), "Method": network_filter}
