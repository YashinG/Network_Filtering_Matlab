"""Multiplex networks of assets: one network per layer.

Builds a multiplex in which every layer links the same assets. A layer can be
a filtered network (PMFG or MST) of any dependence measure (Pearson, Spearman,
Kendall, partial correlation, distance correlation, tail dependence, or your
own function or matrix), or a network you supply as it is (e.g. supply-chain
or ownership links). See :class:`Layer`. It then analyses the layers together with the measures
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

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .estimation import (_check_weights, cov_to_corr, cov_weighted, corr_to_dist,
                         is_equal_weighted, preprocess_returns, qis_weighted)
from .filters import mst, pmfg

MEASURES = ("pearson", "spearman", "kendall")
ROLES = ("focused leaf", "focused hub", "multiplex leaf", "multiplex hub", "isolated")
KINDS = ("correlation", "similarity", "distance", "adjacency")
_DEFAULT = "default"


# ------------------------------------------------------------ built-in measures
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


def partial_correlation(returns, weights=None, shrinkage=True):
    """Partial correlation of each pair of assets given all the others.

    ``rho_ij|rest = -P_ij / sqrt(P_ii P_jj)``, where ``P`` is the inverse of
    the (QIS-shrunk if ``shrinkage``) weighted covariance matrix.
    """
    x = np.asarray(returns, dtype=float)
    w = _check_weights(weights, x.shape[0])
    cov = qis_weighted(x, w) if shrinkage else cov_weighted(x, w)[0]
    P = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return 0.5 * (pc + pc.T)


def distance_correlation(returns, weights=None, chunk=2_000_000):
    """Distance correlation matrix (Székely, Rizzo & Bakirov, 2007), with time weights.

    Uses the V-statistic with each date weighted by ``w`` (equal weights give the
    standard sample distance correlation). Values are in [0, 1] and are 0 only
    under independence, so non-linear dependence is picked up too. Memory is
    limited by processing ``chunk`` (date, date, asset) entries at a time.
    """
    x = np.asarray(returns, dtype=float)
    n, p = x.shape
    w = _check_weights(weights, n)
    rows = max(1, chunk // (n * p))
    # pass 1: weighted row means of |x_s - x_t| per asset
    r = np.zeros((n, p))
    for a in range(0, n, rows):
        b = min(n, a + rows)
        r[a:b] = np.einsum("t,stp->sp", w, np.abs(x[a:b, None, :] - x[None, :, :]))
    g = w @ r
    # pass 2: sum_st w_s w_t A_st B_st with double-centred distances
    G = np.zeros((p, p))
    for a in range(0, n, rows):
        b = min(n, a + rows)
        A = np.abs(x[a:b, None, :] - x[None, :, :]) - r[a:b, None, :] - r[None, :, :] + g
        A *= np.sqrt(w[a:b, None, None] * w[None, :, None])
        A = A.reshape(-1, p)
        G += A.T @ A
    dv = np.sqrt(np.clip(np.diag(G), 0, None))   # sqrt of each distance variance
    den = np.outer(dv, dv)
    dcor = np.sqrt(np.clip(np.divide(G, den, out=np.zeros_like(G), where=den > 0), 0, None))
    np.fill_diagonal(dcor, 1.0)
    return 0.5 * (dcor + dcor.T)


def tail_dependence(returns, weights=None, q=0.1, tail="lower"):
    """Empirical tail dependence: how often two assets are in their tails together.

    Returns are turned into ranks ``u = rank / (n + 1)``. For the lower tail,

        lambda_ij = sum_t w_t [u_it <= q][u_jt <= q] / sqrt(m_i m_j),
        m_i = sum_t w_t [u_it <= q],

    i.e. the share of an asset's worst ``q`` days on which the other asset was
    also in its worst ``q`` days. ``tail='upper'`` uses the best days
    (ranked from the top, so both tails cover the same number of days).
    Values are in [0, 1].
    """
    x = np.asarray(returns, dtype=float)
    n = x.shape[0]
    w = _check_weights(weights, n)
    if tail not in ("lower", "upper"):
        raise ValueError("tail must be 'lower' or 'upper'")
    u = np.apply_along_axis(rankdata, 0, x if tail == "lower" else -x) / (n + 1)
    I = (u <= q).astype(float)  # upper tail = lower tail of -x, so both use the same days count
    joint = I.T @ (I * w[:, None])
    m = np.diag(joint)
    den = np.sqrt(np.outer(m, m))
    lam = np.divide(joint, den, out=np.zeros_like(joint), where=den > 0)
    np.fill_diagonal(lam, 1.0)
    return lam


def _bi_pearson(x, w, shrinkage, **_):
    return _pearson(x, w, shrinkage)


def _bi_spearman(x, w, shrinkage, **_):
    return _pearson(np.apply_along_axis(rankdata, 0, x), w, shrinkage)


def _bi_kendall(x, w, shrinkage, **kw):
    return kendall_tau_weighted(x, w, **kw)


def _bi_partial(x, w, shrinkage, **_):
    return partial_correlation(x, w, shrinkage)


def _bi_dcor(x, w, shrinkage, **kw):
    return distance_correlation(x, w, **kw)


def _bi_lower(x, w, shrinkage, q=0.1, **_):
    return tail_dependence(x, w, q, "lower")


def _bi_upper(x, w, shrinkage, q=0.1, **_):
    return tail_dependence(x, w, q, "upper")


#: Built-in layer measures: name -> (function, kind)
BUILTIN_MEASURES = {
    "pearson": (_bi_pearson, "correlation"),
    "spearman": (_bi_spearman, "correlation"),
    "kendall": (_bi_kendall, "correlation"),
    "partial": (_bi_partial, "correlation"),
    "distance_correlation": (_bi_dcor, "similarity"),
    "lower_tail": (_bi_lower, "similarity"),
    "upper_tail": (_bi_upper, "similarity"),
}


def dependence_matrix(processed_rets, weights=None, measure="pearson", shrinkage=True, **params):
    """Matrix for one built-in measure (see ``BUILTIN_MEASURES``).

    * ``'pearson'``  – weighted (QIS-shrunk if ``shrinkage``) Pearson correlation
    * ``'spearman'`` – the same estimator applied to each asset's ranks
    * ``'kendall'``  – weighted Kendall tau-b (:func:`kendall_tau_weighted`)
    * ``'partial'``  – partial correlation (:func:`partial_correlation`)
    * ``'distance_correlation'`` – :func:`distance_correlation`
    * ``'lower_tail'`` / ``'upper_tail'`` – :func:`tail_dependence` (``q=0.1``)
    """
    if measure not in BUILTIN_MEASURES:
        raise ValueError(f"Unknown measure '{measure}'. Built-ins: {list(BUILTIN_MEASURES)}")
    x = np.asarray(processed_rets, dtype=float)
    w = _check_weights(weights, x.shape[0])
    return BUILTIN_MEASURES[measure][0](x, w, shrinkage, **params)


# ------------------------------------------------------------ layer specs
@dataclass
class Layer:
    """One layer of a multiplex.

    Parameters
    ----------
    name : layer name
    source : where the layer comes from. One of:
        * a built-in measure name (``BUILTIN_MEASURES``), e.g. ``'kendall'``
        * a function ``f(returns, weights, **params) -> (p, p) matrix``, given the
          processed returns (or the raw returns if ``raw_returns``) and time weights
        * a (p, p) numpy array in the order of ``names``
        * a pandas DataFrame indexed and labelled by asset name
        * a networkx Graph whose nodes are asset names (used as it is)
    kind : how to read the matrix (needed for functions and matrices):
        * ``'correlation'`` – values in [-1, 1]; distance ``sqrt(2 (1 - x))``,
          similarity ``1 + x``
        * ``'similarity'``  – larger means more strongly linked
        * ``'distance'``    – smaller means more strongly linked
        * ``'adjacency'``   – already a network: non-zero entries are edges
    network_filter : ``'PMFG'``, ``'MST'``, ``None`` (no filtering) or
        ``'default'`` (use ``build_multiplex``'s filter; ``None`` for adjacency)
    raw_returns : give a function the raw rather than pre-processed returns
    params : extra keyword arguments for the function or built-in measure
        (e.g. ``{'q': 0.05}`` for tail dependence)
    """

    name: str
    source: Any
    kind: Optional[str] = None
    network_filter: Optional[str] = _DEFAULT
    raw_returns: bool = False
    params: dict = field(default_factory=dict)


def _as_layers(layers) -> list:
    """Normalise the ``layers`` argument to a list of :class:`Layer`."""
    if isinstance(layers, (str, Layer)):
        layers = [layers]
    if isinstance(layers, dict):
        layers = [v if isinstance(v, Layer) else Layer(k, v) for k, v in layers.items()]
    out = []
    for item in layers:
        if isinstance(item, Layer):
            out.append(item)
        elif isinstance(item, str):
            out.append(Layer(item, item))
        else:
            raise TypeError("Each layer must be a measure name or a Layer(...)")
    names = [l.name for l in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Layer names must be unique: {names}")
    return out


def _align_matrix(M, names, layer_name):
    if isinstance(M, pd.DataFrame):
        missing = [n for n in names if n not in M.index or n not in M.columns]
        if missing:
            raise ValueError(f"Layer '{layer_name}': matrix has no rows/columns for {missing[:5]}...")
        M = M.loc[names, names].to_numpy(dtype=float)
    M = np.asarray(M, dtype=float)
    if M.shape != (len(names), len(names)):
        raise ValueError(f"Layer '{layer_name}': expected a {len(names)}x{len(names)} matrix, got {M.shape}")
    if np.isnan(M[~np.eye(len(names), dtype=bool)]).any():
        raise ValueError(f"Layer '{layer_name}': matrix contains NaN")
    return M


def _symmetrise(M):
    """Keep the stronger direction of each pair (for directed inputs)."""
    if np.allclose(M, M.T, equal_nan=True):
        return 0.5 * (M + M.T)
    return np.where(np.abs(M) >= np.abs(M.T), M, M.T)


def _filter_inputs(M, kind):
    """Distance and similarity matrices that preserve the ordering implied by ``kind``.

    MST and PMFG only depend on the order of the values, so similarities and
    distances are shifted to be strictly positive off the diagonal.
    """
    off = ~np.eye(M.shape[0], dtype=bool)
    if kind == "correlation":
        D, S = corr_to_dist(np.clip(M, -1, 1)), 1.0 + M
    else:
        score = M if kind == "similarity" else -M
        lo, hi = score[off].min(), score[off].max()
        S, D = score - lo + 1.0, hi - score + 1.0
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(S, 0.0)
    return D, S


def _graph_layer(G, names, layer_name):
    if G.is_directed():
        G = G.to_undirected()
    known = set(names)
    dropped = [n for n in G.nodes if n not in known]
    H = nx.Graph()
    H.add_nodes_from(names)
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True)
                     if u in known and v in known and u != v)
    for u, v, d in H.edges(data=True):
        d.setdefault("weight", 1.0)
    return H, dropped


def build_multiplex(returns, weights, names, layers=MEASURES, network_filter="PMFG",
                    standardise=True, remove_market_mode=True, shrinkage=True):
    """Build a multiplex with one network per layer.

    Returns are pre-processed once (standardised and/or market mode removed).
    Each layer then gives a (p, p) matrix, which is filtered with the MST (on
    distances) or the PMFG (on similarities), or used as it is.

    Parameters
    ----------
    layers : built-in measure names, :class:`Layer` objects, or a mix; or a
        dict ``{name: source}``. Default: Pearson, Spearman and Kendall.
    network_filter : default filter for layers that don't set their own.

    Returns
    -------
    dict with
      ``layers``    – ``{name: networkx.Graph}`` with every asset as a node and
                      edge attributes ``weight`` (the layer's value) and, for
                      filtered layers, ``distance``
      ``matrix``    – ``{name: (p, p) matrix}`` (``None`` for Graph sources);
                      also available as ``corr``
      ``adjacency`` – ``{name: (p, p) 0/1 adjacency matrix}``
      ``specs``     – ``{name: Layer}`` with the kind and filter actually used
      ``inputs``    – inputs, processed returns, and any graph nodes dropped
                      because they are not in ``names``
    """
    returns = np.asarray(returns, dtype=float)
    names = list(names)
    p = len(names)
    w = _check_weights(weights, returns.shape[0])
    processed, *_ = preprocess_returns(returns, w, standardise, remove_market_mode)

    out = {"layers": {}, "matrix": {}, "adjacency": {}, "specs": {}}
    dropped = {}
    for spec in _as_layers(layers):
        src, kind, flt = spec.source, spec.kind, spec.network_filter
        M = None
        if isinstance(src, nx.Graph):
            if kind not in (None, "adjacency") or flt not in (None, _DEFAULT):
                raise ValueError(f"Layer '{spec.name}': a Graph is used as it is "
                                 "(kind='adjacency', network_filter=None)")
            kind, flt = "adjacency", None
            G, dropped[spec.name] = _graph_layer(src, names, spec.name)
        else:
            if isinstance(src, str):
                if src not in BUILTIN_MEASURES:
                    raise ValueError(f"Layer '{spec.name}': unknown measure '{src}'. "
                                     f"Built-ins: {list(BUILTIN_MEASURES)}")
                func, builtin_kind = BUILTIN_MEASURES[src]
                kind = kind or builtin_kind
                M = func(returns if spec.raw_returns else processed, w, shrinkage, **spec.params)
            elif callable(src):
                if kind is None:
                    raise ValueError(f"Layer '{spec.name}': set kind= for a custom function "
                                     f"(one of {KINDS})")
                M = src(returns if spec.raw_returns else processed, w, **spec.params)
            else:
                if kind is None:
                    raise ValueError(f"Layer '{spec.name}': set kind= for a matrix (one of {KINDS})")
                M = src
            if kind not in KINDS:
                raise ValueError(f"Layer '{spec.name}': kind must be one of {KINDS}")
            M = _symmetrise(_align_matrix(M, names, spec.name))
            if flt == _DEFAULT:
                flt = None if kind == "adjacency" else network_filter
            if flt is None:
                A = (M != 0) & ~np.eye(p, dtype=bool)
                D = None
            else:
                if kind == "adjacency":
                    raise ValueError(f"Layer '{spec.name}': an adjacency layer can't be filtered")
                D, S = _filter_inputs(M, kind)
                if flt == "PMFG":
                    A = pmfg(S) != 0
                elif flt == "MST":
                    A = mst(D) != 0
                else:
                    raise ValueError("network_filter must be 'PMFG', 'MST' or None")
            G = nx.Graph()
            G.add_nodes_from(names)
            iu, ju = np.nonzero(np.triu(A, 1))
            for i, j in zip(iu, ju):
                attrs = {"weight": float(M[i, j])}
                if D is not None:
                    attrs["distance"] = float(D[i, j])
                G.add_edge(names[i], names[j], **attrs)
        out["layers"][spec.name] = G
        out["matrix"][spec.name] = M
        out["adjacency"][spec.name] = nx.to_numpy_array(G, nodelist=names, weight=None).astype(int)
        out["specs"][spec.name] = replace(spec, kind=kind, network_filter=flt)
    out["corr"] = out["matrix"]
    out["inputs"] = {"inReturns": returns, "processedRets": processed, "inNames": names,
                     "inTimeWts": w, "layers": list(out["specs"]), "network_filter": network_filter,
                     "standardise": standardise, "remove_market_mode": remove_market_mode,
                     "shrinkage": shrinkage, "time_weighted": not is_equal_weighted(w),
                     "dropped_nodes": {k: v for k, v in dropped.items() if v}}
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
    spread evenly over the ``M`` layers. It is NaN for nodes with no links.
    """
    layers = _layer_cols(degrees)
    M = len(layers)
    if M < 2:
        raise ValueError("Need at least two layers")
    o = degrees["overlapping"].astype(float).replace(0, np.nan)
    return (M / (M - 1) * (1 - degrees[layers].pow(2).sum(axis=1) / o.pow(2))).rename("MPC")


def cartography(degrees: pd.DataFrame, mpc_threshold: float = 0.5, overlap_threshold=None):
    """Classify nodes as focused/multiplex (by MPC) and hub/leaf (by overlapping degree).

    ``overlap_threshold`` defaults to the mean plus two standard deviations of
    the overlapping degree. Nodes with no links in any layer are ``'isolated'``.
    That is the hub rule of Battiston et al. (2014),
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
    role[o == 0] = "isolated"
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
