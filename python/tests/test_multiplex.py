"""Tests for netfilter.multiplex.

The template tests reproduce the results printed in the ngozzi/multiplex
notebook. They need a local clone of that repo. Set MULTIPLEX_TEMPLATE_DATA
to its ``data`` folder; otherwise they are skipped.
"""
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import netfilter as nf  # noqa: E402
from netfilter import multiplex as mx  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return nf.load_data()


@pytest.fixture(scope="module")
def built(data):
    return nf.build_multiplex(data["rets"], None, data["names"])


# ------------------------------------------------------------ correlations
def test_kendall_matches_scipy(data):
    x = data["rets"]
    ref = np.array([[kendalltau(x[:, i], x[:, j])[0] for j in range(20)] for i in range(20)])
    assert np.allclose(mx.kendall_tau_weighted(x), ref, atol=1e-12)
    xt = np.round(x, 2)  # tied values
    ref = np.array([[kendalltau(xt[:, i], xt[:, j])[0] for j in range(20)] for i in range(20)])
    assert np.allclose(mx.kendall_tau_weighted(xt, chunk=500), ref, atol=1e-12)


def test_weighted_kendall_reduces_to_subsample(data):
    # weights of 1 on the last 100 dates and 0 elsewhere = tau on those dates
    x = data["rets"]
    w = np.r_[np.zeros(99), np.ones(100)]
    ref = np.array([[kendalltau(x[99:, i], x[99:, j])[0] for j in range(20)] for i in range(20)])
    assert np.allclose(mx.kendall_tau_weighted(x, w), ref, atol=1e-12)


def test_spearman_matches_scipy(data):
    S = mx.dependence_matrix(data["rets"], None, "spearman", shrinkage=False)
    assert np.allclose(S, spearmanr(data["rets"])[0], atol=1e-12)


def test_pearson_layer_matches_network_filter(data, built):
    # the Pearson PMFG layer is the same network as get_filtered_network's PMFG
    o = nf.get_filtered_network(data["rets"], None, data["names"], "PMFG", True, True,
                                "QIS_correlDistMetric", False)
    assert {frozenset(e) for e in built["layers"]["pearson"].edges()} == \
           {frozenset(e) for e in o["g"].edges()}


# ------------------------------------------------------------ builder
@pytest.mark.parametrize("flt, n_edges", [("PMFG", 54), ("MST", 19)])
def test_build_multiplex(data, flt, n_edges):
    m = nf.build_multiplex(data["rets"], None, data["names"], network_filter=flt)
    for name, G in m["layers"].items():
        assert G.number_of_nodes() == 20 and G.number_of_edges() == n_edges
        assert nx.is_connected(G)
        if flt == "PMFG":
            assert nx.check_planarity(G)[0]
        u, v = next(iter(G.edges()))
        i, j = data["names"].index(u), data["names"].index(v)
        assert np.isclose(G[u][v]["weight"], m["corr"][name][i, j])


def test_build_multiplex_time_weighted(data):
    w, _ = nf.ewma_weights(199, 0.01)
    m = nf.build_multiplex(data["rets"], w, data["names"])
    assert m["inputs"]["time_weighted"]


# ------------------------------------------------------------ analysis
def test_analysis_on_built_multiplex(data, built):
    sectors = dict(zip(data["names"], data["super_sector_names"]))
    res = nf.multiplex_analysis(built["layers"], sectors)
    deg = res["degrees"]
    assert (deg["overlapping"] == deg[["pearson", "spearman", "kendall"]].sum(axis=1)).all()
    assert deg["overlapping"].sum() == 3 * 2 * 54
    assert ((res["MPC"] >= 0) & (res["MPC"] <= 1 + 1e-12)).all()
    assert np.allclose(res["similarity"], 1)  # every asset is in every PMFG layer
    eo = res["edge_overlap"]
    assert np.allclose(np.diag(eo["jaccard"]), 1)
    assert (eo["multiplicity"] * eo["multiplicity"].index).sum() == 3 * 54
    assert (res["sector_degrees"]["overlapping"] > 0).all()


def test_participation_coefficient_limits():
    import pandas as pd
    deg = pd.DataFrame({"a": [3, 2, 0], "b": [0, 2, 5], "c": [0, 2, 0]}, index=list("xyz"))
    deg["overlapping"] = deg.sum(axis=1)
    mpc = mx.participation_coefficient(deg)
    assert np.allclose(mpc.values, [0, 1, 0])


# ------------------------------------------------------------ other measures
def _dcor_naive(a, b):
    A = np.abs(a[:, None] - a[None, :]); B = np.abs(b[:, None] - b[None, :])
    A = A - A.mean(0) - A.mean(1)[:, None] + A.mean(); B = B - B.mean(0) - B.mean(1)[:, None] + B.mean()
    return np.sqrt((A * B).mean() / np.sqrt((A * A).mean() * (B * B).mean()))


def test_distance_correlation(data):
    x = data["rets"]
    D = mx.distance_correlation(x)
    for i, j in [(0, 1), (2, 7), (5, 19)]:
        assert D[i, j] == pytest.approx(_dcor_naive(x[:, i], x[:, j]), abs=1e-12)
    assert np.allclose(mx.distance_correlation(x, chunk=5000), D, atol=1e-14)
    w = np.r_[np.zeros(99), np.ones(100)]  # 0/1 weights = subsample
    assert mx.distance_correlation(x, w)[0, 1] == pytest.approx(_dcor_naive(x[99:, 0], x[99:, 1]), abs=1e-12)
    rng = np.random.default_rng(0)
    z = rng.standard_normal(400)
    y = np.c_[z, z ** 2, rng.standard_normal(400)]
    Dy = mx.distance_correlation(y)
    assert Dy[0, 1] > 0.4 and Dy[0, 2] < 0.15  # finds non-linear dependence


def test_partial_correlation(data):
    x = data["rets"]
    pc = mx.partial_correlation(x, None, shrinkage=False)
    others = list(range(2, 20))
    X = np.c_[np.ones(199), x[:, others]]
    res = [x[:, k] - X @ np.linalg.lstsq(X, x[:, k], rcond=None)[0] for k in (0, 1)]
    assert pc[0, 1] == pytest.approx(np.corrcoef(*res)[0, 1], abs=1e-12)


def test_tail_dependence(data):
    from scipy.stats import rankdata
    x = data["rets"]
    lam = mx.tail_dependence(x, None, q=0.1)
    I = np.apply_along_axis(rankdata, 0, x) / 200 <= 0.1
    assert lam[0, 1] == pytest.approx((I[:, 0] & I[:, 1]).sum() / I[:, 0].sum())
    up = mx.tail_dependence(x, None, q=0.1, tail="upper")
    assert np.allclose(up, mx.tail_dependence(-x, None, q=0.1, tail="lower"))
    assert lam.min() >= 0 and lam.max() <= 1


# ------------------------------------------------------------ user-defined layers
def test_layer_sources(data):
    R, N = data["rets"], data["names"]
    C = np.corrcoef(R.T)
    df = pd.DataFrame(C, index=N, columns=N).iloc[::-1, ::-1]   # different order
    func = lambda r, w: np.corrcoef(r.T)
    m = nf.build_multiplex(R, None, N, [
        nf.Layer("array", C, kind="correlation"),
        nf.Layer("frame", df, kind="correlation"),
        nf.Layer("func", func, kind="correlation", raw_returns=True),
    ])
    E = {k: {frozenset(e) for e in G.edges()} for k, G in m["layers"].items()}
    assert E["array"] == E["frame"] == E["func"]
    assert all(G.number_of_edges() == 54 for G in m["layers"].values())


def test_kinds_preserve_ordering(data):
    R, N = data["rets"], data["names"]
    C = np.corrcoef(R.T)
    m = nf.build_multiplex(R, None, N, [
        nf.Layer("corr", C, kind="correlation"),
        nf.Layer("sim", 5 * C + 3, kind="similarity"),
        nf.Layer("dist", np.sqrt(2 * (1 - C)), kind="distance"),
        nf.Layer("corr_mst", C, kind="correlation", network_filter="MST"),
        nf.Layer("dist_mst", 1 - C, kind="distance", network_filter="MST"),
    ])
    E = {k: {frozenset(e) for e in G.edges()} for k, G in m["layers"].items()}
    assert E["corr"] == E["sim"] == E["dist"]       # PMFG depends only on the order
    assert E["corr_mst"] == E["dist_mst"]


def test_graph_and_adjacency_layers(data):
    R, N = data["rets"], data["names"]
    G = nx.Graph([("Stock_1", "Stock_2"), ("Stock_2", "Stock_3"), ("Stock_1", "Outside")])
    A = np.zeros((20, 20)); A[0, 5] = 2.0                         # directed: one entry only
    m = nf.build_multiplex(R, None, N, ["pearson", nf.Layer("graph", G),
                                        nf.Layer("adj", A, kind="adjacency")])
    g = m["layers"]["graph"]
    assert g.number_of_nodes() == 20 and g.number_of_edges() == 2
    assert m["inputs"]["dropped_nodes"] == {"graph": ["Outside"]}
    assert m["layers"]["adj"].has_edge("Stock_1", "Stock_6")
    assert m["layers"]["adj"]["Stock_1"]["Stock_6"]["weight"] == 2.0
    res = nf.multiplex_analysis(m["layers"])
    assert res["similarity"].at["graph", "graph"] == pytest.approx(3 / 20)
    assert (res["cartography"]["role"] != "isolated").all()      # every asset is in the PMFG


def test_isolated_nodes():
    G1 = nx.Graph([("a", "b")]); G1.add_node("c")
    G2 = nx.Graph([("a", "b")]); G2.add_node("c")
    deg = mx.degree_table({"x": G1, "y": G2})
    assert np.isnan(mx.participation_coefficient(deg)["c"])
    assert mx.cartography(deg, overlap_threshold=1).at["c", "role"] == "isolated"


def test_layer_errors(data):
    R, N = data["rets"], data["names"]
    G = nx.Graph([("Stock_1", "Stock_2")])
    with pytest.raises(ValueError, match="set kind"):
        nf.build_multiplex(R, None, N, [nf.Layer("m", np.eye(20))])
    with pytest.raises(ValueError, match="used as it is"):
        nf.build_multiplex(R, None, N, [nf.Layer("g", G, network_filter="PMFG")])
    with pytest.raises(ValueError, match="unknown measure"):
        nf.build_multiplex(R, None, N, ["nope"])
    with pytest.raises(ValueError, match="expected a 20x20"):
        nf.build_multiplex(R, None, N, [nf.Layer("m", np.eye(3), kind="similarity")])
    with pytest.raises(ValueError, match="unique"):
        nf.build_multiplex(R, None, N, ["pearson", "pearson"])


@pytest.mark.parametrize("measure", list(mx.BUILTIN_MEASURES))
def test_every_builtin_builds(data, measure):
    w, _ = nf.ewma_weights(199, 0.01)
    m = nf.build_multiplex(data["rets"], w, data["names"], [measure])
    G = m["layers"][measure]
    assert G.number_of_edges() == 54 and nx.check_planarity(G)[0]


# ------------------------------------------------------------ template
TEMPLATE = Path(os.environ.get("MULTIPLEX_TEMPLATE_DATA", "/nonexistent"))
needs_template = pytest.mark.skipif(not (TEMPLATE / "edgeslist").exists(),
                                    reason="set MULTIPLEX_TEMPLATE_DATA to ngozzi/multiplex/data")


@pytest.fixture(scope="module")
def template_degrees():
    e = TEMPLATE / "edgeslist"
    return mx.degree_table(mx.read_layers({"pearson": e / "edgeslistP.txt",
                                           "kendall": e / "edgeslistK.txt",
                                           "spearman": e / "edgeslistS.txt"}))


@needs_template
def test_template_top_nodes(template_degrees):
    t = mx.top_nodes(template_degrees, tie_break="notebook")
    expected = {"pearson": [417, 438, 264, 402, 55, 88, 101, 109, 244, 348],
                "kendall": [264, 366, 402, 248, 58, 88, 438, 222, 428, 348],
                "spearman": [402, 264, 438, 88, 248, 366, 109, 348, 428, 222],
                "overlapping": [264, 438, 402, 417, 366, 88, 248, 58, 348, 109]}
    for k, v in expected.items():
        assert [int(n) for n in t[k]] == v


@needs_template
def test_template_similarity_and_cartography(template_degrees):
    Q = mx.layer_similarity(template_degrees)
    assert Q.at["pearson", "kendall"] == pytest.approx(0.8629737609329446)
    assert Q.at["pearson", "spearman"] == pytest.approx(0.8658892128279884)
    assert Q.at["kendall", "spearman"] == pytest.approx(0.9475218658892128)
    c = mx.cartography(template_degrees, 0.5, 60)
    assert c["role"].value_counts().to_dict() == {"multiplex leaf": 270, "multiplex hub": 56,
                                                   "focused leaf": 17}
