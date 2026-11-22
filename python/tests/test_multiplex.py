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
