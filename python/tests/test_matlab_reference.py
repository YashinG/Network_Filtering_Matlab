"""Regression tests against output from the original MATLAB code (R2025b) on Data.mat.

The reference values in tests/data/ were produced by running the MATLAB
functions in ../Functions on the bundled data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import netfilter as nf  # noqa: E402

HERE = Path(__file__).resolve().parent
M = json.loads((HERE / "data" / "matlab_static.json").read_text())
MR = json.loads((HERE / "data" / "matlab_rolling.json").read_text())
TOL = 1e-10


@pytest.fixture(scope="module")
def data():
    return nf.load_data()


def edges(A):
    i, j = np.nonzero(np.triu(A))
    return set(zip((i + 1).tolist(), (j + 1).tolist()))


def test_estimation(data):
    w, ne = nf.ewma_weights(156, 0.005)
    assert np.allclose(w[[0, 1, 155]], M["ewma"], atol=TOL) and np.isclose(ne, M["ewmaNeff"])
    w199, _ = nf.ewma_weights(199, 0.01)
    pr, _, chk, pe = nf.preprocess_returns(data["rets"], w199, True, True)
    assert np.allclose(pr[:3, :4], M["pre"], atol=TOL)
    assert chk == M["preChk"] and np.isclose(pe, M["prePerc"])
    Q = nf.qis_weighted(pr, w199)
    assert np.allclose(np.diag(Q), M["qisDiag"], atol=TOL) and np.allclose(Q[0], M["qisRow1"], atol=TOL)


def test_pmfg_and_metrics(data):
    o = nf.get_filtered_network(data["rets"], None, data["names"], "PMFG", True, True,
                                "QIS_correlDistMetric", False)
    c = o["inputs"]["corrMat"]
    assert np.allclose(c[0], M["corrRow1"], atol=TOL) and np.allclose(c[6], M["corrRow7"], atol=TOL)
    assert edges(o["filteredNetwork"]) == {tuple(e) for e in M["pmfgEdges"]}
    m = o["metrics"]
    assert np.isclose(m["tree"]["treeLenNorm"], M["pmfgTL"])
    w = m["node"]["wtd"]
    for key, ref in [("closeCentrality", "pmfgClose"), ("betwCentrality", "pmfgBetw"),
                     ("pagerankCentrality", "pmfgPR"), ("EV_Centrality", "pmfgEV")]:
        assert np.allclose(w[key], M[ref], atol=TOL), key
    assert np.allclose(m["node"]["hybrid"]["XpY"], M["pmfgXpY"], atol=TOL)


def test_mst(data):
    o = nf.get_filtered_network(data["rets"], None, data["names"], "MST", True, True,
                                "QIS_correlDistMetric", False)
    assert edges(o["filteredNetwork"]) == {tuple(e) for e in M["mstEdges"]}
    assert np.isclose(o["metrics"]["tree"]["treeLenNorm"], M["mstTL"])
    # Stocks 9 and 11 are exactly tied on weighted closeness; MATLAB breaks the
    # tie on floating-point noise (1e-18), the port treats it as a tie.
    xpy, ref = o["metrics"]["node"]["hybrid"]["XpY"], np.array(M["mstXpY"])
    keep = np.ones(20, dtype=bool); keep[[8, 10]] = False
    assert np.allclose(xpy[keep], ref[keep], atol=TOL)
    assert np.isclose(xpy[[8, 10]].sum(), ref[[8, 10]].sum())


@pytest.mark.parametrize("flt", ["PMFG", "TMFG"])
def test_dbht(data, flt):
    cl = nf.get_clusters(data["rets"], None, data["names"], "QIS_correlDistMetric", "DBHT_" + flt,
                         True, True, 15, False, 6, None)
    assert cl["stOutDBHT"]["T8"].tolist() == M["T_" + flt]
    assert np.allclose(cl["stOutDBHT"]["Z_matlab"], M["Z_" + flt])


@pytest.mark.parametrize("link", ["ward", "average"])
def test_standard_linkages(data, link):
    cl = nf.get_clusters(data["rets"], None, data["names"], "QIS_correlDistMetric", link,
                         True, True, 15, False, 6, None)
    assert nf.adjusted_rand_index(cl["clusterIDs"], M["T_" + link]) == pytest.approx(1)
    assert np.allclose(np.sort(cl["Z"][:, 2]), M["h_" + link], atol=TOL)


def test_correlation_analysis(data):
    ca = nf.correlation_analysis(data["rets"], None, data["names"], True, True, "QIS", None, False)
    assert np.allclose(ca["corrDistSummary"], M["caSum"], atol=TOL)
    assert np.isclose(ca["corrAve"], M["caAve"])


def test_rolling(data):
    R, nm, dt = data["rets"], data["names"], data["dates"]
    w, _ = nf.ewma_weights(156, 0.005)
    o = nf.get_filtered_network_dynamic(R, dt, w, 156, 4, nm, "PMFG", True, True,
                                        "QIS_correlDistMetric", False, False)["dynamicNetwork"]
    assert np.allclose(o["nTL"], MR["nTL"], atol=TOL) and np.allclose(o["aveS"], MR["aveS"], atol=TOL)
    assert np.allclose(o["HybridCentrality"][:, -1], MR["hc"], atol=TOL)
    assert np.allclose(o["correlDist"][:, -1], MR["cd"], atol=TOL)
    c = nf.get_clusters_dynamic(R, dt, w, 156, 4, nm, "QIS_correlDistMetric", "DBHT_PMFG", True, True,
                                15, False, None, data["all_sector_num"][2], False)["dynamicClusters"]
    assert c["nClusters"].tolist() == MR["nCl"]
    assert np.allclose(c["adjRandIndex"][0], MR["ari"], atol=TOL)
    ref = np.array(MR["cl"])
    for t in range(ref.shape[1]):
        a, b = c["clusterIDs"][:, t], ref[:, t]
        same = len(np.unique(a)) == len(np.unique(b)) == 1 or nf.adjusted_rand_index(a, b) == pytest.approx(1)
        assert same, t
    ca = nf.correlation_analysis_dynamic(R, dt, w, 156, 4, nm, True, False, "Normal", False,
                                         False)["dynamicCorrelations"]
    assert np.allclose(ca["corrAve"], MR["cAve"], atol=TOL)
    assert np.allclose(ca["corrDistSummary"][:, -1], MR["cSum"], atol=TOL)
    assert np.allclose(ca["corrDistFi"][:, -1], MR["cFi"], atol=TOL)
