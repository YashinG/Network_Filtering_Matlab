"""Property tests for the netfilter port. Run with ``pytest`` from the python/ folder."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, is_valid_linkage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import netfilter as nf  # noqa: E402
from netfilter.estimation import _check_weights  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return nf.load_data()


@pytest.fixture(scope="module")
def dist(data):
    proc, *_ = nf.preprocess_returns(data["rets"], None, True, True)
    return nf.similarity_and_distance(proc, None, "QIS_correlDistMetric")


def test_load_data(data):
    assert data["rets"].shape == (199, 20)
    assert len(data["dates"]) == 199 and len(data["names"]) == 20
    assert data["all_sector_num"].shape == (4, 20)


def test_ewma_weights():
    w, n_eff = nf.ewma_weights(100, 0)
    assert np.allclose(w, 0.01) and np.isclose(n_eff, 100)
    w, n_eff = nf.ewma_weights(100, 0.05)
    assert np.isclose(w.sum(), 1) and np.all(np.diff(w) > 0) and n_eff < 100


def test_cov_weighted_matches_numpy(data):
    cov, _ = nf.cov_weighted(data["rets"])
    assert np.allclose(cov, np.cov(data["rets"].T, bias=True))
    w, _ = nf.ewma_weights(199, 0.01)
    cov_w, _ = nf.cov_weighted(data["rets"], w)
    assert np.allclose(cov_w, np.cov(data["rets"].T, aweights=w, bias=True))


def test_qis_preserves_trace_and_is_pd(data):
    S = nf.qis_weighted(data["rets"])
    sample, _ = nf.cov_weighted(data["rets"])
    assert np.isclose(np.trace(S), np.trace(sample))
    assert np.all(np.linalg.eigvalsh(S) > 0)
    # shrinkage pulls the eigenvalues together
    ev_s, ev_q = np.linalg.eigvalsh(sample), np.linalg.eigvalsh(S)
    assert ev_q.max() / ev_q.min() < ev_s.max() / ev_s.min()


def test_qis_singular_case():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 50))
    S = nf.qis_weighted(x)
    assert np.all(np.linalg.eigvalsh(S) > 0)


@pytest.mark.parametrize("alpha", [0, 0.01])
def test_preprocess(data, alpha):
    w, _ = nf.ewma_weights(199, alpha)
    w = _check_weights(w, 199)
    final, mkt, check, perc = nf.preprocess_returns(data["rets"], w, True, True)
    assert np.allclose(w @ final, 0, atol=1e-12)
    assert np.allclose(np.sqrt(w @ final**2), 1)
    assert check >= 0.5 and 0 < perc < 100
    # residuals are (weighted) orthogonal to the market mode
    mk = mkt - w @ mkt
    assert np.allclose(w @ (final * mk[:, None]), 0, atol=1e-10)


def test_distance_similarity(dist):
    corr, D, S = dist
    assert np.allclose(np.diag(corr), 1) and np.allclose(np.diag(D), 0)
    assert np.allclose(S, 1 + corr)
    assert np.allclose(D, np.sqrt(2 * (1 - corr)))


def test_mst(dist):
    _, D, _ = dist
    T = nf.mst(D)
    G = nx.from_numpy_array(T)
    assert nx.is_tree(G)
    ref = nx.minimum_spanning_tree(nx.from_numpy_array(D))
    assert np.isclose(T.sum() / 2, ref.size(weight="weight"))


def test_pmfg_properties(dist):
    _, D, S = dist
    P = nf.pmfg(S)
    G = nx.from_numpy_array(P)
    N = S.shape[0]
    assert G.number_of_edges() == 3 * (N - 2)
    assert nx.check_planarity(G)[0]
    # the PMFG always contains the MST (Tumminello et al., 2005)
    mst_edges = {tuple(sorted(e)) for e in nx.from_numpy_array(nf.mst(D)).edges}
    assert mst_edges <= {tuple(sorted(e)) for e in G.edges}


def test_pmfg_example_100_stocks():
    from scipy.io import loadmat
    path = Path(__file__).resolve().parents[2] / "External Functions" / "PMFG" / "100stocks.mat"
    r = loadmat(path)["r"]
    W = np.asarray(r.toarray() if hasattr(r, "toarray") else r, dtype=float)  # correlation matrix
    W = (W + W.T) / 2
    G = nx.from_numpy_array(nf.pmfg(W))  # as in External Functions/PMFG/example.m
    assert G.number_of_edges() == 3 * (W.shape[0] - 2) and nx.check_planarity(G)[0]


def test_tmfg_properties(dist):
    _, _, S = dist
    G = nx.from_numpy_array(nf.tmfg(S))
    assert G.number_of_edges() == 3 * (S.shape[0] - 2)
    assert nx.check_planarity(G)[0]


@pytest.mark.parametrize("flt", ["PMFG", "TMFG"])
def test_dbht_consistency(dist, flt):
    _, D, S = dist
    out = nf.dbht(D, S, flt)
    Z = out["Z"]
    assert is_valid_linkage(Z)
    assert np.all(np.diff(Z[:, 2]) >= -1e-12)  # monotonic heights
    K = len(np.unique(out["T8"]))
    T = fcluster(Z, K, criterion="maxclust")
    assert nf.adjusted_rand_index(T, out["T8"]) == pytest.approx(1.0)


def test_dbht_random_inputs():
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = rng.standard_normal((300, 40)) + rng.standard_normal((300, 1)) * 0.3
        C = np.corrcoef(x.T)
        out = nf.dbht(nf.corr_to_dist(C), 1 + C, "PMFG")
        assert is_valid_linkage(out["Z"])


def test_network_metrics_match_networkx(data):
    out = nf.get_filtered_network(data["rets"], None, data["names"], "PMFG", True, True,
                                  "QIS_correlDistMetric", False)
    A = out["filteredNetwork_D"]
    G = nx.from_numpy_array(A)
    m = out["metrics"]["node"]
    N = A.shape[0]
    clo = nx.closeness_centrality(G, distance="weight")
    assert np.allclose(m["wtd"]["closeCentrality"], [clo[i] / (N - 1) for i in range(N)])
    deg = dict(G.degree())
    assert np.allclose(m["uw"]["degree"], [deg[i] for i in range(N)])
    ev = nx.eigenvector_centrality_numpy(nx.from_numpy_array((A != 0).astype(float)))
    ev = np.array([ev[i] for i in range(N)])
    assert np.allclose(m["uw"]["EV_Centrality"], ev / ev.sum())
    h = m["hybrid"]["XpY"]
    assert h.shape == (N,) and np.all((h >= 0) & (h <= 2))


def test_adjusted_rand_index():
    a = [1, 1, 2, 2, 3, 3]
    assert nf.adjusted_rand_index(a, [5, 5, 7, 7, 9, 9]) == pytest.approx(1.0)
    # value from Hubert & Arabie formula, checked by hand
    assert nf.adjusted_rand_index([1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 3, 3]) == pytest.approx(0.2424, abs=1e-4)


@pytest.mark.parametrize("link", ["ward", "single", "average", "complete", "DBHT_PMFG", "DBHT_TMFG"])
def test_get_clusters(data, link):
    out = nf.get_clusters(data["rets"], None, data["names"], "QIS_correlDistMetric", link, True, True,
                          10, False, 4, data["all_sector_num"])
    assert out["allClusterIDs"].shape == (20, 9)
    assert out["adjRandIndex"].shape == (4, 9)
    k = out["nClusters"]
    assert len(np.unique(out["clusterIDs"])) == k
    # ordered labels are a relabelling of the original ones
    assert nf.adjusted_rand_index(out["clusterIDs"], out["clusterIDsOrdered"]) == pytest.approx(1.0)


def test_rolling_windows(data):
    out = nf.get_filtered_network_dynamic(data["rets"], data["dates"], None, 156, 4, data["names"],
                                          "MST", True, True, "QIS_correlDistMetric", False, False)
    dyn = out["dynamicNetwork"]
    assert len(dyn["dates"]) == 11 and dyn["dates"][-1] == data["dates"][-1]
    assert np.allclose(dyn["correlDist"].sum(axis=0), 1)


def test_other_distance_metric(data):
    out = nf.get_filtered_network(data["rets"], None, data["names"], "MST", True, False, "euclidean", False)
    assert nx.is_tree(out["g"])
