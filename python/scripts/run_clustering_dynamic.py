"""Hierarchical clustering on rolling windows (scr_RunClustering_Dynamic.m)."""
import _common  # noqa: F401
import matplotlib.pyplot as plt

import netfilter as nf

data = nf.load_data()

inputs = dict(
    standardise=True,
    remove_market_mode=True,
    ewma_alpha=0.0,                          # {0, 0.005}
    dist_method="QIS_correlDistMetric",      # {'QIS_correlDistMetric', 'correlDistMetric'}
    link_method="ward",                      # {'DBHT_PMFG', 'ward', 'single', 'complete', ...}
    max_clusters=15,
    n_clusters=6,
    plot=True,
    comp_partition=data["all_sector_num"][2],  # third classification level
    window=52 * 3,
    step=4,
)

time_wts, _ = nf.ewma_weights(inputs["window"], inputs["ewma_alpha"])
labels = [f"{n} ({s})" for n, s in zip(data["names"], data["super_sector_names"])]

out = nf.get_clusters_dynamic(data["rets"], data["dates"], time_wts, inputs["window"],
                              inputs["step"], labels, inputs["dist_method"], inputs["link_method"],
                              inputs["standardise"], inputs["remove_market_mode"],
                              inputs["max_clusters"], inputs["plot"], inputs["n_clusters"],
                              inputs["comp_partition"])
dyn = out["dynamicClusters"]
print("Clusters per window:", dyn["nClusters"].tolist())
print("ARI per window:", dyn["adjRandIndex"].round(3).tolist())
plt.show()
