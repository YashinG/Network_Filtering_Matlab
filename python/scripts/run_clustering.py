"""Hierarchical clustering: DBHT, Ward, single, ... (scr_RunClustering.m)."""
import _common  # noqa: F401
import matplotlib.pyplot as plt

import netfilter as nf

data = nf.load_data()

inputs = dict(
    standardise=True,
    remove_market_mode=True,
    ewma_alpha=0.0,                          # {0, 0.005}
    dist_method="QIS_correlDistMetric",      # {'QIS_correlDistMetric', 'correlDistMetric'}
    link_method="ward",                      # {'DBHT_PMFG', 'DBHT_TMFG', 'ward', 'single', 'complete',
                                             #  'average', 'centroid', 'weighted', 'median'}
    max_clusters=15,
    n_clusters=6,
    plot=True,
)

time_wts, _ = nf.ewma_weights(data["rets"].shape[0], inputs["ewma_alpha"])
labels = [f"{n} ({s})" for n, s in zip(data["names"], data["super_sector_names"])]

out = nf.get_clusters(data["rets"], time_wts, labels, inputs["dist_method"], inputs["link_method"],
                      inputs["standardise"], inputs["remove_market_mode"], inputs["max_clusters"],
                      inputs["plot"], inputs["n_clusters"], data["all_sector_num"])

print(out["tables"]["clusterIDsOrdered"])
print("\nAdjusted Rand index vs. classification levels:")
print(out["tables"]["ARI"].round(3))
plt.show()
