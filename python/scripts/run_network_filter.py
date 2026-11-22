"""Run a simple MST or PMFG on the full sample and plot it (scr_RunNetworkFilter.m)."""
import _common  # noqa: F401
import matplotlib.pyplot as plt

import netfilter as nf

data = nf.load_data()

inputs = dict(
    standardise=True,
    remove_market_mode=True,
    ewma_alpha=0.0,                          # {0, 0.005}
    dist_method="QIS_correlDistMetric",      # {'QIS_correlDistMetric', 'correlDistMetric'}
    network_filter="PMFG",                   # {'MST', 'PMFG'}
    plot=True,
)

time_wts, _ = nf.ewma_weights(data["rets"].shape[0], inputs["ewma_alpha"])

out = nf.get_filtered_network(data["rets"], time_wts, data["names"], inputs["network_filter"],
                              inputs["standardise"], inputs["remove_market_mode"],
                              inputs["dist_method"], inputs["plot"])

print(f"{out['filterName']}: {out['g'].number_of_edges()} edges, "
      f"normalised tree length {out['metrics']['tree']['treeLenNorm']:.4f}")
print("Hybrid centrality (X + Y), most central first:")
xpy = out["metrics"]["node"]["hybrid"]["XpY"]
for name, v in sorted(zip(data["names"], xpy), key=lambda t: t[1]):
    print(f"  {name:10s} {v:.3f}")
plt.show()
