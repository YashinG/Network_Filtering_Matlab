"""Bootstrap reliability of filtered-network edges (scr_RunBootstrapNetwork.m)."""
import _common  # noqa: F401
import matplotlib.pyplot as plt

import netfilter as nf

data = nf.load_data()

inputs = dict(
    standardise=True,
    remove_market_mode=True,
    dist_method="QIS_correlDistMetric",      # {'QIS_correlDistMetric', 'correlDistMetric'}
    network_filter="PMFG",                   # {'MST', 'PMFG'}
    plot=True,
    n_sim=10000,                             # the PMFG takes a few minutes at 10,000
    seed=1,
)

out = nf.bootstrap_network(data["rets"], data["names"], inputs["standardise"],
                           inputs["remove_market_mode"], inputs["network_filter"],
                           inputs["dist_method"], inputs["n_sim"], inputs["plot"], inputs["seed"])

print("Least reliable edges:")
names = data["names"]
for (i, j), r in sorted(zip(map(tuple, out["edges"]), out["edgeReliability"]), key=lambda t: t[1])[:10]:
    print(f"  {names[i]:9s} - {names[j]:9s} {r:.3f}")
plt.show()
