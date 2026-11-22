"""Run MST or PMFG on rolling windows (scr_RunNetworkFilter_Dynamic.m)."""
import _common  # noqa: F401
import matplotlib.pyplot as plt

import netfilter as nf

data = nf.load_data()

inputs = dict(
    standardise=True,
    remove_market_mode=True,
    ewma_alpha=0.0,                          # {0, 0.005}
    dist_method="QIS_correlDistMetric",      # {'QIS_correlDistMetric', 'correlDistMetric'}
    network_filter="MST",                    # {'MST', 'PMFG'}
    plot=True,
    window=52 * 3,                           # 3 years of weekly data
    step=4,                                  # about one month
)

time_wts, _ = nf.ewma_weights(inputs["window"], inputs["ewma_alpha"])

out = nf.get_filtered_network_dynamic(data["rets"], data["dates"], time_wts, inputs["window"],
                                      inputs["step"], data["names"], inputs["network_filter"],
                                      inputs["standardise"], inputs["remove_market_mode"],
                                      inputs["dist_method"], inputs["plot"])
dyn = out["dynamicNetwork"]
print(f"{len(dyn['dates'])} windows from {dyn['dates'][0]:%Y-%m-%d} to {dyn['dates'][-1]:%Y-%m-%d}")
plt.show()
