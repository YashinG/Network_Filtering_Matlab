"""Bootstrap distribution of the DBHT number of clusters (scr_RunBootstrapDBHT.m)."""
import _common  # noqa: F401
import matplotlib.pyplot as plt

import netfilter as nf

data = nf.load_data()

inputs = dict(
    standardise=True,
    remove_market_mode=True,
    dist_method="QIS_correlDistMetric",      # {'QIS_correlDistMetric', 'correlDistMetric'}
    link_method="DBHT_PMFG",                 # {'DBHT_PMFG', 'DBHT_TMFG'}
    max_clusters=20,
    plot=True,
    n_sim=10,
    seed=1,
)
labels = [f"{n} ({s})" for n, s in zip(data["names"], data["super_sector_names"])]

out = nf.bootstrap_dbht(data["rets"], labels, inputs["standardise"], inputs["remove_market_mode"],
                        inputs["link_method"], inputs["dist_method"], inputs["max_clusters"],
                        inputs["n_sim"], inputs["plot"], inputs["seed"])
print(f"Full sample: {out['FullPeriod']['nClusters']} clusters; "
      f"bootstrap mean {out['nClusters_ave']:.2f} (sd {out['nClusters_stdev']:.2f})")
plt.show()
