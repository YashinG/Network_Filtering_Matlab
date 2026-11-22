"""Multiplex analysis of Pearson / Spearman / Kendall filtered networks.

Follows the ngozzi/multiplex template: it builds one PMFG (or MST) per
dependence measure from the same returns and then reports:
 - the degree distribution of each layer
 - overlapping degree and the top nodes
 - the multiplex participation coefficient (MPC)
 - multiplex cartography (focused/multiplex hubs and leaves)
 - edge overlap and node-activity similarity between layers
 - average degree by sector

Tables and figures are saved to ``out_dir``.
"""
import _common  # noqa: F401
from pathlib import Path

import matplotlib.pyplot as plt

import netfilter as nf

data = nf.load_data()

inputs = dict(
    measures=("pearson", "spearman", "kendall"),
    network_filter="PMFG",                   # {'MST', 'PMFG'}
    standardise=True,
    remove_market_mode=True,
    shrinkage=True,                          # QIS for Pearson and Spearman
    ewma_alpha=0.0,                          # {0, 0.005}
    top=10,
    mpc_threshold=0.5,
    overlap_threshold=None,                  # None = mean + 2 sd of overlapping degree
    plot=True,
    out_dir=Path(__file__).resolve().parents[1] / "results" / "multiplex",
)

time_wts, _ = nf.ewma_weights(data["rets"].shape[0], inputs["ewma_alpha"])
sectors = dict(zip(data["names"], data["super_sector_names"]))

mplex = nf.build_multiplex(data["rets"], time_wts, data["names"], inputs["measures"],
                           inputs["network_filter"], inputs["standardise"],
                           inputs["remove_market_mode"], inputs["shrinkage"])
res = nf.multiplex_analysis(mplex["layers"], sectors, inputs["top"], inputs["mpc_threshold"],
                            inputs["overlap_threshold"], inputs["plot"])

# ------------------------------------------------------------------ report
print(f"{inputs['network_filter']} multiplex: {len(data['names'])} assets, "
      f"layers {', '.join(inputs['measures'])}")
print("\n### MOST CONNECTED NODES ###\n", res["top_nodes"].to_string())
carto = res["cartography"]
print(f"\n### CARTOGRAPHY ### (hub if overlapping degree >= {carto.attrs['overlap_threshold']:.1f}, "
      f"multiplex if MPC >= {carto.attrs['mpc_threshold']})")
print(carto["role"].value_counts().to_string())
print("\n### EDGE OVERLAP (Jaccard) ###\n", res["edge_overlap"]["jaccard"].round(3).to_string())
print("\nEdges by number of layers they appear in:\n", res["edge_overlap"]["multiplicity"].to_string())
print("\n### AVERAGE DEGREE BY SECTOR ###\n", res["sector_degrees"].round(2).to_string())

# ------------------------------------------------------------------ save
out = inputs["out_dir"]
(out / "figures").mkdir(parents=True, exist_ok=True)
res["degrees"].to_csv(out / "degrees.csv")
carto.to_csv(out / "cartography.csv")
res["edge_overlap"]["jaccard"].to_csv(out / "edge_overlap_jaccard.csv")
res["edge_overlap"]["edge_layers"].to_csv(out / "edge_layers.csv")
res["sector_degrees"].to_csv(out / "sector_degrees.csv")
nf.multiplex.write_layers(mplex["layers"], out / "layers")
for name, fig in res.get("figs", {}).items():
    fig.savefig(out / "figures" / f"{name}.png", dpi=120, bbox_inches="tight")
print(f"\nSaved tables and figures to {out}")
plt.show()
