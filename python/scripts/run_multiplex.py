"""Multiplex analysis of filtered networks.

Inspired by: The multiplex dependency structure of financial markets,
N. Musumeci, V. Nicosia, T. Aste, T. Di Matteo, V. Latora. Jun 15, 2016;
https://arxiv.org/abs/1606.04872v1

Follows the ngozzi/multiplex template. It builds one network per layer from the
same returns (by default Pearson, Spearman and Kendall PMFGs), and then reports:
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
import networkx as nx
import numpy as np

import netfilter as nf
from netfilter import Layer

data = nf.load_data()


# ---- Optional: your own layers -------------------------------------------
# A layer can be a built-in measure name, a function of the returns, a matrix,
# or a ready-made network. Built-ins: pearson, spearman, kendall, partial,
# distance_correlation, lower_tail, upper_tail.
def same_sector_network():
    """Example 'ready-made' layer: link stocks in the same super-sector."""
    G = nx.Graph()
    for a, sa in zip(data["names"], data["super_sector_names"]):
        for b, sb in zip(data["names"], data["super_sector_names"]):
            if a < b and sa == sb:
                G.add_edge(a, b)
    return G


def beta_similarity(rets, wts):
    """Example custom measure: stocks with similar market betas are similar."""
    mkt = rets @ np.full(rets.shape[1], 1 / rets.shape[1])
    betas = np.array([np.cov(rets[:, i], mkt, aweights=wts)[0, 1] / np.cov(mkt, aweights=wts)
                      for i in range(rets.shape[1])])
    return -np.abs(betas[:, None] - betas[None, :])


EXAMPLE_CUSTOM_LAYERS = [
    "pearson",
    Layer("tail", "lower_tail", params={"q": 0.1}),                    # built-in, with a parameter
    Layer("dcor", "distance_correlation", network_filter="MST"),        # own filter for this layer
    Layer("beta", beta_similarity, kind="similarity", raw_returns=True),  # your own function
    Layer("sector", same_sector_network()),                             # network used as it is
]
# --------------------------------------------------------------------------

inputs = dict(
    layers=("pearson", "spearman", "kendall"),  # or EXAMPLE_CUSTOM_LAYERS
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

mplex = nf.build_multiplex(data["rets"], time_wts, data["names"], inputs["layers"],
                           inputs["network_filter"], inputs["standardise"],
                           inputs["remove_market_mode"], inputs["shrinkage"])
res = nf.multiplex_analysis(mplex["layers"], sectors, inputs["top"], inputs["mpc_threshold"],
                            inputs["overlap_threshold"], inputs["plot"])

# ------------------------------------------------------------------ report
print(f"Multiplex: {len(data['names'])} assets, "
      f"layers {', '.join(mplex['layers'])}")
if mplex["inputs"]["dropped_nodes"]:
    print("Nodes dropped (not in the asset list):", mplex["inputs"]["dropped_nodes"])
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
