"""Matplotlib plots used by the analysis functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def _year_ticks(ax, dates):
    d = pd.to_datetime(pd.Series(dates))
    idx = np.nonzero(np.diff(d.dt.year.values))[0]
    ax.set_xticks(idx)
    ax.set_xticklabels([d.iloc[i].strftime("%b%y") for i in idx], rotation=90)
    ax.set_xlim(0, len(d) - 1)


def _titles(ax, title, subtitle):
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)


def plot_network(G, title, subtitle, seed=0):
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx(G, pos, ax=ax, node_size=120, font_size=7, width=0.8)
    _titles(ax, title, subtitle)
    ax.axis("off")
    return fig


def plot_network_reliability(G, edges, reliability, title, subtitle, seed=0):
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=120)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ec = nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, edge_color=reliability,
                                edge_cmap=plt.cm.jet, edge_vmin=0, edge_vmax=1,
                                width=3 * np.asarray(reliability) + 0.1)
    fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(0, 1)), ax=ax,
                 label="Edge reliability")
    _titles(ax, title, subtitle)
    ax.axis("off")
    return fig


def plot_dendrogram(Z, labels, threshold, title, subtitle):
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(labels))))
    dendrogram(Z, labels=labels, orientation="right", color_threshold=threshold, ax=ax)
    ax.set_xlabel("Distance")
    _titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_time_series(dates, y, title, subtitle, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(len(y)), y)
    _year_ticks(ax, dates)
    ax.set_ylabel(ylabel)
    _titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_dynamic_clusters(dates, n_clusters, ari, link_method, subtitle):
    rows = 2 if ari is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(9, 4 * rows), squeeze=False)
    ax = axes[0, 0]
    ax.step(np.arange(len(n_clusters)), n_clusters, "-o", where="post", markersize=3)
    _year_ticks(ax, dates)
    ax.set_ylim(0, max(n_clusters) + 1)
    ax.grid(axis="y")
    ax.set_ylabel("nClusters")
    _titles(ax, f"Dynamic Number of Clusters: {link_method}", subtitle)
    if ari is not None:
        ax = axes[1, 0]
        ax.plot(np.asarray(ari).T)
        _year_ticks(ax, dates)
        ax.grid(axis="y")
        ax.set_ylabel("ARI")
        _titles(ax, f"ARI: {link_method}", subtitle)
    fig.tight_layout()
    return fig


def plot_density(xi, fi, ave, title, subtitle):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xi, fi)
    ax.axvline(ave, color="r", linestyle="--")
    ax.text(ave, ax.get_ylim()[1] * 0.95, f" Average = {ave:.2f}", color="r")
    _titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_density_contour(dates, xi, fi, title, subtitle):
    fig, ax = plt.subplots(figsize=(9, 4))
    cs = ax.contourf(np.arange(fi.shape[1]), xi, fi, cmap="viridis")
    cs.cmap.set_under("white")
    fig.colorbar(cs, ax=ax)
    _year_ticks(ax, dates)
    _titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_histogram(x, title, subtitle, xlabel):
    fig, ax = plt.subplots(figsize=(7, 4))
    vals, counts = np.unique(x, return_counts=True)
    ax.bar(vals, counts / counts.sum())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    _titles(ax, title, subtitle)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- multiplex
_LAYER_COLOURS = {"pearson": "tab:blue", "kendall": "darkorange", "spearman": "darkgreen"}
_ROLE_COLOURS = {"focused leaf": "orange", "focused hub": "steelblue",
                 "multiplex leaf": "indianred", "multiplex hub": "forestgreen"}


def plot_multiplex_degree_distributions(layers):
    """Degree distribution of each layer, side by side."""
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), sharey=True, squeeze=False)
    for ax, (name, G) in zip(axes[0], layers.items()):
        counts = pd.Series(dict(G.degree())).value_counts().sort_index()
        ax.bar(counts.index, counts.values, width=0.8, color=_LAYER_COLOURS.get(name), label=name.title())
        ax.set_xlabel("degree")
        ax.grid(alpha=0.4)
        ax.legend()
    axes[0, 0].set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_overlapping_degree(degrees):
    """Degree in each layer (ordered by the first layer) and the sorted overlapping degree."""
    layers = [c for c in degrees.columns if c != "overlapping"]
    ordered = degrees.sort_values(layers[0], ascending=False, kind="stable")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(ordered))
    for layer in layers:
        ax1.plot(x, ordered[layer].values, label=layer, color=_LAYER_COLOURS.get(layer))
    ax2.plot(x, np.sort(degrees["overlapping"].values)[::-1], label="overlapping", color="orange")
    ax1.set_ylabel("degree"); ax2.set_ylabel("overlapping degree")
    ax1.set_title("Degree by layer"); ax2.set_title("Overlapping degree")
    ax1.legend(); ax2.legend()
    ax2.set_xticks([])
    fig.tight_layout()
    return fig


def plot_participation_coefficient(mpc):
    """Multiplex participation coefficients, sorted from highest to lowest."""
    vals = np.sort(np.asarray(mpc))[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(vals)) / len(vals), vals, marker=".")
    ax.set_title("Multiplex Participation Coefficient")
    ax.set_ylabel("MPC"); ax.set_xlabel("share of nodes")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    return fig


def plot_cartography(carto, labels=None):
    """MPC against overlapping degree, coloured by role.

    ``labels`` (mapping node -> text) annotates the hubs; by default hubs are
    labelled with their node name.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for role, colour in _ROLE_COLOURS.items():
        sub = carto[carto["role"] == role]
        plural = role.replace("leaf", "leaves").replace("hub", "hubs")
        ax.scatter(sub["MPC"], sub["overlapping"], s=50, marker="^", color=colour,
                   label=f"{plural} ({len(sub)})")
    for node, row in carto[carto["role"].str.endswith("hub")].iterrows():
        text = labels.get(node, node) if labels is not None else node
        ax.annotate(str(text), (row["MPC"], row["overlapping"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(carto.attrs.get("overlap_threshold", np.nan), linestyle="--", alpha=0.5)
    ax.axvline(carto.attrs.get("mpc_threshold", 0.5), linestyle="--", alpha=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, max(carto["overlapping"].max(), carto.attrs.get("overlap_threshold", 0)) * 1.08)
    ax.set_xlabel("MPC"); ax.set_ylabel("Overlapping degree")
    ax.set_title("Multiplex Cartography")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    return fig


def plot_edge_overlap(overlap):
    """Jaccard overlap of edge sets between layers, and how many layers each edge is in."""
    jac, mult = overlap["jaccard"], overlap["multiplicity"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    im = ax1.imshow(jac.values, vmin=0, vmax=1, cmap="Blues")
    ax1.set_xticks(range(len(jac))); ax1.set_xticklabels(jac.columns)
    ax1.set_yticks(range(len(jac))); ax1.set_yticklabels(jac.index)
    for i in range(len(jac)):
        for j in range(len(jac)):
            ax1.text(j, i, f"{jac.iat[i, j]:.2f}", ha="center", va="center",
                     color="white" if jac.iat[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.046)
    ax1.set_title("Edge overlap (Jaccard)")
    ax2.bar(mult.index.astype(str), mult.values, color="steelblue")
    ax2.set_xlabel("number of layers containing the edge"); ax2.set_ylabel("edges")
    ax2.set_title("Edge multiplicity")
    fig.tight_layout()
    return fig


def plot_sector_degrees(sector_deg):
    """Average degree per sector, one lollipop chart per layer and overlapping."""
    cols = list(sector_deg.columns)
    ncol = 2
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.5 * nrow + 1), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        s = sector_deg[col].sort_values()
        y = np.arange(1, len(s) + 1)
        ax.hlines(y, 0, s.values, color="skyblue", linewidth=5, zorder=1)
        ax.scatter(s.values, y, s=80, color="darkorange", zorder=2)
        ax.set_yticks(y); ax.set_yticklabels(s.index)
        ax.set_title(f"Average degree - {col.title()}")
        ax.margins(0.05)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
