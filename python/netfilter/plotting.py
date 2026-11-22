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
