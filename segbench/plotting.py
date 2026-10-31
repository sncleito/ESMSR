from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import group_indices, group_msts

GROUP_COLOURS = ["#ed2124", "#0d8140", "#3953a4", "#a53f97", "#1fbcbd",
                 "#f28e2b", "#8c564b", "#7f7f7f", "#bcbd22", "#17becf"]
METRIC_STYLE = {
    "DP": ("#1f77b4", "-"), "KM": ("#ff7f0e", "-"), "CC": ("#2ca02c", "-"), "SI": ("#d62728", "-"),
    "CH": ("#9467bd", "--"), "CV": ("#8c564b", "--"), "CA": ("#e377c2", "--"), "EA": ("#7f7f7f", "--"),
    "NF": ("#bcbd22", "-."), "CE": ("#17becf", "-"), "CO": ("#000000", "-"), "CEC": ("#d4a000", "-"),
    "MI": ("#aec7e8", ":"), "RE": ("#98df8a", ":"),
}


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })


def draw_swarm(ax, X, y, r=None, mst=False, title=None, s=None, lw=0.6):
    X, y = np.asarray(X), np.asarray(y)
    if mst:
        for k, (E, _) in enumerate(group_msts(X, group_indices(y))):
            for a, b in E:
                ax.plot(X[[a, b], 0], X[[a, b], 1], color=GROUP_COLOURS[k % 10], lw=lw, alpha=0.8, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], c=[GROUP_COLOURS[k % 10] for k in y], s=s or 6, lw=0, zorder=2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title)


def save(fig, path_without_ext):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_without_ext}.{ext}")
    plt.close(fig)
