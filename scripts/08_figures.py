"""Generate the figures."""
import argparse
import traceback

import numpy as np
import pandas as pd

from segbench.canonical import load_canonical
from segbench.config import add_common_args, config_from_args, paths
from segbench.controllers.registry import LABELS
from segbench.generators import FAMILY_NAMES, InfeasibleConfiguration, generate
from segbench.metrics import BINARY_CODES, CODES, METRICS, oriented
from segbench.plotting import METRIC_STYLE, draw_swarm, plt, save, setup
from segbench.utils import log, nearest_level


def fig_families(cfg, P):
    fams = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    levels = {"F6": [0, 1, 5]}
    fig, axes = plt.subplots(3, 7, figsize=(7.2, 3.3))
    for j, f in enumerate(fams):
        for i, lv in enumerate(levels.get(f, [0.0, 0.25, 0.5])):
            ax = axes[i, j]
            try:
                inst = generate(f, 5, 30, lv, 7, cfg)
                draw_swarm(ax, inst.X, inst.y, s=2.5)
            except InfeasibleConfiguration:
                ax.axis("off")
            lab = f"q={lv}" if f == "F6" else f"$\\lambda$={lv}"
            ax.set_title((f"{f} {FAMILY_NAMES[f]}\n" if i == 0 else "") + lab, fontsize=6)
    save(fig, P["figures"] / "fig_families")


def _rescaled(df, codes):
    out = {}
    for c in codes:
        o = oriented(c, df[c + "_s"])
        lo, hi = np.nanpercentile(o, 1), np.nanpercentile(o, 99)
        out[c] = np.clip((o - lo) / (hi - lo), 0, 1) if hi > lo else o * np.nan
    return pd.DataFrame(out, index=df.index)


def fig_lambda(cfg, P):
    df = pd.read_csv(P["results"] / "metrics_benchmark.csv.gz")
    codes = [c for c in CODES if c != "RE"]
    R = _rescaled(df, codes)
    fams = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey=True)
    for ax, f in zip(axes.flat, fams):
        d = df[df.family == f]
        x = "level"
        for c in codes:
            g = R.loc[d.index, c].groupby(d[x])
            mu, se = g.mean(), g.sem().fillna(0)
            col, ls = METRIC_STYLE[c]
            ax.plot(mu.index, mu.values, ls, color=col, lw=1, label=c)
            ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.12, lw=0)
        ax.set_title(f"{f} {FAMILY_NAMES[f]}")
        ax.set_xlabel("q (trapped per group)" if f == "F6" else "$\\lambda$")
    axes.flat[-1].axis("off")
    axes.flat[0].set_ylabel("rescaled score (1 = segregated)")
    axes.flat[4].set_ylabel("rescaled score (1 = segregated)")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, ncol=2, loc="center", frameon=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.15)
    save(fig, P["figures"] / "fig_lambda")


def fig_noise(cfg, P):
    df = pd.read_csv(P["results"] / "noise.csv.gz")
    codes = [c for c in BINARY_CODES if c != "RE"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, proto, x, xl in [(axes[0], "A", "fraction", "fraction of noisy agents"),
                             (axes[1], "B", "sigma", "$\\sigma$ [m]")]:
        d = df[df.protocol == proto]
        for c in codes:
            col, ls = METRIC_STYLE[c]
            g = d.groupby(x)[c + "_flip"].mean()
            ax.plot(g.index, g.values, ls, color=col, marker="o", ms=2, lw=1, label=c)
        ax.set_xlabel(xl)
        ax.set_ylabel("verdict change rate")
    axes[1].legend(ncol=2, frameon=False, fontsize=6)
    ax = axes[2]
    d = df[df.sigma > 0]
    flip = d["CO_flip"] == 1
    ax.scatter(d.delta[~flip], d.margin[~flip], s=3, c="#999999", lw=0, label="no change")
    ax.scatter(d.delta[flip], d.margin[flip], s=5, c="#d62728", lw=0, label="CO changes")
    xx = np.linspace(0, d.delta.max() if len(d) else 1, 50)
    ax.plot(xx, 2 * xx, "k--", lw=1, label="margin = 2$\\delta$ (Prop. 3)")
    ax.set_xlabel("largest displacement $\\delta$ [m]")
    ax.set_ylabel("$\\min_k |\\ell^{(k)}_{\\max} - \\rho|$ [m]")
    ax.legend(frameon=False, fontsize=6)
    save(fig, P["figures"] / "fig_noise")


def fig_sensitivity(cfg, P):
    df = pd.read_csv(P["results"] / "sensitivity.csv.gz")
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9))

    def acc(d):
        d = d.dropna(subset=["label", "verdict"])
        return (d.verdict == d.label).mean() * 100 if len(d) else np.nan
    for c in ["CO", "CEC"]:
        g = df[(df.param == "rho_factor") & (df.metric == c)].groupby("value").apply(acc, include_groups=False)
        axes[0].plot(g.index, g.values, "-o", ms=3, color=METRIC_STYLE[c][0], label=c)
    axes[0].set_xlabel("$\\rho / r$")
    axes[0].legend(frameon=False)
    g = df[df.param == "tau_factor"].groupby("value").apply(acc, include_groups=False)
    axes[1].plot(g.index, g.values, "-o", ms=3, color=METRIC_STYLE["CC"][0])
    axes[1].set_xlabel("$\\tau / r$ (CC)")
    d = df[df.param.str.startswith("concave")]
    g = d[d.param == "concave_moreira"].groupby("value").apply(acc, include_groups=False)
    axes[2].plot(g.index, g.values, "-o", ms=3, color=METRIC_STYLE["CV"][0], label="Moreira, k")
    po = d[d.param == "concave_parkoh"]
    if len(po):
        axes[2].axhline(acc(po), color="k", ls=":", lw=1, label="Park-Oh")
    axes[2].set_xlabel("k (CV)")
    axes[2].legend(frameon=False, fontsize=6)
    g = df[df.param == "ellipse_confidence"].groupby("value").apply(acc, include_groups=False)
    axes[3].plot(g.index, g.values, "-o", ms=3, color=METRIC_STYLE["EA"][0])
    axes[3].set_xlabel("confidence (EA)")
    d = df[df.param == "mips_w_max_frac"]
    vals = []
    for w, g in d.groupby("value"):
        rs = [g2.score.corr(g2.lam, method="spearman") for _, g2 in g.groupby("family") if g2.lam.nunique() > 1]
        vals.append((w, -np.nanmean(rs) if rs else np.nan))
    if vals:
        axes[4].plot(*zip(*vals), "-o", ms=3, color="#1f77b4")
    axes[4].set_xlabel("$w_{\\max}/L$ (MI)")
    axes[4].set_ylabel("monotonicity")
    for ax in axes[:4]:
        ax.set_ylabel("accuracy (%)")
    fig.tight_layout()
    save(fig, P["figures"] / "fig_sensitivity")


def fig_runtime(cfg, P):
    df = pd.read_csv(P["results"] / "runtime.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for c, g in df.groupby("metric"):
        col, ls = METRIC_STYLE.get(c, ("k", "-"))
        ax.loglog(g.n, g.seconds, ls, marker="o", ms=2.5, color=col, lw=1, label=c)
    ax.set_xlabel("number of agents n")
    ax.set_ylabel("time [s]")
    ax.legend(ncol=2, frameon=False, fontsize=6)
    save(fig, P["figures"] / "fig_runtime")


def fig_agreement(cfg, P):
    K = pd.read_csv(P["results"] / "agreement_kappa.csv", index_col=0)
    T = pd.read_csv(P["results"] / "agreement_kendall.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, M, title in [(axes[0], K, "Cohen's $\\kappa$ (binary)"), (axes[1], T, "Kendall's $\\tau$ (continuous)")]:
        im = ax.imshow(M.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(M)))
        ax.set_xticklabels(M.columns, rotation=90)
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(M.index)
        for i in range(len(M)):
            for j in range(len(M)):
                v = M.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5,
                            color="white" if abs(v) > 0.6 else "black")
        ax.set_title(title)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.7)
    save(fig, P["figures"] / "fig_agreement")


def fig_controllers(cfg, P):
    df = pd.read_csv(P["results"] / "controllers.csv.gz")
    names = list(df.controller.unique())
    codes = ["CH", "NF", "CE", "CO", "CC", "CEC"]
    fig, axes = plt.subplots(1, len(names), figsize=(1.8 * len(names) + 0.4, 2.2), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], names):
        d = df[df.controller == n]
        for c in codes:
            o = pd.Series(oriented(c, d[c + "_s"]), index=d.index)
            g = o.groupby(d.time)
            mu, se = g.mean(), g.sem().fillna(0)
            if c in ("CH",):
                mu, se = 1 + mu, se
            col, ls = METRIC_STYLE[c]
            ax.plot(mu.index, mu.values, ls, color=col, lw=1, label=c if c != "CH" else "1-CH")
            ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.12, lw=0)
            first = d[d[c + "_v"] == 1].groupby("run").time.min()
            if len(first):
                ax.plot([first.median()], [-0.06], "v", color=col, ms=4, clip_on=False)
        ax.set_title(LABELS.get(n, n))
        ax.set_xlabel("time [s]")
    axes[0][0].set_ylabel("score")
    axes[0][-1].legend(frameon=False, fontsize=6)
    save(fig, P["figures"] / "fig_controllers")


def fig_canonical(cfg, P):
    canon = load_canonical(cfg, P["data"])
    out = P["figures"] / "canonical"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 7, figsize=(7.2, 2.4))
    for ax, inst in zip(axes.flat, canon):
        draw_swarm(ax, inst.X, inst.y, mst=True, s=2, lw=0.4, title=f"{inst.name}")
        f2, a2 = plt.subplots(figsize=(1.2, 1.2))
        draw_swarm(a2, inst.X, inst.y, s=2)
        save(f2, out / inst.name)
    save(fig, P["figures"] / "fig_canonical")


FIGS = [fig_families, fig_lambda, fig_noise, fig_sensitivity, fig_runtime, fig_agreement,
        fig_controllers, fig_canonical]


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--only", nargs="*", help="subset of figure function names")
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    setup()
    for f in FIGS:
        if args.only and f.__name__ not in args.only:
            continue
        try:
            f(cfg, P)
            log(f"{f.__name__}: ok")
        except FileNotFoundError as e:
            log(f"{f.__name__}: skipped ({e.filename} missing)")
        except Exception:
            log(f"{f.__name__}: FAILED")
            traceback.print_exc()


if __name__ == "__main__":
    main()
