"""Render the videos."""
import argparse

import numpy as np
import pandas as pd

from segbench.certify import certify
from segbench.config import add_common_args, config_from_args, paths
from segbench.controllers.registry import LABELS
from segbench.generators import FAMILY_NAMES, InfeasibleConfiguration, generate
from segbench.metrics import BINARY_CODES, Params, evaluate, oriented
from segbench.plotting import METRIC_STYLE, draw_swarm, plt, setup
from segbench.utils import log


def _writer(fps):
    import shutil

    from matplotlib import animation, rcParams
    if not shutil.which("ffmpeg"):
        import imageio_ffmpeg
        rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return animation.FFMpegWriter(fps=fps, bitrate=2400)


def controller_video(cfg, P, name, run):
    f = P["data"] / "controllers" / name / f"run_{run:02d}.npz"
    if not f.exists():
        return False
    z = np.load(f)
    times, traj, types = z["times"], z["traj"], z["types"]
    df = pd.read_csv(P["results"] / "controllers.csv.gz")
    d = df[(df.controller == name) & (df.run == run)].sort_values("time")
    codes = ["CH", "NF", "CE", "CO", "CC", "CEC"]
    W = float(cfg["controllers"]["world"])
    fig = plt.figure(figsize=(8, 4))
    axs = fig.add_axes([0.02, 0.05, 0.45, 0.85])
    axm = fig.add_axes([0.56, 0.15, 0.42, 0.7])
    writer = _writer(int(cfg["videos"]["fps"]))
    out = P["videos"] / f"{name}_run{run:02d}.mp4"
    with writer.saving(fig, str(out), dpi=int(cfg["videos"]["dpi"])):
        for k in range(len(times)):
            axs.clear()
            axm.clear()
            draw_swarm(axs, traj[k], types, mst=True, s=5, lw=0.5)
            axs.set_xlim(-1.03 * W, 1.03 * W)
            axs.set_ylim(-1.03 * W, 1.03 * W)
            axs.plot([-W, W, W, -W, -W], [-W, -W, W, W, -W], "k-", lw=0.8)
            axs.set_title(f"{LABELS.get(name, name)}   t = {times[k]:.1f} s")
            dd = d[d.time <= times[k] + 1e-9]
            for c in codes:
                col, ls = METRIC_STYLE[c]
                y = oriented(c, dd[c + "_s"].values)
                y = 1 + y if c == "CH" else y
                axm.plot(dd.time, y, ls, color=col, lw=1.2, label=("1-CH" if c == "CH" else c))
            axm.set_xlim(0, times[-1])
            axm.set_ylim(-0.05, 1.05)
            axm.set_xlabel("time [s]")
            axm.set_ylabel("score")
            last = d[d.time <= times[k] + 1e-9].iloc[-1]
            verdicts = "  ".join(f"{c}:{'Y' if last[c + '_v'] == 1 else 'N'}" for c in ["CE", "CO", "CEC"])
            axm.set_title(verdicts)
            axm.legend(loc="lower right", frameon=False, fontsize=7, ncol=2)
            writer.grab_frame()
    plt.close(fig)
    return True


def sweep_video(cfg, P, family, frames=41):
    params = Params.from_config(cfg)
    levels = list(range(0, 11)) if family == "F6" else list(np.linspace(0, 1, frames))
    fig = plt.figure(figsize=(8, 4))
    axs = fig.add_axes([0.02, 0.05, 0.45, 0.85])
    axb = fig.add_axes([0.56, 0.2, 0.42, 0.6])
    writer = _writer(max(2, int(cfg["videos"]["fps"]) // 4))
    out = P["videos"] / f"sweep_{family}.mp4"
    codes = [c for c in BINARY_CODES if c != "RE"]
    with writer.saving(fig, str(out), dpi=int(cfg["videos"]["dpi"])):
        for lv in levels:
            try:
                inst = generate(family, 5, 30, lv, 3, cfg)
            except InfeasibleConfiguration:
                continue
            lab, why = certify(inst.X, inst.y, cfg.rho, inst.witness)
            res = evaluate(inst.X, inst.y, params, codes=codes)
            axs.clear()
            axb.clear()
            draw_swarm(axs, inst.X, inst.y, mst=True, s=6, lw=0.5)
            t = f"q = {lv}" if family == "F6" else f"$\\lambda$ = {lv:.2f}"
            labtxt = {1: "segregated", 0: "not segregated", None: "indeterminate"}[lab]
            axs.set_title(f"{family} {FAMILY_NAMES[family]}   {t}\ncertified: {labtxt} ({why})")
            v = [1 if res[c][1] else 0 for c in codes]
            cols = ["#2ca02c" if x else "#d62728" for x in v]
            axb.bar(range(len(codes)), [1] * len(codes), color=cols)
            axb.set_xticks(range(len(codes)))
            axb.set_xticklabels(codes, rotation=90)
            axb.set_yticks([])
            axb.set_title("binary verdicts (green = segregated)")
            writer.grab_frame()
    plt.close(fig)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    setup()
    if (P["results"] / "controllers.csv.gz").exists():
        names = pd.read_csv(P["results"] / "controllers.csv.gz").controller.unique()
        for n in names:
            for r in cfg["videos"]["controllers_runs"]:
                if controller_video(cfg, P, n, int(r)):
                    log(f"video {n} run {r}")
    for f in cfg["videos"]["lambda_sweep_families"]:
        sweep_video(cfg, P, f)
        log(f"video sweep {f}")


if __name__ == "__main__":
    main()
