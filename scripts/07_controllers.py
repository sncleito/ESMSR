"""Simulate the controllers and evaluate the metrics along their trajectories."""
import argparse

import numpy as np
import pandas as pd

from segbench.config import add_common_args, config_from_args, paths
from segbench.controllers.common import cluster_metric, simulate
from segbench.controllers.registry import CONTROLLERS, make
from segbench.metrics import Params, evaluate, flatten
from segbench.utils import log, pmap, stable_seed


def controller_params(cfg):
    cc = cfg["controllers"]
    r = float(cc["robot_radius"])
    return Params.from_config(cfg, r=r, rho=float(cc["rho_factor"]) * r, tau=float(cc["cc_tau"]))


def task(args):
    cfg, name, run, outdir = args
    cc = cfg["controllers"]
    seed = stable_seed(cfg["seed"], "ctrl", run)
    ctrl = make(name, int(cc["robots"]), int(cc["groups"]), float(cc["world"]), seed, cfg)
    times, traj = simulate(ctrl, float(cc["sim_time"]), float(cc["record_every"]))
    f = outdir / name / f"run_{run:02d}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, times=times, traj=traj.astype(np.float32), types=ctrl.types, seed=seed)
    params = controller_params(cfg)
    arena = (-float(cc["world"]), -float(cc["world"]), 2 * float(cc["world"]))
    rows = []
    for t, X in zip(times, traj):
        row = dict(controller=name, run=run, time=float(t),
                   ref_clusters=cluster_metric(X, ctrl.types, float(cc["cc_tau"])))
        row.update(flatten(evaluate(X, ctrl.types, params, arena=arena)))
        rows.append(row)
    return rows


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--with-extra", action="store_true", help="also run the gradient baseline")
    ap.add_argument("--controllers", nargs="+", help="run only these and merge them into the existing results")
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    cc = cfg["controllers"]
    names = args.controllers or list(cc["names"]) + (list(cc["extra"]) if args.with_extra else [])
    missing = [n for n in names if n not in CONTROLLERS]
    if missing:
        log(f"skipping unavailable controllers {missing} (install rvo2 for inacio2019)")
    names = [n for n in names if n in CONTROLLERS]
    tasks = [(cfg, n, r, P["data"] / "controllers") for n in names for r in range(int(cc["runs"]))]
    out = pmap(task, tasks, args.jobs, desc="controllers")
    df = pd.DataFrame([r for rows in out for r in rows])
    f = P["results"] / "controllers.csv.gz"
    if args.controllers and f.exists():
        old = pd.read_csv(f)
        df = pd.concat([old[~old.controller.isin(names)], df], ignore_index=True)
    df.to_csv(f, index=False)
    log(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
