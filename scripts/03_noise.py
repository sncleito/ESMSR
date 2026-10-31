"""Robustness of the verdicts to position noise."""
import argparse

import numpy as np
import pandas as pd

from segbench.canonical import load_canonical
from segbench.config import add_common_args, config_from_args, paths
from segbench.geometry import group_indices, group_msts
from segbench.instance import load_instances
from segbench.metrics import BINARY_CODES, CODES, Params, evaluate
from segbench.noise import perturb
from segbench.utils import log, pmap, require, stable_seed


def _lmax(X, y):
    return np.array([L.max() if len(L) else 0.0 for _, L in group_msts(X, group_indices(y))])


def task(args):
    inst, params, cfg = args
    ncfg = cfg["noise"]
    clean = evaluate(inst.X, inst.y, params, ring_order=inst.ring_order)
    lm0 = _lmax(inst.X, inst.y)
    margin = float(np.min(np.abs(lm0 - params.rho)))
    settings = [("A", ncfg["sigma_fixed"], f) for f in ncfg["fractions"]] + \
               [("B", s, ncfg["frac_fixed"]) for s in ncfg["sigmas"]]
    rows = []
    for proto, sigma, frac in settings:
        for rep in range(int(ncfg["realizations"])):
            rng = np.random.default_rng(stable_seed(inst.name, proto, sigma, frac, rep))
            Xn, delta, _ = perturb(inst.X, sigma, frac, rng)
            res = evaluate(Xn, inst.y, params, ring_order=inst.ring_order)
            lm = _lmax(Xn, inst.y)
            row = dict(name=inst.name, family=inst.family, m=inst.m, nk=inst.nk, protocol=proto,
                       sigma=sigma, fraction=frac, rep=rep, delta=delta, margin=margin,
                       lmax_shift=float(np.max(np.abs(lm - lm0))), label=inst.label)
            for c in CODES:
                s0, v0 = clean[c]
                s1, v1 = res[c]
                row[c + "_ds"] = s1 - s0
                if c in BINARY_CODES:
                    row[c + "_flip"] = np.nan if (v0 is None or v1 is None) else float(v0 != v1)
            rows.append(row)
    return rows


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    require(P["data"] / "benchmark.pkl.gz", "scripts/01_generate_benchmark.py")
    params = Params.from_config(cfg)
    insts = load_instances(P["data"] / "benchmark.pkl.gz")
    lam_set = {float(l) for l in cfg["noise"]["lambdas"]}
    sel, count = [], {}
    for i in insts:
        lvl = i.meta.get("level")
        if i.family == "F6" or float(lvl) not in lam_set:
            continue
        key = (i.family, i.m, i.nk)
        if count.get(key, 0) < int(cfg["noise"]["max_instances_per_cell"]):
            sel.append(i)
            count[key] = count.get(key, 0) + 1
    sel += load_canonical(cfg, P["data"])
    out = pmap(task, [(i, params, cfg) for i in sel], args.jobs, desc="noise")
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(P["results"] / "noise.csv.gz", index=False)
    log(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
