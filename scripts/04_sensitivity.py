"""Sensitivity of the metrics to their parameters."""
import argparse

import numpy as np
import pandas as pd

from segbench.certify import certify
from segbench.config import add_common_args, config_from_args, paths
from segbench.instance import load_instances
from segbench.metrics import Context, Params, m_cc, m_ce, m_co, m_cv, m_ea, m_mi
from segbench.utils import log, pmap, require


def task(args):
    inst, cfg = args
    sc = cfg["sensitivity"]
    base = Params.from_config(cfg)
    ctx = Context(inst.X, inst.y, base, ring_order=inst.ring_order)
    rows = []
    common = dict(name=inst.name, family=inst.family, m=inst.m, nk=inst.nk, lam=inst.lam)

    def add(param, value, metric, sv, label):
        rows.append(dict(common, param=param, value=value, metric=metric, score=sv[0],
                         verdict=np.nan if sv[1] is None else float(sv[1]), label=label))
    ce = m_ce(ctx)
    for f in sc["rho_factors"]:
        rho = f * cfg.r
        lab, _ = certify(inst.X, inst.y, rho, inst.witness)
        co = m_co(ctx, rho)
        add("rho_factor", f, "CO", co, lab)
        add("rho_factor", f, "CEC", (min(ce[0], co[0]), ce[1] and co[1]), lab)
    for f in sc["cc_tau_factors"]:
        add("tau_factor", f, "CC", m_cc(ctx, f * cfg.r), inst.label)
    for alg in sc["concave_algorithms"]:
        for k in (sc["concave_k"] if alg == "moreira" else [None]):
            add(f"concave_{alg}", -1 if k is None else k, "CV", m_cv(ctx, alg, k), inst.label)
    for c in sc["ellipse_confidence"]:
        add("ellipse_confidence", c, "EA", m_ea(ctx, c), inst.label)
    for w in sc["mips_w_max_frac"]:
        mc = dict(cfg["metrics"]["mips"], w_max_frac=w)
        add("mips_w_max_frac", w, "MI", m_mi(ctx, mc), inst.label)
    return rows


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    require(P["data"] / "benchmark.pkl.gz", "scripts/01_generate_benchmark.py")
    insts = load_instances(P["data"] / "benchmark.pkl.gz")
    rng = np.random.default_rng(cfg["seed"])
    k = min(len(insts), int(cfg["sensitivity"]["max_instances"]))
    sel = [insts[i] for i in sorted(rng.choice(len(insts), k, replace=False))]
    out = pmap(task, [(i, cfg) for i in sel], args.jobs, desc="sensitivity")
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(P["results"] / "sensitivity.csv.gz", index=False)
    log(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
