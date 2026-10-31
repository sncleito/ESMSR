"""Check the formal claims of the paper against the results."""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from segbench.config import add_common_args, config_from_args, paths
from segbench.geometry import bichromatic_hull_edges, crossings_between_trees, group_indices, group_msts
from segbench.instance import load_instances
from segbench.metrics import Context, Params, cluster_count, m_co
from segbench.utils import log, require


def implication(df, a, b):
    d = df.dropna(subset=[a + "_v", b + "_v"])
    viol = d[(d[a + "_v"] == 1) & (d[b + "_v"] == 0)]
    return dict(checked=int((d[a + "_v"] == 1).sum()), violations=int(len(viol)),
                examples=viol.name.head(5).tolist())


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--sample", type=int, default=300, help="instances for the recomputed checks")
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    require(P["results"] / "metrics_benchmark.csv.gz", "scripts/02_evaluate.py")
    df = pd.concat([pd.read_csv(P["results"] / "metrics_benchmark.csv.gz"),
                    pd.read_csv(P["results"] / "metrics_canonical.csv")], ignore_index=True)
    big = df[df.nk >= 3]
    out = {"prop1": {f"{a}=>{b}": implication(big, a, b)
                     for a, b in [("CA", "CH"), ("KM", "CH"), ("CH", "CV"), ("CH", "CE")]}}
    neg = df[df.label == 0]
    pos = df[df.label == 1]
    out["prop2"] = {"CEC_positive_on_certified_negative": int((neg.CEC_v == 1).sum()),
                    "CO_negative_on_certified_positive": int((pos.CO_v == 0).sum()),
                    "certified_negative": int(len(neg)), "certified_positive": int(len(pos))}
    out["certification_conflicts"] = int(df.reason.astype(str).str.startswith("conflict").sum())

    insts = load_instances(P["data"] / "benchmark.pkl.gz")
    rng = np.random.default_rng(cfg["seed"])
    sel = [insts[i] for i in rng.choice(len(insts), min(args.sample, len(insts)), replace=False)]
    params = Params.from_config(cfg)
    cc_mismatch, rip_mismatch, rip_checked = 0, 0, 0
    try:
        from ripser import ripser
    except ImportError:
        ripser = None
    tok = dict(checked=0, violations=0)
    for inst in sel:
        ctx = Context(inst.X, inst.y, params)
        co = m_co(ctx)[1]
        cc = cluster_count(ctx, np.nextafter(cfg.rho, 0)) == ctx.m
        cc_mismatch += int(co != cc)
        if ripser is not None:
            for g, (_, L) in zip(ctx.groups, ctx.msts):
                if len(g) < 2:
                    continue
                d0 = ripser(inst.X[g], maxdim=0)["dgms"][0][:, 1]
                d0 = np.sort(d0[np.isfinite(d0)])
                rip_checked += 1
                rip_mismatch += int(not np.allclose(d0, np.sort(L), atol=1e-6))
    for inst in sel:
        G = group_indices(inst.y)
        T = [E for E, _ in group_msts(inst.X, G)]
        for k in range(len(G)):
            for l in range(k + 1, len(G)):
                sub = np.concatenate([G[k], G[l]])
                t = bichromatic_hull_edges(inst.X[sub], inst.y[sub])
                if t >= 4:
                    nc, _ = crossings_between_trees(inst.X, [T[k], T[l]])
                    tok["checked"] += 1
                    tok["violations"] += int(nc < t / 2 - 1)
    out["cc_equals_co_at_rho"] = dict(checked=len(sel), mismatches=cc_mismatch)
    out["co_equals_b0_ripser"] = dict(checked=rip_checked, mismatches=rip_mismatch,
                                      available=ripser is not None)
    out["prop4_tokunaga"] = tok
    nf = P["results"] / "noise.csv.gz"
    if nf.exists():
        nz = pd.read_csv(nf)
        nz = nz.dropna(subset=["CO_flip"])
        bad = nz[(nz.CO_flip == 1) & (nz.margin > 2 * nz.delta)]
        out["prop3_robustness"] = dict(realizations=int(len(nz)), flips=int(nz.CO_flip.sum()),
                                       flips_outside_margin=int(len(bad)))
    ok = (all(v["violations"] == 0 for v in out["prop1"].values())
          and out["prop2"]["CEC_positive_on_certified_negative"] == 0
          and out["prop2"]["CO_negative_on_certified_positive"] == 0
          and out["certification_conflicts"] == 0 and cc_mismatch == 0 and rip_mismatch == 0
          and tok["violations"] == 0
          and out.get("prop3_robustness", {}).get("flips_outside_margin", 0) == 0)
    out["all_checks_passed"] = bool(ok)
    (P["results"] / "validation.json").write_text(json.dumps(out, indent=2))
    log(json.dumps(out, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
