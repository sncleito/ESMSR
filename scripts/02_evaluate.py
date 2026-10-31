"""Evaluate all metrics on the benchmark and the canonical configurations."""
import argparse

import pandas as pd

from segbench.canonical import load_canonical
from segbench.certify import certify
from segbench.config import add_common_args, config_from_args, paths
from segbench.instance import load_instances
from segbench.metrics import Params, evaluate, flatten
from segbench.utils import log, pmap, require


def task(args):
    inst, params, rho = args
    res = flatten(evaluate(inst.X, inst.y, params, ring_order=inst.ring_order, timing=True))
    row = dict(name=inst.name, family=inst.family, m=inst.m, nk=inst.nk, n=inst.n, lam=inst.lam,
               level=inst.meta.get("level", inst.lam), rep=inst.meta.get("rep", 0),
               label=inst.label, reason=inst.label_reason)
    if inst.family == "canonical":
        lab, why = certify(inst.X, inst.y, rho, None)
        row.update(label=lab, reason=why, expected=inst.meta["expected"],
                   description=inst.meta["description"], source=inst.meta["source"])
    row.update(res)
    return row


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    require(P["data"] / "benchmark.pkl.gz", "scripts/01_generate_benchmark.py")
    params = Params.from_config(cfg)
    canon = load_canonical(cfg, P["data"])
    rows = pmap(task, [(i, params, cfg.rho) for i in canon], args.jobs, desc="canonical")
    pd.DataFrame(rows).to_csv(P["results"] / "metrics_canonical.csv", index=False)
    insts = load_instances(P["data"] / "benchmark.pkl.gz")
    rows = pmap(task, [(i, params, cfg.rho) for i in insts], args.jobs, desc="benchmark")
    df = pd.DataFrame(rows)
    df.to_csv(P["results"] / "metrics_benchmark.csv.gz", index=False)
    log(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
