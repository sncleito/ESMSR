"""Generate the benchmark families and certify their ground truth."""
import argparse
import itertools

import pandas as pd

from segbench.certify import certify
from segbench.config import add_common_args, config_from_args, paths
from segbench.generators import InfeasibleConfiguration, family_levels, generate
from segbench.instance import save_instances
from segbench.utils import log, pmap, stable_seed


def task(args):
    cfg, fam, m, nk, lv, rep = args
    if fam == "F6" and lv >= nk:
        return None, dict(family=fam, m=m, nk=nk, level=lv, reason="q >= nk")
    seed = stable_seed(cfg["seed"], fam, m, nk, rep)
    try:
        inst = generate(fam, m, nk, lv, seed, cfg)
    except InfeasibleConfiguration as e:
        return None, dict(family=fam, m=m, nk=nk, level=lv, reason=str(e))
    label, reason, det = certify(inst.X, inst.y, cfg.rho, inst.witness, return_details=True)
    inst.label, inst.label_reason = label, reason
    inst.meta.update(level=lv, rep=rep, **det)
    return inst, None


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    b = cfg["benchmark"]
    tasks = [(cfg, f, m, nk, lv, rep)
             for f in b["families"] for m, nk in itertools.product(b["m"], b["nk"])
             for lv in family_levels(f, cfg) for rep in range(int(b["instances"]))]
    out = pmap(task, tasks, args.jobs, desc="generating instances")
    insts = [i for i, _ in out if i is not None]
    infeasible = [e for _, e in out if e is not None]
    save_instances(insts, P["data"] / "benchmark.pkl.gz")
    rows = [dict(name=i.name, family=i.family, m=i.m, nk=i.nk, n=i.n, lam=i.lam,
                 level=i.meta["level"], rep=i.meta["rep"], seed=i.seed, label=i.label,
                 reason=i.label_reason, cohesive=i.meta["cohesive"], tau_max=i.meta["tau_max"])
            for i in insts]
    idx = pd.DataFrame(rows)
    idx.to_csv(P["results"] / "benchmark_index.csv", index=False)
    inf = pd.DataFrame(infeasible).drop_duplicates(["family", "m", "nk", "reason"]) if infeasible else pd.DataFrame()
    inf.to_csv(P["results"] / "infeasible.csv", index=False)
    log(f"{len(insts)} instances, {len(inf)} infeasible family sizes")
    summ = idx.assign(lab=idx.label.map({1.0: "pos", 0.0: "neg"}).fillna("ind"))
    log("labels per family:\n" + summ.pivot_table(index="family", columns="lab", values="name",
                                                   aggfunc="count", fill_value=0).to_string())
    if idx.reason.str.startswith("conflict").any():
        log("WARNING: conflicting certificates found (see benchmark_index.csv)")


if __name__ == "__main__":
    main()
