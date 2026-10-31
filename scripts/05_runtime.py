"""Running time of each metric against the number of agents."""
import argparse


import numpy as np
import pandas as pd

from segbench.config import add_common_args, config_from_args, paths
from segbench.generators import generate
from segbench.metrics import CODES, Params, evaluate
from segbench.utils import log


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    rc = cfg["runtime"]
    params = Params.from_config(cfg)
    m = int(rc["m"])
    active = [c for c in CODES if c not in ("RE", "CEC")]
    rows = []
    for n in rc["n"]:
        nk = max(2, int(n) // m)
        inst = generate("F1", m, nk, 0.0, int(cfg["seed"]), cfg)
        for c in list(active):
            ts = []
            for _ in range(int(rc["repeats"])):
                res = evaluate(inst.X, inst.y, params, codes=[c], timing=True)
                ts.append(res["t_" + c])
                if ts[-1] > float(rc["time_budget_s"]):
                    break
            rows.append(dict(metric=c, n=inst.n, m=m, seconds=float(np.median(ts))))
            if np.median(ts) > float(rc["time_budget_s"]):
                log(f"{c}: {np.median(ts):.1f}s at n={inst.n}, dropped for larger n")
                active.remove(c)
        log(f"n={inst.n} done")
    df = pd.DataFrame(rows)
    ce = df[df.metric == "CE"].set_index("n").seconds
    co = df[df.metric == "CO"].set_index("n").seconds
    for n in ce.index.intersection(co.index):
        rows.append(dict(metric="CEC", n=n, m=m, seconds=float(ce[n] + co[n])))
    pd.DataFrame(rows).to_csv(P["results"] / "runtime.csv", index=False)


if __name__ == "__main__":
    main()
