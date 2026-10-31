"""Agreement between metrics (Cohen's kappa, Kendall's tau)."""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

from segbench.config import add_common_args, config_from_args, paths
from segbench.metrics import BINARY_CODES, CODES, oriented
from segbench.utils import log, require


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    f = P["results"] / "metrics_benchmark.csv.gz"
    require(f, "scripts/02_evaluate.py")
    df = pd.read_csv(f)
    bc = [c for c in BINARY_CODES if c != "RE"]
    K = pd.DataFrame(np.nan, index=bc, columns=bc)
    for a in bc:
        for b in bc:
            s = df[[a + "_v", b + "_v"]].dropna()
            if len(s) > 1 and s.nunique().min() > 1:
                K.loc[a, b] = cohen_kappa_score(s.iloc[:, 0], s.iloc[:, 1])
            elif len(s) > 1:
                K.loc[a, b] = float((s.iloc[:, 0] == s.iloc[:, 1]).mean() == 1.0)
    cc = [c for c in CODES if c != "RE"]
    T = pd.DataFrame(np.nan, index=cc, columns=cc)
    for a in cc:
        for b in cc:
            s = df[[a + "_s", b + "_s"]].dropna()
            if len(s) > 2:
                T.loc[a, b] = kendalltau(oriented(a, s.iloc[:, 0]), oriented(b, s.iloc[:, 1]))[0]
    K.to_csv(P["results"] / "agreement_kappa.csv")
    T.to_csv(P["results"] / "agreement_kendall.csv")
    log("agreement matrices written")


if __name__ == "__main__":
    main()
