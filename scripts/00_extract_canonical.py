"""Recover the canonical configurations S1-S14 from the original PDF figures into data/canonical/."""
import argparse

import numpy as np

from segbench.canonical import SCENARIOS, extract_from_pdf
from segbench.config import ROOT, add_common_args, config_from_args, paths
from segbench.utils import log


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--pdf-dir", default=None, help="folder with the original PDF figures")
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    pdf_dir = (ROOT / cfg["canonical"]["pdf_dir"]) if args.pdf_dir is None else args.pdf_dir
    out = P["data"] / "canonical"
    out.mkdir(parents=True, exist_ok=True)
    for sid, fname, desc, _ in SCENARIOS:
        pts, g = extract_from_pdf(f"{pdf_dir}/{fname}.pdf")
        np.savetxt(out / f"{sid}.csv", np.c_[pts, g], delimiter=",", header="x,y,group",
                   comments="", fmt=["%.4f", "%.4f", "%d"])
        counts = np.bincount(g).tolist()
        log(f"{sid:4s} {fname:11s} {desc:28s} agents per group {counts}")


if __name__ == "__main__":
    main()
