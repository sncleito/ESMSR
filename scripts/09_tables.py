"""Write Tables 5 and 6 and numbers.tex."""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from segbench.config import add_common_args, config_from_args, paths
from segbench.generators import FAMILY_NAMES
from segbench.metrics import BINARY_CODES, CODES, METRICS, oriented
from segbench.utils import log, nearest_level, require

FAMS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
TB = [c for c in ["DP", "KM", "CC", "CH", "CV", "CA", "EA", "CE", "CO", "CEC", "RE"]]
TC = ["DP", "KM", "CC", "SI", "CH", "CV", "CA", "EA", "NF", "CE", "CO", "CEC", "MI", "RE"]
ARROW = {+1: "$\\uparrow$", -1: "$\\downarrow$"}


def yn(v):
    return "--" if pd.isna(v) else ("Y" if v == 1 else "N")


def accuracy_tables(df):
    acc = pd.DataFrame(index=FAMS, columns=TB, dtype=float)
    ind = {}
    for f in FAMS:
        d = df[df.family == f]
        ind[f] = 100.0 * d.label.isna().mean() if len(d) else np.nan
        cert = d.dropna(subset=["label"])
        for c in TB:
            x = cert.dropna(subset=[c + "_v"])
            acc.loc[f, c] = 100.0 * (x[c + "_v"] == x.label).mean() if len(x) else np.nan
    cert = df.dropna(subset=["label"])
    fpr, fnr = {}, {}
    for c in TB:
        x = cert.dropna(subset=[c + "_v"])
        neg, pos = x[x.label == 0], x[x.label == 1]
        fpr[c] = 100.0 * (neg[c + "_v"] == 1).mean() if len(neg) else np.nan
        fnr[c] = 100.0 * (pos[c + "_v"] == 0).mean() if len(pos) else np.nan
    return acc, pd.Series(ind), pd.Series(fpr), pd.Series(fnr)


def fmt(v, nd=0):
    return "--" if pd.isna(v) else f"{v:.{nd}f}"


def table_binary(canon, df, P):
    acc, ind, fpr, fnr = accuracy_tables(df)
    hdr = " & ".join(f"\\metric{{{c}}}" for c in TB)
    lines = [r"\begin{table*}[p]", r"\centering",
             r"\caption{Binary verdicts. Top: verdicts on the canonical configurations, where Y and N stand for segregated and not segregated, ``Exp.'' is the expected verdict and ``Cert.'' the certified label (Sect.~\ref{sec:ground-truth}), with -- for indeterminate. Bottom: accuracy (\%) of each binary verdict over all certified instances of each family, with the fraction of indeterminate instances in column ``Ind.'', and overall error rates. \metric{RE} applies only to radial configurations. \metric{SI}, \metric{NF} and \metric{MI} have no binary rule and appear only in Table~\ref{tab:continuous}.}",
             r"\label{tab:binary}", r"\scriptsize", r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{c c l c c *{11}{c}}", r"\toprule",
             f" & \\textbf{{Config.}} & \\textbf{{Description}} & \\textbf{{Exp.}} & \\textbf{{Cert.}} & {hdr} \\\\",
             r"\midrule"]
    for _, r in canon.iterrows():
        cells = []
        for c in TB:
            v = r.get(c + "_v")
            if c == "RE" and pd.isna(r.get("RE_s")):
                cells.append("--")
            elif pd.isna(v):
                cells.append(r"\na")
            else:
                cells.append(yn(v))
        h = "1.2cm" if r.source == "5g-c9" else "1.0cm"
        lines.append(f"\\textbf{{{r['name']}}} & \\raisebox{{-.45\\height}}{{\\includegraphics[height={h}]{{{r.source}.pdf}}}} & "
                     f"{r.description.replace('m=2', '$m=2$')} & {yn(r.expected)} & {yn(r.label)} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule",
              r" & \multicolumn{2}{l}{\textbf{Family}} & \textbf{Ind.} & & \multicolumn{11}{c}{\textbf{Accuracy over certified instances (\%)}} \\",
              r"\midrule"]
    for f in FAMS:
        lines.append(f"\\textbf{{{f}}} & \\multicolumn{{2}}{{l}}{{{FAMILY_NAMES[f]}}} & {fmt(ind.get(f))} & & "
                     + " & ".join(fmt(acc.loc[f, c]) for c in TB) + r" \\")
    lines += [r"\midrule",
              r"\multicolumn{3}{l}{All families: False positive rate (\%)} & & & " + " & ".join(fmt(fpr[c]) for c in TB) + r" \\",
              r"\multicolumn{3}{l}{All families: False negative rate (\%)} & & & " + " & ".join(fmt(fnr[c]) for c in TB) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\vspace{2pt}", r"\begin{flushleft}\scriptsize",
              r"\na: the region is undefined for collinear agents.", r"\end{flushleft}", r"\end{table*}"]
    (P["tables"] / "tab_binary.tex").write_text("\n".join(lines) + "\n")
    acc.assign(Ind=ind).to_csv(P["tables"] / "binary_accuracy.csv")
    pd.DataFrame({"FPR": fpr, "FNR": fnr}).to_csv(P["tables"] / "binary_error_rates.csv")
    return acc, ind, fpr, fnr


def criteria(df, noise, runtime):
    rng = {c: np.nanpercentile(oriented(c, df[c + "_s"]), 99) - np.nanpercentile(oriented(c, df[c + "_s"]), 1)
           for c in TC}
    e2, e3, e4, e6 = {}, {}, {}, {}
    for c in TC:
        rs = []
        for (f, m, nk), d in df.groupby(["family", "m", "nk"]):
            d = d.dropna(subset=[c + "_s"])
            if d.level.nunique() > 2:
                rho = spearmanr(oriented(c, d[c + "_s"]), d.level)[0]
                if np.isfinite(rho):
                    rs.append(-rho)
        e2[c] = np.mean(rs) if rs else np.nan
        d6 = df[df.family == "F6"].dropna(subset=[c + "_s"])
        mu = d6.groupby("level")[c + "_s"].mean()
        if len(mu) > 1 and rng[c] > 0:
            o = oriented(c, mu.values)
            e3[c] = 100.0 * abs(o[1] - o[0]) / rng[c]
        else:
            e3[c] = np.nan
        if noise is not None:
            d = noise[(noise.protocol == "B") & np.isclose(noise.sigma, noise.sigma.max())]
            e4[c] = 100.0 * d[c + "_ds"].abs().mean() / rng[c] if rng[c] > 0 and len(d) else np.nan
        if runtime is not None:
            r = runtime[runtime.metric == c]
            if len(r):
                n0 = nearest_level(r.n, 1000)
                e6[c] = 1000.0 * float(r[r.n == n0].seconds.iloc[0])
    return pd.DataFrame({"E2": e2, "E3": e3, "E4": e4, "E6": e6})


def table_continuous(df, cfg, crit, P):
    levels = cfg["benchmark"]["table_levels"]
    hdr = " & ".join(f"\\metric{{{c}}}\\,{ARROW[METRICS[c].direction]}" for c in TC)
    lines = [r"\begin{table}[p]", r"\centering", r"\rotatebox{90}{%", r"\begin{minipage}{0.93\textheight}", r"\centering",
             r"\caption{Continuous scores. Top: mean and standard deviation of each score over the instances of each family at two levels of degradation ($\lambda$ for F1 to F5 and F7, number $q$ of trapped agents per group for F6). Arrows indicate the direction of more segregated configurations. Bottom: summary of criteria E2, E3, E4 and E6 (Sect.~\ref{sec:criteria}). \metric{RE} applies only to radial configurations.}",
             r"\label{tab:continuous}", r"\scriptsize", r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{l l *{14}{c}}", r"\toprule", f"\\textbf{{Family}} & \\textbf{{Level}} & {hdr} \\\\", r"\midrule"]
    rows = []
    for f in FAMS:
        d = df[df.family == f]
        lv_list = levels.get(f, levels["default"])
        for i, lv in enumerate(lv_list):
            if not len(d):
                continue
            lv_eff = nearest_level(d.level, lv)
            e = d[np.isclose(d.level, lv_eff)]
            lab = f"$q={int(lv_eff)}$" if f == "F6" else f"$\\lambda={lv_eff:g}$"
            cells = []
            for c in TC:
                x = e[c + "_s"].dropna()
                cells.append("--" if not len(x) else f"{x.mean():.2f}$\\pm${x.std(ddof=0):.2f}")
                rows.append(dict(family=f, level=lv_eff, metric=c, mean=x.mean() if len(x) else np.nan,
                                 sd=x.std(ddof=0) if len(x) else np.nan))
            first = f"\\multirow{{2}}{{*}}{{\\textbf{{{f}}} {FAMILY_NAMES[f]}}}" if i == 0 else ""
            lines.append(f"{first} & {lab} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[1pt]")
    lines.append(r"\midrule")
    names = {"E2": "E2 Monotonicity, mean Spearman over families",
             "E3": "E3 Change for one trapped agent per group (F6, \\% of range)",
             "E4": "E4 Change under noise, largest $\\sigma$ (\\% of range)",
             "E6": "E6 Runtime, $n \\approx 10^3$ (ms)"}
    for k, nd in [("E2", 2), ("E3", 1), ("E4", 1), ("E6", 1)]:
        lines.append(f"\\multicolumn{{2}}{{l}}{{{names[k]}}} & " + " & ".join(fmt(crit.loc[c, k], nd) for c in TC) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{minipage}}", r"\end{table}"]
    (P["tables"] / "tab_continuous.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(P["tables"] / "continuous_scores.csv", index=False)
    crit.to_csv(P["tables"] / "criteria.csv")


def numbers(df, acc, ind, fpr, fnr, crit, P):
    cert = df.dropna(subset=["label"])
    macros = {
        "resNumInstances": len(df), "resNumCertified": len(cert),
        "resPctIndeterminate": f"{100 * df.label.isna().mean():.1f}",
        "resCOfpr": fmt(fpr["CO"], 1), "resCOfnr": fmt(fnr["CO"], 1),
        "resCECfpr": fmt(fpr["CEC"], 1), "resCECfnr": fmt(fnr["CEC"], 1),
    }
    for c in TB:
        x = cert.dropna(subset=[c + "_v"])
        macros[f"resAcc{c}"] = fmt(100 * (x[c + "_v"] == x.label).mean(), 1) if len(x) else "--"
    for c in TC:
        macros[f"resEthree{c}"] = fmt(crit.loc[c, "E3"], 1)
        macros[f"resEtwo{c}"] = fmt(crit.loc[c, "E2"], 2)
    txt = "% Generated by scripts/09_tables.py\n" + "\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n"
    (P["tables"] / "numbers.tex").write_text(txt)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*constant.*")
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    P = paths(cfg)
    require(P["results"] / "metrics_benchmark.csv.gz", "scripts/02_evaluate.py")
    df = pd.read_csv(P["results"] / "metrics_benchmark.csv.gz")
    canon = pd.read_csv(P["results"] / "metrics_canonical.csv")
    noise = pd.read_csv(P["results"] / "noise.csv.gz") if (P["results"] / "noise.csv.gz").exists() else None
    runtime = pd.read_csv(P["results"] / "runtime.csv") if (P["results"] / "runtime.csv").exists() else None
    acc, ind, fpr, fnr = table_binary(canon, df, P)
    crit = criteria(df, noise, runtime)
    table_continuous(df, cfg, crit, P)
    numbers(df, acc, ind, fpr, fnr, crit, P)
    log("tables written to " + str(P["tables"]))
    print(acc.round(0).assign(Ind=ind.round(0)).to_string())
    print(pd.DataFrame({"FPR": fpr, "FNR": fnr}).round(1).T.to_string())
    print(crit.round(2).T.to_string())


if __name__ == "__main__":
    main()
