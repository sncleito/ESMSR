from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import chi2
from shapely import affinity
from shapely.geometry import Point

from . import concave as _concave
from .geometry import (crossings_between_trees, convex_polygon, delaunay_edges,
                       group_indices, group_msts)
from .mips import mips_index


@dataclass(frozen=True)
class MetricInfo:
    code: str
    name: str
    family: str
    axis: str
    direction: int
    binary: bool


METRICS = {
    "DP": MetricInfo("DP", "Differential potential", "Distance", "S+C", -1, True),
    "KM": MetricInfo("KM", "k-means purity", "Cluster", "S+C", +1, True),
    "CC": MetricInfo("CC", "Same-type cluster count", "Cluster", "C", +1, True),
    "SI": MetricInfo("SI", "Silhouette", "Cluster", "S+C", +1, False),
    "CH": MetricInfo("CH", "Convex hull", "Geometry", "S", -1, True),
    "CV": MetricInfo("CV", "Concave hull", "Geometry", "S", -1, True),
    "CA": MetricInfo("CA", "Circular abstraction", "Geometry", "S", -1, True),
    "EA": MetricInfo("EA", "Ellipsoidal abstraction", "Geometry", "S", -1, True),
    "NF": MetricInfo("NF", "Same-type neighbour fraction", "Neighbourhood", "S", +1, False),
    "CE": MetricInfo("CE", "Crossing Edges", "Graph", "S", +1, True),
    "CO": MetricInfo("CO", "Connectivity", "Graph", "C", +1, True),
    "CEC": MetricInfo("CEC", "CE and CO", "Graph", "S+C", +1, True),
    "MI": MetricInfo("MI", "Per-group MIPS index", "Density", "C", +1, False),
    "RE": MetricInfo("RE", "Radial rank error", "Order", "S", -1, True),
}
CODES = list(METRICS)
BINARY_CODES = [c for c in CODES if METRICS[c].binary]
CONTINUOUS_CODES = CODES


@dataclass
class Params:
    rho: float
    tau: float
    r: float
    concave_algorithm: str = "moreira"
    concave_k: int = 3
    parkoh_concavity: float = 2.0
    ellipse_confidence: float = 0.95
    kmeans_n_init: int = 10
    silhouette_max_n: int = 5000
    mips: dict | None = None
    eps_area_rel: float = 1e-9
    seed: int = 0

    @staticmethod
    def from_config(cfg, **over) -> "Params":
        mc = cfg["metrics"]
        p = Params(rho=cfg.rho, tau=cfg.tau, r=cfg.r,
                   concave_algorithm=mc["concave_algorithm"], concave_k=int(mc["concave_k"]),
                   parkoh_concavity=float(mc["parkoh_concavity"]),
                   ellipse_confidence=float(mc["ellipse_confidence"]),
                   kmeans_n_init=int(mc["kmeans_n_init"]),
                   silhouette_max_n=int(mc["silhouette_max_n"]),
                   mips=dict(mc["mips"]), eps_area_rel=float(cfg["eps_area_rel"]),
                   seed=int(cfg["seed"]))
        for k, v in over.items():
            setattr(p, k, v)
        return p


class Context:
    def __init__(self, X, y, params: Params, ring_order=None, arena=None):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, int)
        self.p = params
        self.m = int(self.y.max()) + 1
        self.groups = group_indices(self.y, self.m)
        self.n = len(self.X)
        self.ring_order = ring_order
        self._arena = arena
        self._msts = None

    @property
    def msts(self):
        if self._msts is None:
            self._msts = group_msts(self.X, self.groups)
        return self._msts

    @property
    def lmax(self) -> np.ndarray:
        return np.array([L.max() if len(L) else 0.0 for _, L in self.msts])

    @property
    def arena(self):
        if self._arena is None:
            lo = self.X.min(0) - self.p.rho
            hi = self.X.max(0) + self.p.rho
            L = float((hi - lo).max())
            c = (lo + hi) / 2
            self._arena = (float(c[0] - L / 2), float(c[1] - L / 2), L)
        return self._arena


def m_dp(ctx: Context):
    X, G = ctx.X, ctx.groups
    within = np.array([pdist(X[g]).mean() if len(g) > 1 else 0.0 for g in G])
    r = -np.inf
    for k in range(ctx.m):
        for l in range(k + 1, ctx.m):
            dkl = cdist(X[G[k]], X[G[l]]).mean()
            r = max(r, max(within[k], within[l]) / dkl)
    return float(r), bool(r < 1.0)


def m_km(ctx: Context):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.cluster import contingency_matrix

    lab = KMeans(n_clusters=ctx.m, n_init=ctx.p.kmeans_n_init, random_state=ctx.p.seed).fit_predict(ctx.X)
    C = contingency_matrix(ctx.y, lab)
    hits = int(C.max(axis=0).sum())
    ctx.ari = float(adjusted_rand_score(ctx.y, lab))
    return hits / ctx.n, bool(hits == ctx.n)


def cluster_count(ctx: Context, tau: float) -> int:
    return int(sum(1 + int(np.sum(L > tau)) for _, L in ctx.msts))


def m_cc(ctx: Context, tau: float | None = None):
    c = cluster_count(ctx, ctx.p.tau if tau is None else tau)
    return ctx.m / c, bool(c == ctx.m)


def m_si(ctx: Context):
    from sklearn.metrics import silhouette_score

    if ctx.m < 2 or min(len(g) for g in ctx.groups) < 2:
        return float("nan"), None
    kw = {}
    if ctx.n > ctx.p.silhouette_max_n:
        kw = dict(sample_size=ctx.p.silhouette_max_n, random_state=ctx.p.seed)
    return float(silhouette_score(ctx.X, ctx.y, **kw)), None


def _region_overlap(ctx: Context, regions):
    if any(R is None or R.is_empty or R.area <= 0 for R in regions):
        return float("nan"), None
    eps = ctx.p.eps_area_rel * max(R.area for R in regions)
    ious, ok = [], True
    for k in range(ctx.m):
        for l in range(k + 1, ctx.m):
            inter = regions[k].intersection(regions[l]).area
            uni = regions[k].union(regions[l]).area
            ious.append(inter / uni if uni > 0 else 0.0)
            ok &= inter <= eps
    return float(np.mean(ious)), bool(ok)


def m_ch(ctx: Context):
    return _region_overlap(ctx, [convex_polygon(ctx.X[g]) for g in ctx.groups])


def m_cv(ctx: Context, algorithm: str | None = None, k: int | None = None):
    alg = algorithm or ctx.p.concave_algorithm
    kk = ctx.p.concave_k if k is None else k
    regs = [_concave.concave_hull(ctx.X[g], alg, kk, ctx.p.parkoh_concavity) for g in ctx.groups]
    return _region_overlap(ctx, regs)


def _lens_area(d, r1, r2):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos(np.clip((d * d + r1 * r1 - r2 * r2) / (2 * d * r1), -1, 1))
    a2 = r2 ** 2 * np.arccos(np.clip((d * d + r2 * r2 - r1 * r1) / (2 * d * r2), -1, 1))
    a3 = 0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    return a1 + a2 - a3


def m_ca(ctx: Context):
    mu, rad = [], []
    for g in ctx.groups:
        P = ctx.X[g]
        c = P.mean(0)
        s = np.mean(np.sum((P - c) ** 2, axis=1))
        mu.append(c)
        rad.append(np.sqrt(len(g) * s))
    ious, ok = [], True
    for k in range(ctx.m):
        for l in range(k + 1, ctx.m):
            d = float(np.linalg.norm(mu[k] - mu[l]))
            inter = _lens_area(d, rad[k], rad[l])
            uni = np.pi * (rad[k] ** 2 + rad[l] ** 2) - inter
            ious.append(inter / uni if uni > 0 else 0.0)
            ok &= d >= rad[k] + rad[l]
    return float(np.mean(ious)), bool(ok)


def ellipse_polygon(P: np.ndarray, confidence: float, quad_segs: int = 64):
    if len(P) < 3:
        return None
    c = P.mean(0)
    S = np.cov(P.T)
    w, V = np.linalg.eigh(S)
    if w.min() <= 1e-12 * max(w.max(), 1e-300):
        return None
    q = chi2.ppf(confidence, 2)
    A = V @ np.diag(np.sqrt(q * w))
    unit = Point(0.0, 0.0).buffer(1.0, quad_segs=quad_segs)
    return affinity.affine_transform(unit, [A[0, 0], A[0, 1], A[1, 0], A[1, 1], c[0], c[1]])


def m_ea(ctx: Context, confidence: float | None = None):
    conf = ctx.p.ellipse_confidence if confidence is None else confidence
    return _region_overlap(ctx, [ellipse_polygon(ctx.X[g], conf) for g in ctx.groups])


def m_nf(ctx: Context):
    E = delaunay_edges(ctx.X)
    f = float(np.mean(ctx.y[E[:, 0]] == ctx.y[E[:, 1]]))
    nk = np.array([len(g) for g in ctx.groups])
    f0 = float(np.sum(nk * (nk - 1)) / (ctx.n * (ctx.n - 1)))
    return (f - f0) / (1 - f0), None


def m_ce(ctx: Context):
    trees = [E for E, _ in ctx.msts]
    ncross, mask = crossings_between_trees(ctx.X, trees)
    ctx.ce_crossings = ncross
    tot = ctx.n - ctx.m
    s = 1.0 - (mask.sum() / tot if tot > 0 else 0.0)
    return float(s), bool(mask.sum() == 0)


def m_co(ctx: Context, rho: float | None = None):
    rho = ctx.p.rho if rho is None else rho
    lm = ctx.lmax
    with np.errstate(divide="ignore"):
        s = float(np.min(np.minimum(1.0, np.where(lm > 0, rho / lm, np.inf))))
    return s, bool(np.all(lm < rho))


def m_cec(ctx: Context, ce=None, co=None):
    ce = ce or m_ce(ctx)
    co = co or m_co(ctx)
    return min(ce[0], co[0]), bool(ce[1] and co[1])


def m_mi(ctx: Context, mcfg: dict | None = None):
    mcfg = mcfg or ctx.p.mips
    vals = [mips_index(ctx.X[g], ctx.arena, ctx.p.r, mcfg, seed=ctx.p.seed) for g in ctx.groups]
    return float(np.nanmean(vals)), None


def radial_rank_error(X, y, ring_order) -> float:
    rank = np.empty(int(max(ring_order)) + 1, int)
    rank[list(ring_order)] = np.arange(len(ring_order))
    g = rank[y]
    rad = np.linalg.norm(X - X.mean(0), axis=1)
    sorted_by_rank = [np.sort(rad[g == a]) for a in range(len(ring_order))]
    count = 0
    for b in range(1, len(ring_order)):
        rj = rad[g == b]
        for a in range(b):
            Ra = sorted_by_rank[a]
            count += int(np.sum(len(Ra) - np.searchsorted(Ra, rj, side="left")))
    return count / len(X) ** 2


def m_re(ctx: Context):
    if ctx.ring_order is None:
        return float("nan"), None
    e = radial_rank_error(ctx.X, ctx.y, ctx.ring_order)
    return float(e), bool(e == 0.0)


FUNCS = {"DP": m_dp, "KM": m_km, "CC": m_cc, "SI": m_si, "CH": m_ch, "CV": m_cv, "CA": m_ca,
         "EA": m_ea, "NF": m_nf, "CE": m_ce, "CO": m_co, "MI": m_mi, "RE": m_re}


def evaluate(X, y, params: Params, codes=None, ring_order=None, arena=None, timing=False) -> dict:
    codes = list(codes or CODES)
    ctx = Context(X, y, params, ring_order=ring_order, arena=arena)
    out = {}
    for c in codes:
        if c == "CEC":
            continue
        t0 = time.perf_counter()
        out[c] = FUNCS[c](ctx)
        if timing:
            out["t_" + c] = time.perf_counter() - t0
    if "CEC" in codes:
        t0 = time.perf_counter()
        out["CEC"] = m_cec(ctx, out.get("CE"), out.get("CO"))
        if timing:
            out["t_CEC"] = time.perf_counter() - t0 + out.get("t_CE", 0) + out.get("t_CO", 0)
    out["ARI"] = getattr(ctx, "ari", float("nan"))
    out["CE_crossings"] = getattr(ctx, "ce_crossings", -1)
    out["lmax_max"] = float(ctx.lmax.max()) if ctx.n else float("nan")
    return out


def flatten(res: dict) -> dict:
    row = {}
    for k, v in res.items():
        if isinstance(v, tuple):
            row[k + "_s"] = v[0]
            row[k + "_v"] = np.nan if v[1] is None else float(v[1])
        else:
            row[k] = v
    return row


def oriented(code: str, s):
    return np.asarray(s, float) * METRICS[code].direction
