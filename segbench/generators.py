from __future__ import annotations

import numpy as np

from .geometry import bfs_tree
from .instance import Instance


class Sampler:
    def __init__(self, min_dist: float, rng: np.random.Generator):
        self.d2 = min_dist ** 2
        self.rng = rng
        self.P = np.zeros((0, 2))
        self.y = np.zeros(0, int)

    def ok(self, p) -> bool:
        return len(self.P) == 0 or np.min(np.sum((self.P - p) ** 2, axis=1)) >= self.d2

    def add(self, p, g):
        self.P = np.vstack([self.P, p])
        self.y = np.append(self.y, g)

    def fill(self, draw, count: int, group: int, max_tries: int = 2000):
        added, tries = 0, 0
        while added < count:
            p = draw(self.rng)
            tries += 1
            if self.ok(p):
                self.add(p, group)
                added += 1
                tries = 0
            elif tries > max_tries:
                raise RuntimeError("sampling failed: region too dense")
        return self


def _disk_draw(c, R):
    def draw(rng):
        a = rng.uniform(0, 2 * np.pi)
        rad = R * np.sqrt(rng.uniform())
        return np.array([c[0] + rad * np.cos(a), c[1] + rad * np.sin(a)])
    return draw


def ring_centers(m: int, D: float) -> np.ndarray:
    if m == 1 or D <= 0:
        return np.zeros((m, 2))
    if m == 2:
        return np.array([[-D / 2, 0.0], [D / 2, 0.0]])
    C = D / (2 * np.sin(np.pi / m))
    a = 2 * np.pi * np.arange(m) / m
    return np.c_[C * np.cos(a), C * np.sin(a)]


def _bfs_witness(P, y, m, rho):
    out = []
    for k in range(m):
        idx = np.flatnonzero(y == k)
        E = bfs_tree(P[idx], rho)
        out.append(None if E is None else (idx[E] if len(E) else np.zeros((0, 2), int)))
    return out


def _path_witness(idx, key):
    order = idx[np.argsort(key)]
    return np.c_[order[:-1], order[1:]] if len(order) > 1 else np.zeros((0, 2), int)


def _finish(S, cfg, family, lam, m, nk, seed, witness, ring_order=None, meta=None):
    name = f"{family}_m{m}_n{nk}_l{lam:.3f}_s{seed}"
    return Instance(X=S.P.copy(), y=S.y.copy(), name=name, family=family, lam=float(lam), m=m,
                    nk=nk, seed=seed, ring_order=ring_order, witness=witness, meta=meta or {})


def gen_F1(m, nk, lam, seed, cfg) -> Instance:
    rng = np.random.default_rng(seed)
    r, rho, s = cfg.r, cfg.rho, cfg.spacing
    R = s * np.sqrt(nk / np.pi)
    D0 = 2 * R + 1.5 * rho
    if lam <= 0.5:
        D, Rk = D0 - (lam / 0.5) * (D0 - 2 * R), R
    else:
        ov = (lam - 0.5) / 0.5
        D, Rk = 2 * R * (1 - ov), R * np.sqrt(1 + (m - 1) * ov)
    C = ring_centers(m, D)
    S = Sampler(2 * r, rng)
    for k in range(m):
        S.fill(_disk_draw(C[k], Rk), nk, k)
    W = _bfs_witness(S.P, S.y, m, rho)
    return _finish(S, cfg, "F1", lam, m, nk, seed, W, meta=dict(centers=C, R=Rk, D=D))


def gen_F2(m, nk, lam, seed, cfg) -> Instance:
    rng = np.random.default_rng(seed)
    r, rho, s = cfg.r, cfg.rho, cfg.spacing
    Rt = s * np.sqrt(m * nk / np.pi)
    width = 2 * np.pi / m
    S = Sampler(2 * r, rng)
    for k in range(m):
        def draw(g, k=k):
            a = k * width + g.uniform(0, width) + g.uniform(-1, 1) * lam * np.pi / m
            rad = Rt * np.sqrt(g.uniform())
            return np.array([rad * np.cos(a), rad * np.sin(a)])
        S.fill(draw, nk, k)
    W = _bfs_witness(S.P, S.y, m, rho)
    return _finish(S, cfg, "F2", lam, m, nk, seed, W, meta=dict(R=Rt))


class InfeasibleConfiguration(RuntimeError):
    pass


def _bands(m, nk, cfg, inner_radius, frac=1.0):
    r, s = cfg.r, cfg.spacing
    g = cfg["ring_gap_factor"] * r
    bands = []
    a = inner_radius
    for k in range(m):
        if k == 0 and inner_radius == 0.0:
            b = s * np.sqrt(nk / (np.pi * frac))
        else:
            b = np.sqrt(a * a + nk * s * s / (np.pi * frac))
        b = max(b, a + 2.5 * r)
        bands.append((a, b))
        a = b + g
    return bands


ANGLE_JITTER = 0.15


def _is_thin(a, b, rho):
    return (b - a) <= rho / 2


def _check_ring_feasible(bands, nk, span, rho, skip_first):
    for k, (a, b) in enumerate(bands):
        if (skip_first and k == 0) or not _is_thin(a, b, rho):
            continue
        arc = span * b / nk * (1 + 2 * ANGLE_JITTER)
        if np.hypot(arc, b - a) >= 0.95 * rho:
            raise InfeasibleConfiguration(f"ring {k} cannot be cohesive (gap {arc:.2f} m)")


def _fill_ring(S, k, nk, a, b, start, span, jitter_amp, rho, max_tries=4000):
    for i in range(nk):
        for t in range(max_tries):
            u = (i + 0.5 + S.rng.uniform(-ANGLE_JITTER, ANGLE_JITTER)) / nk
            th = start + u * span
            rad = abs(np.sqrt(S.rng.uniform(a * a, b * b)) + S.rng.uniform(-1, 1) * jitter_amp)
            p = np.array([rad * np.cos(th), rad * np.sin(th)])
            if S.ok(p):
                S.add(p, k)
                break
        else:
            raise RuntimeError("sampling failed: ring too dense")


def gen_F3(m, nk, lam, seed, cfg) -> Instance:
    rng = np.random.default_rng(seed)
    r, rho = cfg.r, cfg.rho
    bands = _bands(m, nk, cfg, 0.0)
    _check_ring_feasible(bands, nk, 2 * np.pi, rho, skip_first=True)
    mids = np.array([(a + b) / 2 for a, b in bands])
    spacing = float(np.mean(np.diff(mids))) if m > 1 else bands[0][1]
    S = Sampler(2 * r, rng)
    a0, b0 = bands[0]
    S.fill(_disk_draw((0.0, 0.0), b0), nk, 0)
    if lam > 0:
        idx0 = np.flatnonzero(S.y == 0)
        S.P[idx0] *= (1 + rng.uniform(-1, 1, (len(idx0), 1)) * lam * spacing / max(b0, 1e-9))
    for k in range(1, m):
        a, b = bands[k]
        _fill_ring(S, k, nk, a, b, rng.uniform(0, 2 * np.pi), 2 * np.pi, lam * spacing, rho)
    W = []
    for k in range(m):
        idx = np.flatnonzero(S.y == k)
        if k == 0:
            E = bfs_tree(S.P[idx], rho)
            W.append(None if E is None else (idx[E] if len(E) else np.zeros((0, 2), int)))
        else:
            W.append(_path_witness(idx, np.mod(np.arctan2(S.P[idx, 1], S.P[idx, 0]), 2 * np.pi)))
    return _finish(S, cfg, "F3", lam, m, nk, seed, W, ring_order=list(range(m)),
                   meta=dict(bands=bands))


def gen_F4(m, nk, lam, seed, cfg) -> Instance:
    rng = np.random.default_rng(seed)
    r, rho = cfg.r, cfg.rho
    psi = np.deg2rad(cfg["arc_opening_deg"])
    span = 2 * np.pi - psi
    bands = _bands(m, nk, cfg, 2 * cfg.spacing, frac=span / (2 * np.pi))
    _check_ring_feasible(bands, nk, span, rho, skip_first=False)
    mids = np.array([(a + b) / 2 for a, b in bands])
    spacing = float(np.mean(np.diff(mids))) if m > 1 else bands[0][1] - bands[0][0]
    S = Sampler(2 * r, rng)
    W = []
    for k, (a, b) in enumerate(bands):
        phi = 0.0 if k % 2 == 0 else np.pi
        start = phi + psi / 2
        _fill_ring(S, k, nk, a, b, start, span, lam * spacing, rho)
        idx = np.flatnonzero(S.y == k)
        th = np.arctan2(S.P[idx, 1], S.P[idx, 0])
        W.append(_path_witness(idx, np.mod(th - start, 2 * np.pi)))
    return _finish(S, cfg, "F4", lam, m, nk, seed, W, meta=dict(bands=bands, opening=psi))


def gen_F5(m, nk, lam, seed, cfg) -> Instance:
    base = gen_F1(m, nk, 0.0, seed, cfg)
    rng = np.random.default_rng(seed + 10_000)
    y = base.y.copy()
    q = int(round(lam * len(y)))
    if q >= 2:
        sel = rng.choice(len(y), q, replace=False)
        y[sel] = y[sel][rng.permutation(q)]
    W = base.witness if q < 2 else None
    inst = Instance(X=base.X, y=y, name=f"F5_m{m}_n{nk}_l{lam:.3f}_s{seed}", family="F5",
                    lam=float(lam), m=m, nk=nk, seed=seed, witness=W, meta=dict(base.meta))
    return inst


def gen_F6(m, nk, q, seed, cfg) -> Instance:
    base = gen_F1(m, nk, 0.0, seed, cfg)
    rng = np.random.default_rng(seed + 20_000)
    C, R = base.meta["centers"], base.meta["R"]
    X, y = base.X.copy(), base.y.copy()
    moved = np.concatenate([rng.choice(np.flatnonzero(y == k), q, replace=False) for k in range(m)])
    keep = np.setdiff1d(np.arange(len(X)), moved)
    S = Sampler(2 * cfg.r, rng)
    S.P, S.y = X[keep], y[keep]
    for i in moved:
        k = y[i]
        tgt = (k + 1) % m
        S.fill(_disk_draw(C[tgt], R), 1, k)
    return _finish(S, cfg, "F6", q / nk, m, nk, seed, None, meta=dict(base.meta, q=int(q)))


def gen_F7(m, nk, lam, seed, cfg) -> Instance:
    base = gen_F1(m, nk, 0.0, seed, cfg)
    rng = np.random.default_rng(seed + 30_000)
    C, R, rho, r = base.meta["centers"], base.meta["R"], cfg.rho, cfg.r
    X, y = base.X.copy(), base.y.copy()
    q = int(round(lam * nk))
    if lam > 0:
        q = max(q, 1)
    q = min(q, nk - 1)
    if q == 0:
        return _finish_from(base, "F7", lam)
    moved = np.concatenate([rng.choice(np.flatnonzero(y == k), q, replace=False) for k in range(m)])
    keep = np.setdiff1d(np.arange(len(X)), moved)
    S = Sampler(2 * r, rng)
    S.P, S.y = X[keep], y[keep]
    centre = C.mean(0)
    for i in moved:
        k = y[i]
        u = C[k] - centre
        base_ang = np.arctan2(u[1], u[0]) if np.linalg.norm(u) > 1e-9 else rng.uniform(0, 2 * np.pi)

        def draw(gen, k=k, base_ang=base_ang):
            for _ in range(1000):
                a = base_ang + gen.uniform(-np.pi / 3, np.pi / 3)
                d = R + rho * (1.05 + gen.uniform(0, 1))
                p = C[k] + d * np.array([np.cos(a), np.sin(a)])
                if all(np.linalg.norm(p - C[j]) > R + 2 * r for j in range(m) if j != k):
                    return p
            return p
        S.fill(draw, 1, k)
    return _finish(S, cfg, "F7", lam, m, nk, seed, None, meta=dict(base.meta, q=int(q)))


def _finish_from(base, family, lam):
    return Instance(X=base.X, y=base.y, name=base.name.replace("F1", family), family=family,
                    lam=float(lam), m=base.m, nk=base.nk, seed=base.seed, witness=base.witness,
                    meta=dict(base.meta))


GENERATORS = {"F1": gen_F1, "F2": gen_F2, "F3": gen_F3, "F4": gen_F4, "F5": gen_F5, "F6": gen_F6,
              "F7": gen_F7}
FAMILY_NAMES = {"F1": "Clusters", "F2": "Sectors", "F3": "Rings", "F4": "Interleaved arcs",
                "F5": "Relabelling", "F6": "Trapped agents", "F7": "Stragglers"}


def family_levels(family: str, cfg):
    if family == "F6":
        return [int(q) for q in cfg["benchmark"]["f6_q"]]
    return [float(l) for l in cfg["benchmark"]["lambdas"]]


def generate(family, m, nk, level, seed, cfg) -> Instance:
    return GENERATORS[family](m, nk, level, seed, cfg)
