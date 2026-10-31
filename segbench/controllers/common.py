# Port of the shared code of the controllers from github.com/verlab/2021-icra-grf-swarm (see README).
from __future__ import annotations

import numpy as np


def initial_states(robots: int, groups: int, world: float, seed: int):
    rs = np.random.RandomState(seed)
    q = 1.8 * world * (rs.uniform(0.0, 1.0, size=(robots, 2)) - 0.5)
    v = 1.8 * world * (rs.uniform(0.0, 1.0, size=(robots, 2)) - 0.5)
    r = robots / groups
    types = np.floor(np.arange(robots) / r).astype(int)
    return q, v, types


def saturation(v: np.ndarray, norm: float) -> np.ndarray:
    v = np.asarray(v, float)
    x, y = v[..., 0], v[..., 1]
    out = v.copy()
    ax, ay = np.abs(x), np.abs(y)
    big_x = ax > 1.0e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(big_x, ay / np.where(big_x, ax, 1.0), np.inf)
        c1 = big_x & (r >= 1.0) & (ay > norm)
        c2 = big_x & (r < 1.0) & (ax > norm)
        out[..., 1] = np.where(c1, y * norm / np.where(ay > 0, ay, 1.0), out[..., 1])
        out[..., 0] = np.where(c1, x * norm / np.where(r * ax > 0, r * ax, 1.0), out[..., 0])
        out[..., 0] = np.where(c2, x * norm / np.where(ax > 0, ax, 1.0), out[..., 0])
        out[..., 1] = np.where(c2, np.where(ay > 0, y * norm * r / np.where(ay > 0, ay, 1.0), 0.0),
                               out[..., 1])
    return out


def cluster_metric(q: np.ndarray, types: np.ndarray, threshold: float = 0.3) -> int:
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    total = 0
    for k in np.unique(types):
        P = q[types == k]
        if len(P) == 1:
            total += 1
            continue
        pairs = cKDTree(P).query_pairs(threshold, output_type="ndarray")
        from scipy.sparse import coo_matrix
        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
        total += connected_components(A, directed=False)[0]
    return int(total)


def wall_points(p: np.ndarray, radius: float, world: float, res: float = 0.1):
    pts = []
    x, y = p
    for wy in (-world, world):
        dy = wy - y
        if radius * radius - dy * dy > 0:
            h = np.sqrt(radius * radius - dy * dy)
            xs = [x - h, x + h] + list(np.arange(min(x - h, x + h) + res, max(x - h, x + h), res))
            pts += [(xx, wy) for xx in xs]
    for wx in (-world, world):
        dx = wx - x
        if radius * radius - dx * dx > 0:
            h = np.sqrt(radius * radius - dx * dx)
            ys = [y - h, y + h] + list(np.arange(min(y - h, y + h) + res, max(y - h, y + h), res))
            pts += [(wx, yy) for yy in ys]
    return np.asarray(pts, float).reshape(-1, 2)


def coulomb_buckingham(r, eps, eps0, r0, alpha, q1, q2):
    return eps * ((6.0 / (alpha - 6.0)) * np.exp(alpha) * (1.0 - r / r0)
                  - (alpha / (alpha - 6.0)) * np.power(r0 / r, 6)) + (q1 * q2) / (4.0 * np.pi * eps0 * r)


class BaseController:
    name = "base"
    period = 0.02

    def __init__(self, robots, groups, world, seed, **kw):
        self.q, self.v, self.types = initial_states(robots, groups, world, seed)
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

    def step(self):
        raise NotImplementedError

    @property
    def positions(self) -> np.ndarray:
        return self.q.copy()


def simulate(ctrl: BaseController, sim_time: float, record_every: float, progress=None):
    steps = int(round(sim_time / ctrl.period))
    every = max(1, int(round(record_every / ctrl.period)))
    times, traj = [0.0], [ctrl.positions]
    for s in range(1, steps + 1):
        ctrl.step()
        ctrl.t += ctrl.period
        if s % every == 0:
            times.append(ctrl.t)
            traj.append(ctrl.positions)
            if progress:
                progress(s / steps)
    return np.asarray(times), np.asarray(traj)
