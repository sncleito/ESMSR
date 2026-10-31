# Port of grf_swarm/src/grf_swarm.cpp from github.com/verlab/2021-icra-grf-swarm (see README).
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .common import BaseController, coulomb_buckingham, saturation, wall_points

EPS, EPS0, R0, ALPHA = 0.04, 0.04, 0.8, 1.0


class Rezeck2021(BaseController):
    name = "rezeck2021"
    period = 0.02

    def __init__(self, robots, groups, world, seed, chain=100, **kw):
        super().__init__(robots, groups, world, seed)
        self.sensing, self.safezone = 0.5, 0.3
        self.mass, self.vmax, self.dt = 0.3, 1.0, 0.02
        self.chain, self.burnin = chain, int(0.6 * (chain + 1))

    def _energy(self, vc, I, J, same, pred, walls):
        n = len(self.q)
        p = self.q + vc * self.dt
        Us = np.zeros(n)
        for i, W in walls:
            d = np.hypot(*(p[i] - W).T) + 1e-9
            Us[i] = np.sum(coulomb_buckingham(d / 2.0, EPS, EPS0, R0, ALPHA, 16.0, 1.0))
        if len(I) == 0:
            my = np.sum(vc * vc, axis=1)
            return Us + 0.5 * 1e-9 * self.mass + 0.5 * self.mass * (self.vmax - my)
        d = np.hypot(*(p[I] - pred[J]).T) + 1e-9
        sign = np.where(same, 1.0, -1.0)
        skip = (~same) & (d > self.safezone)
        bad = same & (d > self.sensing)
        cb = np.where(skip, 0.0, coulomb_buckingham(d * 0.8, EPS, EPS0, R0, ALPHA, 16.0 * sign, -1.0))
        Ust = np.bincount(I, weights=cb, minlength=n)
        mid = same & (d < self.sensing) & (d > self.safezone)
        gv = np.zeros((n, 2))
        np.add.at(gv, I[mid], self.v[J[mid]] - vc[I[mid]])
        gv = saturation(gv, 1.0)
        gmass = self.mass + self.mass * np.bincount(I[mid], minlength=n)
        gspeed = np.sum(gv * gv, axis=1) + 1e-9
        my = np.sum(vc * vc, axis=1)
        Ust += 0.5 * gspeed * gmass + 0.5 * gmass * (self.vmax - my)
        anybad = np.bincount(I[bad], minlength=n) > 0
        Ust = np.where(anybad, 10000.0, Ust)
        return Us + Ust

    def step(self):
        n = len(self.q)
        pairs = cKDTree(self.q).query_pairs(self.sensing, output_type="ndarray")
        I = np.concatenate([pairs[:, 0], pairs[:, 1]]) if len(pairs) else np.zeros(0, int)
        J = np.concatenate([pairs[:, 1], pairs[:, 0]]) if len(pairs) else np.zeros(0, int)
        same = self.types[I] == self.types[J]
        pred = self.q + self.v * self.dt
        near = np.flatnonzero(np.any(np.abs(self.q) > self.world - self.safezone - 0.05, axis=1))
        walls = [(i, W) for i in near if len(W := wall_points(self.q[i], self.safezone, self.world))]
        cur = self.v.copy()
        U = self._energy(cur, I, J, same, pred, walls)
        acc, cnt = np.zeros((n, 2)), 0
        for t in range(1, self.chain + 1):
            prop = saturation(self.v + self.rng.normal(0.0, 0.2, size=(n, 2)), self.vmax)
            Up = self._energy(prop, I, J, same, pred, walls)
            dE = Up - U
            with np.errstate(over="ignore"):
                ok = (dE < 0) | (self.rng.uniform(size=n) < np.exp(-dE))
            cur = np.where(ok[:, None], prop, cur)
            U = np.where(ok, Up, U)
            if t >= self.burnin:
                acc += cur
                cnt += 1
        newv = saturation(acc / max(cnt, 1), self.vmax)
        self.v = newv
        self.q = self.q + newv * self.dt
