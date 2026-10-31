# Port of grad_swarm/src/grad_swarm.cpp from github.com/verlab/2021-icra-grf-swarm (see README).
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .common import BaseController, saturation, wall_points

EPS, EPS0, R0, ALPHA = 0.04, 0.04, 0.8, 1.0


def _dU(r, q1q2):
    return 6.0 * EPS / (ALPHA - 6.0) * (ALPHA * R0 ** 6 / r ** 7 - np.exp(ALPHA) / R0) \
        - q1q2 / (4 * np.pi * EPS0 * r)


class Gradient(BaseController):
    name = "gradient"
    period = 0.02

    def __init__(self, robots, groups, world, seed, **kw):
        super().__init__(robots, groups, world, seed)
        self.sensing, self.safezone = 0.5, 0.3
        self.mass, self.vmax, self.dt = 100.0, 1.0, 0.02
        self.acc = np.zeros_like(self.q)

    def step(self):
        n = len(self.q)
        q, v, acc = self.q, self.v, self.acc
        pq = q + self.dt * v
        a = np.zeros((n, 2))
        near = np.flatnonzero(np.any(np.abs(q) > self.world - self.safezone - 0.05, axis=1))
        for i in near:
            W = wall_points(q[i], self.safezone, self.world)
            if len(W):
                dx = pq[i] - W
                r = np.hypot(dx[:, 0], dx[:, 1]) + 1e-9
                a[i] += -np.sum((_dU(r, 1.0) / r)[:, None] * dx, axis=0)
        pairs = cKDTree(q).query_pairs(self.sensing, output_type="ndarray")
        if len(pairs):
            I = np.concatenate([pairs[:, 0], pairs[:, 1]])
            J = np.concatenate([pairs[:, 1], pairs[:, 0]])
            dx = pq[I] - q[J]
            r = np.hypot(dx[:, 0], dx[:, 1]) + 1e-9
            Isg = np.where(self.types[I] == self.types[J], 1.0, -1.0)
            keep = ~((Isg < 0) & (r > self.safezone))
            dUst = 6.0 * EPS / (ALPHA - 6.0) * (ALPHA * R0 ** 6 / (r * 1.2) ** 7 - np.exp(ALPHA) / R0) \
                - (16.0 * Isg * -1.0) / (4 * np.pi * EPS0 * r * 1.2)
            contrib = -(dUst / r)[:, None] * dx * keep[:, None]
            np.add.at(a, I, contrib)
            mid = (Isg > 0) & (r < self.sensing) & (r > self.safezone)
            gv = np.zeros((n, 2))
            np.add.at(gv, I[mid], acc[J[mid]] - acc[I[mid]])
            gmass = self.mass + self.mass * np.bincount(I[mid], minlength=n)
        else:
            gv, gmass = np.zeros((n, 2)), np.full(n, self.mass)
        gv = saturation(gv, 1.0)
        gs = np.sum(gv * gv, axis=1) + 1e-9
        ms = np.sum(acc * acc, axis=1) + 1e-9
        a += -((gmass * gs) / gs)[:, None] * gv
        a += -((gmass * (self.vmax - ms)) / ms)[:, None] * acc
        a = 0.1 * a + 0.9 * acc
        self.acc = saturation(a, 1.0)
        self.v = saturation(v + a * self.dt, self.vmax)
        self.q = q + self.dt * self.v
