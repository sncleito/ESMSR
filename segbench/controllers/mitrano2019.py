# Port of min_swarm/src/min_swarm.cpp from github.com/verlab/2021-icra-grf-swarm (see README).
from __future__ import annotations

import numpy as np

from .common import BaseController


def _wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


class Mitrano2019(BaseController):
    name = "mitrano2019"
    period = 0.02

    def __init__(self, robots, groups, world, seed, **kw):
        super().__init__(robots, groups, world, seed)
        self.vmax, self.l, self.r = 1.0, 0.1, 0.07
        self.half_beam = np.deg2rad(15.0)
        self.theta = np.arctan2(self.v[:, 1], self.v[:, 0])
        V, l = self.vmax, self.l
        self.motion = [((4 * V) / (3 * l), -l / 4), ((2 * V) / (3 * l), l), ((2 * V) / l, 0.0)]

    def _fwd(self, x, y, th, omega, R):
        dth = omega * self.period
        xc, yc = x + np.cos(th + np.pi / 2) * R, y + np.sin(th + np.pi / 2) * R
        nx = xc + np.cos(dth) * (x - xc) - np.sin(dth) * (y - yc)
        ny = yc + np.sin(dth) * (x - xc) + np.cos(dth) * (y - yc)
        return nx, ny, _wrap(th + dth)

    def step(self):
        qt, tht = self.q.copy(), self.theta.copy()
        n = len(qt)
        lim = self.world - 2 * self.r
        for i in range(n):
            rel = qt - qt[i]
            dist = np.hypot(rel[:, 0], rel[:, 1])
            ang = _wrap(np.arctan2(rel[:, 1], rel[:, 0]) - tht[i])
            vis = np.abs(ang) <= self.half_beam
            vis[i] = False
            S = 0
            if vis.any():
                j = np.flatnonzero(vis)[np.argmin(dist[vis])]
                S = 1 if self.types[j] == self.types[i] else 2
            omega, R = self.motion[S]
            x, y, th = self._fwd(qt[i, 0], qt[i, 1], tht[i], omega, R)
            x, y = min(max(x, -lim), lim), min(max(y, -lim), lim)
            d = np.hypot(self.q[:, 0] - x, self.q[:, 1] - y)
            d[i] = np.inf
            j = int(np.argmin(d))
            self.theta[i] = th
            if d[j] <= 2 * self.r:
                phi = np.arctan2(self.q[j, 1] - y, self.q[j, 0] - x)
                self.q[i] = self.q[j] + 2 * self.r * np.array([np.cos(phi + np.pi), np.sin(phi + np.pi)])
            else:
                self.q[i] = (x, y)
            self.v[i] = (np.cos(th), np.sin(th))
