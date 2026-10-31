# Port of vgs_swarm/src/vgs_swarm.cpp from github.com/verlab/2021-icra-grf-swarm (see README).
from __future__ import annotations

import numpy as np

from .common import BaseController, saturation


class Santos2020(BaseController):
    name = "santos2020"
    period = 0.02

    def __init__(self, robots, groups, world, seed, velocity_sign="paper", **kw):
        super().__init__(robots, groups, world, seed)
        self.dt = 0.001
        self.substeps = int(round(self.period / self.dt))
        self.vmax = 1.0
        self.alpha = 50.0
        self.dAB = 3.5
        self.dAA = 0.3 if (robots < 100 or groups > 10) else 1.0
        self.rel_sign = -1.0 if velocity_sign == "paper" else 1.0
        same = self.types[:, None] == self.types[None, :]
        self.dij = np.where(same, self.dAA, self.dAB)
        self.off = ~np.eye(robots, dtype=bool)

    def step(self):
        for _ in range(self.substeps):
            q, v = self.q, self.v
            dx = q[:, None, :] - q[None, :, :]
            dsqr = np.sum(dx * dx, axis=2)
            np.fill_diagonal(dsqr, 1.0)
            dist = np.sqrt(dsqr)
            dU = self.alpha * (dist - self.dij + 1.0 / dist - self.dij / dsqr)
            dU[~self.off] = 0.0
            a = -np.sum((dU / dist)[:, :, None] * dx, axis=1)
            dv = v[:, None, :] - v[None, :, :]
            a += self.rel_sign * np.sum(dv * self.off[:, :, None], axis=1)
            disp = saturation(self.dt * v + 0.5 * self.dt * self.dt * a, self.vmax)
            self.q = q + disp
            self.v = saturation(v + a * self.dt, self.vmax)
