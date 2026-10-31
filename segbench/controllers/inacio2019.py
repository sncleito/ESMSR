# Port of pso_swarm/src (agent.cpp, pso_swarm.cpp) from github.com/verlab/2021-icra-grf-swarm (see README).
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .common import BaseController

SCALE = 3.8


class Inacio2019(BaseController):
    name = "inacio2019"
    period = 0.02

    def __init__(self, robots, groups, world, seed, **kw):
        import rvo2

        super().__init__(robots, groups, world, seed)
        self.nd, self.maxn = 0.5, 10
        self.m = int(self.types.max()) + 1
        self.sim = rvo2.PyRVOSimulator(self.period, self.nd, self.maxn, 3.0, 3.0, 0.09, 1.0)
        for i in range(robots):
            self.sim.addAgent(tuple(self.q[i]))
            self.sim.setAgentVelocity(i, tuple(self.v[i]))
            self.sim.setAgentPrefVelocity(i, tuple(self.v[i]))
        n, m = robots, self.m
        self.mem_cnt = np.zeros((n, m), int)
        self.mem_pos = np.zeros((n, m, 2))
        self.oth_cnt = np.zeros(n, int)
        self.oth_pos = self.q.copy()
        self.empty = np.ones(n, int)
        self.gbest = self.q.copy()
        self.goal = self.q.copy()
        self.numn = np.zeros(n, int)
        self.gtime = 0.0
        self._crand = np.random.RandomState(seed)

    def _rand_coord(self):
        return (self._crand.randint(0, 2 ** 31 - 1) % 30) / SCALE - 15 / SCALE

    def step(self):
        n, m = len(self.q), self.m
        P = np.array([self.sim.getAgentPosition(i) for i in range(n)])
        V = np.array([self.sim.getAgentVelocity(i) for i in range(n)])
        tree = cKDTree(P)
        dists, nbrs = tree.query(P, k=min(self.maxn + 1, n), distance_upper_bound=self.nd)
        mem_cnt_prev, mem_pos_prev = self.mem_cnt.copy(), self.mem_pos.copy()
        numn_prev = self.numn.copy()
        fl = self.types
        near_thr = 0.5 / SCALE
        for i in range(n):
            neigh = [j for d, j in zip(np.atleast_1d(dists[i]), np.atleast_1d(nbrs[i]))
                     if j != i and j < n and np.isfinite(d)]
            numN, prev, rep, nrep, gbn = 0, V[i].copy(), np.zeros(2), 0, 0
            nm_cnt, nm_pos = np.zeros(m, int), np.zeros((m, 2))
            for j in neigh:
                d = np.linalg.norm(P[j] - P[i])
                if d > self.nd:
                    continue
                if fl[j] == fl[i] and d < self.nd:
                    numN += 1
                    if numn_prev[j] >= gbn:
                        gbn = numn_prev[j]
                        self.gbest[i] = P[j]
                else:
                    nm_cnt[fl[j]] += 1
                    nm_pos[fl[j]] = P[j]
                    if d < self.nd:
                        rep -= P[j] - P[i]
                        nrep += 1
                    oc, op = mem_cnt_prev[j, fl[i]], mem_pos_prev[j, fl[i]]
                    if oc > self.oth_cnt[i] and np.linalg.norm(op - P[i]) > self.nd:
                        self.empty[i] = 0
                        self.oth_cnt[i], self.oth_pos[i] = oc, op
                if nm_cnt[fl[j]] > self.mem_cnt[i, fl[j]]:
                    self.mem_cnt[i, fl[j]] = nm_cnt[fl[j]]
                    self.mem_pos[i, fl[j]] = nm_pos[fl[j]]
                upd = mem_cnt_prev[j] > self.mem_cnt[i]
                self.mem_cnt[i, upd] = mem_cnt_prev[j, upd]
                self.mem_pos[i, upd] = mem_pos_prev[j, upd]
            if np.linalg.norm(self.oth_pos[i] - P[i]) <= near_thr:
                self.empty[i], self.oth_cnt[i], self.oth_pos[i] = 1, 0, P[i]
            if int(self.gtime) % 50 == 0 or np.linalg.norm(self.gbest[i] - P[i]) <= near_thr:
                if numN == 0:
                    self.gbest[i] = (self._rand_coord(), self._rand_coord())
            goal = self.gbest[i].copy()
            if goal[0] < -16 / SCALE or goal[0] > 16 / SCALE:
                goal[0] = self._rand_coord()
            if goal[1] < -16 / SCALE or goal[1] > 16 / SCALE:
                goal[1] = self._rand_coord()
            gvec = goal - P[i]
            if nrep > 0:
                rep = rep / nrep
            om = self.oth_pos[i] - P[i]
            empty = self.empty[i] == 1
            if len(neigh) == 0:
                if np.linalg.norm(goal - P[i]) < near_thr:
                    goal = np.array([self._rand_coord(), self._rand_coord()])
                pref = gvec + prev if empty else om + prev
            elif numN == 0:
                if np.linalg.norm(om - P[i]) > self.nd and om[0] != 0.0 and om[1] != 0.0:
                    base = gvec if empty else om
                    pref = base if nrep < 5 else base + rep
                else:
                    if int(self.gtime) % 10 == 0:
                        goal = np.array([self._rand_coord(), self._rand_coord()])
                    if empty:
                        pref = gvec + prev if nrep < 5 else gvec + prev + rep
                    else:
                        pref = gvec + om + prev + rep
            else:
                w = 7 // numN
                pref = ((gvec + rep) if empty else (gvec + om + rep)) / numN + w * prev
            self.goal[i] = goal
            self.numn[i] = numN
            self.sim.setAgentPrefVelocity(i, (float(pref[0]), float(pref[1])))
        self.sim.doStep()
        self.gtime += self.period
        self.q = np.array([self.sim.getAgentPosition(i) for i in range(n)])
        self.v = np.array([self.sim.getAgentVelocity(i) for i in range(n)])
