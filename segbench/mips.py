from __future__ import annotations

from functools import lru_cache

import numpy as np


def _integral(P, x0, y0, res, nb):
    H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=nb,
                             range=[[x0, x0 + nb * res], [y0, y0 + nb * res]])
    S = np.zeros((nb + 1, nb + 1))
    S[1:, 1:] = H.cumsum(0).cumsum(1)
    return S


def raw_index(P: np.ndarray, arena, wsizes, res) -> float:
    x0, y0, L = arena
    nb = max(1, int(np.ceil(L / res)))
    S = _integral(P, x0, y0, res, nb)
    out = []
    for w in wsizes:
        c = max(1, int(round(w / res)))
        stride = max(1, c // 2)
        st = np.arange(0, nb - c + 1, stride)
        if len(st) == 0:
            continue
        a, b = st[:, None], st[None, :]
        cnt = S[a + c, b + c] - S[a, b + c] - S[a + c, b] + S[a, b]
        out.append(np.var(cnt / (c * res) ** 2))
    return float(np.mean(out)) if out else 0.0


def _hex_cluster(n, spacing):
    pts = []
    rings = 0
    while len(pts) < n:
        for i in range(-rings, rings + 1):
            for j in range(-rings, rings + 1):
                if max(abs(i), abs(j), abs(i + j)) == rings:
                    pts.append((i + 0.5 * j, j * np.sqrt(3) / 2))
        rings += 1
    pts = np.asarray(pts[:n]) * spacing
    return pts - pts.mean(0)


@lru_cache(maxsize=4096)
def _bounds(n, L, r, wkey, res, n_uniform, seed):
    wsizes = np.asarray(wkey)
    arena = (0.0, 0.0, L)
    rng = np.random.default_rng(seed)
    lo = np.mean([raw_index(rng.uniform(0, L, (n, 2)), arena, wsizes, res) for _ in range(n_uniform)])
    hi = raw_index(_hex_cluster(n, 2 * r) + L / 2, arena, wsizes, res)
    return lo, hi


def window_sizes(L, r, mcfg) -> np.ndarray:
    wmin = mcfg["w_min_factor"] * r
    wmax = max(wmin * 1.01, mcfg["w_max_frac"] * L)
    return np.linspace(wmin, wmax, int(mcfg["n_windows"]))


def mips_index(P: np.ndarray, arena, r: float, mcfg: dict, seed: int = 0) -> float:
    x0, y0, L = arena
    res = mcfg["grid_res_factor"] * r
    ws = window_sizes(L, r, mcfg)
    raw = raw_index(P, arena, ws, res)
    lo, hi = _bounds(len(P), round(float(L), 6), float(r), tuple(np.round(ws, 9)), float(res),
                     int(mcfg["n_uniform"]), int(seed))
    if hi - lo <= 0:
        return float("nan")
    return float((raw - lo) / (hi - lo))
