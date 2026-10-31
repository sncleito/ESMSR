from __future__ import annotations

import numpy as np


def perturb(X: np.ndarray, sigma: float, fraction: float, rng: np.random.Generator):
    X = np.asarray(X, float)
    n = len(X)
    q = int(round(fraction * n))
    mask = np.zeros(n, bool)
    if q > 0 and sigma > 0:
        mask[rng.choice(n, q, replace=False)] = True
    Xn = X.copy()
    Xn[mask] += rng.normal(0.0, sigma, size=(int(mask.sum()), 2))
    disp = np.linalg.norm(Xn - X, axis=1)
    return Xn, float(disp.max()) if n else 0.0, mask


def delta_bound(sigma: float, q: int, alpha: float = 0.05) -> float:
    if q <= 0 or sigma <= 0:
        return 0.0
    return float(sigma * np.sqrt(2.0 * np.log(q / alpha)))
