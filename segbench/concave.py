from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from .geometry import proper_cross

TWO_PI = 2.0 * np.pi


def _moreira_try(pts: np.ndarray, k: int, scale: float):
    n = len(pts)
    first = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    hull = [first]
    current = first
    available = np.ones(n, bool)
    available[first] = False
    prev_angle = 0.0
    step = 2
    while (current != first or step == 2) and available.any():
        if step == 5:
            available[first] = True
        idx = np.flatnonzero(available)
        d = np.hypot(*(pts[idx] - pts[current]).T)
        sel = idx[np.argsort(d, kind="stable")[: min(k, len(idx))]]
        ang = np.arctan2(pts[sel, 1] - pts[current, 1], pts[sel, 0] - pts[current, 0])
        turn = np.mod(prev_angle - ang, TWO_PI)
        turn[turn <= 1e-12] = TWO_PI
        cand = sel[np.argsort(-turn, kind="stable")]
        chosen = None
        H = pts[hull]
        for c in cand:
            last = 1 if c == first else 0
            m = len(hull) - 2 - last
            if m > 0:
                a = H[last:last + m]
                b = H[last + 1:last + 1 + m]
                P0 = np.broadcast_to(pts[current], a.shape)
                P1 = np.broadcast_to(pts[c], a.shape)
                if np.any(proper_cross(P0, P1, a, b)):
                    continue
            chosen = int(c)
            break
        if chosen is None:
            return None
        current = chosen
        hull.append(current)
        prev = hull[-2]
        prev_angle = np.arctan2(pts[prev, 1] - pts[current, 1], pts[prev, 0] - pts[current, 0])
        available[current] = False
        step += 1
    if hull[-1] != first or len(hull) < 4:
        return None
    poly = Polygon(pts[hull[:-1]])
    if not poly.is_valid or poly.area <= 0:
        return None
    if not poly.buffer(1e-9 * scale).covers(MultiPoint(pts)):
        return None
    return poly


def moreira(P: np.ndarray, k: int = 3, max_k: int | None = None):
    pts = np.unique(np.asarray(P, float), axis=0)
    n = len(pts)
    if n < 3:
        return None, k
    if n == 3:
        poly = Polygon(pts)
        return (poly if poly.area > 0 else None), k
    scale = float(np.ptp(pts, axis=0).max())
    k = max(int(k), 3)
    max_k = n - 1 if max_k is None else min(max_k, n - 1)
    while k <= max_k:
        poly = _moreira_try(pts, k, scale)
        if poly is not None:
            return poly, k
        k += 1
    hull = MultiPoint(pts).convex_hull
    return (hull if hull.area > 0 else None), n


def parkoh(P: np.ndarray, concavity: float = 2.0):
    from concave_hull import concave_hull as _ch

    pts = np.asarray(P, float)
    if len(np.unique(pts, axis=0)) < 3:
        return None
    ring = np.asarray(_ch(pts, concavity=concavity))
    if len(ring) < 3:
        return None
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if poly.area > 0 else None


def geos(P: np.ndarray, ratio: float = 0.3):
    import shapely

    g = shapely.concave_hull(MultiPoint(np.asarray(P, float)), ratio=ratio)
    return g if g.geom_type == "Polygon" and g.area > 0 else None


def concave_hull(P: np.ndarray, algorithm: str = "moreira", k: int = 3, concavity: float = 2.0):
    if algorithm == "moreira":
        return moreira(P, k)[0]
    if algorithm == "parkoh":
        return parkoh(P, concavity)
    if algorithm == "geos":
        return geos(P)
    raise ValueError(f"unknown concave hull algorithm {algorithm!r}")
