from __future__ import annotations

from collections import deque

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import MultiPoint

_EPS_W = 1e-12


def _mst_dense(P: np.ndarray):
    n = len(P)
    D = squareform(pdist(P)) + _EPS_W
    np.fill_diagonal(D, 0.0)
    T = minimum_spanning_tree(D).tocoo()
    return np.c_[T.row, T.col].astype(int)


def delaunay_edges(P: np.ndarray, joggle_on_fail: bool = True) -> np.ndarray:
    try:
        tri = Delaunay(P)
    except QhullError:
        if not joggle_on_fail:
            raise
        tri = Delaunay(P, qhull_options="QJ")
    s = tri.simplices
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def emst(P: np.ndarray):
    P = np.asarray(P, float)
    n = len(P)
    if n < 2:
        return np.zeros((0, 2), int), np.zeros(0)
    E = None
    if n > 3:
        try:
            e = delaunay_edges(P, joggle_on_fail=False)
            if len(np.unique(e)) == n:
                w = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1) + _EPS_W
                T = minimum_spanning_tree(coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n))).tocoo()
                if T.nnz == n - 1:
                    E = np.c_[T.row, T.col].astype(int)
        except QhullError:
            E = None
    if E is None:
        E = _mst_dense(P)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    return E, L


def group_indices(y: np.ndarray, m: int | None = None):
    m = int(y.max()) + 1 if m is None else m
    return [np.flatnonzero(y == k) for k in range(m)]


def group_msts(X: np.ndarray, groups):
    out = []
    for idx in groups:
        E, L = emst(X[idx])
        out.append((idx[E] if len(E) else np.zeros((0, 2), int), L))
    return out


def _orient(a, b, c):
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def proper_cross(a, b, c, d) -> np.ndarray:
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def crossing_pairs(X: np.ndarray, edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if len(edges) < 2:
        return np.zeros((0, 2), int)
    A, B = X[edges[:, 0]], X[edges[:, 1]]
    geoms = shapely.linestrings(np.stack([A, B], axis=1))
    tree = shapely.STRtree(geoms)
    i, j = tree.query(geoms, predicate="intersects")
    keep = (i < j) & (labels[i] != labels[j])
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return np.zeros((0, 2), int)
    ok = proper_cross(A[i], B[i], A[j], B[j])
    return np.c_[i[ok], j[ok]]


def crossings_between_trees(X: np.ndarray, trees) -> tuple[int, np.ndarray]:
    edges = [e for e in trees if len(e)]
    if not edges:
        return 0, np.zeros(0, bool)
    E = np.vstack(edges)
    lab = np.concatenate([np.full(len(e), k) for k, e in enumerate(edges)])
    pairs = crossing_pairs(X, E, lab)
    mask = np.zeros(len(E), bool)
    if len(pairs):
        mask[pairs[:, 0]] = True
        mask[pairs[:, 1]] = True
    return len(pairs), mask


def convex_polygon(P: np.ndarray):
    return MultiPoint(np.asarray(P, float)).convex_hull


def bichromatic_hull_edges(P: np.ndarray, labels: np.ndarray) -> int:
    if len(P) < 3:
        return 0 if len(set(labels.tolist())) < 2 else 2
    try:
        v = ConvexHull(P).vertices
    except QhullError:
        return 0
    lv = labels[v]
    return int(np.sum(lv != np.roll(lv, -1)))


def bfs_tree(P: np.ndarray, rho: float):
    n = len(P)
    if n == 1:
        return np.zeros((0, 2), int)
    tree = cKDTree(P)
    nbrs = tree.query_ball_point(P, r=np.nextafter(rho, 0))
    seen = np.zeros(n, bool)
    seen[0] = True
    q = deque([0])
    edges = []
    while q:
        u = q.popleft()
        for v in sorted(nbrs[u], key=lambda w: (P[w, 0] - P[u, 0]) ** 2 + (P[w, 1] - P[u, 1]) ** 2):
            if not seen[v]:
                seen[v] = True
                edges.append((u, v))
                q.append(v)
    if not seen.all():
        return None
    return np.asarray(edges, int).reshape(-1, 2)


def nearest_neighbour_distances(P: np.ndarray) -> np.ndarray:
    d, _ = cKDTree(P).query(P, k=2)
    return d[:, 1]
