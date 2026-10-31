from __future__ import annotations

import numpy as np

from .geometry import (bfs_tree, bichromatic_hull_edges, crossings_between_trees,
                       group_indices, group_msts)


def valid_witness(X, groups, trees, rho) -> bool:
    if trees is None or any(t is None for t in trees):
        return False
    for idx, E in zip(groups, trees):
        E = np.asarray(E, int).reshape(-1, 2)
        if len(E) != len(idx) - 1:
            return False
        if len(E) and np.any(np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) >= rho):
            return False
        if len(E):
            parent = {int(v): int(v) for v in idx}

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            for a, b in E:
                parent[find(int(a))] = find(int(b))
            if len({find(int(v)) for v in idx}) != 1:
                return False
    ncross, _ = crossings_between_trees(X, [np.asarray(t, int).reshape(-1, 2) for t in trees])
    return ncross == 0


def bfs_witness(X, groups, rho):
    out = []
    for idx in groups:
        E = bfs_tree(X[idx], rho)
        if E is None:
            return None
        out.append(idx[E] if len(E) else np.zeros((0, 2), int))
    return out


def certify(X, y, rho, witness=None, return_details=False):
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = group_indices(y)
    m = len(groups)
    det = {}
    lmax = np.array([L.max() if len(L) else 0.0 for _, L in group_msts(X, groups)])
    det["cohesive"] = bool(np.all(lmax < rho))
    tau_max = 0
    for k in range(m):
        for l in range(k + 1, m):
            sel = np.concatenate([groups[k], groups[l]])
            tau_max = max(tau_max, bichromatic_hull_edges(X[sel], y[sel]))
    det["tau_max"] = int(tau_max)
    pos_reason = None
    if witness is not None and valid_witness(X, groups, witness, rho):
        pos_reason = "witness-region"
    else:
        bw = bfs_witness(X, groups, rho)
        if bw is not None and valid_witness(X, groups, bw, rho):
            pos_reason = "witness-bfs"
        elif witness is not None and bw is not None:
            hyb = [w if w is not None else b for w, b in zip(witness, bw)]
            if valid_witness(X, groups, hyb, rho):
                pos_reason = "witness-hybrid"
    neg_reason = None
    if not det["cohesive"]:
        neg_reason = "cohesion"
    elif tau_max >= 4:
        neg_reason = "tokunaga"
    if pos_reason and neg_reason:
        label, reason = None, f"conflict({pos_reason},{neg_reason})"
    elif pos_reason:
        label, reason = 1, pos_reason
    elif neg_reason:
        label, reason = 0, neg_reason
    else:
        label, reason = None, "indeterminate"
    return (label, reason, det) if return_details else (label, reason)
