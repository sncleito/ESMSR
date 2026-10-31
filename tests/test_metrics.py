import numpy as np
import pytest

from segbench.certify import certify
from segbench.generators import generate
from segbench.geometry import (bichromatic_hull_edges, crossings_between_trees, emst,
                               group_indices, group_msts)
from segbench.metrics import CODES, Context, Params, cluster_count, evaluate, m_co

INVARIANT = [c for c in CODES if c not in ("MI", "RE", "KM", "CV")]


def _instances(cfg, fams=("F1", "F2", "F3", "F4", "F5", "F6", "F7"), seeds=(0, 1)):
    for f in fams:
        for lv in ([0, 1, 5] if f == "F6" else [0.0, 0.5, 1.0]):
            for s in seeds:
                yield generate(f, 3, 30, lv, s, cfg)


def test_emst_matches_dense():
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 10, (200, 2))
    E, L = emst(P)
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    assert np.isclose(L.sum(), minimum_spanning_tree(squareform(pdist(P))).sum())
    assert len(E) == len(P) - 1


def test_invariance_rigid_motion_and_labels(cfg, params):
    inst = generate("F4", 3, 30, 0.3, 4, cfg)
    base = evaluate(inst.X, inst.y, params, codes=INVARIANT)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    X2 = inst.X @ R.T + np.array([3.0, -1.0])
    perm = np.random.default_rng(1).permutation(inst.n)
    relabel = np.array([2, 0, 1])
    other = evaluate(X2[perm], relabel[inst.y[perm]], params, codes=INVARIANT)
    for c in INVARIANT:
        s0, v0 = base[c]
        s1, v1 = other[c]
        assert v0 == v1, c
        if np.isfinite(s0):
            assert np.isclose(s0, s1, rtol=1e-6, atol=1e-6), c


def test_prop1_hierarchy(cfg, params):
    for inst in _instances(cfg):
        r = evaluate(inst.X, inst.y, params, codes=["CA", "KM", "CH", "CV", "CE"])
        v = {c: r[c][1] for c in r if isinstance(r[c], tuple)}
        if v["CH"] is None:
            continue
        if v["CA"]:
            assert v["CH"]
        if v["KM"]:
            assert v["CH"]
        if v["CH"]:
            assert v["CV"] and v["CE"], inst.name


def test_prop2_certificate(cfg, params):
    for inst in _instances(cfg):
        lab, _ = certify(inst.X, inst.y, cfg.rho, inst.witness)
        r = evaluate(inst.X, inst.y, params, codes=["CE", "CO", "CEC"])
        if r["CEC"][1]:
            assert lab != 0, inst.name
        if lab == 1:
            assert r["CO"][1], inst.name


def test_cc_equals_co_at_rho(cfg, params):
    for inst in _instances(cfg, seeds=(3,)):
        ctx = Context(inst.X, inst.y, params)
        assert (cluster_count(ctx, np.nextafter(cfg.rho, 0)) == ctx.m) == m_co(ctx)[1]


def test_co_equals_zeroth_persistence(cfg):
    ripser = pytest.importorskip("ripser").ripser
    inst = generate("F2", 3, 30, 0.4, 2, cfg)
    for E_L, g in zip(group_msts(inst.X, group_indices(inst.y)), group_indices(inst.y)):
        d = ripser(inst.X[g], maxdim=0)["dgms"][0][:, 1]
        assert np.allclose(np.sort(d[np.isfinite(d)]), np.sort(E_L[1]))


def test_prop3_robustness(cfg, params):
    rng = np.random.default_rng(5)
    for inst in _instances(cfg, fams=("F1", "F7"), seeds=(0,)):
        ctx = Context(inst.X, inst.y, params)
        lm0 = ctx.lmax
        v0 = m_co(ctx)[1]
        for _ in range(20):
            delta = rng.uniform(0.01, 0.5)
            u = rng.normal(size=inst.X.shape)
            u *= (delta * rng.uniform(size=(inst.n, 1))) / np.linalg.norm(u, axis=1, keepdims=True)
            c2 = Context(inst.X + u, inst.y, params)
            assert np.all(np.abs(c2.lmax - lm0) <= 2 * delta + 1e-9)
            if np.min(np.abs(lm0 - params.rho)) > 2 * delta:
                assert m_co(c2)[1] == v0


def test_prop4_tokunaga_bound(cfg):
    for inst in _instances(cfg, seeds=(0, 1, 2)):
        G = group_indices(inst.y)
        T = [E for E, _ in group_msts(inst.X, G)]
        for k in range(len(G)):
            for l in range(k + 1, len(G)):
                sub = np.concatenate([G[k], G[l]])
                t = bichromatic_hull_edges(inst.X[sub], inst.y[sub])
                if t >= 4:
                    assert crossings_between_trees(inst.X, [T[k], T[l]])[0] >= t / 2 - 1


def test_concave_hull_not_rotation_invariant(cfg, params):
    inst = generate("F4", 3, 30, 0.3, 4, cfg)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s0 = evaluate(inst.X, inst.y, params, codes=["CV"])["CV"][0]
    s1 = evaluate(inst.X @ R.T, inst.y, params, codes=["CV"])["CV"][0]
    assert abs(s0 - s1) < 0.05
