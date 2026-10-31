import numpy as np
import pytest
from shapely.geometry import MultiPoint

from segbench.canonical import SCENARIOS, load_canonical
from segbench.certify import certify
from segbench.concave import moreira
from segbench.config import ROOT
from segbench.generators import InfeasibleConfiguration, generate


@pytest.mark.parametrize("fam", ["F1", "F2", "F3", "F4"])
def test_families_segregated_at_lambda0(cfg, fam):
    inst = generate(fam, 3, 30, 0.0, 0, cfg)
    assert certify(inst.X, inst.y, cfg.rho, inst.witness)[0] == 1
    d = np.linalg.norm(inst.X[:, None] - inst.X[None], axis=2) + np.eye(inst.n) * 1e9
    assert d.min() >= 2 * cfg.r - 1e-9


def test_trapped_and_stragglers_not_segregated(cfg):
    for fam, lv in [("F6", 1), ("F7", 0.2)]:
        inst = generate(fam, 3, 30, lv, 0, cfg)
        assert certify(inst.X, inst.y, cfg.rho, inst.witness)[0] == 0


def test_infeasible_rings_flagged(cfg):
    with pytest.raises(InfeasibleConfiguration):
        generate("F3", 5, 10, 0.0, 0, cfg)


def test_moreira_valid_and_covering():
    rng = np.random.default_rng(3)
    t = rng.uniform(0, 1.5 * np.pi, 150)
    P = np.c_[np.cos(t), np.sin(t)] * rng.uniform(0.8, 1.0, (150, 1))
    poly, k = moreira(P, 3)
    assert poly.is_valid and poly.buffer(1e-9).covers(MultiPoint(P))
    assert poly.area < MultiPoint(P).convex_hull.area


def test_canonical_counts(cfg):
    canon = load_canonical(cfg, ROOT / "data")
    assert len(canon) == len(SCENARIOS)
    for inst in canon:
        counts = np.bincount(inst.y)
        assert len(set(counts)) == 1
        assert inst.n in (25, 100, 250)


@pytest.mark.parametrize("name", ["santos2020", "mitrano2019", "rezeck2021", "gradient", "inacio2019"])
def test_controllers_run(name):
    from segbench.controllers.common import simulate
    from segbench.controllers.registry import CONTROLLERS, make
    if name not in CONTROLLERS:
        pytest.skip("rvo2 not installed")
    c = make(name, 30, 3, 5.0, 0)
    t, traj = simulate(c, 0.2, 0.1)
    assert np.isfinite(traj).all() and traj.shape[1:] == (30, 2)
