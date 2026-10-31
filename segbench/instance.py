from __future__ import annotations

import gzip
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Instance:
    X: np.ndarray
    y: np.ndarray
    name: str
    family: str = ""
    lam: float = 0.0
    m: int = 0
    nk: int = 0
    seed: int = 0
    ring_order: list | None = None
    witness: list | None = None
    label: int | None = None
    label_reason: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.X = np.asarray(self.X, float)
        self.y = np.asarray(self.y, int)
        if not self.m:
            self.m = int(self.y.max()) + 1
        if not self.nk:
            self.nk = int(np.bincount(self.y).max())

    @property
    def n(self) -> int:
        return len(self.X)

    def arena(self, margin: float):
        lo = self.X.min(0) - margin
        hi = self.X.max(0) + margin
        L = float((hi - lo).max())
        c = (lo + hi) / 2
        return float(c[0] - L / 2), float(c[1] - L / 2), L

    def to_dict(self) -> dict:
        d = asdict(self)
        d["X"] = self.X.astype(np.float32)
        d["y"] = self.y.astype(np.int16)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        return Instance(**d)


def save_instances(instances, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump([i.to_dict() for i in instances], f, protocol=pickle.HIGHEST_PROTOCOL)


def load_instances(path: Path):
    with gzip.open(path, "rb") as f:
        return [Instance.from_dict(d) for d in pickle.load(f)]
