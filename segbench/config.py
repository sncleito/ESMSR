from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    @property
    def r(self) -> float:
        return float(self["robot_radius"])

    @property
    def rho(self) -> float:
        return float(self["rho_factor"]) * self.r

    @property
    def tau(self) -> float:
        return float(self["cc_tau_factor"]) * self.r

    @property
    def spacing(self) -> float:
        return float(self["spacing_factor"]) * self.r


def load_config(path=None, quick: bool = False) -> Config:
    base = yaml.safe_load(open(path or ROOT / "configs" / "default.yaml"))
    if quick:
        base = _merge(base, yaml.safe_load(open(ROOT / "configs" / "quick.yaml")))
    return Config(base)


def paths(cfg: Config) -> dict:
    p = {k: ROOT / v for k, v in cfg["paths"].items()}
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def add_common_args(parser):
    parser.add_argument("--config", default=None, help="base YAML configuration")
    parser.add_argument("--quick", action="store_true", help="use configs/quick.yaml")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers (joblib)")
    return parser


def config_from_args(args) -> Config:
    return load_config(args.config, quick=args.quick)
