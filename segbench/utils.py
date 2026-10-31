from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_T0 = time.time()


def log(msg: str):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def pmap(func, items, jobs: int = -1, desc: str = ""):
    items = list(items)
    if desc:
        log(f"{desc}: {len(items)} tasks")
    if jobs == 1 or len(items) < 2:
        return [func(x) for x in items]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=jobs, batch_size="auto", verbose=0)(delayed(func)(x) for x in items)


def stable_seed(*parts) -> int:
    import hashlib

    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def require(path: Path, hint: str):
    if not Path(path).exists():
        sys.exit(f"missing {path}. Run {hint} first.")


def nearest_level(values, target):
    values = np.asarray(sorted(set(values)), float)
    return float(values[np.argmin(np.abs(values - target))])
