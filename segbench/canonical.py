from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import nearest_neighbour_distances
from .instance import Instance

SCENARIOS = [
    ("S1", "5g-c2", "Collinear columns", 1),
    ("S2", "5g-c1", "Separated clusters", 1),
    ("S3", "5g-c13", "Chained clusters", 1),
    ("S4", "5g-c3", "Concentric rings", 1),
    ("S5", "5g-c4", "Sectors", 1),
    ("S6", "5g-c5", "Interleaved arcs", 1),
    ("S7", "5g-c35", "Clusters with strays", 0),
    ("S8", "5g-c11", "Interpenetrating clusters", 0),
    ("S9", "5g-c10", "Adjacent clusters", 1),
    ("S10", "5g-c9", "Overlapping clusters", 0),
    ("S11", "5g-c7", "Partially merged", 0),
    ("S12", "5g-c14", "Mixed pairs", 0),
    ("S13", "5g-c6", "Random mixture", 0),
    ("S14", "kumar_erro", "Trapped agents (m=2)", 0),
]
RADIAL = {"S4"}

VECTOR_COLOURS = [(0.931, 0.13, 0.141), (0.052, 0.506, 0.251), (0.225, 0.326, 0.645),
                  (0.646, 0.246, 0.593), (0.12, 0.738, 0.742)]
RASTER_COLOURS = np.array([[255, 0, 0], [0, 128, 0], [0, 0, 255], [191, 0, 191], [0, 191, 191]])


def _extract_vector(page):
    pts, cols = [], []
    for d in page.get_drawings():
        fc = d.get("fill")
        if not fc or tuple(round(c, 3) for c in fc) == (1.0, 1.0, 1.0) or len(d["items"]) != 8:
            continue
        r = d["rect"]
        pts.append(((r.x0 + r.x1) / 2, -(r.y0 + r.y1) / 2))
        c = np.array(fc)
        cols.append(int(np.argmin([np.linalg.norm(c - np.array(v)) for v in VECTOR_COLOURS])))
    return np.array(pts), np.array(cols)


def _extract_raster(img: np.ndarray, colour_tol: float = 60.0):
    from scipy import ndimage
    from sklearn.cluster import KMeans

    im = img.astype(float)
    d = np.linalg.norm(im[:, :, None, :] - RASTER_COLOURS[None, None], axis=3)
    lab, ok = d.argmin(2), d.min(2) < colour_tol
    comps, areas = [], []
    for g in range(len(RASTER_COLOURS)):
        cc, n = ndimage.label(ok & (lab == g))
        sizes = ndimage.sum(np.ones_like(cc), cc, range(1, n + 1))
        for i, s in enumerate(sizes):
            if s >= 20:
                comps.append((g, i + 1, cc, s))
                areas.append(s)
    a25 = np.percentile(areas, 25)
    single = np.median([a for a in areas if a < 1.6 * a25])
    pts, cols = [], []
    for g, i, cc, s in comps:
        ys, xs = np.nonzero(cc == i)
        P = np.c_[xs, -ys].astype(float)
        k = max(1, int(round(s / single)))
        centres = P.mean(0, keepdims=True) if k == 1 else \
            KMeans(k, n_init=4, random_state=0).fit(P).cluster_centers_
        pts.extend(centres)
        cols.extend([g] * len(centres))
    return np.array(pts), np.array(cols)


def extract_from_pdf(pdf: Path):
    import pymupdf

    doc = pymupdf.open(str(pdf))
    page = doc[0]
    P, g = _extract_vector(page)
    if len(P) == 0:
        imgs = page.get_images(full=True)
        if not imgs:
            raise ValueError(f"no markers found in {pdf}")
        pix = pymupdf.Pixmap(doc, imgs[0][0])
        if pix.n - pix.alpha > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        P, g = _extract_raster(img)
    uniq = sorted(set(g.tolist()))
    g = np.array([uniq.index(v) for v in g])
    return P, g


def load_canonical(cfg, data_dir: Path):
    out = []
    s_ref = float(cfg["canonical"]["spacing_factor"]) * cfg.r
    for sid, fname, desc, exp in SCENARIOS:
        f = Path(data_dir) / "canonical" / f"{sid}.csv"
        A = np.loadtxt(f, delimiter=",", skiprows=1)
        P, g = A[:, :2], A[:, 2].astype(int)
        scale = s_ref / np.median(nearest_neighbour_distances(P))
        X = (P - P.mean(0)) * scale
        ring = None
        if sid in RADIAL:
            rad = np.linalg.norm(X - X.mean(0), axis=1)
            ring = list(np.argsort([rad[g == k].mean() for k in range(g.max() + 1)]))
        out.append(Instance(X=X, y=g, name=sid, family="canonical", seed=0, ring_order=ring,
                            meta=dict(source=fname, description=desc, expected=exp, scale=scale)))
    return out
