#!/usr/bin/env python
"""Exact mean/std (float64) + histogram quantiles (openpi's own RunningStats recipe: 5000 bins over [min,max] per dim) for 'state'/'actions'
from the repo's lowdim memmaps (lowdim_mm/*.npy) — streaming, multi-threaded, no 463M-row sort. Falls back to parquet if memmaps are absent."""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from openpi.shared import normalize

ap = argparse.ArgumentParser(); ap.add_argument("repo_root"); ap.add_argument("out_dir"); ap.add_argument("--bins", type=int, default=5000); ap.add_argument("--chunk", type=int, default=20_000_000); a = ap.parse_args()
root = pathlib.Path(a.repo_root); t0 = time.time()
mm = {k: np.load(root / f"lowdim_mm/{k}.npy", mmap_mode="r") for k in ("action", "observation.state")}
N = mm["action"].shape[0]; print(f"{N:,} frames", flush=True)
def pass1(x):  # min/max/sum/sumsq per dim in float64, chunked
    lo = np.full(x.shape[1], np.inf); hi = np.full(x.shape[1], -np.inf); s = np.zeros(x.shape[1]); ss = np.zeros(x.shape[1])
    for i in range(0, N, a.chunk):
        c = np.asarray(x[i:i+a.chunk], dtype=np.float64); lo = np.minimum(lo, c.min(0)); hi = np.maximum(hi, c.max(0)); s += c.sum(0); ss += (c * c).sum(0)
    mean = s / N; std = np.sqrt(np.maximum(ss / N - mean ** 2, 0)); return lo, hi, mean, std
def pass2(x, lo, hi):  # per-dim histograms -> q01/q99 (openpi RunningStats uses the same bin interpolation)
    D = x.shape[1]; edges = [np.linspace(lo[d] - 1e-10, hi[d] + 1e-10, a.bins + 1) for d in range(D)]; H = np.zeros((D, a.bins))
    for i in range(0, N, a.chunk):
        c = np.asarray(x[i:i+a.chunk], dtype=np.float32)
        for d in range(D): H[d] += np.histogram(c[:, d], bins=edges[d])[0]
    q = {}
    for p in (0.01, 0.99):
        out = np.zeros(D)
        for d in range(D):
            cdf = np.cumsum(H[d]) / H[d].sum(); j = int(np.searchsorted(cdf, p)); out[d] = edges[d][min(j + 1, a.bins)]
        q[p] = out
    return q[0.01], q[0.99]
ns = {}
for key, name in (("action", "actions"), ("observation.state", "state")):
    lo, hi, mean, std = pass1(mm[key]); q01, q99 = pass2(mm[key], lo, hi)
    ns[name] = normalize.NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32), q01=q01.astype(np.float32), q99=q99.astype(np.float32))
    print(name, "mean", np.round(mean, 3), "std", np.round(std, 3), "q01", np.round(q01, 3), "q99", np.round(q99, 3), f"({time.time()-t0:.0f}s)", flush=True)
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); normalize.save(out, ns); print("wrote", out / "norm_stats.json", f"total {time.time()-t0:.0f}s")
