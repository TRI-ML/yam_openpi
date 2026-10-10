#!/usr/bin/env python
"""Exact openpi norm stats (mean/std/q01/q99) for 'state' and 'actions' straight from the assembled repo's parquet
(14-D raw joints, i.e. exactly what YAMInputs emits before Normalize), instead of sampling frames through video decode."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "src")
import pyarrow.compute as pc

from openpi.shared import normalize


def col(t, name):
    c = t[name].combine_chunks(); flat = pc.list_flatten(c).to_numpy(zero_copy_only=False).astype(np.float32)
    return flat.reshape(len(c), -1)

def load(p):
    t = pq.read_table(p, columns=["action", "observation.state"])
    return col(t, "action"), col(t, "observation.state")

def stats(x):
    # float64 accumulation: a float32 axis-0 sum saturates at ~1.7e7 elements (observed: means collapsing to 0.43/0.107 at 156M frames)
    mean = x.mean(0, dtype=np.float64); std = np.sqrt(((x - mean.astype(np.float32)) ** 2).mean(0, dtype=np.float64))
    return normalize.NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32), q01=np.quantile(x, 0.01, axis=0).astype(np.float32), q99=np.quantile(x, 0.99, axis=0).astype(np.float32))

ap = argparse.ArgumentParser(); ap.add_argument("repo_root"); ap.add_argument("out_dir"); ap.add_argument("--workers", type=int, default=64); a = ap.parse_args()
files = sorted(glob.glob(f"{a.repo_root}/data/chunk-*/episode_*.parquet")); t0 = time.time()
with ThreadPoolExecutor(a.workers) as ex: parts = list(ex.map(load, files))
A = np.concatenate([p[0] for p in parts]); S = np.concatenate([p[1] for p in parts]); del parts
print(f"{len(files)} episodes, {len(A):,} frames loaded in {time.time()-t0:.0f}s")
ns = {"actions": stats(A), "state": stats(S)}
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); normalize.save(out, ns)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for k, v in ns.items(): print(k, "mean", v.mean, "\n   std", v.std, "\n   q01", v.q01, "\n   q99", v.q99)
print("wrote", out / "norm_stats.json", f"total {time.time()-t0:.0f}s")
