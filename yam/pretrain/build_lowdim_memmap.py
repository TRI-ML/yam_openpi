#!/usr/bin/env python
"""Build per-repo float32 memmaps of the 14-D `action` / `observation.state` columns in global frame order
(<repo>/lowdim_mm/{action,observation.state}.npy). lerobot's per-sample `hf_dataset.select()` costs ~1.6 s on a
463M-frame / 141-shard table; the patched _query_hf_dataset reads these memmaps instead (O(1), page-cache shared)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import pathlib
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("repo_root"); ap.add_argument("--keys", nargs="+", default=["action", "observation.state"]); ap.add_argument("--workers", type=int, default=48); a = ap.parse_args()
    root = pathlib.Path(a.repo_root); info = json.load(open(root / "meta/info.json")); N = info["total_frames"]
    out = root / "lowdim_mm"; out.mkdir(exist_ok=True)
    dims = {k: info["features"][k]["shape"][0] for k in a.keys}
    mms = {k: np.lib.format.open_memmap(out / f"{k}.npy.tmp", mode="w+", dtype=np.float32, shape=(N, dims[k])) for k in a.keys}
    files = sorted(glob.glob(str(root / "data/chunk-*/episode_*.parquet"))); t0 = time.time()
    def do(p):
        t = pq.read_table(p, columns=a.keys + ["index"]); idx = t["index"].to_numpy(); i0, i1 = int(idx[0]), int(idx[-1]) + 1
        assert i1 - i0 == t.num_rows and np.all(np.diff(idx) == 1), p
        for k in a.keys:
            c = t[k].combine_chunks(); mms[k][i0:i1] = pc.list_flatten(c).to_numpy(zero_copy_only=False).astype(np.float32).reshape(t.num_rows, -1)
        return t.num_rows
    with ThreadPoolExecutor(a.workers) as ex: n = sum(ex.map(do, files, chunksize=64))
    assert n == N, (n, N)
    for k in a.keys: mms[k].flush(); del mms[k]; (out / f"{k}.npy.tmp").rename(out / f"{k}.npy")
    json.dump({"total_frames": N, "keys": a.keys, "dims": dims, "files": len(files), "seconds": round(time.time() - t0, 1)}, open(out / "meta.json", "w"), indent=1)
    print(f"{root.name}: {N:,} frames x {dims} from {len(files)} files in {time.time()-t0:.0f}s -> {out}")

if __name__ == "__main__": main()
