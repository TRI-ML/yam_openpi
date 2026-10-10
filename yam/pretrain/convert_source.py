#!/usr/bin/env python
"""Stage A: LeRobot v3.0 repo (Zheyuan v3ar schema) -> per-episode v2.1-style files, keeping the SOURCE episode ids.
Output <out>/ep_XXXXXX.parquet (14-D joints promoted to action/observation.state), ep_XXXXXX.<cam>.mp4 (lossless
stream copy of the episode's packets; every episode starts on a keyframe in these files, verified), episodes.jsonl.
Videos are cut with ffmpeg -ss <from_ts> -c copy -frames:v <len>; packet count is verified, mismatch -> re-encode fallback."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = ("top", "left_wrist", "right_wrist")
FF = os.environ.get("FFMPEG", "ffmpeg"); FP = os.environ.get("FFPROBE", "ffprobe")

def nb_packets(p):
    out = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(p)], capture_output=True, text=True, check=False).stdout.strip()
    return int(out or 0)

def cut(src, t0, n, dst):
    cmd = [FF, "-v", "error", "-y", "-ss", f"{t0:.6f}", "-i", str(src), "-frames:v", str(n), "-c:v", "copy", "-an", str(dst)]
    subprocess.run(cmd, check=True)
    if nb_packets(dst) == n: return "copy"
    subprocess.run([FF, "-v", "error", "-y", "-ss", f"{t0:.6f}", "-i", str(src), "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "2", "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
    assert nb_packets(dst) == n, f"{dst}: frame count mismatch after re-encode"
    return "reencode"

def stats_of(a):
    a = np.asarray(a, dtype=np.float32)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [len(a)]}

def do_file(args):
    src, out, fps, rows, tasks = args   # rows: list of episode meta dicts sharing one data file
    src, out = Path(src), Path(out); done = []
    df_all = pq.read_table(src / rows[0]["data_path"]).to_pandas()
    for e in rows:
        ei = int(e["episode_index"]); n = int(e["length"])
        if (out / f"ep_{ei:06d}.done").exists(): done.append(ei); continue
        df = df_all[df_all["episode_index"] == ei].sort_values("frame_index")
        assert len(df) == n, (src, ei, len(df), n)
        ja = np.stack(df["joint_action"].to_numpy()).astype(np.float32); js = np.stack(df["joint_state"].to_numpy()).astype(np.float32)
        tname = tasks[int(df["task_index"].iloc[0])]
        out_df = pd.DataFrame({"action": list(ja), "observation.state": list(js), "timestamp": (df["frame_index"].to_numpy() / fps).astype(np.float32),
                               "frame_index": df["frame_index"].to_numpy().astype(np.int64), "episode_index": np.full(n, ei, np.int64),
                               "index": np.zeros(n, np.int64), "task_index": np.zeros(n, np.int64)})
        schema = pa.schema([("action", pa.list_(pa.float32(), 14)), ("observation.state", pa.list_(pa.float32(), 14)), ("timestamp", pa.float32()),
                            ("frame_index", pa.int64()), ("episode_index", pa.int64()), ("index", pa.int64()), ("task_index", pa.int64())])
        pq.write_table(pa.Table.from_pandas(out_df, schema=schema, preserve_index=False), out / f"ep_{ei:06d}.parquet")
        modes = {}
        for c in CAMS:
            k = f"observation.images.{c}"; vsrc = src / f"videos/{k}/chunk-{int(e[f'videos/{k}/chunk_index']):03d}/file-{int(e[f'videos/{k}/file_index']):03d}.mp4"
            modes[c] = cut(vsrc, float(e[f"videos/{k}/from_timestamp"]), n, out / f"ep_{ei:06d}.{c}.mp4")
        rec = {"episode_index": ei, "task": tname, "length": n, "stats": {"action": stats_of(ja), "observation.state": stats_of(js)}, "video_mode": modes}
        (out / f"ep_{ei:06d}.json").write_text(json.dumps(rec)); (out / f"ep_{ei:06d}.done").touch(); done.append(ei)
    return done

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=32); a = ap.parse_args()
    src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    info = json.load(open(src / "meta/info.json")); fps = info["fps"]
    tasks = {int(r["task_index"]): r["task"] for r in pq.read_table(src / "meta/tasks.parquet").to_pylist()}
    eps = pd.concat([pq.read_table(p).to_pandas() for p in sorted(glob.glob(str(src / "meta/episodes/*/*.parquet")))])
    eps = eps[[c for c in eps.columns if not c.startswith("stats/")]]
    eps["data_path"] = [info["data_path"].format(chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])) for _, r in eps.iterrows()]
    groups = [(str(src), str(out), fps, g.to_dict("records"), tasks) for _, g in eps.groupby("data_path")]
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for done in ex.map(do_file, groups): n += len(done)
    recs = [json.loads(open(p).read()) for p in sorted(out.glob("ep_*.json"))]
    with open(out / "episodes.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    json.dump({"source": str(src), "fps": fps, "episodes": len(recs), "frames": int(sum(r["length"] for r in recs)), "video_modes": pd.Series([m for r in recs for m in r["video_mode"].values()]).value_counts().to_dict(),
               "video_shape": info["features"]["observation.images.top"]["shape"], "seconds": round(time.time() - t0, 1)}, open(out / "source.json", "w"), indent=1)
    print(f"{src.name}: {len(recs)} eps, {sum(r['length'] for r in recs)} frames, {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__": main()
