#!/usr/bin/env python
"""Stage B: assemble staged sources (convert_source.py outputs) into ONE LeRobot v2.1 repo with contiguous episode ids,
global task table and global frame `index`; videos are hardlinked (fallback symlink), parquets rewritten (lowdim only)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = ("top", "left_wrist", "right_wrist"); FPS = 30
NAMES14 = ["left_joint_0.pos", "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos", "left_joint_4.pos", "left_joint_5.pos", "left_gripper.pos",
           "right_joint_0.pos", "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos", "right_joint_4.pos", "right_joint_5.pos", "right_gripper.pos"]

def link(src, dst):
    if dst.exists() or dst.is_symlink(): dst.unlink()
    try: os.link(src, dst)
    except OSError: os.symlink(src, dst)

def do_ep(job):
    sdir, old, new, frame0, tid, root = job
    sdir, root = Path(sdir), Path(root); ch = new // 1000
    t = pq.read_table(sdir / f"ep_{old:06d}.parquet"); n = t.num_rows
    t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", pa.array(np.full(n, new, np.int64)))
    t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(np.arange(frame0, frame0 + n, dtype=np.int64)))
    t = t.set_column(t.schema.get_field_index("task_index"), "task_index", pa.array(np.full(n, tid, np.int64)))
    p = root / f"data/chunk-{ch:03d}/episode_{new:06d}.parquet"; p.parent.mkdir(parents=True, exist_ok=True); pq.write_table(t, p)
    for c in CAMS:
        d = root / f"videos/chunk-{ch:03d}/observation.images.{c}/episode_{new:06d}.mp4"; d.parent.mkdir(parents=True, exist_ok=True)
        link(sdir / f"ep_{old:06d}.{c}.mp4", d)
    return n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("sources", nargs="+"); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--robot-type", default="yam_bimanual")
    ap.add_argument("--skip", nargs="*", default=[], help="source dir names to skip (e.g. camera-less tasks)")
    ap.add_argument("--exclude", nargs="*", default=[], help="<source_name>:<file of source episode ids to drop> (val holdout)")
    ap.add_argument("--only", nargs="*", default=None, help="<source_name>:<file of source episode ids to KEEP> (val repo)")
    a = ap.parse_args()
    excl = {k: set(int(x) for x in open(f).read().split()) for k, f in (e.split(":", 1) for e in a.exclude)}
    only = {k: set(int(x) for x in open(f).read().split()) for k, f in (e.split(":", 1) for e in (a.only or []))}
    root = Path(a.root)
    if root.exists(): shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    tasks, episodes, jobs, frame0, new = [], [], [], 0, 0
    per_source = {}
    dropped = {}
    for s in a.sources:
        sdir = Path(s)
        if sdir.name in a.skip: dropped[sdir.name] = "skipped"; continue
        recs = [json.loads(l) for l in open(sdir / "episodes.jsonl")]; recs.sort(key=lambda r: r["episode_index"])
        if sdir.name in excl: n0 = len(recs); recs = [r for r in recs if r["episode_index"] not in excl[sdir.name]]; dropped[sdir.name] = f"excluded {n0-len(recs)} episodes"
        if sdir.name in only: recs = [r for r in recs if r["episode_index"] in only[sdir.name]]
        per_source[sdir.name] = {"episodes": len(recs), "frames": sum(r["length"] for r in recs), "first_new_episode": new, "source_episode_ids": [r["episode_index"] for r in recs]}
        for r in recs:
            if r["task"] not in tasks: tasks.append(r["task"])
            tid = tasks.index(r["task"])
            jobs.append((str(sdir), r["episode_index"], new, frame0, tid, str(root)))
            episodes.append({"episode_index": new, "tasks": [r["task"]], "length": r["length"], "stats": r["stats"]})
            frame0 += r["length"]; new += 1
    with ProcessPoolExecutor(a.workers) as ex: total = sum(ex.map(do_ep, jobs, chunksize=64))
    assert total == frame0
    with open(root / "meta/episodes.jsonl", "w") as f:
        for e in episodes: f.write(json.dumps({k: e[k] for k in ("episode_index", "tasks", "length")}) + "\n")
    with open(root / "meta/episodes_stats.jsonl", "w") as f:
        for e in episodes: f.write(json.dumps({"episode_index": e["episode_index"], "stats": e["stats"]}) + "\n")
    with open(root / "meta/tasks.jsonl", "w") as f:
        for i, t in enumerate(tasks): f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    vid = {"dtype": "video", "shape": [256, 416, 3], "names": ["height", "width", "channels"],
           "info": {"video.fps": FPS, "video.height": 256, "video.width": 416, "video.channels": 3, "video.codec": "av1", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False,
                    "note": "width varies per source task (256 x 352..448); the trainer resizes to 224x224"}}
    feats = {"action": {"dtype": "float32", "shape": [14], "names": NAMES14}, "observation.state": {"dtype": "float32", "shape": [14], "names": NAMES14},
             **{f"observation.images.{c}": vid for c in CAMS},
             "timestamp": {"dtype": "float32", "shape": [1], "names": None}, "frame_index": {"dtype": "int64", "shape": [1], "names": None},
             "episode_index": {"dtype": "int64", "shape": [1], "names": None}, "index": {"dtype": "int64", "shape": [1], "names": None}, "task_index": {"dtype": "int64", "shape": [1], "names": None}}
    n = len(episodes)
    info = {"codebase_version": "v2.1", "robot_type": a.robot_type, "total_episodes": n, "total_frames": frame0, "total_tasks": len(tasks), "total_videos": n * len(CAMS),
            "total_chunks": (n - 1) // 1000 + 1, "chunks_size": 1000, "fps": FPS, "splits": {"train": f"0:{n}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": feats}
    json.dump(info, open(root / "meta/info.json", "w"), indent=2)
    json.dump({"sources": per_source, "dropped": dropped, "total_episodes": n, "total_frames": frame0, "total_tasks": len(tasks)}, open(root / "meta/assembly.json", "w"), indent=1)
    print(f"{root}: {n} episodes, {frame0} frames, {len(tasks)} tasks; sources={len(per_source)}; dropped={dropped}")

if __name__ == "__main__": main()
