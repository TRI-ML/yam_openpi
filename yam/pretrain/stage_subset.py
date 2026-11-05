#!/usr/bin/env python
"""Write an s5cmd command file that stages a FRAC:SEED episode subset of a LeRobot v2.1 repo on S3 (all meta/* + each
sampled episode's parquet and camera videos) into a local repo dir. The subset is the one assemble_v21.py
`--sample LABEL=FRAC:SEED` picks (same sample_episodes on the same full meta/episodes.jsonl).

  python yam/pretrain/stage_subset.py <local meta dir> <s3 repo uri> <local repo dir> --sample 0.10:0 --out cmds.txt
  s5cmd run cmds.txt
"""
import argparse
import json
from pathlib import Path

from assemble_v21 import parse_sample, sample_episodes

ap = argparse.ArgumentParser()
ap.add_argument("meta_dir", help="local copy of the repo's meta/ (episodes.jsonl, info.json, ...)")
ap.add_argument("s3_repo"); ap.add_argument("local_repo"); ap.add_argument("--sample", required=True, help="FRAC[:SEED]")
ap.add_argument("--out", required=True)
a = ap.parse_args()
meta = Path(a.meta_dir); info = json.load(open(meta / "info.json"))
eps = {r["episode_index"]: r["length"] for r in map(json.loads, open(meta / "episodes.jsonl"))}
frac, seed = parse_sample(a.sample)
keep = sample_episodes(eps, frac, seed)
vids = [k for k, v in info["features"].items() if v["dtype"] == "video"]
src, dst = a.s3_repo.rstrip("/"), a.local_repo.rstrip("/")
lines = [f'cp "{src}/meta/*" {dst}/meta/']
for ep in keep:
    ch = ep // info["chunks_size"]
    rels = [info["data_path"].format(episode_chunk=ch, episode_index=ep)]
    rels += [info["video_path"].format(episode_chunk=ch, video_key=k, episode_index=ep) for k in vids]
    lines += [f"cp {src}/{r} {dst}/{r}" for r in rels]
Path(a.out).write_text("\n".join(lines) + "\n")
print(f"{len(keep)}/{len(eps)} episodes (frac {frac}, seed {seed}), {sum(eps[e] for e in keep):,} frames, "
      f"{len(lines) - 1} files + meta/* -> {a.out}")
