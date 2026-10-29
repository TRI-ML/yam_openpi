#!/usr/bin/env python
"""Stage B: assemble staged sources (convert_source.py outputs) into ONE LeRobot v2.1 repo with contiguous episode ids,
global task table and global frame `index`; videos are hardlinked (fallback symlink), parquets rewritten (lowdim only).

Stage-2 mode (`--repo LABEL=DIR`, docs/STAGE2.md): the inputs are LeRobot v2.1 repos instead of staged dirs, and every
output parquet gains `source` (string), `quality` (1 success / 0 failure) and `domain` (0 real / 1 sim). LABEL picks
the per-input rules:
  abcmolmo       the abc_molmo_v21 cache; per-episode source molmo|abc from its meta/assembly.json; success, real
  oopsie_<lab>   an oopsie-YAM unit export (18-D, padded): keep dims 0:14, cameras base_0_rgb/left_wrist_0_rgb/
                 right_wrist_0_rgb -> top/left_wrist/right_wrist; quality = `success` column; real
  absim          ABC-sim; quality from meta/episode_labels.jsonl by GLOBAL episode_index, trivial_success dropped; sim
Inputs with the same label must be adjacent (each source is one contiguous episode/frame range; the trainer's
source-mix sampler reads those ranges from meta/assembly.json)."""
import argparse
import hashlib
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

# Princeton_PRISM: its `action` is a copy of the state; Berkeley_HybridRobotics: single-arm (needs action masking).
EXCLUDED_OOPSIE = {"Princeton_PRISM": "action is a copy of observation.state", "Berkeley_HybridRobotics": "single arm (would need action masking)"}
OOPSIE_CAMS = {"base_0_rgb": "top", "left_wrist_0_rgb": "left_wrist", "right_wrist_0_rgb": "right_wrist"}
S2_KEEP = ("timestamp", "frame_index")


def sample_episodes(episode_indices, frac, seed=0):
    """Uniform, deterministic FRAC subset of an input's episode indices (sorted). Shared by --sample and stage_subset.py."""
    ids = np.asarray(sorted(episode_indices), np.int64)
    n = int(round(frac * len(ids)))
    return sorted(np.random.default_rng(seed).choice(ids, size=n, replace=False).tolist())


def parse_sample(spec):
    """'FRAC[:SEED]' -> (frac, seed)."""
    frac, _, seed = spec.partition(":")
    return float(frac), int(seed or 0)


def _fsl14(col, n):
    flat = col.combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    return pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(flat[:, :14]).ravel()), 14), flat


def do_repo_ep(job):
    """One v2.1 input episode -> output parquet with 14-D lowdim + source/quality/domain, videos linked under the new names."""
    in_root, chunks, old, new, frame0, tid, src, q, dom, cams, root = job
    in_root, root = Path(in_root), Path(root); ch_in, ch = old // chunks, new // 1000
    t = pq.read_table(in_root / f"data/chunk-{ch_in:03d}/episode_{old:06d}.parquet"); n = t.num_rows
    if "action_mask" in t.column_names:  # oopsie: refuse anything that is not a full 14-D bimanual episode
        m = _fsl14(t["action_mask"], n)[1]
        assert (m[:, :14] == 1).all() and (m[:, 14:] == 0).all(), f"{in_root} ep {old}: action_mask is not 14 bimanual dims"
    out = pa.table({"action": _fsl14(t["action"], n)[0], "observation.state": _fsl14(t["observation.state"], n)[0],
                    **{k: t[k] for k in S2_KEEP},
                    "episode_index": pa.array(np.full(n, new, np.int64)), "index": pa.array(np.arange(frame0, frame0 + n, dtype=np.int64)),
                    "task_index": pa.array(np.full(n, tid, np.int64)), "source": pa.array([src] * n, pa.string()),
                    "quality": pa.array(np.full(n, q, np.int64)), "domain": pa.array(np.full(n, dom, np.int64))})
    p = root / f"data/chunk-{ch:03d}/episode_{new:06d}.parquet"; p.parent.mkdir(parents=True, exist_ok=True); pq.write_table(out, p)
    for c_in, c in cams.items():
        d = root / f"videos/chunk-{ch:03d}/observation.images.{c}/episode_{new:06d}.mp4"; d.parent.mkdir(parents=True, exist_ok=True)
        link(in_root / f"videos/chunk-{ch_in:03d}/observation.images.{c_in}/episode_{old:06d}.mp4", d)
    return n


def _slice_stats(st, cams):
    out = {}
    for k, v in st.items():
        if k.startswith("observation.images."):
            c = k.split(".", 2)[2]
            if c in cams: out[f"observation.images.{cams[c]}"] = v
        elif k in ("action", "observation.state"):
            out[k] = {s: (x[:14] if s != "count" else x) for s, x in v.items()}
        elif k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            out[k] = v
    return out


def repo_episodes(label, d, a):
    """-> (records, dropped note): one record per kept input episode with its source/quality/domain."""
    d = Path(d); info = json.load(open(d / "meta/info.json"))
    if info["fps"] != FPS: print(f"WARNING {d}: fps {info['fps']} != {FPS} (timestamps kept from the source; action chunks span {FPS / info['fps']:.1f}x the wall time)")
    eps = [json.loads(l) for l in open(d / "meta/episodes.jsonl")]; eps.sort(key=lambda r: r["episode_index"])
    sample = None
    if label in a.sample:  # first, on the input's full episode list (identical to stage_subset.py's pick)
        frac, seed = a.sample[label]; keep = set(sample_episodes([e["episode_index"] for e in eps], frac, seed))
        sample = {"frac": frac, "seed": seed, "episodes": len(keep), "of": len(eps),
                  "sha1": hashlib.sha1(json.dumps(sorted(keep)).encode()).hexdigest()}
        eps = [e for e in eps if e["episode_index"] in keep]
    want = {e["episode_index"] for e in eps}
    stats = {r["episode_index"]: r["stats"] for r in map(json.loads, open(d / "meta/episodes_stats.jsonl")) if r["episode_index"] in want}
    note = None
    if label == "abcmolmo":
        cams = {c: c for c in CAMS}
        src_of = {}
        for name, v in json.load(open(d / "meta/assembly.json"))["sources"].items():
            for i in range(v["first_new_episode"], v["first_new_episode"] + v["episodes"]): src_of[i] = name
        if a.cache_sources: n0 = len(eps); eps = [e for e in eps if src_of[e["episode_index"]] in a.cache_sources]; note = f"kept {len(eps)}/{n0} episodes of cache sources {a.cache_sources}"
        lab = lambda e: ("molmo" if src_of[e["episode_index"]] == "molmo_yam_v3" else "abc", 1, 0)
    elif label.startswith("oopsie_"):
        lab_name = label[len("oopsie_"):]
        if lab_name in EXCLUDED_OOPSIE: raise SystemExit(f"{label} is excluded: {EXCLUDED_OOPSIE[lab_name]}")
        cams = OOPSIE_CAMS
        def lab(e):
            ep = e["episode_index"]; ch = ep // info["chunks_size"]
            s = pq.read_table(d / f"data/chunk-{ch:03d}/episode_{ep:06d}.parquet", columns=["success"])["success"].to_numpy()
            assert s.min() == s.max(), f"{d} ep {ep}: success not constant within the episode"
            return label, int(s[0] >= 0.5), 0
    elif label == "absim":
        cams = {c: c for c in CAMS}
        L = {r["episode_index"]: r for r in map(json.loads, open(d / "meta/episode_labels.jsonl"))}  # GLOBAL index; never the root labels.jsonl
        n0 = len(eps); eps = [e for e in eps if not L[e["episode_index"]].get("trivial_success")]; note = f"dropped {n0 - len(eps)} trivial_success episodes"
        Q = {"success_suboptimal": 1, "failure": 0}
        lab = lambda e: ("absim", Q[L[e["episode_index"]]["outcome"]], 1)
    else:
        raise SystemExit(f"unknown --repo label {label!r} (abcmolmo | oopsie_<lab> | absim)")
    if label in a.limit: eps = eps[: a.limit[label]]
    recs = []
    for e in eps:
        src, q, dom = lab(e)
        assert len(e["tasks"]) == 1, (d, e["episode_index"])
        recs.append({"dir": str(d), "chunks": info["chunks_size"], "old": e["episode_index"], "task": e["tasks"][0], "length": e["length"],
                     "source": src, "quality": q, "domain": dom, "cams": cams, "stats": _slice_stats(stats[e["episode_index"]], cams), "fps": info["fps"]})
    if sample: note = f"sampled {sample}" + (f"; {note}" if note else "")
    return recs, note


def main_repos(a):
    root = Path(a.root)
    if root.exists(): shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    tasks, tidx, episodes, jobs, frame0, new, per_source, dropped, seen = [], {}, [], [], 0, 0, {}, {}, []
    for spec in a.repo:
        label, d = spec.split("=", 1)
        if seen and label in seen and seen[-1] != label: raise SystemExit(f"inputs with label {label} must be adjacent")
        seen.append(label)
        recs, note = repo_episodes(label, d, a)
        if note: dropped[d] = note
        for r in recs:
            if r["task"] not in tidx: tidx[r["task"]] = len(tasks); tasks.append(r["task"])
            s = per_source.setdefault(r["source"], {"episodes": 0, "frames": 0, "first_new_episode": new, "first_frame": frame0,
                                                    "success_episodes": 0, "failure_episodes": 0, "domain": "sim" if r["domain"] else "real", "fps": r["fps"], "inputs": []})
            assert s["first_new_episode"] + s["episodes"] == new, f"source {r['source']} is not contiguous (order the --repo inputs by source)"
            if d not in s["inputs"]: s["inputs"].append(d)
            s["episodes"] += 1; s["frames"] += r["length"]; s["success_episodes" if r["quality"] else "failure_episodes"] += 1
            jobs.append((r["dir"], r["chunks"], r["old"], new, frame0, tidx[r["task"]], r["source"], r["quality"], r["domain"], r["cams"], str(root)))
            episodes.append({"episode_index": new, "tasks": [r["task"]], "length": r["length"], "stats": r["stats"]})
            frame0 += r["length"]; new += 1
    for s in per_source.values(): s["end_new_episode"] = s["first_new_episode"] + s["episodes"]; s["end_frame"] = s["first_frame"] + s["frames"]
    with ProcessPoolExecutor(a.workers) as ex: total = sum(ex.map(do_repo_ep, jobs, chunksize=64))
    assert total == frame0
    extra = {"source": {"dtype": "string", "shape": [1], "names": None}, "quality": {"dtype": "int64", "shape": [1], "names": None},
             "domain": {"dtype": "int64", "shape": [1], "names": None}}
    write_meta(root, a.robot_type, episodes, tasks, frame0, extra, codec="mixed (av1 / h264)",
               assembly={"sources": per_source, "dropped": {**dropped, **{f"oopsie_{k}": f"excluded: {v}" for k, v in EXCLUDED_OOPSIE.items()}},
                         "inputs": a.repo, "columns": {"quality": "1 success / 0 failure", "domain": "0 real / 1 sim"}})


def write_meta(root, robot_type, episodes, tasks, frame0, extra_feats, codec, assembly):
    with open(root / "meta/episodes.jsonl", "w") as f:
        for e in episodes: f.write(json.dumps({k: e[k] for k in ("episode_index", "tasks", "length")}) + "\n")
    with open(root / "meta/episodes_stats.jsonl", "w") as f:
        for e in episodes: f.write(json.dumps({"episode_index": e["episode_index"], "stats": e["stats"]}) + "\n")
    with open(root / "meta/tasks.jsonl", "w") as f:
        for i, t in enumerate(tasks): f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    vid = {"dtype": "video", "shape": [256, 416, 3], "names": ["height", "width", "channels"],
           "info": {"video.fps": FPS, "video.height": 256, "video.width": 416, "video.channels": 3, "video.codec": codec, "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False,
                    "note": "size varies per source (168x224 .. 256x448); the trainer resizes to 224x224"}}
    feats = {"action": {"dtype": "float32", "shape": [14], "names": NAMES14}, "observation.state": {"dtype": "float32", "shape": [14], "names": NAMES14},
             **{f"observation.images.{c}": vid for c in CAMS},
             "timestamp": {"dtype": "float32", "shape": [1], "names": None}, "frame_index": {"dtype": "int64", "shape": [1], "names": None},
             "episode_index": {"dtype": "int64", "shape": [1], "names": None}, "index": {"dtype": "int64", "shape": [1], "names": None}, "task_index": {"dtype": "int64", "shape": [1], "names": None},
             **extra_feats}
    n = len(episodes)
    info = {"codebase_version": "v2.1", "robot_type": robot_type, "total_episodes": n, "total_frames": frame0, "total_tasks": len(tasks), "total_videos": n * len(CAMS),
            "total_chunks": (n - 1) // 1000 + 1, "chunks_size": 1000, "fps": FPS, "splits": {"train": f"0:{n}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": feats}
    json.dump(info, open(root / "meta/info.json", "w"), indent=2)
    json.dump({**assembly, "total_episodes": n, "total_frames": frame0, "total_tasks": len(tasks)}, open(root / "meta/assembly.json", "w"), indent=1)
    print(f"{root}: {n} episodes, {frame0} frames, {len(tasks)} tasks; sources={len(assembly['sources'])}; dropped={assembly['dropped']}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("sources", nargs="*")
    ap.add_argument("--repo", action="append", default=[], help="stage-2 mode: LABEL=<v2.1 repo dir>, repeatable (see module docstring)")
    ap.add_argument("--cache-sources", nargs="*", default=None, help="stage-2: keep only these abc_molmo_v21 cache sources (its assembly.json keys)")
    ap.add_argument("--sample", nargs="*", default=[], help="stage-2: LABEL=FRAC[:SEED] keeps a uniform random FRAC of that input's episodes (seed default 0)")
    ap.add_argument("--limit", nargs="*", default=[], help="stage-2: LABEL=N keeps the first N episodes of each input with that label"); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--robot-type", default="yam_bimanual")
    ap.add_argument("--skip", nargs="*", default=[], help="source dir names to skip (e.g. camera-less tasks)")
    ap.add_argument("--exclude", nargs="*", default=[], help="<source_name>:<file of source episode ids to drop> (val holdout)")
    ap.add_argument("--only", nargs="*", default=None, help="<source_name>:<file of source episode ids to KEEP> (val repo)")
    a = ap.parse_args()
    if a.repo:
        assert not a.sources, "--repo (stage-2) and staged source dirs cannot be mixed"
        a.limit = {k: int(v) for k, v in (x.split("=", 1) for x in a.limit)}
        a.sample = {k: parse_sample(v) for k, v in (x.split("=", 1) for x in a.sample)}
        return main_repos(a)
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
    write_meta(root, a.robot_type, episodes, tasks, frame0, {}, codec="av1", assembly={"sources": per_source, "dropped": dropped})

if __name__ == "__main__": main()
