#!/usr/bin/env python
"""Build the multi-task YAM fine-tune sources from our raiden episode bundles (tri-yam/episodes/<eid>/), in the layout
yam/pretrain/assemble_v21.py consumes (ep_NNNNNN.parquet + ep_NNNNNN.{top,left_wrist,right_wrist}.mp4 + ep_NNNNNN.json + episodes.jsonl).

Contract = the pretrain repo (local/abc_mm2_train_v21): LeRobot v2.1, 30 fps, 14-D absolute commanded joints `action` /
measured `observation.state` ([L j0-5, L grip, R j0-5, R grip], grippers 0..1 with 1 = open), cameras top (= ego_camera),
left_wrist, right_wrist (the bundle videos already store the right wrist in the same orientation as the pretrain data —
verified against the foundry shard, MAE 1.6/255 — so NO rotation is applied). Videos are re-encoded 1280x720 -> WxH
(default 456x256, aspect kept) h264 and cut to the parquet row count (bundle mp4s carry one trailing extra frame).

Inputs per task: <meta>/<Task>.uris (episode S3 uris, one per line) and <meta>/<Task>.status lines
`<episode_id> <status> <num_frames> <instruction_with_underscores>`; only status == success is used. The task string
written to the repo = the episode's instruction (every episode of a task shares one).
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = {"top": "ego_camera", "left_wrist": "left_wrist_camera", "right_wrist": "right_wrist_camera"}
SCHEMA = pa.schema([("action", pa.list_(pa.float32(), 14)), ("observation.state", pa.list_(pa.float32(), 14)), ("timestamp", pa.float32()),
                    ("frame_index", pa.int64()), ("episode_index", pa.int64()), ("index", pa.int64()), ("task_index", pa.int64())])

def stats_of(a):
    a = np.asarray(a, dtype=np.float32)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [len(a)]}

def s3cp(uri, dst):
    subprocess.run(["aws", "s3", "cp", uri, str(dst), "--only-show-errors"], check=True)

def nb_frames(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    return int(out or 0)

def do_episode(job):
    uri, ei, instr, out, size, crf = job
    out = Path(out)
    if (out / f"ep_{ei:06d}.done").exists():
        return ei, json.loads((out / f"ep_{ei:06d}.json").read_text())["length"], "cached"
    with tempfile.TemporaryDirectory(prefix="mt9_", dir="/dev/shm" if Path("/dev/shm").exists() else None) as td:
        td = Path(td)
        s3cp(f"{uri}/lerobot/data/chunk-000/episode_000000.parquet", td / "ep.parquet")
        t = pq.read_table(td / "ep.parquet"); n = t.num_rows
        a = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
        s = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
        assert a.shape == (n, 14) and s.shape == (n, 14), (uri, a.shape, s.shape)
        fi = t["frame_index"].to_numpy().astype(np.int64)
        tbl = pa.Table.from_arrays([pa.array(list(a), type=pa.list_(pa.float32(), 14)), pa.array(list(s), type=pa.list_(pa.float32(), 14)),
                                    pa.array((fi / 30.0).astype(np.float32)), pa.array(fi), pa.array(np.full(n, ei, np.int64)),
                                    pa.array(np.zeros(n, np.int64)), pa.array(np.zeros(n, np.int64))], schema=SCHEMA)
        pq.write_table(tbl, out / f"ep_{ei:06d}.parquet")
        w, h = size
        for cam, src in CAMS.items():
            s3cp(f"{uri}/video/{src}.mp4", td / f"{src}.mp4")
            dst = out / f"ep_{ei:06d}.{cam}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(td / f"{src}.mp4"), "-frames:v", str(n), "-vf", f"scale={w}:{h}",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-g", "15", "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
            got = nb_frames(dst)
            assert got == n, f"{dst}: {got} frames, parquet {n} rows (source {nb_frames(td / f'{src}.mp4')})"
        rec = {"episode_index": ei, "task": instr, "length": n, "stats": {"action": stats_of(a), "observation.state": stats_of(s)}, "source_uri": uri}
        (out / f"ep_{ei:06d}.json").write_text(json.dumps(rec)); (out / f"ep_{ei:06d}.done").touch()
    return ei, n, "built"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--size", default="456x256"); ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.05, help="per-task holdout; writes <out>/<task>/val_ids.txt + train_ids.txt")
    a = ap.parse_args(); meta, out = Path(a.meta), Path(a.out); w, h = (int(x) for x in a.size.split("x"))
    summary = {}
    for task in a.tasks:
        uris = [l.strip() for l in open(meta / f"{task}.uris") if l.strip()]
        status = {}
        for l in open(meta / f"{task}.status"):
            p = l.split()
            if len(p) >= 3: status[p[0]] = (p[1], int(p[2]) if p[2].isdigit() else 0, p[3].replace("_", " ") if len(p) > 3 else "")
        eps = sorted((u for u in uris if status.get(u.rstrip("/").rsplit("/", 1)[-1], ("?",))[0] == "success"), key=lambda u: u.rsplit("/", 1)[-1])
        from collections import Counter
        instrs = Counter(status[u.rsplit("/", 1)[-1]][2] for u in eps)
        instr = instrs.most_common(1)[0][0]
        assert instr, (task, instrs)
        if len(instrs) > 1:   # a task whose instruction text was edited mid-collection: use the majority string for every episode
            print(f"{task}: {len(instrs)} instruction variants, using the majority one for all episodes: {dict(instrs)}", flush=True)
        odir = out / task; odir.mkdir(parents=True, exist_ok=True)
        jobs = [(u, i, instr, str(odir), (w, h), a.crf) for i, u in enumerate(eps)]
        t0 = time.time(); frames = 0; built = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for fut in as_completed([ex.submit(do_episode, j) for j in jobs]):
                ei, n, how = fut.result(); frames += n; built += how == "built"
        recs = [json.loads((odir / f"ep_{i:06d}.json").read_text()) for i in range(len(eps))]
        with open(odir / "episodes.jsonl", "w") as f:
            for r in recs: f.write(json.dumps({k: r[k] for k in ("episode_index", "task", "length", "stats")}) + "\n")
        val = [i for i in range(len(eps)) if i % max(1, round(1 / a.val_frac)) == 0] if a.val_frac > 0 else []
        (odir / "val_ids.txt").write_text(" ".join(map(str, val))); (odir / "train_ids.txt").write_text(" ".join(str(i) for i in range(len(eps)) if i not in set(val)))
        summary[task] = {"instruction": instr, "episodes": len(eps), "frames": frames, "hours": round(frames / 30 / 3600, 2), "val_episodes": len(val),
                         "status_counts": {k: sum(1 for v in status.values() if v[0] == k) for k in {v[0] for v in status.values()}}, "seconds": round(time.time() - t0, 1)}
        json.dump({"task": task, **summary[task], "video": {"size": a.size, "codec": "h264", "crf": a.crf}}, open(odir / "source.json", "w"), indent=1)
        print(f"{task:28s} success={len(eps):4d} frames={frames:8d} h={frames/30/3600:5.2f} val={len(val):3d} built={built} ({time.time()-t0:.0f}s) instr={instr!r}", flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
