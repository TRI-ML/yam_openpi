import json
import pathlib
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).parent


def _fake_v21(d, tasks, n=5):
    """Tiny merged v2.1 dir as the stage-1 assembly of convert_source.py --keep-flags writes it (one episode per task)."""
    (d / "meta").mkdir(parents=True)
    for ep in range(len(tasks)):
        a = np.arange(n * 14, dtype=np.float32).reshape(n, 14)
        t = pa.table({"action": pa.FixedSizeListArray.from_arrays(pa.array(a.ravel()), 14),
                      "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(a.ravel()), 14),
                      "timestamp": pa.array(np.arange(n, dtype=np.float32) / 30), "frame_index": pa.array(np.arange(n)),
                      "episode_index": pa.array(np.full(n, ep)), "index": pa.array(np.arange(n)), "task_index": pa.array(np.full(n, ep)),
                      "is_demo": pa.array(np.ones(n, bool)), "is_intervention": pa.array(np.arange(n) < ep)})
        p = d / f"data/chunk-000/episode_{ep:06d}.parquet"; p.parent.mkdir(parents=True, exist_ok=True); pq.write_table(t, p)
        for c in ("top", "left_wrist", "right_wrist"):
            v = d / f"videos/chunk-000/observation.images.{c}/episode_{ep:06d}.mp4"; v.parent.mkdir(parents=True, exist_ok=True); v.write_bytes(b"x")
    (d / "meta/info.json").write_text(json.dumps({"fps": 30, "chunks_size": 1000}))
    (d / "meta/episodes.jsonl").write_text("".join(json.dumps({"episode_index": i, "tasks": [t], "length": n}) + "\n" for i, t in enumerate(tasks)))
    (d / "meta/episodes_stats.jsonl").write_text("".join(json.dumps({"episode_index": i, "stats": {}}) + "\n" for i in range(len(tasks))))


def test_absimteleop(tmp_path):
    _fake_v21(tmp_path / "in", ["hang the mug", "spell cat", "hang the mug"])
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(HERE / "assemble_v21.py"), str(out), "--repo", f"absimteleop={tmp_path / 'in'}", "--workers", "1"], check=True)
    t = pq.read_table(out / "data/chunk-000/episode_000002.parquet")
    assert set(t["source"].to_pylist()) == {"absim_teleop"}
    assert set(t["quality"].to_pylist()) == {1}
    assert set(t["domain"].to_pylist()) == {1}
    assert t["index"].to_pylist() == list(range(10, 15))
    assert "is_intervention" not in t.column_names
    s = json.loads((out / "meta/assembly.json").read_text())["sources"]["absim_teleop"]
    assert (s["episodes"], s["frames"], s["success_episodes"], s["domain"]) == (3, 15, 3, "sim")
    assert s["task_episodes"] == {"hang the mug": 2, "spell cat": 1}
    assert s["intervention_frac"] == round(3 / 15, 6)  # episode i flags its first i frames
    assert (out / "videos/chunk-000/observation.images.right_wrist/episode_000001.mp4").exists()
