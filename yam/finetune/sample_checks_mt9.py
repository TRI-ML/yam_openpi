"""Sample checks on the mt9 fine-tune repo through openpi's actual pipeline: repack -> YAMInputs -> Normalize (PRETRAIN stats)
-> model transforms. One sample per task (= per assembly.json source): image keys/shapes, 14-D state/action bit-exact vs the
staged source parquet, normalised state within [-1, 1] under the pretrain quantiles, prompt tokenised, action->state lag,
gripper range. Env: HF_LEROBOT_HOME, YAM_PRETRAIN_ASSETS_DIR, MT9_SOURCES (stage dir with <task>/ep_*.parquet), optional CFG."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
import lerobot.common.datasets.lerobot_dataset as L  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from openpi import transforms as T  # noqa: E402
from openpi.training import config as C  # noqa: E402
from openpi.training import data_loader as DL  # noqa: E402

cfg = C.get_config(os.environ.get("CFG", "pi05_yam_mt9_h32_ft_smoke")); dc = cfg.data.create(cfg.assets_dirs, cfg.model)
ds = DL.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
meta = L.LeRobotDatasetMetadata(dc.repo_id)
asm = json.load(open(f"{os.environ['HF_LEROBOT_HOME']}/{dc.repo_id}/meta/assembly.json"))
print("repo:", dc.repo_id, meta.total_episodes, "eps", meta.total_frames, "frames", len(meta.tasks), "tasks | norm stats:", dc.norm_stats is not None)
full = DL.transform_dataset(ds, dc)
pre = T.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])
np.set_printoptions(precision=3, suppress=True, linewidth=220)
S = os.environ["MT9_SOURCES"]; starts = np.cumsum([0] + [meta.episodes[i]["length"] for i in range(meta.total_episodes)])
ok_all = True
for name, info in asm["sources"].items():
    if not info["episodes"]:
        continue
    new_ep = info["first_new_episode"] + min(2, info["episodes"] - 1); src_ep = info["source_episode_ids"][new_ep - info["first_new_episode"]]
    t = pq.read_table(f"{S}/{name}/ep_{src_ep:06d}.parquet").to_pandas().sort_values("frame_index"); ja = np.stack(t.action); js = np.stack(t["observation.state"])
    fr = min(300, len(t) - 40); i = int(starts[new_ep]) + fr
    t0 = time.time(); r = pre(ds[i]); dt = time.time() - t0; f = full[i]
    st_ok = np.array_equal(np.asarray(r["state"], np.float32), js[fr].astype(np.float32)); ac_ok = np.array_equal(np.asarray(r["actions"][0], np.float32), ja[fr].astype(np.float32))
    ns = np.asarray(f["state"])[:14]; in_range = bool(np.all(np.abs(ns) <= 1.05))
    lag = int(np.argmin([np.abs(ja[:-10] - js[k:len(js) - 10 + k]).mean() for k in range(10)]))
    ok_all &= st_ok and ac_ok and in_range
    print(f"\n[{name}] new_ep {new_ep} (src {src_ep}) frame {fr}: fetch {dt*1000:.0f} ms | prompt={r['prompt']!r}")
    print("  images:", {k: tuple(v.shape) for k, v in r["image"].items()}, "-> model:", {k: tuple(v.shape) for k, v in f["image"].items()}, f["image"]["base_0_rgb"].dtype)
    print("  state bit-exact:", st_ok, "| actions[0] bit-exact:", ac_ok, "| actions", tuple(np.asarray(r["actions"]).shape), "| tokenized:", tuple(np.asarray(f["tokenized_prompt"]).shape) if "tokenized_prompt" in f else "n/a")
    print("  normalized state[:14]:", ns, "| all within [-1.05,1.05]:", in_range)
    print("  action->state lag:", lag, "frames | gripper state L/R min..max", js[:, 6].min().round(3), js[:, 6].max().round(3), js[:, 13].min().round(3), js[:, 13].max().round(3), "(1=open)")
print("\nALL CHECKS PASS:", ok_all)
