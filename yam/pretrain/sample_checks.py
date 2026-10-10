"""Sample checks on the assembled repo through openpi's actual pipeline: repack -> YAMInputs -> (model transforms incl. resize)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
import lerobot.common.datasets.lerobot_dataset as L
import pyarrow.parquet as pq

from openpi import transforms as T
from openpi.training import config as C
from openpi.training import data_loader as DL

cfg = C.get_config("pi05_yam_abc_molmo_full"); dc = cfg.data.create(cfg.assets_dirs, cfg.model)
ds = DL.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
meta = L.LeRobotDatasetMetadata(dc.repo_id); asm = json.load(open(f"{os.environ['HF_LEROBOT_HOME']}/local/abc_molmo_v21/meta/assembly.json"))
print("repo:", meta.total_episodes, "eps", meta.total_frames, "frames", len(meta.tasks), "tasks | sources:", {k: v["episodes"] for k, v in asm["sources"].items()})
full = DL.transform_dataset(ds, dc)          # repack + YAMInputs + Normalize + model transforms (resize/pad/tokenize)
pre = T.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])   # repack + YAMInputs only (14-D, native image size)
np.set_printoptions(precision=3, suppress=True, linewidth=220)
W = os.environ["W"]
def src_row(stage, ep, fr):
    t = pq.read_table(f"{W}/stage/{stage}/ep_{ep:06d}.parquet").to_pandas(); r = t[t.frame_index == fr].iloc[0]; return np.asarray(r["action"], np.float32), np.asarray(r["observation.state"], np.float32)
# one molmo episode (new ep 7 = source ep 7) and one ABC episode (first episode of the 2nd source)
abc_name = [k for k in asm["sources"] if k != "molmo_yam_v3"][0]; abc_first = asm["sources"][abc_name]["first_new_episode"]
for name, new_ep, src_ep, fr in [("molmo_yam_v3", 7, 7, 300), (abc_name, abc_first + 3, None, 300)]:
    if src_ep is None:  # map new -> source episode id via the source's sorted episodes.jsonl
        recs = sorted(json.loads(l)["episode_index"] for l in open(f"{W}/stage/{name}/episodes.jsonl")); src_ep = recs[new_ep - abc_first]
    i = int(sum(meta.episodes[k]["length"] for k in range(new_ep))) + fr   # episodes are contiguous 0..N-1
    t0 = time.time(); r = pre(ds[i]); dt = time.time() - t0; f = full[i]
    a_src, s_src = src_row(name, src_ep, fr)
    img = r["image"]["base_0_rgb"]; print(f"\n[{name}] new_ep {new_ep} (src ep {src_ep}) frame {fr}: fetch {dt*1000:.0f} ms")
    print("  raw image (YAMInputs):", {k: tuple(v.shape) for k, v in r["image"].items()}, img.dtype, "| after model transforms:", {k: tuple(v.shape) for k, v in f["image"].items()}, f["image"]["base_0_rgb"].dtype)
    print("  state 14-D bit-exact vs source:", np.array_equal(np.asarray(r["state"], np.float32), s_src), "| actions[0] bit-exact:", np.array_equal(np.asarray(r["actions"][0], np.float32), a_src), "| actions shape", tuple(np.asarray(r["actions"]).shape), "| padded state/actions:", tuple(np.asarray(f["state"]).shape), tuple(np.asarray(f["actions"]).shape))
    print("  prompt:", repr(r["prompt"]), "| tokenized:", tuple(np.asarray(f["tokenized_prompt"]).shape) if "tokenized_prompt" in f else "n/a")
    print("  normalized state[:7]:", np.asarray(f["state"])[:7], "(quantile-normed, expect ~[-1,1])")
    # lag + gripper over the episode (from stage parquet)
    t = pq.read_table(f"{W}/stage/{name}/ep_{src_ep:06d}.parquet").to_pandas().sort_values("frame_index"); ja = np.stack(t.action); js = np.stack(t["observation.state"])
    print("  action->state best lag:", int(np.argmin([np.abs(ja[:-10]-js[k:len(js)-10+k]).mean() for k in range(10)])), "frames | gripper L state min/max", js[:,6].min().round(3), js[:,6].max().round(3), "(1=open)")
