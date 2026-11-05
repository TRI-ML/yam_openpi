#!/usr/bin/env python
"""Offline validation of a pi05 checkpoint on the held-out YAM val repo (local/abc_mm2_val_v21).
Per source (abc / molmo): action MSE & MAE (all / joints / gripper), per horizon step, gripper open-close accuracy (thr 0.5).
Logs to a wandb run <exp>_eval (same project) with x = ckpt_step, and writes <ckpt>/eval_val.json."""
import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np

sys.path.insert(0, "src")
import lerobot.common.datasets.lerobot_dataset as L

from openpi import transforms as T
from openpi.policies import policy_config as PC
from openpi.training import config as C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="pi05_yam_abc_mm2_h32"); ap.add_argument("--ckpt", required=True, help="checkpoint step dir")
    ap.add_argument("--val-repo", default="local/abc_mm2_val_v21"); ap.add_argument("--n-per-source", type=int, default=192); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="openpi"); ap.add_argument("--wandb-run", default=None); ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--group-by", default="abc_molmo", choices=["abc_molmo", "source"], help="source = one group per assembly.json source (e.g. per task for the mt9 repos)")
    ap.add_argument("--out-json", default=None, help="where to write results (default <ckpt>/eval_val.json)"); a = ap.parse_args()
    cfg = C.get_config(a.config); H = cfg.model.action_horizon; ckpt = pathlib.Path(a.ckpt); step = int(ckpt.name)
    meta = L.LeRobotDatasetMetadata(a.val_repo)
    ds = L.LeRobotDataset(a.val_repo, delta_timestamps={"action": [i / meta.fps for i in range(H)]}, video_backend="pyav")
    pft = T.PromptFromLeRobotTask(meta.tasks)
    asm = json.load(open(pathlib.Path(os.environ["HF_LEROBOT_HOME"]) / a.val_repo / "meta/assembly.json"))
    mol = asm["sources"].get("molmo_yam_v3"); mol_range = range(mol["first_new_episode"], mol["first_new_episode"] + mol["episodes"]) if mol else range(0)
    starts = np.cumsum([0] + [meta.episodes[i]["length"] for i in range(meta.total_episodes)])
    rng = random.Random(a.seed)
    def sample(src):
        eps = [e for e in range(meta.total_episodes) if (e in mol_range) == (src == "molmo")]
        out = []
        if not eps: return out   # source absent from this val repo (e.g. smoke without molmo)
        for _ in range(a.n_per_source):
            e = rng.choice(eps); n = meta.episodes[e]["length"]; fr = rng.randrange(0, max(1, n - H)); out.append(int(starts[e]) + fr)
        return out
    if a.group_by == "source":
        def sample_range(lo, hi):
            out = []
            for _ in range(a.n_per_source):
                e = rng.randrange(lo, hi); n = meta.episodes[e]["length"]; fr = rng.randrange(0, max(1, n - H)); out.append(int(starts[e]) + fr)
            return out
        idx = {name: sample_range(sinfo["first_new_episode"], sinfo["first_new_episode"] + sinfo["episodes"]) for name, sinfo in asm["sources"].items() if sinfo["episodes"]}
    else:
        idx = {k: v for k, v in {"abc": sample("abc"), "molmo": sample("molmo")}.items() if v}
    policy = PC.create_trained_policy(cfg, ckpt)
    res = {}; t0 = time.time()
    for src, ids in idx.items():
        se = []; ae = []; gt_g = []; pr_g = []; ok = 0; n_valid = 0
        for i in ids:
            s = pft(ds[i]); gt = np.asarray(s["action"], np.float32); pad = np.asarray(s["action_is_pad"]).astype(bool)
            obs = {"observation/image_head": np.asarray(s["observation.images.top"]), "observation/image_left_wrist": np.asarray(s["observation.images.left_wrist"]),
                   "observation/image_right_wrist": np.asarray(s["observation.images.right_wrist"]), "observation/state": np.asarray(s["observation.state"], np.float32), "prompt": s["prompt"]}
            pred = np.asarray(policy.infer(obs)["actions"], np.float32)[:H]
            m = ~pad; d = (pred - gt)[m]; se.append(np.full((H, 14), np.nan, np.float32)); se[-1][m] = d ** 2; ae.append(np.where(m[:, None], np.abs(pred - gt), np.nan))
            gt_g.append(gt[m][:, [6, 13]]); pr_g.append(pred[m][:, [6, 13]])
        se = np.stack(se); ae = np.stack(ae); gt_g = np.concatenate(gt_g); pr_g = np.concatenate(pr_g)
        J = [i for i in range(14) if i not in (6, 13)]; G = [6, 13]
        r = {"mse": float(np.nanmean(se)), "mae": float(np.nanmean(ae)), "mse_joints": float(np.nanmean(se[:, :, J])), "mae_joints": float(np.nanmean(ae[:, :, J])),
             "mse_gripper": float(np.nanmean(se[:, :, G])), "mae_gripper": float(np.nanmean(ae[:, :, G])),
             "gripper_openclose_acc": float(np.mean((pr_g > 0.5) == (gt_g > 0.5))), "n_samples": len(ids),
             "mse_per_h": [float(x) for x in np.nanmean(se, axis=(0, 2))], "mae_per_h": [float(x) for x in np.nanmean(ae, axis=(0, 2))]}
        res[src] = r; print(src, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if "per_h" not in k}, flush=True)
    res["eval_seconds"] = time.time() - t0; res["ckpt_step"] = step; json.dump(res, open(a.out_json or (ckpt / "eval_val.json"), "w"), indent=1)
    if not a.no_wandb:
        import wandb
        run = wandb.init(project=a.wandb_project, name=a.wandb_run or f"{cfg.exp_name if hasattr(cfg,'exp_name') else a.config}_eval", id=os.environ.get("EVAL_WANDB_ID"), resume="allow", config={"config": a.config, "val_repo": a.val_repo, "n_per_source": a.n_per_source})
        wandb.define_metric("ckpt_step"); wandb.define_metric("val/*", step_metric="ckpt_step")
        log = {"ckpt_step": step}
        for src, r in res.items():
            if not isinstance(r, dict): continue
            for k, v in r.items():
                if k.endswith("per_h"):
                    for h, x in enumerate(v): log[f"val/{src}/{k}/h{h:02d}"] = x
                else: log[f"val/{src}/{k}"] = v
        wandb.log(log); run.finish()
    print("done", res["eval_seconds"], "s")

if __name__ == "__main__": main()
