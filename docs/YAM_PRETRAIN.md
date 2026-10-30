# pi05 pretraining on the public YAM corpora (ABC-130k + MolmoAct2)

Full-parameter pi05 (from `pi05_base`) on **Zheyuan's LeRobot v3.0 reprocessing** of the two large bimanual-YAM datasets, in the
**14-D absolute joint action space** shared with our station fine-tunes and the LFV team's Cosmos3 world-action model.
Everything here lives under `yam/` (tooling), `src/openpi/training/config.py` (`pi05_yam_abc_mm2_h32*`), `scripts/train.py`
(multi-host + telemetry) and `src/openpi/training/data_loader.py` (`YamLeRobotDataset`).

## 1. Data

| source | S3 (LeRobot v3.0, Zheyuan 2026-10-07) | episodes / frames | images |
|---|---|---|---|
| ABC-130k train | `s3://sagemaker-us-west-2-124224456861/zheyuan.hu/data/abc130k_v3ar/<task>/` (201 task repos) | 131,251 / 388M | 3 cams AV1 256x416 (some 256x352) |
| MolmoAct2-BimanualYAM | `.../zheyuan.hu/data/molmo_yam_v3ar/molmo_yam_v3/` (one repo) | 32,246 / 76M | 3 cams AV1 256x448 |
| ABC-130k val (official) | `.../zheyuan.hu/data/abc130k_val_v3ar/<task>/` (189 repos) | 1,671 | same |

Schema ("loong schema 3"): the default `action` / `observation.state` columns are a **20-D end-effector pose** (synthesised FK) — we
do **not** use them. The joints are `joint_action` (commanded, leads measured by ~3 frames) and `joint_state` (measured), both
`[L joints 0-5, L gripper, R joints 0-5, R gripper]` in radians, grippers fraction-open 0..1 (**1 = open**, verified on frames).
Cameras `observation.images.{top,left_wrist,right_wrist}`; the right wrist is stored rotated 180° (same as every YAM export).
Values are bit-identical to our own foundry shards for the same recordings.

Equivalence with the AnyData webbed eef32 record (Cosmos3): `L_xyz(3) L_rot6(6) R_xyz(3) R_rot6(6) L_grip R_grip L_joints(6) R_joints(6)`
→ `joint_action == concat(eef32[20:26], eef32[18], eef32[26:32], eef32[19])`.

### Build the training repo (v3 → v2.1, lossless)

openpi pins LeRobot **v2.1**; lerobot 0.4 (the v3 reader) cannot be installed next to openpi's pins. So the v3 repos are converted
to **one merged v2.1 repo** without touching pixels:

```
yam/pretrain/convert_source.py  <v3 repo dir> <stage dir> --workers 48   # per source: per-episode parquet (joint_action->action,
                                                                           # joint_state->observation.state, 20-D EE dropped) + per-camera
                                                                           # mp4 cut with `ffmpeg -c copy` at the episode's keyframe
yam/pretrain/assemble_v21.py    <stage dirs...> --out <HF_LEROBOT_HOME>/local/abc_mm2_train_v21 [--exclude-molmo-episodes molmo_val_ids.txt]
                                                                           # contiguous episode ids (molmo first, ABC largest-first), global
                                                                           # task table, hard-linked videos, v2.1 meta, assembly.json
yam/pretrain/build_lowdim_memmap.py <repo>                                 # <repo>/lowdim_mm/{action,observation.state}.npy (see §3)
yam/pretrain/norm_stats_exact.py / norm_stats_fast.py <repo> --config ...  # exact float64 quantile stats over ALL frames (§4)
```

Validation repo `local/abc_mm2_val_v21` = ABC official val + the **323 MolmoAct2 episodes of the webbed val split**
(`yam/pretrain/molmo_val_ids.txt`, mapped by index + task name; excluded from train) → 1,915 episodes / 5.85M frames. The merged
repos are cached at `s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/abc_molmo_v21/{train,val}/` (+`_COMPLETE`); SageMaker jobs and
DGX nodes fetch the cache instead of rebuilding (~100 min from the v3 sources at ~1.9 GB/s).

Mix: uniform frame sampling over the merged repo = **frame-proportional** (84 % ABC / 16 % MolmoAct2; the 10 largest ABC tasks are
27 %). Zubair's WAM used the same proportions.

## 2. Configs (`src/openpi/training/config.py`)

| name | use | batch | steps | notes |
|---|---|---|---|---|
| `pi05_yam_abc_mm2_h32` | 1 node (8xH100/A100-80GB) | 128 | 200k | 25.6M samples; SageMaker run `pi05_full_yam_abc_mm2_h32_sm_p5` |
| `pi05_yam_abc_mm2_h32_3node` | 3 nodes multi-host | 384 (128/node) | 66.7k | same samples; DGX run `pi05_full_yam_abc_mm2_h32_dgx3node` |
| `pi05_yam_abc_mm2_h32_smoke` | LoRA twin, one 48 GB card | 8 | 200 | repo `local/abc_mm2_smoke_v21` (any assembled subset) |
| `pi05_yam_abc_mm2_h32_ft` | fine-tune template on a station repo | 32 | 15k | §6 |
| `pi05_yam_s2_plain` / `pi05_yam_s2_meta` | stage 2 from step 66699 (+ oopsie-YAM, ABC-sim; quality/domain prompt) | 128 | 60k | docs/STAGE2.md |

Shared: pi05, `action_horizon=32` (1.07 s at 30 Hz = the Cosmos3 WAM chunk; openpi default YAM configs use 10), images 224x224
resize-with-pad (aspect preserved), prompt = task string, cosine lr 2.5e-5 → 2.5e-6 with 1k warmup, AdamW clip 1.0, EMA 0.99,
bf16, FSDP over the 8 local devices, `prompt_from_task`, `action_sequence_keys=("action",)`.

## 3. Data-loader scale fixes (`YamLeRobotDataset`, `make_lerobot_dataset`)

At 463M frames / 141 parquet shards the pinned lerobot v2.1 has two problems; both are handled in `data_loader.py` without
patching lerobot:

* `hf_dataset.select()` per sample = **2.7 s** (deep-copies the dataset info incl. a 163k-entry `download_checksums` dict and rebuilds
  the shard schema). `YamLeRobotDataset._query_hf_dataset` reads the state/action window from `<repo>/lowdim_mm/*.npy` memmaps when
  present (**0.12 s/sample**, bit-exact) and drops `download_checksums`. Env `LEROBOT_LOWDIM_MEMMAP=0` disables.
* the constructor's timestamp-sync check materialises every timestamp through the per-row torch transform (hours).
  `LEROBOT_SKIP_TS_CHECK=1` skips it (only for repos assembled by `assemble_v21.py`, whose timestamps are `frame_index/fps` exactly).

Without the memmaps a 3-node run sat at 4–6 s/step with idle GPUs; with them the loader waits 0.06–0.4 s/step.

## 4. Norm stats

pi05 uses quantile normalisation (q01→-1, q99→+1) of the 14-D state/action (padded to 32). `scripts/compute_norm_stats.py` samples
through the full pipeline (hours at this scale, and it now drops image keys so mixed widths collate); `yam/pretrain/norm_stats_exact.py`
computes exact stats straight from the parquet/memmaps in minutes. **Accumulate in float64**: a float32 axis-0 mean saturates at ~1.7e7
elements and silently returns wrong means. Stats are one pooled set over both sources (same robot, units and gripper convention) and
live in the checkpoint under `assets/<asset_id>/norm_stats.json`; downstream fine-tunes should reuse them (§6).

## 5. Launch

**Single node** (any 8-GPU host):
```
LEROBOT_SKIP_TS_CHECK=1 HF_LEROBOT_HOME=<...> XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true" \
uv run scripts/train.py pi05_yam_abc_mm2_h32 --exp-name <exp> --fsdp-devices 8 [--resume|--overwrite]
```
**Multi-host (JAX multi-process, one process per node)** — `yam/multihost/launch_multihost.sh` (ssh + tmux, coordinator = first node,
NCCL over IB) and `mh_run.sh`; `scripts/train.py` initialises `jax.distributed` from `OPENPI_NPROC / OPENPI_COORD_ADDR / OPENPI_PROC_ID`
(default off), wandb/tqdm/checkpoint-dir creation happen on process 0 with an NFS visibility barrier, and the torch loader uses a
`DistributedSampler` per process (disjoint shards; `[multihost-check]` logs a per-process batch hash). Measured 1.91x on 2 nodes
(96 % scaling), 2.25 s/step at global batch 384 on 24 A100s. Details, pitfalls (>10 rsync streams to one DGX node truncate copies; NFS
negative-lookup race on the checkpoint dir) and the switch procedure: `yam/multihost/MULTIHOST.md`.

**SageMaker (1x ml.p5.48xlarge via the AWS Batch service queues)** — `yam/sagemaker/`: `Dockerfile` (build from the repo root),
`sm_entry.sh` (stage from the v3 sources or the S3 cache → convert → assemble → memmaps → norm stats → train; checkpoints in
`/opt/ml/checkpoints` so SageMaker mirrors them and a spot restart `--resume`s; final params + eval jsons + logs to
`s3://tri-ml-datasets-uw2/tri-yam/train_runs/<cfg>/<exp>/`), `launch_pi05.py` (submits to the `vla` and `vla-spot` queues at
priority 300 (OFFICIAL tier); the on-demand queue needs `SM_USE_RESERVED_CAPACITY=1`), `race_watch.sh` (terminates the losing twin once one job
trains). Measured 1.06–1.18 s/step at batch 128 on 8xH100.

Telemetry logged to wandb (`tri/openpi`): `dataset/*` composition once (from `YAM_ASSEMBLY_JSON`), `perf/step_time_s`,
`perf/samples_per_s`, `perf/loader_wait_s`, `perf/gpu_mem_max_gb` every `log_interval`.

## 6. Eval and fine-tuning

* `yam/pretrain/eval_ckpt.py --config <cfg> --ckpt <step dir> --wandb-run <exp>_eval` scores a checkpoint on the val repo:
  `val/{abc,molmo}/{mse,mae,mse_joints,mae_joints,mse_gripper,mae_gripper,gripper_openclose_acc}` + per-horizon `mse_per_h/hNN`.
  `eval_watcher.sh <exp>` runs it on every finished checkpoint (CPU is fine, ~40 min per checkpoint). Reference: the Cosmos3 WAM
  pretraining ended at action MSE 0.035–0.040 on the ABC buckets.
* Kept checkpoints: `s3://tri-ml-datasets-uw2/tri-yam/train_runs/pi05_yam_abc_mm2_h32/<exp>/checkpoints/<step>/{params,assets,...}`.
* **Fine-tune** on a station task with `pi05_yam_abc_mm2_h32_ft`: `YAM_PRETRAIN_PARAMS=<local copy of .../checkpoints/<step>/params>`
  `YAM_PRETRAIN_ASSETS_DIR=<.../checkpoints/<step>/assets>` `YAM_FT_REPO_ID=local/<task>_yam` — the pretrain norm stats are reused via
  `AssetsConfig`, horizon stays 32, cameras are our `scene_camera/left_wrist_camera/right_wrist_camera` keys.
* **Serving**: `scripts/write_deploy_sidecar.py --checkpoint-dir <step dir> --config-name <cfg> [--task-instruction ...]` writes the
  `deploy_metadata.json` YAM Eval reads; `scripts/serve_policy.py policy:checkpoint --policy.config=<cfg> --policy.dir=<step dir>`
  serves it (chunk `(action_horizon, 14)`; the station runner learns the chunk length from the first reply). The server logs a
  `READY:` line, exits cleanly on SIGTERM, and ignores the bare-TCP liveness probes YAM Eval sends.

## 7. Runs so far (2026-10-09/10)

| run | hardware | s/step | status |
|---|---|---|---|
| `pi05_full_yam_abc_mm2_h32_dgx3node` (wandb tri/openpi `tr1xq2uf`) | dgx02+dgx03+dgx06, 24xA100 | 2.25 @ 384 | 66.7k steps, finishes 2026-10-11 |
| `pi05_full_yam_abc_mm2_h32_sm_p5` | SageMaker 1x p5 (spot) | 1.06–1.18 @ 128 | 200k steps, finishes 2026-10-13 |
