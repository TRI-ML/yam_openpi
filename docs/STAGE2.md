# Stage-2 YAM pretraining (`pi05_yam_s2_plain` / `pi05_yam_s2_meta`)

Continues `pi05_yam_abc_mm2_h32` (docs/YAM_PRETRAIN.md) from the 3-node run's step 66699 on a wider YAM mix with success /
failure and sim / real data. The two configs are identical except for the prompt:

| config | prompt |
|---|---|
| `pi05_yam_s2_plain` | `<task>` |
| `pi05_yam_s2_meta` | `<quality=success\|failure> <domain=sim\|real> <task>`; the whole prefix is dropped with p = 0.3 (CFG style, deterministic per frame `index`) |

Shared (differences from `pi05_yam_abc_mm2_h32`): init = params of
`s3://tri-ml-datasets-uw2/tri-yam/train_runs/pi05_yam_abc_mm2_h32/pi05_full_yam_abc_mm2_h32_dgx3node/checkpoints/66699/params`
(params only, fresh optimizer; override with `YAM_S2_INIT_PARAMS=<dir>` or `--weight-loader.params-path <dir>`; an `s3://`
default is synced with the aws cli into `$OPENPI_DATA_HOME`), norm stats = the ones 66699 was trained with
(`.../checkpoints/66699/assets`, asset id `local/abc_mm2_train_v21`; override with `YAM_S2_ASSETS_DIR=<dir>` or
`--data.assets.assets-dir <dir>`), repo `local/yam_s2_train_v21`, lr 1e-5 cosine → 1e-6, 500 warmup,
60k steps, AdamW weight decay 1e-4, checkpoints every 10k (keep every 20k + the last 2), state dropout 0.1 (normalized state
zeroed per sample, after `Normalize`), per-sample source mix, wandb entity `tri`. Unchanged: pi05, horizon 32, action dim 32
(14 used), absolute actions, 3 cameras 224², bf16, FSDP 8, EMA 0.99, batch 128. There is no in-training val hook (offline
`yam/pretrain/eval_ckpt.py` as before), so no val-every-2.5k.

## Data

| source label | input (LeRobot v2.1) | quality | domain |
|---|---|---|---|
| `abc`, `molmo` | 10 % uniform episode subset (seed 0) of `s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/abc_molmo_v21/train` (label `abcmolmo`; split by its `meta/assembly.json`) | success | real |
| `oopsie_<lab>` | `s3://tri-ml-datasets-uw2/anthonyliang/oopsie_clean/yam/lerobot/{SE3Labs,Enact,AI2_MOLMOACT2,CMU_Aire/chunk_000..031}` | `success` column | real |
| `absim` | `s3://tri-ml-datasets-uw2/anthonyliang/oopsie/abc-sim/lerobot_v21` | `meta/episode_labels.jsonl` by global `episode_index` (`success_suboptimal` → success); 54 `trivial_success` dropped | sim |

* oopsie exports are 18-D: dims 0:14 are kept (the assembler asserts `action_mask` is exactly the 14 bimanual dims), cameras are
  renamed `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb` → `top/left_wrist/right_wrist` (videos linked, not re-encoded). Frame 0
  of every source has both wrist grippers at the bottom of the image, so the orientation matches ABC.
* Excluded: `Princeton_PRISM` (its `action` is a copy of the state) and `Berkeley_HybridRobotics` (single arm; needs action
  masking). The assembler refuses both labels.
* Never use the root `abc-sim/labels.jsonl` (its `episode_index` is per-source).
* Mix (`DataConfig.source_mix`, `SourceMixSampler` in `data_loader.py`): per sample 0.35 `abc+molmo` (frame-proportional inside) /
  0.35 `oopsie_*` (split equally over the 4 labs) / 0.30 `absim`, drawn with replacement from the per-source frame ranges in
  `meta/assembly.json`. Multi-host: each process draws its own stream (seed + process id) instead of the DistributedSampler.

## Build

Download the inputs onto the filesystem that holds `$HF_LEROBOT_HOME` (videos are hardlinked; symlinks otherwise). For ABC+Molmo
only the sampled 10 % is staged: `stage_subset.py` writes an s5cmd file for exactly the episodes `--sample abcmolmo=0.10:0`
keeps (both call `sample_episodes` in assemble_v21.py on the cache's full `meta/episodes.jsonl`; all of `meta/` is staged so
the assembler sees the same list). 0.10:0 = 16,317 / 163,174 episodes, 46.1M frames, 65,268 files, ~740 GB (estimated from a
16-episode staging) + ~400 MB meta. On a DGX node that already has the stage-1 repo, pass
`$HF_LEROBOT_HOME/local/abc_mm2_train_v21` as the `abcmolmo` input instead (same `--sample` picks the same episodes).

```
S=s3://tri-ml-datasets-uw2/anthonyliang; C=s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/abc_molmo_v21/train; IN=$HF_LEROBOT_HOME/../s2_inputs
mkdir -p $IN/cache_meta; for f in episodes.jsonl info.json tasks.jsonl assembly.json; do AWS_PROFILE=claude-profile aws s3 cp $C/meta/$f $IN/cache_meta/; done
python yam/pretrain/stage_subset.py $IN/cache_meta $C $IN/abc_molmo_train --sample 0.10:0 --out $IN/abc_molmo_s5cmd.txt
AWS_PROFILE=claude-profile AWS_REGION=us-west-2 s5cmd run $IN/abc_molmo_s5cmd.txt     # s5cmd needs the region (bucket is us-west-2)
for u in SE3Labs Enact AI2_MOLMOACT2 CMU_Aire; do aws s3 sync $S/oopsie_clean/yam/lerobot/$u $IN/oopsie/$u; done
aws s3 sync $S/oopsie/abc-sim/lerobot_v21 $IN/abc_sim

python yam/pretrain/assemble_v21.py $HF_LEROBOT_HOME/local/yam_s2_train_v21 --workers 48 \
  --repo abcmolmo=$IN/abc_molmo_train \
  --repo oopsie_SE3Labs=$IN/oopsie/SE3Labs --repo oopsie_Enact=$IN/oopsie/Enact --repo oopsie_AI2_MOLMOACT2=$IN/oopsie/AI2_MOLMOACT2 \
  $(for d in $IN/oopsie/CMU_Aire/chunk_*; do echo --repo oopsie_CMU_Aire=$d; done) \
  --repo absim=$IN/abc_sim --sample abcmolmo=0.10:0 --absim-keep failure
python yam/pretrain/build_lowdim_memmap.py $HF_LEROBOT_HOME/local/yam_s2_train_v21
```

Norm stats are not recomputed: both configs load the 66699 checkpoint's (identical to the cache's
`meta/norm_stats_pi05_yam_abc_mm2_h32.json` up to 4e-4 in the quantiles). Optional alternative, exact stats over the mix:
`python yam/pretrain/norm_stats_exact.py <repo> /local/s2_assets/local/yam_s2_train_v21` then train with
`YAM_S2_ASSETS_DIR=/local/s2_assets` and `--data.assets.asset-id local/yam_s2_train_v21` (the action/state scaling then
differs from what the init was trained with).

Every output parquet has 14-D `action` / `observation.state` plus `source` (abc|molmo|oopsie_<lab>|absim), `quality` (1/0) and
`domain` (0 real / 1 sim); `task_index` points into one global task table. `meta/assembly.json` has per-source episode / frame
counts, success / failure counts and the `[first_new_episode, end_new_episode)` / `[first_frame, end_frame)` ranges.

## Train (one 8-GPU DGX node)

```
export LEROBOT_SKIP_TS_CHECK=1 HF_LEROBOT_HOME=<...> XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true" \
  YAM_ASSEMBLY_JSON=$HF_LEROBOT_HOME/local/yam_s2_train_v21/meta/assembly.json
uv run scripts/train.py pi05_yam_s2_plain --exp-name <exp> --fsdp-devices 8
uv run scripts/train.py pi05_yam_s2_meta  --exp-name <exp> --fsdp-devices 8
# resume (same env): uv run scripts/train.py <cfg> --exp-name <exp> --fsdp-devices 8 --resume
```

wandb (`tri/openpi`): `dataset/*` once from `YAM_ASSEMBLY_JSON` (totals + `dataset/<source>/{episodes,frames,success_episodes,
failure_episodes}`); every `log_interval` (50): `loss`, `loss/{abc,oopsie,absim}` (per-sample loss reduced by source group,
abc includes molmo), `lr`, `grad_norm`, `param_norm`, `perf/{step_time_s,samples_per_s,loader_wait_s,gpu_mem_max_gb}`, and
`prefix/dropped_frac` on the meta run. Val per source: `yam/pretrain/eval_ckpt.py` on `local/abc_mm2_val_v21` (abc / molmo only;
there is no oopsie / ABC-sim val split).

## Smoke (one node, 200 steps)

The 135-episode `s3://tri-ml-datasets-uw2/anthonyliang/oopsie_clean/lerobot` subset contains no YAM episodes (embodiments
2/10/11/12/13 with 6-18 masked dims; the assembler's `action_mask` check refuses it), so the smoke takes its oopsie part from
the first 50 episodes of two YAM units instead, plus one ABC task (`abc130k_t121_roll_the_t_shirts`, 271 episodes in cache
chunk 152) and 300 ABC-sim episodes.

```
S=s3://tri-ml-datasets-uw2/anthonyliang; C=s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/abc_molmo_v21/train; IN=/tmp/s2_smoke_in
aws s3 sync $C $IN/abc --exclude '*' --include 'meta/*' --include 'data/chunk-152/*' --include 'videos/chunk-152/*'
for u in SE3Labs Enact; do aws s3 sync $S/oopsie_clean/yam/lerobot/$u $IN/$u --exclude '*' --include 'meta/*' --include '*/episode_0000[0-9]?.*'; done
aws s3 sync $S/oopsie/abc-sim/lerobot_v21 $IN/sim --exclude '*' --include 'meta/*' --include '*/episode_000[0-3]??.*'
python yam/pretrain/assemble_v21.py $HF_LEROBOT_HOME/local/yam_s2_smoke_v21 --workers 16 \
  --repo abcmolmo=$IN/abc --cache-sources abc130k_t121_roll_the_t_shirts \
  --repo oopsie_SE3Labs=$IN/SE3Labs --repo oopsie_Enact=$IN/Enact --repo absim=$IN/sim \
  --limit oopsie_SE3Labs=50 oopsie_Enact=50 absim=300
python yam/pretrain/build_lowdim_memmap.py $HF_LEROBOT_HOME/local/yam_s2_smoke_v21
YAM_ASSEMBLY_JSON=$HF_LEROBOT_HOME/local/yam_s2_smoke_v21/meta/assembly.json <train env as above> \
uv run scripts/train.py pi05_yam_s2_meta --exp-name s2_smoke --fsdp-devices 8 --overwrite \
  --data.repo-id local/yam_s2_smoke_v21 --num-train-steps 200 --lr-schedule.warmup-steps 20 --save-interval 100 --log-interval 10
```

Expect `source mix (pool -> sample share)` in the log, `loss/{abc,oopsie,absim}` and `prefix/dropped_frac` ≈ 0.3 in wandb.


## ABC-sim: failures only (2026-10-29)
ABC-sim outcomes are `success_suboptimal` / `failure`; a policy rollout that merely completed the task is not teleop-grade, so it must not
train `<quality=success>`. The assembler keeps only the 13,948 failures by default (`--absim-keep failure`; `--absim-keep all` restores the old
behaviour). Consequence: `<quality=success> <domain=sim>` has no training data until ABC's ~400 h sim teleop is added; sim eval prompts it anyway
(compositional). Mix is 0.40 / 0.35 / 0.25 (abc+molmo / oopsie / absim).

## SageMaker (1x ml.p5.48xlarge per run, AWS Batch `vla` + `vla-spot` race)

`yam/sagemaker/sm_entry.sh` treats any `CFG=pi05_yam_s2*` as stage 2: it does not build anything. It needs
`$DATA_CACHE_URI/_COMPLETE` (default `s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/yam_s2_train_v21`), copies
`$DATA_CACHE_URI/train/` (31,940 eps, 883 GB, with `meta/assembly.json` and `lowdim_mm/`) to
`$HF_LEROBOT_HOME/local/yam_s2_train_v21`, and exports `YAM_ASSEMBLY_JSON`. It skips norm stats and the abc_mm2 asset copy;
the config pulls the 66699 params and assets from S3 with the image's aws cli. Then the usual steps: resume, train, and publish the final step + logs to
`s3://tri-ml-datasets-uw2/tri-yam/train_runs/<cfg>/<exp>/{final,logs}/`.

**Checkpoints** (`yam/sagemaker/ckpt_sync.sh`; self-check `bash yam/sagemaker/test_ckpt_sync.sh`). `/opt/ml/checkpoints` and
`checkpoint_s3_uri` are not used. The first plain run died at its 10k save with an orbax `JSONDecodeError` and "Artifact upload failed":
SageMaker's sync agent was uploading orbax's in-progress files, so `.orbax-checkpoint-tmp-*` dirs ended up on S3. Now train.py
writes to local scratch (`$W/checkpoints/<cfg>/<exp>`) with synchronous orbax saves (`OPENPI_ASYNC_CKPT=0`). Every 5 min
(`CKPT_SYNC_EVERY`), plus once after training, a background uploader copies each committed step to
`CK_URI = s3://tri-ml-datasets-uw2/tri-yam/train_runs/<cfg>/<exp>/checkpoints_<qkey>/<cfg>/<exp>/<step>/`. A committed step has an
all-digit name and contains `_CHECKPOINT_METADATA`; on S3 the uploader writes `_UPLOADED` after all its other files. The uploader
also copies `wandb_id.txt` and prunes S3 to the config's keep policy: every 20k plus the last 2 (`KEEP_PERIOD`, `KEEP_LAST`).
`final/<step>/` is written the same way. There is no val watcher (no `abc_mm2_val_v21` is staged).
`--mode smoke` = 200 steps (warmup 20, save every 100) on the same repo, which still fetches all 883 GB.

**`_COMPLETE` marker** (same convention as `abc_molmo_v21`): write it at the prefix root only after the `train/` upload has finished
and been checked, because jobs refuse to start without it.
```
aws s3 ls --recursive --summarize s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/yam_s2_train_v21/train/ | tail -2   # ~883 GB
echo "$(date -u) yam_s2_train_v21 31940 eps" | aws s3 cp - s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/yam_s2_train_v21/_COMPLETE
```

**Image** (repo root on this branch; `COPY .` plus `uv sync --frozen`, awscli v2 + s5cmd included; the build imports both s2 configs):
```
export AWS_PROFILE=sagemaker AWS_REGION=us-west-2; ECR=124224456861.dkr.ecr.us-west-2.amazonaws.com
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR
IMG=$ECR/anthonyliang-openpi:s2-$(git rev-parse --short HEAD)
docker build -f yam/sagemaker/Dockerfile -t $IMG . && docker push $IMG
```

**Launch** (the launcher needs the sagemaker SDK, e.g. `~/Documents/vla_foundry/.venv/bin/python`; it reads `WANDB_API_KEY` from
`~/Documents/vla_foundry/secrets.env` and sets wandb `tri/openpi`). The default is priority 300 (OFFICIAL). Each run submits 2 jobs
(on-demand + spot), so both runs together hit the 4-node cap until the race watchers kill the losers.
```
PY=~/Documents/vla_foundry/.venv/bin/python; export AWS_PROFILE=sagemaker
OV="TRAIN_OVERRIDES=--fsdp-devices 8 --batch-size 128 --num-train-steps 60000 --lr-schedule.decay-steps 60000 --num-workers 32"
for v in plain meta; do
  $PY yam/sagemaker/launch_pi05.py --image $IMG --cfg pi05_yam_s2_$v --exp pi05_yam_s2_${v}_sm_p5 --max-run-days 2 --env "$OV"
  nohup bash yam/sagemaker/race_watch.sh /tmp/pi05_sm/jobs_pi05_yam_s2_${v}_sm_p5.json > /dev/null 2>&1 &
done
# winner: /tmp/pi05_sm/jobs_<exp>.winner; log: /tmp/pi05_sm/logs/race_watch.log
# smoke first (optional): --mode smoke --exp pi05_yam_s2_meta_sm_smoke --max-run-days 0.5, without --env "$OV"
```

**Expected time** (an estimate, not measured on p5): stage 2 on 8xA100 runs at 1.82 s/step at 16 samples/device. Stage 1 (same
model and batch) ran at 1.06-1.18 s/step on 8xH100. Assuming stage 2 is about as fast on H100, 60k steps take about 18-20 h.
Add the 883 GB fetch and JIT compile (assumed under 1 h) on every (re)start. `--max-run-days 2` leaves room for a couple of spot restarts.

**Resume.** On every start, sm_entry restores the latest step under `RESUME_FROM_URI` (default `CK_URI`) that has `_UPLOADED`,
together with `wandb_id.txt` (so it continues the same wandb run), and passes `--resume`. Orbax tmp dirs and partial uploads are ignored.
- No committed step: fresh start. `--overwrite` is never passed.
- Committed steps exist but the restore fails: the job exits (code 6) rather than start over on top of them.

Managed spot restarts the container on its own. A restart re-fetches the repo and loses at most the steps since the last uploaded
save (saves are every 10k; add `--save-interval 5000` to `TRAIN_OVERRIDES` to halve that). For spot, `max_wait` equals `max_run`
and also counts the time spent waiting. If the job ends before 60k, relaunch with the same `--cfg`/`--exp`:
- On the queue that won, which then resumes from its own `checkpoints_<qkey>`.
- Or on both queues, with `--env RESUME_FROM_URI=<the winner's CK_URI>` so that either twin resumes from it.

The first plain run (2026-10-30) left only an uncommitted `5000.orbax-checkpoint-tmp-0` (incomplete train_state), so its
relaunch starts from step 0. That leftover is ignored, and can be deleted with `aws s3 rm --recursive`.
