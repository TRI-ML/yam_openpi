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
`python yam/pretrain/norm_stats_exact.py <repo> <assets_dir>/local/yam_s2_train_v21` then train with
`YAM_S2_ASSETS_DIR=<assets_dir>` and `--data.assets.asset-id local/yam_s2_train_v21` (the action/state scaling then
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


## v2: + ABC sim teleop (`pi05_yam_s2v2_plain` / `pi05_yam_s2v2_meta`)

Same as the s2 pair (66699 params + assets, 60k steps, batch 128, lr, prompts, state dropout, FSDP 8) except the repo
`local/yam_s2v2_train_v21` and the mix 0.35 `abc+molmo` / 0.30 `oopsie_*` / **0.20 `absim_teleop`** / 0.15 `absim` (failures).
This gives `<quality=success> <domain=sim>` real training data.

New source: `s3://sagemaker-us-west-2-124224456861/zheyuan.hu/data/abc_sim_v3ar/abc_sim_<task>/`, 24 LeRobot **v3.0** task repos,
54,064 episodes / 44.4M frames / 412 h at 30 fps, 348 GB, robot_type `yam_bimanual_sim`, the same loong schema as abc130k_v3ar
(`joint_action` / `joint_state` [14] are used; the 20-D EE `action` / `observation.state` are dropped). Cameras
top/left_wrist/right_wrist are 256x352 AV1 (g=2). Per-frame bool `is_demo` / `is_intervention` columns. Staged on dgx06 at
`/datasets/aliangdw/stage2/abc_sim_v3ar/<repo>/` (same layout as S3).

| source label | `--repo` label | input | quality | domain |
|---|---|---|---|---|
| `absim_teleop` | `absimteleop` | `/datasets/aliangdw/stage2/abc_sim_teleop_v21`: ONE v2.1 dir merging all 24 task repos | 1 (teleop success) | 1 (sim) |

The assembler refuses nothing for this label. It prints the `is_intervention` frame fraction and records it as
`sources.absim_teleop.intervention_frac` in `meta/assembly.json` (information only). Every source in assembly.json now also has
`task_episodes` (episodes per task string). The loss telemetry gets a 4th group, `loss/absim_teleop`.

All commands below run on dgx06 from the repo root, with the openpi venv (`PY=.venv/bin/python`). Keep `$HF_LEROBOT_HOME` on
`/datasets` so videos are hardlinked instead of symlinked. `ffmpeg`/`ffprobe` must be on PATH (or set `FFMPEG` / `FFPROBE`).

**(a) v3.0 → v2.1 (stage-1 path, lossless `ffmpeg -c copy`).** This is the same converter as stage 1, plus `--keep-flags` to keep
`is_demo` / `is_intervention`. Do not pass the stage-1 molmo options (`--exclude molmo_yam_v3:molmo_val_ids.txt` / `--only`):
ABC sim has no val holdout. `--skip` is not needed either, since all 24 repos have the 3 cameras. The real stage-1 CLI takes the
output root positionally; the `--out` / `--exclude-molmo-episodes` spelling in YAM_PRETRAIN.md §1 is shorthand. `convert_source.py`
parallelises over the v3 data files: there are 7-27 per repo, 305 in total. So run 3 repos at a time, as `sm_entry.sh` does. A
re-run resumes, because finished episodes leave `ep_*.done`.
```
PY=.venv/bin/python; V3=/datasets/aliangdw/stage2/abc_sim_v3ar; ST=/datasets/aliangdw/stage2/abc_sim_teleop_stage
mkdir -p $ST; ls $V3 | grep ^abc_sim_ | xargs -P 3 -I{} sh -c "$PY yam/pretrain/convert_source.py $V3/{} $ST/{} --workers 48 --keep-flags >> $ST/convert.log 2>&1 || echo {} >> $ST/failed.txt"
for d in $V3/abc_sim_*; do [ -f $ST/$(basename $d)/source.json ] || echo "MISSING $d"; done     # expect nothing; 24 source.json
jq -s 'map(.episodes) | add' $ST/abc_sim_*/source.json                                           # 54064
jq -r '.video_modes | keys[]' $ST/abc_sim_*/source.json | sort | uniq -c                    # expect "copy" only
$PY yam/pretrain/assemble_v21.py /datasets/aliangdw/stage2/abc_sim_teleop_v21 $ST/abc_sim_* --workers 48 --robot-type yam_bimanual_sim
```
The intermediate's `meta/assembly.json` lists each `abc_sim_<task>` with its episode count. Its parquets keep the two flag
columns, which its info.json does not list, so only the stage-2 assembler should read it, not lerobot.

**(b) Merged stage-2 repo.** Use the exact v1 inputs (from `yam_s2_train_v21/train/meta/assembly.json` `inputs`) plus the new
label. `--sample abcmolmo=0.10:0` picks the same 16,317 episodes (sha1 `a7de6582…`).
```
I=/datasets/aliangdw/stage2; R=$HF_LEROBOT_HOME/local/yam_s2v2_train_v21
$PY yam/pretrain/assemble_v21.py $R --workers 48 \
  --repo abcmolmo=$I/abc_molmo_train \
  --repo oopsie_SE3Labs=$I/oopsie_v21/SE3Labs --repo oopsie_Enact=$I/oopsie_v21/Enact --repo oopsie_AI2_MOLMOACT2=$I/oopsie_v21/AI2_MOLMOACT2 \
  $(for c in 000 002 014 015 017 020 021 026; do echo --repo oopsie_CMU_Aire=$I/oopsie_v21/CMU_Aire/chunk_$c; done) \
  --repo absim=$I/absim_v21 --repo absimteleop=$I/abc_sim_teleop_v21 \
  --sample abcmolmo=0.10:0 --absim-keep failure
jq -c '.sources | map_values({episodes, frames, intervention_frac})' $R/meta/assembly.json
jq -r '.dropped["'$I'/abc_molmo_train"]' $R/meta/assembly.json                                  # sha1 a7de6582dac23aa1f88a7863f6f53c39e61a2bb6
```
Expected: v1's 31,940 episodes / 75.0M frames (molmo 3,269, abc 13,048, oopsie 1,015 / 329 / 75 / 256, absim 13,948), plus
absim_teleop 54,064 / ~44.4M. That totals ~86,004 episodes / ~119.4M frames, ~1.23 TB.

**(c) Memmaps.** `$PY yam/pretrain/build_lowdim_memmap.py $R --workers 48`

**(d) Upload + `_COMPLETE`.** Same convention as v1: write the marker last, after checking the upload.
```
U=s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/yam_s2v2_train_v21
AWS_REGION=us-west-2 s5cmd --numworkers 256 --retry-count 10 cp "$R/*" $U/train/
find $R -type f -o -type l | wc -l; aws s3 ls --recursive --summarize $U/train/ | tail -2           # same object count, ~1.23 TB
echo "$(date -u) yam_s2v2_train_v21 86004 eps" | aws s3 cp - $U/_COMPLETE
```

**(e) SageMaker.** For `CFG=pi05_yam_s2v2*`, `sm_entry.sh` defaults `DATA_CACHE_URI` to `$U` and the repo dir to
`local/yam_s2v2_train_v21`; it is otherwise the s2 path. The image must be built from a commit that has the s2v2 configs (see
"Image" above; tag e.g. `s2v2-$(git rev-parse --short HEAD)`). Launch at priority 300 with the same overrides:
```
PY=~/Documents/vla_foundry/.venv/bin/python; export AWS_PROFILE=sagemaker
OV="TRAIN_OVERRIDES=--fsdp-devices 8 --batch-size 128 --num-train-steps 60000 --lr-schedule.decay-steps 60000 --num-workers 32"
for v in plain meta; do
  $PY yam/sagemaker/launch_pi05.py --image $IMG --cfg pi05_yam_s2v2_$v --exp pi05_yam_s2v2_${v}_sm_p5 --max-run-days 2 --priority 300 --env "$OV"
  nohup bash yam/sagemaker/race_watch.sh /tmp/pi05_sm/jobs_pi05_yam_s2v2_${v}_sm_p5.json > /dev/null 2>&1 &
done
```
Every (re)start now fetches ~1.23 TB instead of 883 GB.

**(f) Smoke (200 steps).** Run this after (c), before (d) / (e). It uses dgx06's 8 GPUs on the built repo, so only the 66699 params and assets are fetched:
```
export LEROBOT_SKIP_TS_CHECK=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true" YAM_ASSEMBLY_JSON=$R/meta/assembly.json
uv run scripts/train.py pi05_yam_s2v2_meta --exp-name s2v2_smoke --fsdp-devices 8 --overwrite \
  --num-train-steps 200 --lr-schedule.warmup-steps 20 --save-interval 100 --log-interval 10
```
In the log, `source mix (pool -> sample share)` should show abc+molmo 0.35, the 4 oopsie labs 0.075 each, absim_teleop 0.20 and
absim 0.15. In wandb, expect `loss/{abc,oopsie,absim,absim_teleop}` and `prefix/dropped_frac` ≈ 0.3. The same smoke on SageMaker
(it fetches the full repo) is `--cfg pi05_yam_s2v2_meta --mode smoke --exp pi05_yam_s2v2_meta_sm_smoke --max-run-days 0.5`, without `--env "$OV"`.

**Conversion time (estimate, not measured).** Stage 1 staged and converted 8 TB in ~100 min on a p5. Here the job is 348 GB, read
and rewritten on dgx06 NFS (~0.75-1 GB/s aggregate measured for DGX NFS copies), so ≥ 12 min of IO. It also runs ~54k × 6
ffmpeg/ffprobe processes. Expect ~20-40 min for (a) and a few minutes for the hardlink assembly.

## v3: stratified mix + canonical sim prompts + per-field prefix dropout (`pi05_yam_s3_meta`)

Same v2 repo (`local/yam_s2v2_train_v21`, same cache) and warm start; what changes is how it is sampled and prompted.
Motivation (v2 at 120k): equal lab slots gave AI2 MolmoAct's 75 episodes ~13 epochs while sim teleop saw 0.07, ABC-real loss
drifted up after 85k, and the sim-failure prompts ("sim pouring beads", "grab the X and place it in the box", "put the A left
of the B") differed in phrasing from the 24 teleop task names, so `<quality=failure>` was confounded with prompt style.

- **Sampler** `StratifiedMixSampler` (`data_config.strata_mix`): batch share fixed per stratum (domain x quality)
  `real_success .30 / real_failure .20 / sim_success .30 / sim_failure .20`; inside a stratum each (source, quality) pool is
  weighted `frames**strata_alpha` (0.5). It reads the per-episode sidecar `meta/episode_strata.jsonl`:
  ```
  $PY yam/pretrain/build_episode_strata.py $HF_LEROBOT_HOME/local/yam_s2v2_train_v21 --workers 48      # ~1 min on NFS
  aws s3 cp $R/meta/episode_strata.jsonl $U/train/meta/episode_strata.jsonl                             # additive; cache otherwise untouched
  ```
  Realised shares at 120k x 128 (epochs): abc 16.1% (0.07), molmo 7.2% (0.14), oopsie succ SE3/Enact/CMU ~2% each (0.4-0.5),
  AI2 succ 0.3% (3.5); oopsie fail CMU 10.3% (0.69), SE3 4.6% (1.5), Enact 3.3% (2.2), AI2 1.9% (3.8); sim teleop 30% (0.10);
  sim failures 20% (0.13). Failures = 40% of batches.
- **Prompt remap** `yam/pretrain/absim_prompt_remap.json` (`data_config.prompt_remap_json`): all 1,431 ABC-sim rollout prompts ->
  8 teleop task names (6 `sim ...` -> their teleop name, `grab the … box` -> `grab clutter`, `put the A <rel> the B` -> `put relative`).
  Applied in `QualityDomainPrefixPrompt`, the cache's `tasks.jsonl` is not edited.
  The path in the config is derived from `config.py`'s location; the loader fails at once with `FileNotFoundError` if the
  file is not in the checkout.
- **Loader robustness** (opt-in, set on the stage-2/3 configs only): `video_tolerance_s=0.05` (lerobot's 1e-4 killed the run on a
  stream-copied mp4 one frame short of its parquet, ep 60186) and `skip_bad_samples=True` (`SkipBadSamples`: a failed read is
  replaced by a random other index, up to 8 tries, warned once per episode, counted as `data/skipped_samples`). Other configs keep
  lerobot's defaults and fail loudly.
- **Dropout** `meta_prefix_field_dropout=True`: quality and domain dropped independently (p=0.3 each) so inference can guide on
  one axis; telemetry `prefix/quality_dropped_frac`, `prefix/domain_dropped_frac`, `prefix/dropped_frac` (= 1-0.7^2 = 0.51).
- **Schedule**: one cosine 500 warmup 1e-5 -> 1e-6 over 120k steps, batch 128 (no warm restart). Meta model only.

Smoke (one GPU, real repo): `scripts/train.py pi05_yam_s3_meta --exp-name s3_smoke --fsdp-devices 1 --batch-size 8 --num-train-steps 60
--lr-schedule.warmup-steps 10 --lr-schedule.decay-steps 60 --save-interval 50 --log-interval 10` and check the log for the strata
table, "prompt remap … 1431 of 1783 task strings remapped", `prefix/*_frac` ~0.3/0.3/0.51 and a committed checkpoint.
SageMaker: image `s3-<sha>`, `launch_pi05.py --cfg pi05_yam_s3_meta --exp pi05_yam_s3_meta_sm_p5 --priority 300 --max-run-days 3
--data-cache-uri s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/yam_s2v2_train_v21 --env "TRAIN_OVERRIDES=… --num-train-steps 120000
--lr-schedule.decay-steps 120000 …"`. ~29 h at 0.86 s/step.

## v3 sim-only (`pi05_yam_s3sim_meta`, 2026-11-06)

Twin of `pi05_yam_s3_meta` with `strata_mix = {real_success: 0, real_failure: 0, sim_success: .5, sim_failure: .5}`: every batch is
ABC-sim teleop : ABC-sim failure rollouts exactly 1:1, no real data. Zero-share strata are dropped by the sampler (their sources are
simply never drawn; logged at init). Same warm start, norm stats, prompt remap (the only thing separating the two pools' prompts is
then the `<quality=…>` token) and per-field dropout. At 120k x 128: teleop 7.68M draws / 44.4M frames = 0.17 epochs, failures
7.68M / 23.9M = 0.32 epochs. Per-task overlap after the remap: failures exist for only 8 of the 24 teleop task names (46% of teleop
frames); a task-matched 1:1 would need the other 16 tasks' failures (not available) or dropping those tasks. Smoke / launch as v3 with
`CFG=pi05_yam_s3sim_meta EXP=pi05_yam_s3sim_meta_sm_p5`, image `s3sim-<sha>`; dgx06 `/local` was full, so the smoke writes checkpoints
to NFS via `--checkpoint-base-dir /datasets/aliangdw/stage2/smoke_ckpts`.
