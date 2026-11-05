# Notes for coding agents (yam_openpi)

yam_openpi is our fork of openpi for training pi05 on YAM data: stage-1 pretraining on ABC + MolmoAct2
(`pi05_yam_abc_mm2_h32*`) and stage-2 mid-training on the YAM mix (`pi05_yam_s2_{plain,meta}`), on DGX nodes or on
SageMaker (via AWS Batch). Read `docs/YAM_PRETRAIN.md` (stage 1, data cache, multi-host, SageMaker) and
`docs/STAGE2.md` (stage-2 data, configs, SageMaker path, checkpoints) before changing anything.

## Where things live

| Path | What |
|---|---|
| `src/openpi/training/config.py` | train configs (`pi05_yam_abc_mm2_h32*`, `pi05_yam_s2_*`) |
| `src/openpi/transforms.py`, `src/openpi/training/data_loader.py` | YAM transforms, `SourceMixSampler` |
| `scripts/train.py` | the trainer (FSDP, orbax checkpoints, W&B) |
| `yam/pretrain/` | v2.1 repo assembler (`assemble_v21.py`), `stage_subset.py`, memmap/norm-stat tools |
| `yam/sagemaker/` | `launch_pi05.py`, `sm_entry.sh`, `ckpt_sync.sh` (+ `test_ckpt_sync.sh`), `race_watch.sh`, Dockerfile |
| `yam/multihost/` | JAX multi-process launch over ssh + tmux (`MULTIHOST.md`) |
| `yam/oopsie/reference/` | unported code from the retired dsrl_openpi_flip fork (reference only) |

## Skills

Skills under `.claude/skills/` are numbered procedures that end in a receipt.

| Skill | Use it for |
|---|---|
| `/launch-run` | one training run on a DGX node (survey, tmux) or SageMaker (priority 300 + race watcher) |
| `/run-status` | one table: W&B step/loss/step time, Batch job status, newest committed S3 checkpoint; no actions |
| `/find-ckpt` | committed checkpoints on S3 (`checkpoints_<q>/`, `final/`) and local `checkpoints/`; committed vs tmp |
| `/pr-ready` | pre-PR checklist: scope + report hygiene, `ruff --no-fix`, tests, docs row, PR body |

## Hard rules (from incidents)

- SageMaker: never write orbax checkpoints under `/opt/ml/checkpoints` (the sync agent uploads in-progress files and
  corrupts them). train.py writes to the local scratch dir; `ckpt_sync.sh` uploads committed steps only.
- On S3 only a step with `_UPLOADED` is committed. Resume from the newest committed step; a
  `*.orbax-checkpoint-tmp-*` dir is not a checkpoint.
- Root `abc-sim/labels.jsonl` is indexed per source: join labels via `lerobot_v21/meta/episode_labels.jsonl`.
- oopsie PRISM actions are a copy of the state and Berkeley is single-arm: the assembler refuses both; keep it that way.
- LeRobot v3 ABC-130k: train from `joint_action` / `joint_state`, not the 20-D EE columns.
- AWS Batch: priority 300 for Anthony's (official) project, at most 4 nodes in flight; every launch runs the race
  watcher so the losing on-demand/spot twin is killed.
- DGX: one 8-way FSDP run per node; check `nvidia-smi` for other users first; never `pkill -f` a pattern that
  matches your own ssh command (kill by pid).
- Report hygiene: experiment launch scripts, logs, PDFs, reports, backups (`*.bak*`) and personal paths
  (`/local/aliangdw`, `/home/...`) stay out of PRs. A PR carries implementation, tests and docs.
- Evidence: a claim of launched, fixed or verified carries one line of evidence (pid, W&B run, S3 path, test count);
  without it, label the claim `GUESS`.
- W&B: entity `tri`, project `openpi`.
- `pyproject.toml` has no ruff `fix = true`, but lint with `uvx ruff check --no-fix <files>` anyway and fix by hand.
