---
name: find-ckpt
description: "Locate committed pi05 checkpoints on S3 (checkpoints_<queue>/, final/) and in local checkpoints/ dirs; tell committed steps from orbax tmp dirs and partial uploads. Usage: /find-ckpt <cfg> [exp] [step]"
---

# find-ckpt

## Where things live

- SageMaker runs (`sm_entry.sh`): `s3://tri-ml-datasets-uw2/tri-yam/train_runs/<cfg>/<exp>/`
  - `checkpoints_<queue>/<cfg>/<exp>/<step>/` rolling saves (keep last 2 + every 20000), plus `wandb_id.txt`
  - `final/<step>/` the last committed step, published at the end (`params/`, `assets/`)
  - `logs/`, `logs_prep/`, `eval/eval_val_<step>.json`
- DGX runs: `<checkout>/checkpoints/<cfg>/<exp>/<step>/` (`params/`, `train_state/`, `assets/`, `_CHECKPOINT_METADATA`).

## Committed or not

- S3: committed = the step dir has `_UPLOADED` (written last by `ckpt_sync.sh`). A step without it is a partial
  upload; never resume from or serve it.
- Local: committed = all-digit name + `_CHECKPOINT_METADATA`. `<step>.orbax-checkpoint-tmp-*` is an in-progress save,
  not a checkpoint (old runs that used `/opt/ml/checkpoints` may have these on S3: ignore them).

## Procedure

1. List: `aws s3 ls s3://tri-ml-datasets-uw2/tri-yam/train_runs/<cfg>/<exp>/` then each `checkpoints_*/<cfg>/<exp>/`
   and `final/`.
2. For each candidate step: `aws s3 ls <step>/_UPLOADED` (S3) or `ls <step>/_CHECKPOINT_METADATA` (local).
3. Size check of the newest committed step vs the previous one (`aws s3 ls --recursive --summarize <step>/ | tail -2`).
4. For serving or init, the path is `<...>/<step>/params` (what `CheckpointWeightLoader` takes).

## Report format (mandatory)

```
run      : <cfg> / <exp>
newest   : <uri>/<step>  _UPLOADED: yes  size <GB>
final    : <uri>/final/<step>/params | none
other    : <committed steps>   ignored: <tmp / partial dirs>
```
Anything not listed on S3 or disk during this run is `GUESS`.
