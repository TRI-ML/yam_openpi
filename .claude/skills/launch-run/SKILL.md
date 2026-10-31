---
name: launch-run
description: "Launch one pi05 training run on a DGX node (scripts/train.py in tmux) or on SageMaker (yam/sagemaker/launch_pi05.py, priority 300, race watcher), with the node survey and data-cache check. Usage: /launch-run <cfg> <exp> [dgx node | sagemaker]"
---

# launch-run

Launch from a clean `main` checkout (`git pull --ff-only`) and state the sha in the receipt. Config names live in
`src/openpi/training/config.py`; data prep is in `docs/YAM_PRETRAIN.md` / `docs/STAGE2.md`.

## Procedure

1. Data. The training repo must be complete before anything starts:
   - SageMaker / S3 cache: `aws s3 ls <DATA_CACHE_URI>/_COMPLETE` must exist (`sm_entry.sh` refuses otherwise).
   - DGX: `$HF_LEROBOT_HOME/<repo_id>/meta/info.json` exists and the episode count matches `meta/assembly.json`.
2. DGX node survey, right before launching:
   ```bash
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
   ps -o user=,pid=,etime=,cmd= -p <pid> | cut -c1-120   # every pid -> user
   df -h . ; cat /proc/loadavg
   ```
   One 8-way FSDP run per node. Any foreign process on the GPUs: report user + pids and ask; do not kill.
3. DGX launch, one tmux session named after the exp:
   ```bash
   tmux new-session -d -s <exp> "cd <checkout> && LEROBOT_SKIP_TS_CHECK=1 HF_LEROBOT_HOME=<...> \
     XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py <cfg> --exp-name <exp> --fsdp-devices 8 \
     2>&1 | tee checkpoints/<cfg>/<exp>.log"
   ```
   Resume = same command with `--resume` (never `--overwrite` on a run that has committed steps).
   Cleanup kills by pid (`pgrep -f '^<venv>/bin/python .*scripts/train.py'`), never a bare `pkill -f`.
4. SageMaker launch (launcher needs the sagemaker SDK venv; see docs/STAGE2.md "SageMaker"):
   ```bash
   $PY yam/sagemaker/launch_pi05.py --image $IMG --cfg <cfg> --exp <exp> --priority 300 --max-run-days 2 --env "$OV"
   nohup bash yam/sagemaker/race_watch.sh /tmp/pi05_sm/jobs_<exp>.json > /dev/null 2>&1 &
   ```
   Each launch submits two jobs (on-demand + spot); keep at most 4 nodes in flight in total. The race watcher must
   run for every launch; it terminates the losing twin and writes `/tmp/pi05_sm/jobs_<exp>.winner`.
   Never point checkpoints at `/opt/ml/checkpoints`; `ckpt_sync.sh` owns uploads.
5. Post-launch (within 10 min, again after JIT): DGX pid alive and GPUs at the expected memory; SageMaker job
   `Training`; W&B run visible in `tri/openpi` with a step > 0.

## Receipt (mandatory)

```
run      : <cfg> / <exp>   W&B tri/openpi <run id>
tree     : <checkout> @ <sha>
placement: <node> tmux <session> pid <pid> | SageMaker <training job> (batch <id>, queue <q>, priority 300)
data     : <repo or DATA_CACHE_URI> _COMPLETE: yes
evidence : <one live log line or W&B step with timestamp>
```
A launch is running only with a live pid / Training status and a log line; otherwise it is `GUESS: launched`.
