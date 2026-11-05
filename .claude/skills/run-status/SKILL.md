---
name: run-status
description: "Report one status table for pi05 runs: W&B step / loss / step time, AWS Batch service-job status, newest committed S3 checkpoint. Reports only, no actions. Usage: /run-status [exp filter]"
---

# run-status

## Procedure

1. Inventory: W&B `tri/openpi` runs in state running/crashed in the last 48 h matching the filter, plus the jobs
   json files under `/tmp/pi05_sm/jobs_*.json` (and their `.winner`).
2. W&B per run (one script, summary only): `_step`, `loss`, `step_time` (or the per-interval step delta),
   heartbeat age, state.
3. Batch per job: `aws batch describe-service-job --job-id <id> --query '[status,latestAttempt.serviceResourceId.value]'`,
   then `aws sagemaker describe-training-job --training-job-name <name> --query SecondaryStatus`.
4. Checkpoint per run: newest step under `<OUT_URI>/<exp>/checkpoints_<q>/<cfg>/<exp>/` that has `_UPLOADED`
   (see `/find-ckpt`); `final/<step>/` if the run finished. DGX runs: newest all-digit dir in
   `checkpoints/<cfg>/<exp>/` with `_CHECKPOINT_METADATA`.
5. Classify: healthy (step advancing), JIT/fetch window (job Training, no W&B step yet, < 1 h), frozen (step
   unchanged across two looks >= 20 min apart), dead (job FAILED / pid gone, not at budget), duplicate (two live
   jobs for one exp after the race should have settled: report at once), finished (final/ present).

## Report format (mandatory)

```
| exp | where | job/pid status | step | loss | s/step | hb | newest committed ckpt | state |
|-----|-------|----------------|------|------|--------|----|------------------------|-------|
```
One row per run, then one evidence line per anomaly (`<exp>: <what> -- <metric or log line, timestamp>`).
This skill launches, kills and resumes nothing.
