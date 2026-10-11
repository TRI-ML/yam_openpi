#!/usr/bin/env python
"""Submit the pi05 YAM pretraining as SageMaker training jobs through the AWS Batch 'vla' service queues (on-demand + spot race).
Usage: launch_pi05.py --image <ecr uri or digest> [--exp NAME] [--queues vla vla-spot] [--priority 1001] [--max-run-days 5] [--mode full|smoke]
Checkpoints: the container writes to scratch and mirrors finished steps to CKPT_S3_URI (= <OUT_URI>/<exp>/checkpoints) itself
(yam/sagemaker/ckpt_sync.sh). SageMaker's own checkpoint sync (checkpoint_s3_uri) is deliberately NOT used: its agent plants
`*.sagemaker-uploaded` markers inside orbax tmp dirs and breaks checkpoint finalization. Resubmitting with the same --exp resumes."""
import argparse
from datetime import datetime
import json
import os
import re

import boto3
from sagemaker import Session
from sagemaker.aws_batch.training_queue import TrainingQueue
from sagemaker.estimator import Estimator

ROLE = "arn:aws:iam::124224456861:role/Robotics-LBM-SagemakerAccess-us-west-2"
REGION = "us-west-2"; USER = "anthonyliang"
QUEUE = {"vla": "fss-vla-p5-48xlarge-us-west-2", "vla-spot": "fss-vla-spot-p5-48xlarge-us-west-2",
         "cv-wfm-spot": "fss-cv-wfm-spot-p5-48xlarge-us-west-2"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--exp", default="pi05_full_yam_abc_mm2_h32_sm_p5")
    ap.add_argument("--cfg", default="pi05_yam_abc_mm2_h32"); ap.add_argument("--mode", default="full")
    ap.add_argument("--queues", nargs="+", default=["vla", "vla-spot"]); ap.add_argument("--priority", type=int, default=1001)
    ap.add_argument("--max-run-days", type=float, default=5); ap.add_argument("--volume-size", type=int, default=500)
    ap.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE for the container")
    ap.add_argument("--num-workers", type=int, default=32, help="torch DataLoader workers (loader_wait was 0.3-0.4 s/step at 32)")
    ap.add_argument("--train-overrides", default="", help="extra tyro flags for scripts/train.py, e.g. '--num-train-steps 60 --save-interval 20'")
    ap.add_argument("--out", default="/tmp/pi05_sm/jobs.json")
    a = ap.parse_args()
    os.environ.setdefault("AWS_PROFILE", "claude-profile"); os.environ["AWS_DEFAULT_REGION"] = REGION
    sess = Session(boto_session=boto3.Session(profile_name=os.environ["AWS_PROFILE"], region_name=REGION))
    secrets = {}
    for line in open(os.path.expanduser("~/Documents/vla_foundry/secrets.env")):
        if "=" in line and not line.startswith("#"): k, v = line.strip().split("=", 1); secrets[k] = v.strip().strip("\"'")
    base_env = {"CFG": a.cfg, "EXP": a.exp, "MODE": a.mode, "WANDB_API_KEY": secrets["WANDB_API_KEY"], "WANDB_ENTITY": "tri",
                "WANDB_PROJECT": "openpi", "FSDP_DEVICES": "8", "HF_TOKEN": secrets.get("HF_TOKEN", ""),
                "OUT_URI": f"s3://tri-ml-datasets-uw2/tri-yam/train_runs/{a.cfg}", "NUM_WORKERS": str(a.num_workers),
                "CKPT_S3_URI": f"s3://tri-ml-datasets-uw2/tri-yam/train_runs/{a.cfg}/{a.exp}/checkpoints"}
    if a.train_overrides: base_env["TRAIN_OVERRIDES"] = a.train_overrides
    for kv in a.env: k, v = kv.split("=", 1); base_env[k] = v
    max_run = int(a.max_run_days * 86400); submitted = []
    for qkey in a.queues:
        spot = "spot" in qkey; qname = QUEUE[qkey]
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", f"pi05-yam-{a.mode}-{qkey}-{USER}-{stamp}")[:63].strip("-")
        ckpt_uri = base_env["CKPT_S3_URI"]   # same prefix for every queue: the race leaves exactly one writer
        # reserved-capacity (on-demand) queues require SM_USE_RESERVED_CAPACITY=1; spot/training-plan queues want 0 (same rule as vla_foundry)
        env = dict(base_env, SM_QUEUE=qkey, SM_USE_RESERVED_CAPACITY=("0" if spot else "1"))
        est = Estimator(image_uri=a.image, role=ROLE, instance_count=1, instance_type="ml.p5.48xlarge", sagemaker_session=sess,
                        volume_size=a.volume_size, max_run=max_run, max_wait=max_run if spot else None, use_spot_instances=spot,
                        environment=env,   # no checkpoint_s3_uri on purpose (see module docstring)
                        output_path=f"s3://tri-ml-datasets-uw2/tri-yam/train_runs/{a.cfg}/{a.exp}/sm_output_{qkey}",
                        keep_alive_period_in_seconds=None, debugger_hook_config=False, disable_profiler=True,
                        tags=[{"Key": "tri.project", "Value": "MM:PJ-0077"}, {"Key": "tri.owner.email", "Value": "anthony.liang@tri.global"},
                              {"Key": "run", "Value": a.exp}])
        q = TrainingQueue(queue_name=qname)
        qj = q.submit(est, inputs=None, job_name=job_name, priority=a.priority, share_identifier="default",
                      timeout={"attemptDurationSeconds": max_run})
        jid = getattr(qj, "job_arn", None) or getattr(qj, "job_id", None) or str(qj)
        print(f"submitted {job_name} -> {qname} id={jid}")
        submitted.append({"queue": qname, "queue_key": qkey, "job_name": job_name, "batch_job": str(jid), "ckpt_uri": ckpt_uri, "spot": spot})
    json.dump({"exp": a.exp, "cfg": a.cfg, "image": a.image, "jobs": submitted, "submitted_at": datetime.now().isoformat()}, open(a.out, "w"), indent=1)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
