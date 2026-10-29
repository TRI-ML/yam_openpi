#!/usr/bin/env bash
# SageMaker entrypoint for YAM pi05 fine-tunes. Configuration comes in as environment variables (set by the
# yam_eval train driver) and input channels; SageMaker's `train` argument is ignored.
#   OPENPI_CONFIG      openpi TrainConfig name (e.g. pi05_yam_pickbanana52 — the YAM LeRobot v2.1 recipe)
#   OPENPI_EXP_NAME    experiment name (checkpoints land in /opt/ml/checkpoints/<config>/<exp>/<step>/)
#   OPENPI_REPO_ID     LeRobot repo id of the dataset mounted at the `lerobot` channel (e.g. local/ToastBread_v1)
#   OPENPI_TRAIN_ARGS  extra tyro overrides, e.g. "--num-train-steps 300 --batch-size 32 --save-interval 300"
#   OPENPI_NORM_STATS_MAX_FRAMES  cap for the in-container norm-stats pass (default 200000; 0 = skip, stats must exist)
# Channels: lerobot -> /opt/ml/input/data/lerobot/{meta,data,videos}; config (provenance json) -> /opt/ml/input/data/config/.
set -euo pipefail
: "${OPENPI_CONFIG:?OPENPI_CONFIG must be set}"
: "${OPENPI_EXP_NAME:?OPENPI_EXP_NAME must be set}"
: "${OPENPI_REPO_ID:?OPENPI_REPO_ID must be set}"
cd /opt/ml/code
echo "=== openpi git sha: ${OPENPI_GIT_SHA:-?} ==="; echo "=== GPUs ==="; nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo "=== data channels ==="; ls /opt/ml/input/data 2>/dev/null || echo "(none)"
DATA_CH="${OPENPI_DATA_CHANNEL:-/opt/ml/input/data/lerobot}"
test -f "$DATA_CH/meta/info.json" || { echo "no LeRobot dataset at $DATA_CH (meta/info.json missing)"; ls -R /opt/ml/input/data | head -50; exit 2; }
# LeRobotDataset resolves <HF_LEROBOT_HOME>/<repo_id>; point that at the channel.
export HF_LEROBOT_HOME=/opt/lerobot_home
mkdir -p "$HF_LEROBOT_HOME/$(dirname "$OPENPI_REPO_ID")"
ln -sfn "$DATA_CH" "$HF_LEROBOT_HOME/$OPENPI_REPO_ID"
ASSETS=/opt/ml/assets
CKPT=/opt/ml/checkpoints
mkdir -p "$ASSETS" "$CKPT"
# Everything below shares ONE argv so the config the stats/preflight/train see is identical.
ARGS=("$OPENPI_CONFIG" --exp-name="$OPENPI_EXP_NAME" --data.repo-id="$OPENPI_REPO_ID" --data.assets.asset-id="$OPENPI_REPO_ID"
      --assets-base-dir="$ASSETS" --checkpoint-base-dir="$CKPT")
# shellcheck disable=SC2206
EXTRA=(${OPENPI_TRAIN_ARGS:-})
if [ "${1:-}" = "preflight" ]; then
  echo "=== preflight (CPU) ==="
  exec python scripts/sagemaker/yam_preflight.py -- "${ARGS[@]}" "${EXTRA[@]}"
fi
MAXF="${OPENPI_NORM_STATS_MAX_FRAMES:-200000}"
if [ "$MAXF" != "0" ]; then
  echo "=== norm stats (max_frames=$MAXF) ==="
  python scripts/sagemaker/yam_norm_stats.py --max-frames "$MAXF" -- "${ARGS[@]}" "${EXTRA[@]}"
fi
# A requeued spot attempt finds /opt/ml/checkpoints restored from S3: --resume continues, fresh start otherwise.
RESUME_ARG="--resume"; case " ${OPENPI_TRAIN_ARGS:-} " in *" --resume"*|*" --overwrite"*) RESUME_ARG="";; esac
echo "=== train ==="
exec python scripts/train.py "${ARGS[@]}" $RESUME_ARG "${EXTRA[@]}"
