#!/bin/bash
# Per-node runner for multi-host openpi training. Called by launch_multihost.sh inside tmux on each node.
# Usage: mh_run.sh <nproc> <proc_id> <coord_host:port> <cfg> <exp> <ckpt_base> [tyro overrides...]
NPROC=$1; PID=$2; COORD=$3; CFG=$4; EXP=$5; CKPT_BASE=$6; shift 6
W=${W:-/raid/aliangdw/pi05_abc_molmo}; source ${ENV_SH:-$W/scripts/env.sh}
mkdir -p $W/logs
export OPENPI_NPROC=$NPROC OPENPI_PROC_ID=$PID OPENPI_COORD_ADDR=$COORD OPENPI_LOCAL_DEVICES=${OPENPI_LOCAL_DEVICES:-8}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0} NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export WANDB_ENTITY=tri WANDB_TAGS=${WANDB_TAGS:-"pi05,full_ft,pretrain,yam,abc130k_v3ar,molmoact2_v3ar,lerobot_v3,joint14_abs,horizon32,multihost"}
export PYTHONUNBUFFERED=1
echo "$(date +%F_%T) mh_run host=$(hostname) nproc=$NPROC pid=$PID coord=$COORD cfg=$CFG exp=$EXP ckpt=$CKPT_BASE args=$* XLA_FLAGS=$XLA_FLAGS" | tee -a $W/logs/mh_${EXP}_p${PID}.log
exec $PY scripts/train.py $CFG --exp-name $EXP --checkpoint-base-dir $CKPT_BASE "$@" 2>&1 | tee -a $W/logs/mh_${EXP}_p${PID}.log
