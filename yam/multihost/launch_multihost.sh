#!/bin/bash
# Launch one JAX process per node (8 GPUs each), coordinator on the first node.
# Usage: NODES="03 06" [CFG=pi05_yam_abc_mm2_h32] CKPT_BASE=/home/aliangdw/pi05_mh_ckpt launch_multihost.sh <exp> [tyro overrides...]
# Requirements: identical $W (data repo, venv, code+patches, assets) on every node; CKPT_BASE on a filesystem shared by all
# nodes (/home or /datasets: orbax writes per-process shards into one directory); node-to-node ssh as aliangdw.
set -euo pipefail
NODES=${NODES:?'e.g. NODES="03 06"'}; EXP=${1:?exp_name}; shift
CFG=${CFG:-pi05_yam_abc_mm2_h32}; CKPT_BASE=${CKPT_BASE:?shared checkpoint base dir}; PORT=${PORT:-12355}
W=${W:-/raid/aliangdw/pi05_abc_molmo}   # node work dir: env.sh, logs/, lerobot_home/
read -ra ARR <<< "$NODES"; N=${#ARR[@]}; COORD="tri-hq-ml-dgx-${ARR[0]}:$PORT"
echo "launching $N procs: coord $COORD cfg $CFG exp $EXP ckpt $CKPT_BASE/$CFG/$EXP extra: $*"
for i in "${!ARR[@]}"; do
  node=tri-hq-ml-dgx-${ARR[$i]}
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node" "tmux has-session -t mh 2>/dev/null || tmux new-session -d -s mh; \
    tmux kill-window -t mh:${EXP}_p${i} 2>/dev/null; tmux new-window -d -t mh: -n ${EXP}_p${i} \"env XLA_FLAGS='${XLA_FLAGS:-}' NCCL_DEBUG='${NCCL_DEBUG:-WARN}' WANDB_TAGS='${WANDB_TAGS:-}' bash $W/scripts/mh_run.sh $N $i $COORD $CFG $EXP $CKPT_BASE $*; sleep 86400\""
  echo "  proc $i -> $node  (tmux mh:${EXP}_p${i}; log $W/logs/mh_${EXP}_p${i}.log)"; [ "$i" = 0 ] && sleep 3
done
echo "stop: for n in $NODES; do ssh tri-hq-ml-dgx-\$n \"tmux kill-window -t mh:${EXP}_p\\\$(...)\"; done  (or pkill -f 'exp-name $EXP' on each node)"
