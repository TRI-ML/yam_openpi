#!/bin/bash
# Evaluate every finished checkpoint of <exp> once (every 10k steps). Device: EVAL_DEVICE=cpu (default) or a GPU index.
W=${W:-/raid/aliangdw/pi05_abc_molmo}; source ${ENV_SH:-$W/scripts/env.sh}; S=${S:-$(cd "$(dirname "$0")" && pwd)}; EXP=${1:?exp}; CFG=${CFG:-pi05_yam_abc_mm2_h32}; CK=${CKPT_BASE:-$W/checkpoints}/$CFG/$EXP; DONE=$W/logs/eval_done_$EXP.txt; LOG=$W/logs/eval_$EXP.log; touch $DONE
export EVAL_WANDB_ID=${EVAL_WANDB_ID:-$(cat $W/logs/eval_wandb_id_$EXP.txt 2>/dev/null || (tr -dc a-z0-9 </dev/urandom | head -c 8 | tee $W/logs/eval_wandb_id_$EXP.txt))}
if [ "${EVAL_DEVICE:-cpu}" = "cpu" ]; then export JAX_PLATFORMS=cpu; else export CUDA_VISIBLE_DEVICES=$EVAL_DEVICE XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2; fi
while true; do
  for d in $(ls -d $CK/[0-9]* 2>/dev/null | sort -t/ -k1 -V); do
    st=$(basename $d); [[ "$st" =~ ^[0-9]+$ ]] || continue; grep -qx "$st" $DONE && continue; [ -d $d/params ] && [ -f $d/_CHECKPOINT_METADATA ] || continue
    echo "$(date +%F_%T) eval step $st on ${EVAL_DEVICE:-cpu}" >> $LOG
    $PY $S/eval_ckpt.py --config $CFG --ckpt $d --wandb-run ${EXP}_eval >> $LOG 2>&1 && echo "$st" >> $DONE || echo "$(date +%F_%T) eval FAILED step $st" >> $LOG
  done
  [ -f $W/STOP_EVAL ] && break; sleep 600
done
