#!/bin/bash
# SageMaker entrypoint: stage Zheyuan's v3ar LeRobot repos -> lossless v2.1 train/val repos -> exact norm stats -> pi05 training
# with orbax checkpoints in local scratch ($W/checkpoints, never /opt/ml/checkpoints: SageMaker's sync agent raced orbax's writes); a background
# uploader (ckpt_sync.sh) copies committed steps to CK_URI every 5 min; on (re)start the latest committed step of RESUME_FROM_URI is restored + --resume.
# Stage 2 (CFG=pi05_yam_s2_*, docs/STAGE2.md): fetches the prebuilt merged repo from DATA_CACHE_URI (needs _COMPLETE), no norm stats
# (init params + assets are S3 paths in the config), then the same resume/train/publish steps.
# Env (set by the launcher): CFG EXP MODE(full|smoke) SRC_ROOT DATA_CACHE_URI OUT_URI OPENPI_CACHE_URI WANDB_* TRAIN_OVERRIDES
#   CK_URI (default $OUT_URI/$EXP/checkpoints_$SM_QUEUE/$CFG/$EXP) RESUME_FROM_URI (default CK_URI) CKPT_SYNC_EVERY (s, 300)
#   SMOKE_ABC_TASKS (smoke: space-separated ABC task dirs) SMOKE_VAL_TASKS FSDP_DEVICES
set -uo pipefail
[ "${1:-}" = "train" ] && shift
CFG=${CFG:-pi05_yam_abc_mm2_h32}; EXP=${EXP:-pi05_full_yam_abc_mm2_h32_sm_p5}; MODE=${MODE:-full}
SRC_ROOT=${SRC_ROOT:-s3://sagemaker-us-west-2-124224456861/zheyuan.hu/data}
S2=0; case $CFG in pi05_yam_s2*) S2=1;; esac
if [ $S2 = 1 ]; then DATA_CACHE_URI=${DATA_CACHE_URI:-s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/yam_s2_train_v21}
else DATA_CACHE_URI=${DATA_CACHE_URI:-s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/abc_molmo_v21}; fi
OUT_URI=${OUT_URI:-s3://tri-ml-datasets-uw2/tri-yam/train_runs/$CFG}
OPENPI_CACHE_URI=${OPENPI_CACHE_URI:-s3://tri-ml-datasets-uw2/tri-yam/pretrain_data/openpi_cache}
MOLMO_WEBBED=${MOLMO_WEBBED:-s3://tri-ml-datasets-uw2/geyan/anydata/cv_webbed/videos/molmoact2_v2}
FSDP_DEVICES=${FSDP_DEVICES:-8}
CODE=${CODE:-/opt/ml/code/yam_openpi}; S=$CODE/yam/pretrain; PY=$CODE/.venv/bin/python
CK_URI=${CK_URI:-$OUT_URI/$EXP/checkpoints_${SM_QUEUE:-default}/$CFG/$EXP}; RESUME_FROM_URI=${RESUME_FROM_URI:-$CK_URI}
log(){ echo "$(date -u +%FT%TZ) [sm_entry] $*"; }

# 0. scratch = the biggest free filesystem (p5: NVMe under /opt/ml or /tmp)
best=""; bestf=0
for d in ${SCRATCH_CANDIDATES:-/opt/ml /tmp /mnt /scratch}; do [ -d $d ] || continue; f=$(df -Pk $d 2>/dev/null | awk 'NR==2{print $4}'); [ "${f:-0}" -gt "$bestf" ] && { best=$d; bestf=$f; }; done
export W=${W_OVERRIDE:-$best/pi05_work}; mkdir -p $W/{data,stage,stage_val,lerobot_home/local,openpi_cache,hf_home,jax_cache,logs}
log "cfg=$CFG exp=$EXP mode=$MODE scratch=$W ($((bestf/1024/1024)) GB free) ckpt_uri=$CK_URI resume_from=$RESUME_FROM_URI"; df -h $W | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -8
export HF_LEROBOT_HOME=$W/lerobot_home OPENPI_DATA_HOME=$W/openpi_cache HF_HOME=$W/hf_home JAX_COMPILATION_CACHE_DIR=$W/jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9} LEROBOT_SKIP_TS_CHECK=1 FFMPEG=/opt/tools/ffmpeg FFPROBE=/opt/tools/ffprobe
export WANDB_ENTITY=${WANDB_ENTITY:-tri} WANDB_DIR=$W/logs
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true}" LEROBOT_LOWDIM_MEMMAP=1
export OPENPI_ASYNC_CKPT=${OPENPI_ASYNC_CKPT:-0}  # synchronous orbax saves: a step dir is final when save() returns
S5="s5cmd --numworkers ${S5_WORKERS:-256} --retry-count 10"; source $CODE/yam/sagemaker/ckpt_sync.sh
cd $CODE

# 1. pi05_base + tokenizer
if [ ! -f $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params/_METADATA ] && [ ! -d $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params ]; then
  log "fetching openpi cache from $OPENPI_CACHE_URI"; $S5 cp "$OPENPI_CACHE_URI/*" $OPENPI_DATA_HOME/ >/dev/null || { log "FATAL openpi cache fetch failed"; exit 2; }
fi
ls $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params | head -3

TRAIN=$HF_LEROBOT_HOME/local/abc_mm2_train_v21; VAL=$HF_LEROBOT_HOME/local/abc_mm2_val_v21
stage_convert(){ # $1 = s3 repo uri, $2 = local repo dir, $3 = stage out dir
  if [ -f $3/source.json ]; then return 0; fi
  [ -f $2/meta/info.json ] || { mkdir -p $2; $S5 cp --exclude "*lowdim_pack/*" "$1/*" $2/ >/dev/null || { log "download FAILED $1"; return 1; }; }
  $PY $S/convert_source.py $2 $3 --workers ${CONV_WORKERS:-48} >> $W/logs/convert.log 2>&1 || { log "convert FAILED $1"; return 1; }
  [ "${KEEP_RAW:-0}" = "1" ] || rm -rf $2
}
if [ $S2 = 1 ]; then
  # 2'. stage 2: fetch only (the repo is built offline, docs/STAGE2.md "Build"); smoke = 200 steps on the same repo
  TRAIN=$HF_LEROBOT_HOME/local/yam_s2_train_v21
  if [ ! -f $TRAIN/meta/assembly.json ]; then
    $S5 ls "$DATA_CACHE_URI/_COMPLETE" >/dev/null 2>&1 || { log "FATAL no $DATA_CACHE_URI/_COMPLETE (upload unfinished?)"; exit 2; }
    log "data cache hit: $DATA_CACHE_URI/train -> $TRAIN"; t0=$(date +%s); mkdir -p $TRAIN
    $S5 cp "$DATA_CACHE_URI/train/*" $TRAIN/ >/dev/null || { log "FATAL s2 repo fetch failed"; exit 2; }
    log "fetched in $(( $(date +%s)-t0 ))s"; df -h $W | tail -1
  fi
  [ -f $TRAIN/meta/assembly.json ] || { log "FATAL $TRAIN/meta/assembly.json missing"; exit 2; }
  [ -f $TRAIN/lowdim_mm/action.npy ] || $PY $S/build_lowdim_memmap.py $TRAIN --workers 48 >> $W/logs/memmap.log 2>&1 || { log "FATAL train memmap failed"; exit 5; }
  [ "$MODE" = "smoke" ] && TRAIN_OVERRIDES="--num-train-steps 200 --lr-schedule.warmup-steps 20 --save-interval 100 --log-interval 10 ${TRAIN_OVERRIDES:-}"
  WANDB_TAGS=${WANDB_TAGS:-pi05,full_ft,pretrain,yam,stage2,yam_s2_train_v21,joint14_abs,horizon32,sagemaker,h100}
else
t_stage0=$(date +%s)
# 2. data: cache first
if [ "$MODE" = "full" ] && $S5 ls "$DATA_CACHE_URI/_COMPLETE" >/dev/null 2>&1 && [ ! -f $TRAIN/meta/info.json ]; then
  log "data cache hit: $DATA_CACHE_URI -> $HF_LEROBOT_HOME/local"; $S5 cp "$DATA_CACHE_URI/train/*" $TRAIN/ >/dev/null && $S5 cp "$DATA_CACHE_URI/val/*" $VAL/ >/dev/null || { log "cache fetch failed, falling back to build"; rm -rf $TRAIN $VAL; }
fi
if [ ! -f $TRAIN/meta/info.json ]; then
  if [ "$MODE" = "smoke" ]; then
    ABC_TASKS="${SMOKE_ABC_TASKS:?}"; VAL_TASKS="${SMOKE_VAL_TASKS:?}"; DO_MOLMO=${SMOKE_MOLMO:-0}
  else
    ABC_TASKS=$(awk '{print $1}' $S/abc_tasks_by_eps.txt); DO_MOLMO=1
    VAL_TASKS=$($S5 ls "$SRC_ROOT/abc130k_val_v3ar/" | awk '{print $NF}' | sed 's|/$||' | grep ^abc130k)
  fi
  log "staging+converting: molmo=$DO_MOLMO abc_tasks=$(echo $ABC_TASKS | wc -w) val_tasks=$(echo $VAL_TASKS | wc -w)"
  # molmo (1.4 TB) first, in background, while ABC tasks stream through a parallel pool
  if [ "$DO_MOLMO" = "1" ]; then ( stage_convert $SRC_ROOT/molmo_yam_v3ar/molmo_yam_v3 $W/data/molmo_yam_v3 $W/stage/molmo_yam_v3 && log "molmo converted" ) & MPID=$!; fi
  export -f stage_convert log; export S5 PY S W CONV_WORKERS KEEP_RAW
  printf "%s\n" $ABC_TASKS | xargs -P ${STAGE_PAR:-3} -I{} bash -c 'stage_convert '$SRC_ROOT'/abc130k_v3ar/{} $W/data/{} $W/stage/{} || echo "{}" >> $W/logs/failed_tasks.txt'
  printf "%s\n" $VAL_TASKS | xargs -P ${STAGE_PAR:-3} -I{} bash -c 'stage_convert '$SRC_ROOT'/abc130k_val_v3ar/{} $W/data/val_{} $W/stage_val/{} || echo "val {}" >> $W/logs/failed_tasks.txt'
  [ -n "${MPID:-}" ] && wait $MPID
  log "staging done in $(( $(date +%s)-t_stage0 ))s; failed: $(cat $W/logs/failed_tasks.txt 2>/dev/null | wc -l)"; df -h $W | tail -1
  # 3. assemble train (skip camera-less, hold out molmo webbed-val eps) + val
  SRCS=""; for t in $ABC_TASKS; do [ -f $W/stage/$t/source.json ] && SRCS="$SRCS $W/stage/$t"; done
  EXCL=""; if [ "$DO_MOLMO" = "1" ] && [ -f $W/stage/molmo_yam_v3/source.json ]; then SRCS="$W/stage/molmo_yam_v3 $SRCS"; EXCL="--exclude molmo_yam_v3:$S/molmo_val_ids.txt"; fi
  echo "$SRCS" | tr ' ' '\n' | grep . > $W/logs/train_sources.txt
  log "assembling TRAIN from $(wc -l < $W/logs/train_sources.txt) sources"
  $PY $S/assemble_v21.py $TRAIN $SRCS --workers 64 $EXCL >> $W/logs/assemble.log 2>&1 || { log "FATAL train assemble failed"; tail -20 $W/logs/assemble.log; exit 3; }
  VSRCS=""; for d in $W/stage_val/*/; do [ -f $d/source.json ] && VSRCS="$VSRCS ${d%/}"; done
  ONLY=""; if [ "$DO_MOLMO" = "1" ] && [ -f $W/stage/molmo_yam_v3/source.json ]; then VSRCS="$W/stage/molmo_yam_v3 $VSRCS"; ONLY="--only molmo_yam_v3:$S/molmo_val_ids.txt"; fi
  [ -n "$VSRCS" ] && { $PY $S/assemble_v21.py $VAL $VSRCS --workers 32 $ONLY >> $W/logs/assemble.log 2>&1 || log "WARN val assemble failed"; }
  # 3b. lowdim memmaps (O(1) per-sample lowdim lookups; lerobot select() is seconds/sample at 460M frames)
  $PY $S/build_lowdim_memmap.py $TRAIN --workers 48 >> $W/logs/memmap.log 2>&1 || { log "FATAL train memmap failed"; exit 5; }
  [ -f $VAL/meta/info.json ] && { $PY $S/build_lowdim_memmap.py $VAL --workers 32 >> $W/logs/memmap.log 2>&1 || log "WARN val memmap failed"; }
  # 4. exact norm stats (float64) for the train repo
  $PY $S/norm_stats_exact.py $TRAIN assets/$CFG/local/abc_mm2_train_v21 --workers 64 >> $W/logs/norm_stats.log 2>&1 || { log "FATAL norm stats failed"; exit 4; }
  cp assets/$CFG/local/abc_mm2_train_v21/norm_stats.json $TRAIN/meta/norm_stats_$CFG.json
  # 5. cache the built repos (background, low priority) so restarts / future runs skip all of the above
  if [ "$MODE" = "full" ] && [ "${CACHE_UPLOAD:-1}" = "1" ]; then
    ( nice -n 19 $S5 --numworkers 64 cp $TRAIN/ "$DATA_CACHE_URI/train/" >/dev/null && nice -n 19 $S5 --numworkers 64 cp $VAL/ "$DATA_CACHE_URI/val/" >/dev/null \
      && echo "$(date -u) $EXP" | $S5 pipe "$DATA_CACHE_URI/_COMPLETE" && log "data cache uploaded" || log "WARN data cache upload failed" ) >> $W/logs/cache_upload.log 2>&1 &
  fi
else
  log "train repo present ($(jq -r .total_episodes $TRAIN/meta/info.json) eps); norm stats from repo"
  [ -f $TRAIN/lowdim_mm/action.npy ] || $PY $S/build_lowdim_memmap.py $TRAIN --workers 48 >> $W/logs/memmap.log 2>&1
  [ -f $VAL/meta/info.json ] && [ ! -f $VAL/lowdim_mm/action.npy ] && $PY $S/build_lowdim_memmap.py $VAL --workers 32 >> $W/logs/memmap.log 2>&1
  mkdir -p assets/$CFG/local/abc_mm2_train_v21; cp $TRAIN/meta/norm_stats_$CFG.json assets/$CFG/local/abc_mm2_train_v21/norm_stats.json 2>/dev/null || \
    $PY $S/norm_stats_exact.py $TRAIN assets/$CFG/local/abc_mm2_train_v21 --workers 64 >> $W/logs/norm_stats.log 2>&1
fi
fi  # S2
jq -c '{total_episodes,total_frames,total_tasks,fps}' $TRAIN/meta/info.json; [ -f $VAL/meta/info.json ] && jq -c '{val_episodes:.total_episodes,val_frames:.total_frames}' $VAL/meta/info.json
export YAM_ASSEMBLY_JSON=$TRAIN/meta/assembly.json
cp $TRAIN/meta/assembly.json $TRAIN/meta/info.json assets/$CFG/local/abc_mm2_train_v21/norm_stats.json $W/logs/ 2>/dev/null
$S5 cp $W/logs/ "$OUT_URI/$EXP/logs_prep/" >/dev/null 2>&1 || true

# 6. resume: local scratch is empty on a fresh instance, so restore the latest committed step of RESUME_FROM_URI (orbax tmp dirs are ignored)
CK=$W/checkpoints; CKD=$CK/$CFG/$EXP; RESUME=""
rm -rf $CKD/*.orbax-checkpoint-tmp-* 2>/dev/null
if [ -n "$(local_steps $CKD)" ] && [ -f $CKD/wandb_id.txt ]; then RESUME="--resume"; log "RESUMING from local $(local_steps $CKD | tail -1)"
else
  rm -rf $CKD; st=$(resume_latest "$RESUME_FROM_URI" $CKD); rrc=$?
  case $rrc in
    0) RESUME="--resume"; log "RESUMING from $RESUME_FROM_URI/$st";;
    1) rm -rf $CKD; log "no committed step under $RESUME_FROM_URI: fresh start";;
    *) log "FATAL committed steps exist under $RESUME_FROM_URI but the restore failed; not starting fresh over them"; exit 6;;
  esac
fi
mkdir -p $CK
( while true; do sleep ${CKPT_SYNC_EVERY:-300}; [ -d $CKD ] && sync_pass $CKD "$CK_URI"; done ) >> $W/logs/ckpt_sync.log 2>&1 & SYNC_PID=$!

# 7. val eval watcher (CPU) on finished checkpoints -> wandb run ${EXP}_eval; stops (after a final pass) when training ends
eval_pass(){ for st in $(local_steps $CKD); do d=$CKD/$st; grep -qx "$st" $W/logs/eval_done.txt && continue
    [ -f $d/_CHECKPOINT_METADATA ] && [ -d $d/params ] || continue
    log "eval ckpt $st"; t0=$(date +%s)
    if JAX_PLATFORMS=cpu $PY $S/eval_ckpt.py --config $CFG --ckpt $d --val-repo local/abc_mm2_val_v21 --wandb-run ${EXP}_eval --n-per-source ${EVAL_N:-192} >> $W/logs/eval.log 2>&1; then
      echo $st >> $W/logs/eval_done.txt; log "eval ckpt $st done in $(( $(date +%s)-t0 ))s"; $S5 cp $d/eval_val.json "$OUT_URI/$EXP/eval/eval_val_$st.json" >/dev/null 2>&1
    else log "eval FAILED $st (see logs/eval.log)"; echo $st >> $W/logs/eval_done.txt; fi; done; }
EVAL_PID=""
if [ -f $VAL/meta/info.json ] && [ "${EVAL_WATCH:-1}" = "1" ]; then
  touch $W/logs/eval_done.txt
  ( while true; do eval_pass; [ -f $W/STOP_EVAL ] && { eval_pass; break; }; sleep ${EVAL_POLL:-300}; done ) & EVAL_PID=$!
fi

# 8. train
ls -la $TRAIN/lowdim_mm/ 2>/dev/null | tail -2
log "starting training: $CFG exp=$EXP fsdp=$FSDP_DEVICES $RESUME ${TRAIN_OVERRIDES:-}"
export WANDB_TAGS=${WANDB_TAGS:-pi05,full_ft,pretrain,yam,abc130k_v3ar,molmoact2_v3ar,lerobot_v3,joint14_abs,horizon32,sagemaker,h100}
$PY scripts/train.py $CFG --exp-name $EXP --checkpoint-base-dir $CK --fsdp-devices $FSDP_DEVICES $RESUME ${TRAIN_OVERRIDES:-} 2>&1 | tee -a $W/logs/train.log
rc=${PIPESTATUS[0]}; log "train exited rc=$rc"
touch $W/STOP_EVAL; [ -n "$EVAL_PID" ] && { log "waiting for the eval watcher's final pass"; timeout ${EVAL_FINAL_TIMEOUT:-7200} tail --pid=$EVAL_PID -f /dev/null; }
# 9. last checkpoint sync (stop the loop first: one writer), then final params (committed steps only) + logs
pkill -TERM -P $SYNC_PID 2>/dev/null; kill $SYNC_PID 2>/dev/null; wait $SYNC_PID 2>/dev/null; sync_pass $CKD "$CK_URI" >> $W/logs/ckpt_sync.log 2>&1
LAST=$(local_steps $CKD | tail -1)
if [ -n "$LAST" ]; then log "publishing $LAST -> $OUT_URI/$EXP/final/"; put_step $CKD/$LAST "$OUT_URI/$EXP/final" || log "WARN final copy failed"; fi
$S5 cp $W/logs/ "$OUT_URI/$EXP/logs/" >/dev/null 2>&1 || true
kill $(jobs -p) 2>/dev/null; exit $rc
