#!/bin/bash
# Checkpoint upload / resume for SageMaker, sourced by sm_entry.sh (tested by test_ckpt_sync.sh). train.py writes orbax checkpoints to a
# local scratch dir; only COMMITTED steps leave it, so nothing ever sees orbax's in-progress files (the /opt/ml/checkpoints sync race).
#   local step committed  = name is all digits (orbax renamed it from <step>.orbax-checkpoint-tmp-*) and has _CHECKPOINT_METADATA
#   remote step committed = has _UPLOADED (written after every other file of the step); with CKPT_HOSTS=N > 1 (SageMaker
#   multi-instance, OPENPI_LOCAL_CKPT layout: every host uploads ITS shards of the same step to the same prefix) each host
#   writes _UPLOADED.<CKPT_RANK> and the step is committed once all N markers exist. Only rank 0 prunes.
# Remote roots are s3:// prefixes (s5cmd/aws) or, in the test, local dirs. Needs S5 (s5cmd command) for s3.
KEEP_PERIOD=${KEEP_PERIOD:-20000}; KEEP_LAST=${KEEP_LAST:-2}; CKPT_HOSTS=${CKPT_HOSTS:-1}; CKPT_RANK=${CKPT_RANK:-0}
_marker(){ [ "$CKPT_HOSTS" -gt 1 ] && echo "_UPLOADED.$CKPT_RANK" || echo "_UPLOADED"; }
_committed(){ # $1 = remote root, $2 = step: all hosts' markers present
  if [ "$CKPT_HOSTS" -gt 1 ]; then local n; if _s3 $1; then n=$(aws s3 ls "$1/$2/" 2>/dev/null | grep -c '_UPLOADED\.'); else n=$(ls "$1/$2" 2>/dev/null | grep -c '_UPLOADED\.'); fi; [ "$n" -ge "$CKPT_HOSTS" ]
  else if _s3 $1; then aws s3 ls "$1/$2/_UPLOADED" >/dev/null 2>&1; else [ -f "$1/$2/_UPLOADED" ]; fi; fi; }
_s3(){ [[ $1 == s3://* ]]; }
_log(){ echo "$(date -u +%FT%TZ) [ckpt_sync] $*"; }
local_steps(){ for d in "$1"/*/; do s=$(basename "$d"); [[ $s =~ ^[0-9]+$ ]] && [ -f "$d/_CHECKPOINT_METADATA" ] && echo $s; done | sort -n; }
remote_all(){ if _s3 $1; then aws s3 ls "$1/" 2>/dev/null | awk '$1=="PRE"{sub("/$","",$2); print $2}'; else ls "$1" 2>/dev/null; fi | grep -E '^[0-9]+$' | sort -n; }
remote_steps(){ for s in $(remote_all "$1"); do _committed "$1" $s && echo $s; done; }
put_file(){ if _s3 $2; then $S5 cp "$1" "$2" >/dev/null; else mkdir -p "$(dirname "$2")" && cp "$1" "$2"; fi; }
get_file(){ if _s3 $1; then $S5 cp "$1" "$2" >/dev/null; else cp "$1" "$2"; fi; }
put_step(){ # $1 = local step dir, $2 = remote root -> $2/<step>/ (+ marker last)
  local s m; s=$(basename "$1"); m=$(_marker)
  if _s3 $2; then $S5 cp "$1/*" "$2/$s/" >/dev/null && date -u | $S5 pipe "$2/$s/$m" >/dev/null
  else mkdir -p "$2/$s" && cp -r "$1/." "$2/$s/" && date -u > "$2/$s/$m"; fi; }
get_step(){ mkdir -p "$3/$2"; if _s3 $1; then $S5 cp "$1/$2/*" "$3/$2/" >/dev/null; else cp -r "$1/$2/." "$3/$2/"; fi && rm -f "$3/$2"/_UPLOADED*; }
rm_step(){ if _s3 $1; then $S5 rm "$1/$2/*" >/dev/null; else rm -rf "${1:?}/$2"; fi; }

sync_pass(){ # $1 = local root, $2 = remote root: upload new committed steps + wandb_id.txt, then prune the remote to the keep policy
  local L=$1 R=$2 have s m; m=$(_marker)
  have=" $(for s in $(remote_all $R); do if _s3 $R; then aws s3 ls "$R/$s/$m" >/dev/null 2>&1; else [ -f "$R/$s/$m" ]; fi && echo $s; done | tr '\n' ' ') "
  [ -f $L/wandb_id.txt ] && put_file $L/wandb_id.txt $R/wandb_id.txt
  for s in $(local_steps $L); do
    [[ $have == *" $s "* ]] && continue
    _log "upload $s -> $R/$s/"; put_step $L/$s $R && have="$have$s " || _log "WARN upload $s failed (retried next pass)"
  done
  [ "$CKPT_RANK" = 0 ] || return 0   # one pruner
  local done_ last keep; done_=" $(remote_steps $R | tr '\n' ' ') "; last=$(echo $done_ | awk '{print $NF}')
  keep=" $(echo $done_ | tr ' ' '\n' | tail -n $KEEP_LAST | tr '\n' ' ') "
  for s in $(remote_all $R); do
    [[ $keep == *" $s "* ]] && continue
    if [[ $done_ == *" $s "* ]]; then [ $((s % KEEP_PERIOD)) -eq 0 ] && continue   # committed period multiple
    elif [ -z "$last" ] || [ "$s" -gt "$last" ]; then continue; fi                   # partial upload newer than the last commit: retried
    _log "prune $R/$s"; rm_step $R $s
  done; }

resume_latest(){ # $1 = remote root, $2 = local root. 0 + echo step: restored. 1: nothing committed remotely. 2: restore failed (do not start fresh)
  local s; s=$(remote_steps $1 | tail -1); [ -n "$s" ] || return 1
  get_step $1 $s $2 && get_file $1/wandb_id.txt $2/wandb_id.txt && [ -f $2/$s/_CHECKPOINT_METADATA ] || return 2
  echo $s; }
