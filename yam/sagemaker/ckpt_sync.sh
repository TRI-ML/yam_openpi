#!/bin/bash
# Checkpoint syncer: mirror FINISHED orbax steps from a local checkpoint dir to S3 (or a local dir, for tests).
# Why: SageMaker's /opt/ml/checkpoints sync agent drops zero-byte `<file>.sagemaker-uploaded` markers inside orbax's
# in-progress `*.orbax-checkpoint-tmp-*` dirs; orbax json-decodes every file under array_metadatas/ and dies
# (JSONDecodeError) at finalize. So training writes to a scratch dir and THIS script is the only thing that touches S3.
# A step is finished when its dir name is all digits (orbax renames the tmp dir atomically at finalize) AND
# `_CHECKPOINT_METADATA` exists AND no `*.orbax-checkpoint-tmp-*` remains inside. Markers are excluded defensively.
# Usage: ckpt_sync.sh <local ckpt dir (…/<cfg>/<exp>)> <dest (s3://… or /path)> [--once] [--interval 60]
set -uo pipefail
SRC=${1:?local ckpt dir}; DST=${2:?dest}; shift 2; ONCE=0; INTERVAL=60
while [ $# -gt 0 ]; do case $1 in --once) ONCE=1;; --interval) INTERVAL=$2; shift;; esac; shift; done
S5="s5cmd --numworkers ${S5_WORKERS:-128} --retry-count 10"
log(){ echo "$(date -u +%FT%TZ) [ckpt_sync] $*"; }
copy_dir(){ # $1 local dir, $2 dest prefix (no trailing slash)
  if [[ "$2" == s3://* ]]; then $S5 sync --exclude "*.sagemaker-uploaded" --exclude ".synced" "$1/" "$2/" >/dev/null
  else mkdir -p "$2" && rsync -a --exclude "*.sagemaker-uploaded" --exclude ".synced" "$1/" "$2/"; fi
}
put_file(){ if [[ "$2" == s3://* ]]; then $S5 cp "$1" "$2" >/dev/null; else mkdir -p "$(dirname "$2")" && cp "$1" "$2"; fi; }
exists(){ if [[ "$1" == s3://* ]]; then $S5 ls "$1" >/dev/null 2>&1; else [ -e "$1" ]; fi; }
finished(){ # $1 = step dir
  local d=$1; [[ "$(basename "$d")" =~ ^[0-9]+$ ]] || return 1
  [ -f "$d/_CHECKPOINT_METADATA" ] || return 1
  [ -z "$(find "$d" -maxdepth 2 -name '*.orbax-checkpoint-tmp-*' -print -quit 2>/dev/null)" ] || return 1
  return 0
}
pass(){
  [ -d "$SRC" ] || return 0
  [ -f "$SRC/wandb_id.txt" ] && ! exists "$DST/wandb_id.txt" && put_file "$SRC/wandb_id.txt" "$DST/wandb_id.txt" && log "synced wandb_id.txt"
  for d in $(ls -d "$SRC"/[0-9]* 2>/dev/null | sort -V); do
    [ -f "$d/.synced" ] && continue; finished "$d" || continue
    local st; st=$(basename "$d"); log "syncing step $st -> $DST/$st/"; local t0=$(date +%s)
    if copy_dir "$d" "$DST/$st"; then
      # the sentinel is written LAST: readers treat a step as complete only when it exists
      echo "$(date -u +%FT%TZ) $(du -sb "$d" | cut -f1)" > "$d/.synced" && put_file "$d/.synced" "$DST/$st/.complete"
      log "step $st synced in $(( $(date +%s)-t0 ))s"
    else log "WARN sync of step $st failed (retry next pass)"; rm -f "$d/.synced"; fi
  done
}
if [ $ONCE = 1 ]; then pass; exit 0; fi
while true; do pass; [ -f "$SRC/.stop_sync" ] && { pass; log "stop requested; final pass done"; exit 0; }; sleep "$INTERVAL"; done
