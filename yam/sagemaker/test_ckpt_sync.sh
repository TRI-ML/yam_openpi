#!/bin/bash
# Self-check of ckpt_sync.sh on fake local trees (remote = a local dir; the s3 branch differs only in the copy commands).
# Run: bash yam/sagemaker/test_ckpt_sync.sh
set -u; source "$(dirname "$0")/ckpt_sync.sh"; T=$(mktemp -d); trap 'rm -rf $T' EXIT; L=$T/local/cfg/exp R=$T/remote/cfg/exp; fails=0
ok(){ if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails+1)); fi; }
mk(){ mkdir -p $L/$1/params; echo p > $L/$1/params/x; [ "${2:-}" = nometa ] || echo m > $L/$1/_CHECKPOINT_METADATA; }
mk 10000; mk 20000; mk 30000.orbax-checkpoint-tmp-1; mk 40000 nometa; echo wid > $L/wandb_id.txt
sync_pass $L $R > /dev/null
ok "uploads only committed steps" '[ "$(remote_steps $R | xargs)" = "10000 20000" ] && [ "$(ls $R | xargs)" = "10000 20000 wandb_id.txt" ]'
ok "step content + marker" '[ -f $R/20000/params/x ] && [ -f $R/20000/_CHECKPOINT_METADATA ] && [ -f $R/20000/_UPLOADED ]'
rm -rf $L/10000 $L/30000.orbax-checkpoint-tmp-1 $L/40000; mk 30000; mk 40000          # orbax committed 30k/40k and dropped 10k
mkdir -p $R/50000 $R/15000 $R/60000.orbax-checkpoint-tmp-0                             # partial uploads (no _UPLOADED) + a stray tmp dir
sync_pass $L $R > /dev/null
ok "keep = last 2 + every 20k; older partial pruned; newer partial kept" '[ "$(remote_all $R | xargs)" = "20000 30000 40000 50000" ] && [ "$(remote_steps $R | xargs)" = "20000 30000 40000" ]'
N=$T/new/cfg/exp; st=$(resume_latest $R $N); rc=$?
ok "resume = latest committed step" '[ $rc = 0 ] && [ "$st" = 40000 ] && [ -f $N/40000/_CHECKPOINT_METADATA ] && [ ! -f $N/40000/_UPLOADED ] && [ "$(cat $N/wandb_id.txt)" = wid ]'
ok "restored step is a committed local step" '[ "$(local_steps $N)" = 40000 ]'
P=$T/plain/cfg/exp; mkdir -p $P/5000.orbax-checkpoint-tmp-0; echo m > $P/5000.orbax-checkpoint-tmp-0/_CHECKPOINT_METADATA; echo old > $P/wandb_id.txt
resume_latest $P $T/n2 > /dev/null; rc=$?; ok "only an orbax tmp dir remotely -> fresh start (rc 1)" '[ $rc = 1 ]'
rm $R/wandb_id.txt; resume_latest $R $T/n3 > /dev/null 2>&1; rc=$?; ok "committed step but no wandb_id -> refuse (rc 2)" '[ $rc = 2 ]'
[ $fails = 0 ] && echo "all ok" || { echo "$fails failed"; exit 1; }
