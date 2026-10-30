#!/bin/bash
# Race watcher v2: poll Batch jobs in jobs.json; a job WINS when its SageMaker training job reaches SecondaryStatus Downloading/Training
# (i.e. has an instance and is past provisioning). Then terminate every other Batch job. Logs to logs/race_watch.log.
export AWS_PROFILE=${AWS_PROFILE:-claude-profile}; R="--region us-west-2"; SM=${SM_DIR:-/tmp/pi05_sm}; J=${1:-$SM/jobs.json}; LOG=$SM/logs/race_watch.log; mkdir -p $SM/logs
ids=$(python3 -c "import json; print(' '.join(j['batch_job'].split('/')[-1] for j in json.load(open('$J'))['jobs']))")
while true; do
  winner=""; line=""; alive=0
  for id in $ids; do
    read st res <<< "$(aws batch describe-service-job $R --job-id $id --query '[status, latestAttempt.serviceResourceId.value]' --output text 2>/dev/null)"
    sec="-"; if [ -n "$res" ] && [ "$res" != "None" ]; then sec=$(aws sagemaker describe-training-job $R --training-job-name ${res##*/} --query 'SecondaryStatus' --output text 2>/dev/null); fi
    line="$line $id:$st/$sec"; case $st in SUBMITTED|PENDING|RUNNABLE|SCHEDULED|STARTING|RUNNING) alive=$((alive+1));; esac
    case $sec in Downloading|Training|Uploading) [ -z "$winner" ] && winner=$id;; esac
    [ "$st" = "RUNNING" ] && [ -z "$winner" ] && winner=$id
  done
  echo "$(date +%F_%T)$line" >> $LOG
  if [ -n "$winner" ]; then
    for id in $ids; do [ "$id" != "$winner" ] && aws batch terminate-service-job $R --job-id $id --reason "race-loser (winner $winner)" >/dev/null 2>&1 && echo "$(date +%F_%T) terminated loser $id" >> $LOG; done
    echo "$(date +%F_%T) WINNER $winner" >> $LOG; echo $winner > ${J%.json}.winner; break
  fi
  [ $alive -eq 0 ] && { echo "$(date +%F_%T) all jobs dead" >> $LOG; break; }
  sleep 60
done
