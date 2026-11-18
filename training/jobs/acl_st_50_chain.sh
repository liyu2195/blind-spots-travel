#!/bin/bash
# Wave-2 RL run chained after this VM's wave-1 job. It defers the wave-1 job's final shutdown (a
# /usr/local/sbin/shutdown wrapper that only logs), waits for WAITJOB's .done marker, then runs acl_st_49_grpo3.sh with
# this job's control file, removes the wrapper and shuts the VM down.
set -u
WAITJOB=acl_st_46_grpo2
eval "$(gcloud storage cat $BUCKET/acl_st/control/st_${HOST}_$JOB 2> /dev/null | grep -E '^WAITJOB=' )"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p /usr/local/sbin
printf '#!/bin/sh\necho "shutdown deferred $(date -u) $*" >> /opt/cert/deferred_shutdown.log\nexit 0\n' > /usr/local/sbin/shutdown
chmod +x /usr/local/sbin/shutdown; hash -r
log "shutdown deferred; waiting for $WAITJOB"
until gcloud storage ls $BUCKET/status/$HOST/$WAITJOB.done > /dev/null 2>&1; do sleep 60; done
log "$WAITJOB done; starting the wave-2 run"; sleep 30
pkill -f "vllm serve" 2> /dev/null; sleep 5
gcloud storage cp -q $BUCKET/jobs/_lib/acl_st_49_grpo3.sh $WORK/jobs/_w2_run.sh > /dev/null 2>&1
bash $WORK/jobs/_w2_run.sh
log "wave-2 run finished"
rm -f /usr/local/sbin/shutdown; hash -r
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
sleep 15; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
/usr/sbin/shutdown -h now || shutdown -h now
