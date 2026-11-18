#!/bin/bash
# Judge server for RL students on other VMs (no 2-GPU capacity): serves JM over the internal network,
# publishes its URL to gs://.../acl_st/serve/<host>.url, and shuts down once every client run in CLIENTS has RL.done
# (or after CAP seconds).
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
JM=""; JQ=""; CLIENTS=""; CAP=25200; start=$(date +%s)
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
if [ ! -f $D/env/.ok ]; then
  if [ ! -f /usr/include/python3.10/Python.h ] || ! python3 -c "import ensurepip" > /dev/null 2>&1; then
    { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv python3.10-dev build-essential; } > /dev/null 2>&1
  fi
  rm -rf $D/env; python3 -m venv $D/env && . $D/env/bin/activate
  pip install -q "vllm==0.30.0" "peft==0.21.2" accelerate 2>&1 | tail -2
  python -c "import vllm" && touch $D/env/.ok
fi
. $D/env/bin/activate
serve(){ nohup vllm serve $JM $JQ --host 0.0.0.0 --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.90 --max-logprobs 20 "$@" > $D/server.log 2>&1 & echo $!; }
SP=$(serve --limit-mm-per-prompt '{"image":0,"video":0}')
for i in $(seq 1 90); do curl -s localhost:8000/v1/models > /dev/null 2>&1 && break; kill -0 $SP 2> /dev/null || break; sleep 10; done
if ! curl -s localhost:8000/v1/models > /dev/null 2>&1; then kill $SP 2> /dev/null; sleep 5; SP=$(serve); for i in $(seq 1 90); do curl -s localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done; fi
IP=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip)
curl -s localhost:8000/v1/models > /dev/null 2>&1 && echo -n "http://$IP:8000" | gcloud storage cp - $GS/serve/$HOST.url > /dev/null 2>&1 && log "serving $JM at http://$IP:8000"
until [ $(( $(date +%s) - start )) -gt $CAP ]; do
  all=1; for c in $CLIENTS; do gcloud storage ls $GS/run1/rl/$c/RL.done > /dev/null 2>&1 || all=0; done
  [ $all = 1 ] && break; sleep 120
done
log "clients done or cap reached; stopping"; gcloud storage rm -q $GS/serve/$HOST.url > /dev/null 2>&1
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
sleep 10; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
shutdown -h now
