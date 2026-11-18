#!/bin/bash
# First-token margins of the UNTRAINED students (st_logit.py): how far each question is from
# "I don't know". One GPU. Control file keys: STUDENTS="tag=model ..." FILES="tqa_1500 pqa_1500".
# Output: gs://.../acl_st/run1/base/<tag>_<file>_logit.jsonl. Always shuts the VM down at the end.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
STUDENTS="phi4m=microsoft/Phi-4-mini-instruct smol3=HuggingFaceTB/SmolLM3-3B"; FILES="tqa_1500 pqa_1500"
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
log "first-token margins on $HOST: students [$STUDENTS] files [$FILES]"
finish() {
  gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
  log "shutting the VM down"; sleep 15; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
  shutdown -h now; exit 0
}
if [ ! -f $D/env/.ok ]; then
  if [ ! -f /usr/include/python3.10/Python.h ] || ! python3 -c "import ensurepip" > /dev/null 2>&1; then
    { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv python3.10-dev build-essential; } > /dev/null 2>&1
  fi
  rm -rf $D/env; python3 -m venv $D/env && . $D/env/bin/activate
  pip install -q "vllm==0.30.0" "peft==0.21.2" accelerate 2>&1 | tail -2
  python -c "import vllm, peft" && touch $D/env/.ok
fi
[ -f $D/env/.ok ] || { log "environment failed"; finish; }
. $D/env/bin/activate
C=$D/code_logit; mkdir -p $C $D/base; gcloud storage cp -q "$GS/code/*.py" $C/ > /dev/null 2>&1
[ -s $C/st_logit.py ] || { log "code missing"; finish; }
for F in $FILES; do
  gcloud storage cp -q $GS/data/$F.jsonl $D/ > /dev/null 2>&1
  [ -s $D/$F.jsonl ] || { log "data file $F missing"; continue; }
  for sm in $STUDENTS; do
    tag=${sm%%=*}; M=${sm#*=}; O=${tag}_${F}_logit.jsonl
    if gcloud storage ls $GS/run1/base/$O > /dev/null 2>&1; then log "$O exists: skipping"; continue; fi
    log "computing $O"
    CUDA_VISIBLE_DEVICES=0 timeout 3600 python $C/st_logit.py --student $M --data $D/$F.jsonl --out $D/base/$O 2>&1 | grep -E "wrote|rror|IDK reply" | tail -4
    [ -s $D/base/$O ] && gcloud storage cp -q $D/base/$O $GS/run1/base/$O > /dev/null 2>&1 && log "uploaded $O ($(wc -l < $D/base/$O) rows)" || log "FAILED $O"
  done
done
finish
