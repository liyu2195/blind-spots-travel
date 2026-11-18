#!/bin/bash
# Judges as TEST-TIME FILTERS: several judge models score the greedy answers of untrained and
# RL-trained students (one call per answer, P(Yes), no reference answer). One GPU.
# Control file keys: JUDGES="tag=model[:quant] ..." CANDS="run:ck:datafile ..." (datafile = tqa_1500 or pqa_1500).
# Output: gs://.../acl_st/run1/filter/<judge tag>__<datafile>.jsonl (rows: st = <run>__<ck>, idx, s = -1, py, pn).
# Always shuts the VM down at the end.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
JUDGES=""; CANDS=""
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
log "filter scoring on $HOST: judges [$JUDGES]; $(echo $CANDS | wc -w) candidate files"
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
C=$D/code_filter; mkdir -p $C; gcloud storage cp -q "$GS/code/*.py" $C/ > /dev/null 2>&1
[ -s $C/st_judge.py ] || { log "code missing"; finish; }
F=$D/filt; rm -rf $F; mkdir -p $F
for c in $CANDS; do
  run=${c%%:*}; rest=${c#*:}; ck=${rest%%:*}; df=${rest#*:}
  mkdir -p $F/$df; [ -s $D/$df.jsonl ] || gcloud storage cp -q $GS/data/$df.jsonl $D/ > /dev/null 2>&1
  gcloud storage cp -q $GS/run1/rl/$run/eval_rl_$ck.jsonl $F/tmp.jsonl > /dev/null 2>&1 || { log "missing $run $ck"; continue; }
  python -c "import json; [print(l, end='') for l in open('$F/tmp.jsonl') if json.loads(l)['s'] < 0]" > $F/$df/${run}__$ck.jsonl
done
for df in $(ls $F | grep -v tmp.jsonl); do log "$df: $(ls $F/$df | wc -l) files, $(cat $F/$df/*.jsonl | wc -l) answers"; done
for jm in $JUDGES; do
  tag=${jm%%=*}; rest=${jm#*=}; M=${rest%%:*}; QF=""; [ "$rest" != "$M" ] && QF="--quant ${rest#*:}"
  for df in $(ls $F | grep -v tmp.jsonl); do
    O=${tag}__$df.jsonl
    if gcloud storage ls $GS/run1/filter/$O > /dev/null 2>&1; then log "$O exists: skipping"; continue; fi
    log "judge $tag ($M $QF) on $df"
    CUDA_VISIBLE_DEVICES=0 timeout 2400 python $C/st_judge.py --judge $M $QF --task trivia --data $D/$df.jsonl --cands $(ls $F/$df/*.jsonl) --out $F/$O --include_greedy 2>&1 | grep -E "scored [0-9]+/|rror|to score" | tail -3
    [ -s $F/$O ] && gcloud storage cp -q $F/$O $GS/run1/filter/$O > /dev/null 2>&1 && log "uploaded $O ($(wc -l < $F/$O) rows)" || log "FAILED $O"
  done
done
finish
