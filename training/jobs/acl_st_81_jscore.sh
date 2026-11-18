#!/bin/bash
# Judge scores of the UNTRAINED students' sampled answers (the base files made by acl_st_80_base3):
# P(Yes) / P(No) of the key-free judge for every sampled answer and the greedy answer, with st_judge.py (same prompt and the same
# first-token rule as the judge reward in st_grpo.py). Used to choose judge thresholds WITHOUT labels (share of the untrained
# student's answers the judge accepts) and for the acceptance rates of the confirmation plan (bar B4).
# One GPU. Control file keys: JUDGE (default Qwen/Qwen3.5-9B) QUANT (default fp8) JTAG (default Q) STUDENTS="qwen7 llama8"
# SETS="tqa pqa nqo" N=32 SEED=1.
# Output: gs://.../acl_st/run1/base/score_<JTAG>_<student>_<set>_1500_n<N>_s<SEED>.jsonl. Always shuts the VM down at the end.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
JUDGE="Qwen/Qwen3.5-9B"; QUANT=fp8; JTAG=Q; STUDENTS="qwen7 llama8"; SETS="tqa pqa nqo"; N=32; SEED=1
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
log "judge scores of base samples on $HOST: judge $JUDGE ($QUANT) tag $JTAG students [$STUDENTS] sets [$SETS] n $N seed $SEED"
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
C=$D/code_js; mkdir -p $C $D/js; gcloud storage cp -q "$GS/code/*.py" $C/ > /dev/null 2>&1
[ -s $C/st_judge.py ] || { log "code missing"; finish; }
QA=""; [ -n "$QUANT" ] && [ "$QUANT" != none ] && QA="--quant $QUANT"
for S in $SETS; do
  gcloud storage cp -q $GS/data/${S}_1500.jsonl $D/ > /dev/null 2>&1
  [ -s $D/${S}_1500.jsonl ] || { log "data file ${S}_1500 missing"; continue; }
  for st in $STUDENTS; do
    F=${st}_${S}_1500_n${N}_s${SEED}; O=score_${JTAG}_$F.jsonl
    if gcloud storage ls $GS/run1/base/$O > /dev/null 2>&1; then log "$O exists: skipping"; continue; fi
    gcloud storage cp -q $GS/run1/base/$F.jsonl $D/js/ > /dev/null 2>&1
    [ -s $D/js/$F.jsonl ] || { log "base file $F missing"; continue; }
    log "scoring $F"
    CUDA_VISIBLE_DEVICES=0 timeout 3600 python $C/st_judge.py --judge $JUDGE $QA --task trivia --data $D/${S}_1500.jsonl --cands $D/js/$F.jsonl \
      --out $D/js/$O --include_greedy 2>&1 | grep -E "to score|scored [0-9]+/|rror|failed" | tail -4
    [ -s $D/js/$O ] && gcloud storage cp -q $D/js/$O $GS/run1/base/$O > /dev/null 2>&1 && log "uploaded $O ($(wc -l < $D/js/$O) rows)" || log "FAILED $O"
  done
done
finish
