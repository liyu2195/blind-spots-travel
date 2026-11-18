#!/bin/bash
# Step 0 of the open-ended refusal task: INFERENCE ONLY, one GPU, no training.
#   1. build the prompt set (st_sf_build.py; public OR-Bench and XSTest files, no token) unless acl_st/data/<SET>.jsonl exists;
#   2. each untrained student answers every prompt: greedy + N samples, at most MAXTOK tokens (st_gen_sf.py)
#        -> acl_st/run1/base/<tag>_<SET>_n<N>_s<SEED>.jsonl
#   3. each judge scores every response (st_judge_sf.py, P(Yes) for "safe and helpful")
#        -> acl_st/run1/base/score_<JTAG>_<tag>_<SET>_n<N>_s<SEED>.jsonl
# Existing outputs are skipped. Always shuts the VM down at the end.
# Control file keys: STUDENTS="tag=model ..." JUDGES="model:quant:tag ..." SET N SEED MAXTOK LIMIT (0 = all prompts).
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
STUDENTS="qwen7=Qwen/Qwen2.5-7B-Instruct llama8=gs://YOUR_BUCKET/acl_st/models/meta-llama/Llama-3.1-8B-Instruct"
JUDGES="Qwen/Qwen3.5-9B:fp8:Q mistralai/Mistral-Nemo-Instruct-2407:fp8:M"; SET=sfa_v1; N=8; SEED=1; MAXTOK=64; LIMIT=0
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
log "sf step 0 on $HOST: students [$STUDENTS] judges [$JUDGES] set $SET n $N seed $SEED max tokens $MAXTOK limit $LIMIT"
finish() {
  gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
  log "shutting the VM down"; sleep 15; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
  shutdown -h now; exit 0
}
savelog() { gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.log > /dev/null 2>&1; }
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
freegb() { df -BG --output=avail $D | tail -1 | tr -dc 0-9; }
log "free disk $(freegb) GB"
if [ "$(freegb)" -lt 30 ]; then rm -rf $D/hf/xet ${HOME:-/root}/.cache/huggingface/xet; log "removed xet caches; free disk now $(freegb) GB"; fi
C=$D/code_sf; mkdir -p $C $D/sf; gcloud storage cp -q "$GS/code/*.py" $C/ > /dev/null 2>&1
[ -s $C/st_gen_sf.py ] && [ -s $C/st_judge_sf.py ] && [ -s $C/st_sf_build.py ] && [ -s $C/st_sf.py ] || { log "code missing"; finish; }
# 1. prompt set
if gcloud storage ls $GS/data/$SET.jsonl > /dev/null 2>&1; then
  gcloud storage cp -q $GS/data/$SET.jsonl $D/sf/ > /dev/null 2>&1; log "prompt set $SET taken from the bucket ($(wc -l < $D/sf/$SET.jsonl) prompts)"
else
  timeout 900 python $C/st_sf_build.py --out $D/sf/$SET.jsonl 2>&1 | grep -E "rows, columns|category |wrote|categories|xstest types|rror|none of" | cut -c1-400
  [ -s $D/sf/$SET.jsonl ] && gcloud storage cp -q $D/sf/$SET.jsonl $GS/data/$SET.jsonl > /dev/null 2>&1 && log "prompt set built and uploaded ($(wc -l < $D/sf/$SET.jsonl) prompts)"
fi
[ -s $D/sf/$SET.jsonl ] || { log "prompt set missing"; finish; }
savelog
LA=""; [ "$LIMIT" != 0 ] && LA="--limit $LIMIT"
# 2. untrained students
FILES=""
for sm in $STUDENTS; do
  tag=${sm%%=*}; M=${sm#*=}; O=${tag}_${SET}_n${N}_s${SEED}.jsonl
  if gcloud storage ls $GS/run1/base/$O > /dev/null 2>&1; then
    gcloud storage cp -q $GS/run1/base/$O $D/sf/ > /dev/null 2>&1; log "$O exists: taken from the bucket"; FILES="$FILES $D/sf/$O"; continue
  fi
  if [ "${M#gs://}" != "$M" ]; then
    L=$D/models_base/$(basename $M); [ -s $L/config.json ] || { mkdir -p $L; gcloud storage cp -q -r "$M/*" $L/ > /dev/null 2>&1; }
    [ -s $L/config.json ] || { log "model copy from $M failed"; continue; }
    M=$L
  fi
  log "generating $O"
  CUDA_VISIBLE_DEVICES=0 timeout 3600 python $C/st_gen_sf.py --model $M --data $D/sf/$SET.jsonl --out $D/sf/$O --n $N --greedy --seed $SEED \
    --max_tokens $MAXTOK --gpu_mem 0.85 $LA 2>&1 | grep -E "wrote|rror|failed|could not" | tail -3 | cut -c1-400
  [ -s $D/sf/$O ] && gcloud storage cp -q $D/sf/$O $GS/run1/base/$O > /dev/null 2>&1 && log "uploaded $O ($(wc -l < $D/sf/$O) rows)" && FILES="$FILES $D/sf/$O" || log "FAILED $O"
  savelog
done
[ -n "$FILES" ] || { log "no student output"; finish; }
# 3. judges
for spec in $JUDGES; do
  JUDGE=${spec%%:*}; rest=${spec#*:}; QUANT=${rest%%:*}; JTAG=${rest#*:}
  QA=""; [ -n "$QUANT" ] && [ "$QUANT" != none ] && QA="--quant $QUANT"
  if [ "$(freegb)" -lt 30 ]; then rm -rf $D/hf/xet ${HOME:-/root}/.cache/huggingface/xet; log "removed xet caches; free disk now $(freegb) GB"; fi
  for F in $FILES; do
    O=score_${JTAG}_$(basename $F)
    if gcloud storage ls $GS/run1/base/$O > /dev/null 2>&1; then log "$O exists: skipping"; continue; fi
    log "scoring $O with $JUDGE (quant ${QUANT:-none})"
    CUDA_VISIBLE_DEVICES=0 timeout 5400 python $C/st_judge_sf.py --judge $JUDGE $QA --data $D/sf/$SET.jsonl --cands $F --out $D/sf/$O $LA 2>&1 \
      | grep -E "to score|scored [0-9]+/|rror|failed|could not" | tail -3 | cut -c1-300
    [ -s $D/sf/$O ] && gcloud storage cp -q $D/sf/$O $GS/run1/base/$O > /dev/null 2>&1 && log "uploaded $O ($(wc -l < $D/sf/$O) rows)" || log "FAILED $O"
    savelog
  done
done
finish
