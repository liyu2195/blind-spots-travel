#!/bin/bash
# adeval: evaluate KEPT adapters (acl_st/run1/adapters/<run>/ck<step>/) on any question file, greedy, no training.
# For every run named in the list file acl_st/adeval/<LIST>.txt (one run name per line) whose adapter is in the bucket and whose output does not exist yet:
#   output acl_st/run1/adeval/<EVALNAME>/<run>.jsonl (the rows of st_gen.py: one greedy answer per question of acl_st/data/<EVALF>).
# The student model is taken from the run name's prefix (qwen7_ / llama8_ / phi4m_ / smol3_). Up to 6 adapters per vLLM process, the "idk" prompt as in the
# workers' evaluation. One GPU. Always shuts the VM down at the end.
# Control file keys: LIST (name without .txt), EVALF (file in acl_st/data/), EVALNAME (output folder name), UNTRAINED=1 also writes <student>_ck0.jsonl.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
LIST=""; EVALF=""; EVALNAME=""; UNTRAINED=1
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
log "adeval on $HOST: list [$LIST] questions [$EVALF] output [$EVALNAME]"
finish() {
  gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
  log "shutting the VM down"; sleep 15; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
  shutdown -h now; exit 0
}
[ -n "$LIST" ] && [ -n "$EVALF" ] && [ -n "$EVALNAME" ] || { log "LIST, EVALF and EVALNAME are required"; finish; }
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
C=$D/code_ae; A=$D/adeval; mkdir -p $C $A/out; gcloud storage cp -q "$GS/code/*.py" $C/ > /dev/null 2>&1
[ -s $C/st_gen.py ] || { log "code missing"; finish; }
gcloud storage cp -q $GS/data/$EVALF $D/ > /dev/null 2>&1; [ -s $D/$EVALF ] || { log "question file $EVALF missing"; finish; }
gcloud storage cp -q $GS/adeval/$LIST.txt $A/$LIST.txt > /dev/null 2>&1; [ -s $A/$LIST.txt ] || { log "list $LIST.txt missing"; finish; }
have=$(gcloud storage ls "$GS/run1/adeval/$EVALNAME/*.jsonl" 2> /dev/null | sed 's#.*/##; s#\.jsonl$##')
ads=$(gcloud storage ls "$GS/run1/adapters/*/*/adapter_model.safetensors" 2> /dev/null)
model_of() {  # run name -> student model (hub name or local copy of the bucket folder)
  case "$1" in
    qwen7_*)  echo "Qwen/Qwen2.5-7B-Instruct";;
    phi4m_*)  echo "microsoft/Phi-4-mini-instruct";;
    smol3_*)  echo "HuggingFaceTB/SmolLM3-3B";;
    llama8_*) local L=$D/models_w0/Llama-3.1-8B-Instruct
              [ -s $L/config.json ] || { mkdir -p $L; gcloud storage cp -q -r "$GS/models/meta-llama/Llama-3.1-8B-Instruct/*" $L/ > /dev/null 2>&1; }
              [ -s $L/config.json ] && echo $L;;
  esac
}
run_batch() {  # $1 student tag, $2 model, $3 "tag=dir,tag=dir"
  [ -n "$3" ] || return 0
  CUDA_VISIBLE_DEVICES=0 timeout 3600 python $C/st_gen.py --model $2 --task trivia --data $D/$EVALF --out "$A/out/{tag}.jsonl" \
    --n 0 --greedy --loras "$3" --prompt idk --gpu_mem 0.85 2>&1 | grep -E "wrote|rror" | tail -8
  local t f
  for t in $(echo "$3" | tr ',' '\n' | sed 's#=.*##'); do
    f=$A/out/$t.jsonl
    if [ -s $f ]; then
      if [ "$t" = ck0 ]; then gcloud storage cp -q $f $GS/run1/adeval/$EVALNAME/${1}_ck0.jsonl > /dev/null 2>&1; else gcloud storage cp -q $f $GS/run1/adeval/$EVALNAME/$t.jsonl > /dev/null 2>&1; fi
      nok=$((nok+1)); rm -f $f
    else log "no output for $t"; nbad=$((nbad+1)); fi
  done
  gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.log > /dev/null 2>&1
}
nok=0; nbad=0; nskip=0; nmiss=0
for S in qwen7 llama8 phi4m smol3; do
  runs=$(grep "^${S}_" $A/$LIST.txt | sort -u)
  [ -n "$runs" ] || continue
  M=$(model_of ${S}_x); [ -n "$M" ] || { log "no model for $S"; continue; }
  L=""; k=0
  if [ "$UNTRAINED" = 1 ] && ! echo "$have" | grep -qxF "${S}_ck0"; then L="ck0="; k=1; fi
  for r in $runs; do
    if echo "$have" | grep -qxF "$r"; then nskip=$((nskip+1)); continue; fi
    src=$(echo "$ads" | grep -F "/adapters/$r/" | sort -t/ -k8 -V | tail -1)
    [ -n "$src" ] || { nmiss=$((nmiss+1)); continue; }
    ck=$(echo "$src" | sed 's#.*/adapters/[^/]*/\([^/]*\)/.*#\1#'); mkdir -p $A/ad/$r
    [ -s $A/ad/$r/adapter_model.safetensors ] || gcloud storage cp -q "$GS/run1/adapters/$r/$ck/*" $A/ad/$r/ > /dev/null 2>&1
    [ -s $A/ad/$r/adapter_model.safetensors ] && [ -s $A/ad/$r/adapter_config.json ] || { log "adapter copy of $r failed"; nbad=$((nbad+1)); continue; }
    if [ -n "$L" ]; then L="$L,$r=$A/ad/$r"; else L="$r=$A/ad/$r"; fi
    k=$((k+1))
    if [ $k -ge 6 ]; then run_batch $S $M "$L"; L=""; k=0; fi
  done
  run_batch $S $M "$L"
  rm -rf $A/ad
done
log "adeval finished: $nok outputs written, $nskip already there, $nmiss adapters not in the bucket yet, $nbad failed"
finish
