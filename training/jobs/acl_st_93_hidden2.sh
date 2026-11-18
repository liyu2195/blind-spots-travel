#!/bin/bash
# = acl_st_73_hidden.sh, plus: a student given as gs://... is copied to local disk first (Llama). Hidden states
# analysis. One GPU. Control file keys: STUDENTS="tag=model ..." FILES="tqa_1500 pqa_1500".
# Output: gs://.../acl_st/run1/base/<tag>_<file>_hidden.npz. Always shuts the VM down at the end.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $WORK/acl_st/tmp; export TMPDIR=$WORK/acl_st/tmp
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
STUDENTS="phi4m=microsoft/Phi-4-mini-instruct smol3=HuggingFaceTB/SmolLM3-3B"; FILES="tqa_1500 pqa_1500"
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
log "hidden states on $HOST: students [$STUDENTS] files [$FILES]"
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
# free disk: if under 45 GB, remove cached Hugging Face models other than the two students (judge models from earlier jobs on this VM)
freegb() { df -BG --output=avail $D | tail -1 | tr -dc 0-9; }
if [ "$(freegb)" -lt 45 ]; then
  for h in $D/hf/hub ${HOME:-/root}/.cache/huggingface/hub /root/.cache/huggingface/hub; do
    [ -d "$h" ] || continue
    for m in "$h"/models--*; do case "$m" in *Qwen2.5-7B-Instruct*) ;; *) [ -d "$m" ] && rm -rf "$m" && log "removed cache $m";; esac; done
  done
  log "free disk now $(freegb) GB"
fi
C=$D/code_hidden; mkdir -p $C $D/base; gcloud storage cp -q "$GS/code/*.py" $C/ > /dev/null 2>&1
[ -s $C/st_hidden.py ] || { log "code missing"; finish; }
rm -rf $D/models_h   # a partial copy from an earlier failed attempt
log "disk use: $(du -sh $D/* 2> /dev/null | sort -h | tail -6 | tr '\n' ' ')"
for sm in $STUDENTS; do                       # students outer: each student's model is removed from disk after its files
  tag=${sm%%=*}; M=${sm#*=}
  case "$M" in gs://*) if [ -s $D/models_w0/$(basename $M)/config.json ]; then M=$D/models_w0/$(basename $M)   # copy left by an adeval job
                       else L=$D/models_h/$(basename $M); [ -s $L/config.json ] || { mkdir -p $L; gcloud storage cp -q -r "$M/*" $L/ > /dev/null 2>&1; }; M=$L; fi;; esac
  log "student $tag from $M; free disk $(freegb) GB"
  for F in $FILES; do
    O=${tag}_${F}_hidden.npz
    if gcloud storage ls $GS/run1/base/$O > /dev/null 2>&1; then log "$O exists: skipping"; continue; fi
    [ -s $D/$F.jsonl ] || gcloud storage cp -q $GS/data/$F.jsonl $D/ > /dev/null 2>&1
    [ -s $D/$F.jsonl ] || { log "data file $F missing"; continue; }
    log "computing $O"
    CUDA_VISIBLE_DEVICES=0 timeout 3600 python $C/st_hidden.py --student $M --data $D/$F.jsonl --out $D/base/$O 2>&1 | grep -E "wrote|rror|layers kept" | tail -4
    [ -s $D/base/$O ] && gcloud storage cp -q $D/base/$O $GS/run1/base/$O > /dev/null 2>&1 && log "uploaded $O ($(stat -c %s $D/base/$O) bytes)" && rm -f $D/base/$O || log "FAILED $O"
  done
  case "$M" in $D/models_h/*) rm -rf "$M";; *) for h in $D/hf/hub ${HOME:-/root}/.cache/huggingface/hub /root/.cache/huggingface/hub; do rm -rf "$h/models--${M//\//--}"; done;; esac
  log "removed $tag model; free disk $(freegb) GB"
done
finish
