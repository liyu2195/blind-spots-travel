#!/bin/bash
# jscore4 = jscore3 plus control keys JTASK (trivia | gsm8k, passed to st_judge.py --task) and SFX (dataset-name suffix, default "_1500"; "" for gsm8k_platinum). jscore3 = jscore2 plus control key JFLAGS (extra st_judge.py flags, e.g. "--strict"). jscore2 = acl_st_81_jscore generalised: several judges in one job, and besides the untrained students' samples
# (base files) also candidate files of distinct final answers of trained runs (data/cand_runs_<set>.jsonl, built by work/final/build_cand_runs.py).
# For every judge spec "model:quant:tag" (quant "none" = no on-the-fly quantisation):
#   DO_BASE=1  -> score run1/base/<student>_<set>_1500_n<N>_s<SEED>.jsonl            -> run1/base/score_<tag>_<student>_<set>_1500_n<N>_s<SEED>.jsonl
#   DO_CANDS=1 -> score data/cand_runs_<set>.jsonl (all rows)                         -> run1/base/score_<tag>_cand_runs_<set>.jsonl
# Existing outputs are skipped. Same judge prompt and first-token rule as the judge reward (st_judge.py). One GPU. Always shuts the VM down at the end.
# Control file keys: JUDGES="model:quant:tag ..." DO_BASE DO_CANDS STUDENTS="qwen7 llama8" SETS="tqa pqa nqo" N SEED.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
JUDGES="Qwen/Qwen3.5-9B:fp8:Q"; DO_BASE=1; DO_CANDS=1; STUDENTS="qwen7 llama8"; SETS="tqa pqa nqo"; N=32; SEED=1; JFLAGS=""; JTASK=trivia; SFX=_1500
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
log "jscore2 on $HOST: judges [$JUDGES] base $DO_BASE cands $DO_CANDS students [$STUDENTS] sets [$SETS] n $N seed $SEED"
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
freegb() { df -BG --output=avail $D | tail -1 | tr -dc 0-9; }
if [ "$(freegb)" -lt 30 ]; then   # free disk: the Hugging Face download (xet) cache is only a cache
  log "free disk $(freegb) GB; hf usage: $(du -sh $D/hf/* 2> /dev/null | tr '\n' ' ')"; rm -rf $D/hf/xet ${HOME:-/root}/.cache/huggingface/xet; log "removed xet caches; free disk now $(freegb) GB"
fi
C=$D/code_js2; mkdir -p $C $D/js2; gcloud storage cp -q "$GS/code/*.py" $C/ > /dev/null 2>&1
[ -s $C/st_judge.py ] || { log "code missing"; finish; }
for S in $SETS; do gcloud storage cp -q $GS/data/${S}${SFX}.jsonl $D/ > /dev/null 2>&1; done
score() {   # $1 candidates file (local), $2 data file (local), $3 output name
  local O=$3
  if gcloud storage ls $GS/run1/base/$O > /dev/null 2>&1; then log "$O exists: skipping"; return; fi
  log "scoring $O with $JUDGE"
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python $C/st_judge.py --judge $JUDGE $QA --task $JTASK --data $2 --cands $1 \
    --out $D/js2/$O --include_greedy 2>&1 | grep -E "to score|scored [0-9]+/|rror|failed|could not" | tail -4
  [ -s $D/js2/$O ] && gcloud storage cp -q $D/js2/$O $GS/run1/base/$O > /dev/null 2>&1 && log "uploaded $O ($(wc -l < $D/js2/$O) rows)" || log "FAILED $O"
  gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.log > /dev/null 2>&1
}
for spec in $JUDGES; do
  JUDGE=${spec%%:*}; rest=${spec#*:}; QUANT=${rest%%:*}; JTAG=${rest#*:}
  QA=""; [ -n "$QUANT" ] && [ "$QUANT" != none ] && QA="--quant $QUANT"; [ -n "$JFLAGS" ] && QA="$QA $JFLAGS"
  log "judge $JUDGE (quant ${QUANT:-none}, tag $JTAG)"
  for S in $SETS; do
    [ -s $D/${S}${SFX}.jsonl ] || { log "data file ${S}${SFX} missing"; continue; }
    if [ "$DO_CANDS" = 1 ]; then
      gcloud storage cp -q $GS/data/cand_runs_$S.jsonl $D/js2/ > /dev/null 2>&1
      [ -s $D/js2/cand_runs_$S.jsonl ] && score $D/js2/cand_runs_$S.jsonl $D/${S}${SFX}.jsonl score_${JTAG}_cand_runs_$S.jsonl || log "cand_runs_$S missing"
    fi
    if [ "$DO_BASE" = 1 ]; then
      for st in $STUDENTS; do
        F=${st}_${S}${SFX}_n${N}_s${SEED}
        [ -s $D/js2/$F.jsonl ] || gcloud storage cp -q $GS/run1/base/$F.jsonl $D/js2/ > /dev/null 2>&1
        [ -s $D/js2/$F.jsonl ] && score $D/js2/$F.jsonl $D/${S}${SFX}.jsonl score_${JTAG}_$F.jsonl || log "base file $F missing"
      done
    fi
  done
done
finish
