#!/bin/bash
# Simulated-grader RL worker: claims run configs from a shared queue (gs://.../acl_st/queue/sim/*),
# atomically (create-if-absent claim marker), runs each on ONE GPU (GPU from control), evaluates greedy answers at ck0
# (IDK prompt) and ck120, uploads, repeats until the queue is empty. Shuts the VM down when all workers on it are done.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
GPU=0; eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
if [ ! -f $D/env/.ok ]; then
  if [ "$GPU" != 0 ]; then for i in $(seq 1 120); do [ -f $D/env/.ok ] && break; sleep 30; done; fi
  if [ ! -f $D/env/.ok ]; then
    if [ ! -f /usr/include/python3.10/Python.h ] || ! python3 -c "import ensurepip" > /dev/null 2>&1; then
      { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv python3.10-dev build-essential; } > /dev/null 2>&1
    fi
    rm -rf $D/env; python3 -m venv $D/env && . $D/env/bin/activate
    pip install -q "vllm==0.30.0" "peft==0.21.2" accelerate 2>&1 | tail -2
    python -c "import vllm, peft" && touch $D/env/.ok
  fi
fi
[ -f $D/env/.ok ] || { log "environment failed"; exit 1; }
. $D/env/bin/activate
mkdir -p $D/code_w$GPU; gcloud storage cp -q "$GS/code/*.py" $D/code_w$GPU/ > /dev/null 2>&1; C=$D/code_w$GPU
gcloud storage cp -q $GS/data/tqa_1500.jsonl $GS/data/tqa_h1.jsonl $GS/data/qwen_knows_h1.jsonl $D/ > /dev/null 2>&1
n=0
while true; do
  got=""
  for q in $(gcloud storage ls "$GS/queue/sim/*" 2> /dev/null | sort); do
    cfg=$(basename $q)
    if echo "$HOST/$JOB $(date -u)" | gcloud storage cp - $GS/claims/sim/$cfg --if-generation-match=0 > /dev/null 2>&1; then got=$cfg; break; fi
  done
  [ -z "$got" ] && break
  STUDENT=""; TAG=""; FPK=0; FPU=0; FN=0; ADV=std; IDK=0.5; SEED=0; OUTNAME=""
  eval "$(gcloud storage cat $GS/queue/sim/$got 2> /dev/null)"
  W=$D/rl/$OUTNAME; R=$GS/run1/rl/$OUTNAME; mkdir -p $W
  log "GPU$GPU run $OUTNAME (fp $FPK/$FPU fn $FN adv $ADV idk $IDK seed $SEED)"
  CUDA_VISIBLE_DEVICES=$GPU timeout 14400 python $C/st_grpo.py --student $STUDENT --reward synth --data $D/tqa_h1.jsonl --out $W \
    --steps 120 --lr 5e-5 --ck 120 --seed $SEED --fp_known $FPK --fp_unknown $FPU --knows_file $D/qwen_knows_h1.jsonl \
    --fn_rate $FN --adv_norm $ADV --prompt idk --idk_credit $IDK 2>&1 | grep -v -E "Warning|warn" | tail -3
  gcloud storage cp -q $W/train_log.jsonl $R/ > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python $C/st_gen.py --model $STUDENT --task trivia --data $D/tqa_1500.jsonl --out "$W/eval_rl_{tag}.jsonl" \
    --n 0 --greedy --loras "ck0=,ck120=$W/ck120" --prompt idk --gpu_mem 0.85 2>&1 | grep -E "wrote|rror" | tail -3
  gcloud storage cp -q $W/eval_rl_ck0.jsonl $W/eval_rl_ck120.jsonl $R/ > /dev/null 2>&1
  [ -s $W/eval_rl_ck120.jsonl ] && echo "done $(date -u)" | gcloud storage cp - $R/RL.done > /dev/null 2>&1
  n=$((n+1))
done
log "queue empty after $n runs"
touch $WORK/worker_$JOB.done
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
for j in $WORK/jobs/acl_st_61_worker*.sh; do [ -f $WORK/worker_$(basename $j .sh).done ] || exit 0; done
sleep 15; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
shutdown -h now
