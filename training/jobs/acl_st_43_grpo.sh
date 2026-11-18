#!/bin/bash
# RL with a judge reward. 2-GPU VM: GPU1 serves the reward judge (vLLM OpenAI server),
# GPU0 runs GRPO on the LoRA student (st_grpo.py) on TriviaQA (TRAIN, default tqa_1500), then evaluates checkpoints (greedy + 8
# samples) and scores them with both judges (Qwen3.5-9B FP8, Gemma-4-12B). Reads control/st_<host>_<job>:
#   STUDENT TAG REWARD(judgeQ|judgeG|gold|major) STEPS LR CK CAP. Outputs gs://.../acl_st/run1/rl/<TAG>_<REWARD>/.
# Shuts the VM down at the end.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
start=$(date +%s)
STUDENT=""; TAG=""; REWARD=judgeQ; STEPS=150; LR=5e-5; CK=50; CAP=18000; NOSHUTDOWN=0; TRAIN=tqa_1500; OUTNAME=""; SEED=0
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
left(){ echo $(( CAP - ($(date +%s) - start) )); }
run(){ timeout $(left) "$@"; }
QJ="Qwen/Qwen3.5-9B"; GJ="google/gemma-4-12B-it-qat-w4a16-ct"
if [ "$REWARD" = judgeG ]; then JM=$GJ; JQ=""; else JM=$QJ; JQ="--quantization fp8"; fi
RW=$REWARD; [ "$REWARD" = judgeQ ] || [ "$REWARD" = judgeG ] && RW=judge
[ -n "$OUTNAME" ] || OUTNAME=${TAG}_$REWARD
W=$D/rl/$OUTNAME; R=$GS/run1/rl/$OUTNAME; mkdir -p $W
log "host $HOST student $STUDENT reward $REWARD ($RW, judge server $JM) train $TRAIN steps $STEPS lr $LR"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
if [ ! -f $D/env/.ok ]; then
  if [ ! -f /usr/include/python3.10/Python.h ] || ! python3 -c "import ensurepip" > /dev/null 2>&1; then
    { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv python3.10-dev build-essential; } > /dev/null 2>&1
  fi
  rm -rf $D/env; python3 -m venv $D/env && . $D/env/bin/activate
  pip install -q "vllm==0.30.0" "peft==0.21.2" accelerate 2>&1 | tail -2
  python -c "import vllm, transformers, torch, peft; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" && touch $D/env/.ok
fi
[ -f $D/env/.ok ] || { log "environment failed"; exit 1; }
. $D/env/bin/activate
mkdir -p $D/code; gcloud storage cp -q "$GS/code/*.py" $D/code/ > /dev/null 2>&1; C=$D/code
gcloud storage cp -q $GS/data/tqa_1500.jsonl $GS/data/$TRAIN.jsonl $D/ > /dev/null 2>&1
gcloud storage cp -q $GS/run1/trivia/$TAG/eval_base.jsonl $W/eval_rl_ck0.jsonl > /dev/null 2>&1
# ---- reward judge server on GPU1
serve(){ CUDA_VISIBLE_DEVICES=1 nohup vllm serve $JM $JQ --max-model-len 2048 --gpu-memory-utilization 0.90 --port 8000 --max-logprobs 20 "$@" > $W/server.log 2>&1 & echo $!; }
SP=$(serve --limit-mm-per-prompt '{"image":0,"video":0}')
for i in $(seq 1 90); do curl -s localhost:8000/v1/models > /dev/null 2>&1 && break; kill -0 $SP 2> /dev/null || break; sleep 10; done
if ! curl -s localhost:8000/v1/models > /dev/null 2>&1; then
  log "server failed with the multimodal limit, retrying without"; kill $SP 2> /dev/null; sleep 5; tail -3 $W/server.log
  SP=$(serve); for i in $(seq 1 90); do curl -s localhost:8000/v1/models > /dev/null 2>&1 && break; sleep 10; done
fi
curl -s localhost:8000/v1/models > /dev/null 2>&1 && log "judge server up" || { log "judge server failed"; tail -5 $W/server.log; }
# ---- GRPO on GPU0
log "GRPO training"
CUDA_VISIBLE_DEVICES=0 run python $C/st_grpo.py --student $STUDENT --reward $RW --data $D/$TRAIN.jsonl --out $W \
  --judge_url http://127.0.0.1:8000 --judge_model $JM --steps $STEPS --lr $LR --ck $CK --seed $SEED 2>&1 | grep -v -E "Warning|warn" | tail -40
gcloud storage cp -q $W/train_log.jsonl $W/server.log $R/ > /dev/null 2>&1
for d in $W/ck*/; do gcloud storage cp -q -r ${d%/} $R/ > /dev/null 2>&1; done
kill $SP 2> /dev/null; sleep 5; pkill -f "vllm serve" ; sleep 10
# ---- evaluate checkpoints (greedy + 8 samples on all 1500 questions)
L=$(for d in $(ls -d $W/ck*/ 2> /dev/null | grep -v "/ck0/"); do n=$(basename $d); printf "%s=%s," $n ${d%/}; done | sed 's/,$//')
log "eval of [$L]"
[ -n "$L" ] && CUDA_VISIBLE_DEVICES=0 run python $C/st_gen.py --model $STUDENT --task trivia --data $D/tqa_1500.jsonl --out "$W/eval_rl_{tag}.jsonl" --n 8 --greedy --loras "$L" 2>&1 | grep -E "wrote|rror" | tail -6
gcloud storage cp -q $W/eval_rl_ck*.jsonl $R/ > /dev/null 2>&1
# ---- score every checkpoint (and base) with both judges, in parallel on the two GPUs
CANDS=$(ls $W/eval_rl_ck*.jsonl)
log "scoring with both judges"
( CUDA_VISIBLE_DEVICES=1 run python $C/st_judge.py --judge $QJ --quant fp8 --task trivia --data $D/tqa_1500.jsonl --cands $CANDS --out $W/score_rl_Q.jsonl --include_greedy 2>&1 | grep -E "scored [0-9]+/|rror" | tail -2 ) &
( CUDA_VISIBLE_DEVICES=0 run python $C/st_judge.py --judge $GJ --task trivia --data $D/tqa_1500.jsonl --cands $CANDS --out $W/score_rl_G.jsonl --include_greedy 2>&1 | grep -E "scored [0-9]+/|rror" | tail -2 ) &
wait
gcloud storage cp -q $W/score_rl_Q.jsonl $W/score_rl_G.jsonl $R/ > /dev/null 2>&1
[ -s $W/score_rl_Q.jsonl ] && [ -s $W/score_rl_G.jsonl ] && echo "done $(date -u)" | gcloud storage cp - $R/RL.done > /dev/null 2>&1
log "finished"
if [ "$NOSHUTDOWN" = 1 ]; then exit 0; fi
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
sleep 15; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
shutdown -h now
