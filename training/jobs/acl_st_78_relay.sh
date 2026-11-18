#!/bin/bash
# RL with an OUTSIDE judge through a relay: no judge model and no API key on this VM. Each step's new (question, answer)
# pairs go to gs://.../acl_st/relay/<OUTNAME>/req_<tag>_<step>.tsv; the trainer waits for resp_<tag>_<step>.tsv (st_grpo.py v9, --reward relay).
# Control keys as in acl_st_74_grpo6 plus RCACHE (file in acl_st/relay/ with earlier verdicts) and RYES (level that counts as YES, default 4). One GPU is enough.
# v4 text: RL with a judge reward, v4 (= acl_st_66_grpo4 plus EVALF: evaluation question file, default tqa_1500; PopQA: pqa_1500).
# v3 text: RL with a judge reward, v3. JT=<tau> turns the judge's
# P(Yes) into a 0/1 verdict at that threshold (st_grpo.py --judge_thresh); SCOREG=0 skips the final Gemma scoring.
# v2 text: adds the answer-key judge (JMODE=ref), the commit-first judge
# (JMODE=own: the judge first answers each training question itself), and the "I don't know" outlet (PROMPT=idk, IDK=credit). 2-GPU VM: GPU1 serves the reward judge (vLLM OpenAI server),
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
JMODE=free; PROMPT=plain; IDK=""; LOGJ=G; JMODEL=""; JT=""; SCOREG=1; FMT=0; EVALF=tqa_1500; TEMP=0.8; TOPP=0.95; TOPK=-1; KLB=0; RCACHE=""; RYES=4
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
left(){ echo $(( CAP - ($(date +%s) - start) )); }
run(){ timeout $(left) "$@"; }
QJ="Qwen/Qwen3.5-9B"; GJ="google/gemma-4-12B-it-qat-w4a16-ct"
if [ "$REWARD" = judgeG ] || { [ "$REWARD" = gold_strict ] && [ "$LOGJ" = G ]; }; then JM=$GJ; JQ=""; GQ=""; else JM=$QJ; JQ="--quantization fp8"; GQ="--quant fp8"; fi
[ -n "$JMODEL" ] && { JM=$JMODEL; JQ=""; GQ=""; }
RW=$REWARD; [ "$REWARD" = judgeQ ] || [ "$REWARD" = judgeG ] && RW=judge
[ -n "$OUTNAME" ] || OUTNAME=${TAG}_$REWARD
W=$D/rl/$OUTNAME; R=$GS/run1/rl/$OUTNAME; mkdir -p $W
log "host $HOST student $STUDENT reward $REWARD ($RW, judge server $JM, mode $JMODE) prompt $PROMPT idk_credit [$IDK] train $TRAIN steps $STEPS lr $LR seed $SEED judge_thresh [${JT:--1}] strict_format $FMT eval $EVALF temp $TEMP top_p $TOPP top_k $TOPK"
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
gcloud storage cp -q $GS/data/$EVALF.jsonl $GS/data/$TRAIN.jsonl $D/ > /dev/null 2>&1
[ "$PROMPT" = idk ] || gcloud storage cp -q $GS/run1/trivia/$TAG/eval_base.jsonl $W/eval_rl_ck0.jsonl > /dev/null 2>&1
RD=$GS/relay/$OUTNAME; RT=$(date +%s)
[ -n "$RCACHE" ] && gcloud storage cp -q $GS/relay/$RCACHE $W/relay_cache.jsonl > /dev/null 2>&1
log "relay folder $RD tag $RT cache [$RCACHE] $(wc -l < $W/relay_cache.jsonl 2> /dev/null) verdicts; YES = level >= $RYES"
log "GRPO training (relay reward)"
CUDA_VISIBLE_DEVICES=0 run python $C/st_grpo.py --student $STUDENT --reward relay --data $D/$TRAIN.jsonl --out $W \
  --steps $STEPS --lr $LR --ck $CK --seed $SEED --relay_dir $RD --relay_cache $W/relay_cache.jsonl --relay_yes $RYES --relay_tag $RT \
  --prompt $PROMPT --idk_credit ${IDK:--1} --strict_format $FMT --temperature $TEMP --top_p $TOPP --top_k $TOPK --kl_beta $KLB 2>&1 | grep -v -E "Warning|warn" | tail -60
echo "done $(date -u)" | gcloud storage cp - $RD/DONE > /dev/null 2>&1
gcloud storage cp -q $W/train_log.jsonl $W/relay_verdicts.jsonl $R/ > /dev/null 2>&1
for d in $W/ck*/; do gcloud storage cp -q -r ${d%/} $R/ > /dev/null 2>&1; done
# ---- evaluate checkpoints (greedy + 8 samples on all 1500 questions)
L=$(for d in $(ls -d $W/ck*/ 2> /dev/null | grep -v "/ck0/"); do n=$(basename $d); printf "%s=%s," $n ${d%/}; done | sed 's/,$//')
[ "$PROMPT" = idk ] && L="ck0=,$L"
log "eval of [$L]"
[ -n "$L" ] && CUDA_VISIBLE_DEVICES=0 run python $C/st_gen.py --model $STUDENT --task trivia --data $D/$EVALF.jsonl --out "$W/eval_rl_{tag}.jsonl" --n 8 --greedy --loras "$L" --prompt $PROMPT 2>&1 | grep -E "wrote|rror" | tail -6
gcloud storage cp -q $W/eval_rl_ck*.jsonl $R/ > /dev/null 2>&1
[ -s $W/eval_rl_ck$STEPS.jsonl ] && echo "done $(date -u)" | gcloud storage cp - $R/RL.done > /dev/null 2>&1
log "finished"
if [ "$NOSHUTDOWN" = 1 ]; then exit 0; fi
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
sleep 15; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
shutdown -h now
