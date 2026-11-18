#!/bin/bash
# PopQA long tail: the three students answer 3000 popularity-stratified PopQA questions (plain prompt:
# greedy + 8 samples; IDK prompt: greedy), then the open judges score the plain greedy answers with and without the key.
# Output: gs://.../acl_st/run1/popqa/. One GPU. Shuts down at the end.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
[ -f $D/env/.ok ] || { log "no environment"; exit 1; }
. $D/env/bin/activate
mkdir -p $D/code $D/popqa; gcloud storage cp -q "$GS/code/*.py" $D/code/ > /dev/null 2>&1; C=$D/code; P=$D/popqa
gcloud storage cp -q $GS/data/popqa_3000.jsonl $D/ > /dev/null 2>&1
for spec in "granite:ibm-granite/granite-4.2-3b" "smol3:HuggingFaceTB/SmolLM3-3B" "phi4m:microsoft/Phi-4-mini-instruct"; do
  s=${spec%%:*}; m=${spec#*:}; mkdir -p $P/$s
  log "generate $s"
  timeout 3000 python $C/st_gen.py --model $m --task trivia --data $D/popqa_3000.jsonl --out $P/$s/eval_base.jsonl --n 8 --greedy 2>&1 | grep -E "wrote|rror" | tail -2
  timeout 1800 python $C/st_gen.py --model $m --task trivia --data $D/popqa_3000.jsonl --out $P/$s/eval_base_idk.jsonl --n 0 --greedy --prompt idk 2>&1 | grep -E "wrote|rror" | tail -2
  python -c "import json; [print(l,end='') for l in open('$P/$s/eval_base.jsonl') if json.loads(l)['s']<0]" > $P/cand_$s.jsonl
  gcloud storage cp -q -r $P/$s $GS/run1/popqa/ > /dev/null 2>&1; gcloud storage cp -q $P/cand_$s.jsonl $GS/run1/popqa/ > /dev/null 2>&1
done
CANDS="$P/cand_granite.jsonl $P/cand_smol3.jsonl $P/cand_phi4m.jsonl"
for spec in "Qwen/Qwen3.5-9B:q9:--quant fp8:" "Qwen/Qwen3.5-9B:q9:--quant fp8:--ref" "google/gemma-4-12B-it-qat-w4a16-ct:g12::" "google/gemma-4-12B-it-qat-w4a16-ct:g12::--ref" "Qwen/Qwen3.5-4B:q4::" "Qwen/Qwen3.5-2B:q2::"; do
  IFS=: read -r J T Q R <<< "$spec"; tag=${T}_$( [ -n "$R" ] && echo key || echo free )
  log "score $tag"
  timeout 3000 python $C/st_judge.py --judge $J $Q $R --task trivia --data $D/popqa_3000.jsonl --cands $CANDS --out $P/score_$tag.jsonl --include_greedy 2>&1 | grep -E "scored [0-9]+/|rror" | tail -2
  gcloud storage cp -q $P/score_$tag.jsonl $GS/run1/popqa/ > /dev/null 2>&1
done
log "done"
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
sleep 10; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
shutdown -h now
