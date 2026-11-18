#!/bin/bash
# Measure each judge's false-positive / false-negative rates with and without the answer key, on the
# three students' base greedy TriviaQA answers (1500 questions each). One GPU. Output: gs://.../acl_st/run1/keyfp/.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
[ -f $D/env/.ok ] || { log "no environment"; exit 1; }
. $D/env/bin/activate
mkdir -p $D/code $D/keyfp; gcloud storage cp -q "$GS/code/*.py" $D/code/ > /dev/null 2>&1; C=$D/code
gcloud storage cp -q $GS/data/tqa_1500.jsonl $D/ > /dev/null 2>&1
for s in granite smol3 phi4m; do
  gcloud storage cp -q $GS/run1/rl/${s}_goldstrict_h1/eval_rl_ck0.jsonl $D/keyfp/all_$s.jsonl > /dev/null 2>&1
  python -c "import json; [print(l,end='') for l in open('$D/keyfp/all_$s.jsonl') if json.loads(l)['s']<0]" > $D/keyfp/cand_$s.jsonl
done
CANDS="$D/keyfp/cand_granite.jsonl $D/keyfp/cand_smol3.jsonl $D/keyfp/cand_phi4m.jsonl"
for spec in "Qwen/Qwen3.5-9B:q9:--quant fp8:--ref" "google/gemma-4-12B-it-qat-w4a16-ct:g12::--ref" "Qwen/Qwen3.5-2B:q2::--ref" "Qwen/Qwen3.5-2B:q2::" "Qwen/Qwen3.5-4B:q4::--ref" "Qwen/Qwen3.5-4B:q4::"; do
  IFS=: read -r J T Q R <<< "$spec"; tag=${T}_$( [ -n "$R" ] && echo key || echo free )
  log "scoring with $J ($tag)"
  timeout 3000 python $C/st_judge.py --judge $J $Q $R --task trivia --data $D/tqa_1500.jsonl --cands $CANDS --out $D/keyfp/score_$tag.jsonl --include_greedy 2>&1 | grep -E "scored [0-9]+/|rror" | tail -2
  gcloud storage cp -q $D/keyfp/score_$tag.jsonl $GS/run1/keyfp/ > /dev/null 2>&1
done
log "done"
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
sleep 10; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
shutdown -h now
