#!/bin/bash
# Paraphrase transfer: the TriviaQA round-2 students (all filters, judges Qwen/Gemma/Qwen-reasoning)
# answer meaning-preserving rewrites of the questions they were trained on (tqa_para, 2 per question), greedy + 8
# samples, plus the base model. Inference only. Outputs gs://.../acl_st/run1/para/<TAG>/; shuts down.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
STUDENT=""; TAG=""
eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
. $D/env/bin/activate; mkdir -p $D/code; gcloud storage cp -q "$GS/code/*.py" $D/code/ > /dev/null 2>&1; C=$D/code
W=$D/para/$TAG; R=$GS/run1/para/$TAG; mkdir -p $W/ad
gcloud storage cp -q $GS/data/tqa_para.jsonl $D/ > /dev/null 2>&1
L=""
for spec in ":gold:A_gold" ":judge:A_judge" ":noise:A_noise" ":major:A_major" "_g4:judge:B_judge" "_g4:noise:B_noise" "_qr:judge:R_judge" "_qr:noise:R_noise"; do
  IFS=: read suf c name <<< "$spec"
  gcloud storage cp -q -r $GS/run1/trivia/$TAG$suf/ad_r2_$c $W/ad/ > /dev/null 2>&1 && mv $W/ad/ad_r2_$c $W/ad/$name && L="$L$name=$W/ad/$name,"
done
L=${L%,}; log "student $STUDENT adapters [$L]"
timeout 5400 python $C/st_gen.py --model $STUDENT --task trivia --data $D/tqa_para.jsonl --out $W/para_base.jsonl --n 8 --greedy 2>&1 | grep -E "wrote|rror" | tail -2
timeout 9000 python $C/st_gen.py --model $STUDENT --task trivia --data $D/tqa_para.jsonl --out "$W/para_{tag}.jsonl" --n 8 --greedy --loras "$L" 2>&1 | grep -E "wrote|rror" | tail -9
gcloud storage cp -q $W/para_*.jsonl $R/ > /dev/null 2>&1 && log "uploaded $(ls $W/para_*.jsonl | wc -l) files"
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
sleep 10; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
shutdown -h now
