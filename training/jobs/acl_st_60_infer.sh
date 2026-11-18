#!/bin/bash
# Inference-only follow-ups for the "judges can check math, not facts" story. MODE from
# control/st_<host>_<job>: exploit (both judges score rewritten answer strategies), logp (judges' own likelihood of
# pilot candidates), rescoreQ / rescoreG (reasoning-mode judge re-scores RL students' greedy answers). Shuts down.
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
MODE=""; eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
. $D/env/bin/activate; mkdir -p $D/code; gcloud storage cp -q "$GS/code/*.py" $D/code/ > /dev/null 2>&1; C=$D/code
gcloud storage cp -q $GS/data/tqa_1500.jsonl $GS/data/abl_gsm200.jsonl $GS/data/abl_tq300.jsonl $D/ > /dev/null 2>&1
QJ="Qwen/Qwen3.5-9B"; GJ="google/gemma-4-12B-it-qat-w4a16-ct"; O=$D/x60; mkdir -p $O; R=$GS/x60
log "mode $MODE"
case $MODE in
exploit)
  mkdir -p $O/var; gcloud storage cp -q "$GS/variants/*.jsonl" $O/var/ > /dev/null 2>&1
  timeout 5400 python $C/st_judge.py --judge $QJ --quant fp8 --task trivia --data $D/tqa_1500.jsonl --cands $O/var/cand_*.jsonl --out $O/score_var_Q.jsonl 2>&1 | grep -E "scored [0-9]+/|rror" | tail -2
  rm -rf $HF_HOME/hub/models--Qwen--Qwen3.5-9B
  timeout 5400 python $C/st_judge.py --judge $GJ --task trivia --data $D/tqa_1500.jsonl --cands $O/var/cand_*.jsonl --out $O/score_var_G.jsonl 2>&1 | grep -E "scored [0-9]+/|rror" | tail -2
  gcloud storage cp -q $O/score_var_*.jsonl $R/ > /dev/null 2>&1 ;;
logp)
  mkdir -p $O/gsm_c $O/tq_c
  for s in S1 S2 S3 S4 S5; do gcloud storage cp -q $BUCKET/acl_sj/out/cand_$s.jsonl $O/gsm_c/ > /dev/null 2>&1; gcloud storage cp -q $BUCKET/acl_sj/tq/out/cand_$s.jsonl $O/tq_c/ > /dev/null 2>&1; done
  for spec in "$QJ:q359:--quant fp8" "$GJ:g412:"; do j=${spec%%:*}; rest=${spec#*:}; tag=${rest%%:*}; q=${rest#*:}
    timeout 3600 python $C/st_logp.py --judge $j $q --task gsm8k --data $D/abl_gsm200.jsonl --cand_dir $O/gsm_c --items 200 --out $O/logp_${tag}_gsm.jsonl 2>&1 | grep -E "candidates in|rror" | tail -2
    timeout 3600 python $C/st_logp.py --judge $j $q --task trivia --data $D/abl_tq300.jsonl --cand_dir $O/tq_c --items 300 --out $O/logp_${tag}_tq.jsonl 2>&1 | grep -E "candidates in|rror" | tail -2
  done
  gcloud storage cp -q $O/logp_*.jsonl $R/ > /dev/null 2>&1 ;;
rescoreQ|rescoreG)
  mkdir -p $O/rl; gcloud storage cp -q "$GS/rlscore/cand_*.jsonl" $O/rl/ > /dev/null 2>&1
  if [ $MODE = rescoreQ ]; then j=$QJ; q="--quant fp8"; t=Q; else j=$GJ; q=""; t=G; fi
  timeout 9000 python $C/st_judge.py --judge $j $q --reason --max_model_len 4096 --include_greedy --task trivia --data $D/tqa_1500.jsonl --cands $O/rl/cand_*.jsonl --out $O/score_rl_reason_$t.jsonl 2>&1 | grep -E "scored [0-9]+/|rror" | tail -2
  gcloud storage cp -q $O/score_rl_reason_$t.jsonl $R/ > /dev/null 2>&1 ;;
esac
log "done"
gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
sleep 10; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
shutdown -h now
