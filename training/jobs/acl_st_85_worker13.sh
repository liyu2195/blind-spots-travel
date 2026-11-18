#!/bin/bash
# v13 = v12 plus the control-file key STOPMIN (default 0 = no limit): once the VM has been up for STOPMIN minutes this
# worker claims nothing more and finishes (on-demand VMs stop by themselves 24 h after they start, so a run claimed too late would be cut).
# v12 = v11 plus the config key CKEVERY (default 0 = only the final checkpoint, exactly as v11): with CKEVERY=N the
# trainer saves a checkpoint every N steps (STEPS must be a multiple of N), every checkpoint is evaluated (greedy, all questions) as
# eval_rl_ck<step>.jsonl next to eval_rl_ck0.jsonl, six adapters per evaluation call, and the checkpoint folders are removed afterwards.
# v11 = v10 plus the config key KNOWSF: the file (in acl_st/data/) that tells the simulated grader on which
# questions the false-positive rate FPK applies (knows = true) and on which FPU (knows = false); default qwen_knows_h1.jsonl as before.
# Used for graders whose errors depend on the question (question type, popularity).
# v10 = v9 plus the config key REWARD (default synth = the simulated graders of v9, unchanged):
#   REWARD=api    frontier judge through the Anthropic Messages API (st_grpo.py --reward api, st_api.py). Config keys: APIMODEL (required),
#                 APIVER (default 2026-10-21), APIMAX (spend guard: most judge calls per run, default 12000), APIYES (default 4),
#                 APIWORKERS (default 8). The key is read by the trainer from Secret Manager into memory; it needs the secret-access
#                 permission of the VM's service account (project YOUR_GCP_PROJECT only), so api configs live in their own queue.
#   REWARD=judge  key-free open judge served on THIS VM (vLLM OpenAI server on a second GPU). Config keys: JT (threshold on P(Yes),
#                 required), JMODE (default free). The control file of the worker must name the server's GPU: JGPU=1 (with GPU=0), and
#                 may set JMODEL (default Qwen/Qwen3.5-9B) and JQUANT (default fp8; "none" = no quantization). A worker without JGPU
#                 refuses judge configs. One judge worker per 2-GPU VM.
# Default queue list "v6". Suggested: judge configs in queue v7, api configs in queue v8.
# v9 = v8 plus the config keys MICRO (st_grpo.py --micro: sequences per backward pass, default 64; larger students
# need 8-16 on one L4), GENCHUNK (st_grpo.py --gen_chunk: prompts per sampling call, default 0 = all 256 at once), CAP / ECAP (time limits of training / evaluation in seconds, defaults 18000 / 3600), and STUDENT given as a
# gs:// folder (copied to local disk first; used for gated models that live in the private bucket). Default queue list "v6".
# v8 = v7 plus the config key PROMPT (idk = refuse up front with "I don't know", default; flag = answer first, then add
# "(unsure)"; st_grpo.py / st_gen.py --prompt). Default queue list "v5 v4 v3 v2"; flag configs live ONLY in queue v5.
# v7 text: v7 = v6 plus the config key KLB (st_grpo.py --kl_beta: KL to the untrained model; default 0 = off).
# v6 text: v6 = v5 plus the config keys TRAINF / EVALF (training and evaluation question files in acl_st/data/, default
# TriviaQA tqa_h1.jsonl / tqa_1500.jsonl; PopQA: pqa_h1.jsonl / pqa_1500.jsonl); default queues "v3 v2". v5 text:
# Unified simulated-grader RL worker, v5 (= acl_st_64_worker3 plus the config keys FMT, TEMP, TOPP, TOPK,
# FORCEMODE, LRATE; default queue "v2"; FMT=1 is the strict output format of st_grpo.py --strict_format). v3 text:
# Unified simulated-grader RL worker (replaces the 61 and 63 workers).
# Claims run configs from the shared queues gs://.../acl_st/queue/<q>/* (q in $QUEUES, in that order) with an atomic
# create-if-absent claim marker, runs each on ONE GPU, evaluates greedy answers at ck0 and at the final checkpoint
# (IDK prompt), uploads, and repeats until nothing is claimable. GPU and QUEUES come from the control file.
# Safety rules (added after the gpu-vm incident, where two workers shared a GPU, failed 17 configs in 2 minutes each
# and kept the claims, then powered the VM off under two healthy runs):
#   - one worker per GPU per VM (file lock); a duplicate exits without claiming anything
#   - never claims while the GPU is busy with another process
#   - a failed run releases its own claim so another worker can take it (after 2 failures the config stays
#     quarantined), and a worker stops after 2 failures in a row
#   - the VM is shut down only when no other job on it is still working and no GPU process is running
set -u
GS=$BUCKET/acl_st; D=$WORK/acl_st; mkdir -p $D && cd $D
export HF_HOME=$D/hf HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
ME="$HOST/$JOB"
GPU=0; QUEUES="v6"; JGPU=""; JMODEL="Qwen/Qwen3.5-9B"; JQUANT=fp8; STOPMIN=0; eval "$(gcloud storage cat $GS/control/st_${HOST}_$JOB 2> /dev/null)"
JPORT=$((8000 + ${JGPU:-0})); JPID=""
log "worker $ME on GPU $GPU, queues: $QUEUES${JGPU:+, judge server $JMODEL ($JQUANT) on GPU $JGPU port $JPORT}, uptime limit for claiming: $STOPMIN min (0 = none)"

finish() {  # mark this worker finished; shut the VM down unless something else is still working on it
  if [ -n "$JPID" ]; then kill $JPID 2> /dev/null; sleep 5; pkill -f "vllm serve.*--port $JPORT" 2> /dev/null; sleep 10; fi
  touch /run/aclw_done_$JOB
  gcloud storage cp $WORK/logs/$JOB.log $BUCKET/status/$HOST/$JOB.done > /dev/null 2>&1
  local pid name
  for pid in $(pgrep -f "jobs/acl_st_[^ ]*\.sh"); do
    name=$(tr '\0' ' ' < /proc/$pid/cmdline 2> /dev/null | grep -o "jobs/acl_st_[^ ]*\.sh" | head -1 | sed 's#jobs/##; s#\.sh$##')
    [ -z "$name" ] && continue
    [ "$name" = "$JOB" ] && continue
    [ -f /run/aclw_done_$name ] || { log "job $name is still working: leaving the VM up"; exit 0; }
  done
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" ]; then
    log "a GPU process is still running: leaving the VM up"; exit 0
  fi
  log "nothing else is working: shutting the VM down"
  sleep 15; gcloud storage cp $WORK/logs/*.log $BUCKET/status/$HOST/ > /dev/null 2>&1
  shutdown -h now
  exit 0
}

# one worker per GPU on this VM
exec 9> /run/aclw_gpu$GPU.lock
flock -n 9 || { log "another worker already holds GPU $GPU on this VM: exiting without claiming"; finish; }
if [ -n "$JGPU" ]; then
  [ "$JGPU" != "$GPU" ] || { log "JGPU must differ from GPU"; finish; }
  exec 8> /run/aclw_gpu$JGPU.lock
  flock -n 8 || { log "another worker already holds GPU $JGPU on this VM: exiting without claiming"; finish; }
fi

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
[ -f $D/env/.ok ] || { log "environment failed"; finish; }
. $D/env/bin/activate
mkdir -p $D/code_w$GPU; gcloud storage cp -q "$GS/code/*.py" $D/code_w$GPU/ > /dev/null 2>&1; C=$D/code_w$GPU
gcloud storage cp -q $GS/data/tqa_1500.jsonl $GS/data/tqa_h1.jsonl $GS/data/qwen_knows_h1.jsonl $D/ > /dev/null 2>&1
[ -s $C/st_grpo.py ] && [ -s $C/st_gen.py ] && [ -s $D/tqa_h1.jsonl ] && [ -s $D/tqa_1500.jsonl ] || { log "code or data missing"; finish; }

judge_up() { curl -s -m 10 localhost:$JPORT/v1/models > /dev/null 2>&1; }
serve_judge() {  # (re)start the judge server on GPU $JGPU; returns 0 when it answers
  local i jq=""; [ -n "$JQUANT" ] && [ "$JQUANT" != none ] && jq="--quantization $JQUANT"
  judge_up && return 0
  [ -n "$JPID" ] && { kill $JPID 2> /dev/null; sleep 5; }
  mkdir -p $D/jserve
  CUDA_VISIBLE_DEVICES=$JGPU nohup vllm serve $JMODEL $jq --max-model-len 2048 --gpu-memory-utilization 0.90 --port $JPORT --max-logprobs 20 \
    --limit-mm-per-prompt '{"image":0,"video":0}' > $D/jserve/server_$JOB.log 2>&1 &
  JPID=$!
  for i in $(seq 1 120); do judge_up && break; kill -0 $JPID 2> /dev/null || break; sleep 10; done
  if ! judge_up; then   # same fallback as the earlier judge jobs: retry without the multimodal limit
    log "judge server did not start with the multimodal limit; retrying without"; tail -3 $D/jserve/server_$JOB.log
    kill $JPID 2> /dev/null; sleep 5
    CUDA_VISIBLE_DEVICES=$JGPU nohup vllm serve $JMODEL $jq --max-model-len 2048 --gpu-memory-utilization 0.90 --port $JPORT --max-logprobs 20 > $D/jserve/server_$JOB.log 2>&1 &
    JPID=$!
    for i in $(seq 1 120); do judge_up && break; kill -0 $JPID 2> /dev/null || break; sleep 10; done
  fi
  judge_up
}
if [ -n "$JGPU" ]; then
  serve_judge && log "judge server up" || { log "judge server failed"; tail -5 $D/jserve/server_$JOB.log; gcloud storage cp -q $D/jserve/server_$JOB.log $BUCKET/status/$HOST/ > /dev/null 2>&1; finish; }
fi

gpu_free() {  # true when GPU $GPU has (almost) no memory in use
  local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2> /dev/null | head -1 | tr -d ' ')
  [ -n "$u" ] && [ "$u" -lt 1500 ] 2> /dev/null
}
wait_gpu() { local i; for i in $(seq 1 20); do gpu_free && return 0; sleep 15; done; return 1; }

SKIP=" "   # configs this worker already failed on: never claim them again
claim_next() {  # sets Q (queue) and got (config name); returns 1 when nothing is claimable
  local q f cfg claimed
  got=""; Q=""
  for q in $QUEUES; do
    claimed=$(gcloud storage ls "$GS/claims/$q/*" 2> /dev/null | sed 's#.*/##')
    for f in $(gcloud storage ls "$GS/queue/$q/*" 2> /dev/null | sort); do
      cfg=$(basename $f)
      case "$SKIP" in *" $q/$cfg "*) continue;; esac
      echo "$claimed" | grep -qxF "$cfg" && continue
      if echo "$ME $(date -u)" | gcloud storage cp - $GS/claims/$q/$cfg --if-generation-match=0 > /dev/null 2>&1; then got=$cfg; Q=$q; return 0; fi
    done
  done
  return 1
}

run_cfg() {  # $1 queue, $2 config name; returns 0 when the final evaluation exists (or the run was already finished)
  local q=$1 cfg=$2 W R
  STUDENT=""; TAG=""; FPK=0; FPU=0; FN=0; ADV=std; IDK=0.5; SEED=0; OUTNAME=""; FPMODE=random; SWITCH=0; FPA=0; FNA=0; FORCE=0; STEPS=120; FMT=1; TEMP=0.8; TOPP=0.95; TOPK=-1; FORCEMODE=train; LRATE=5e-5; TRAINF=tqa_h1.jsonl; EVALF=tqa_1500.jsonl; KLB=0; PROMPT=idk
  MICRO=64; CAP=18000; ECAP=3600; GENCHUNK=0
  REWARD=synth; JT=""; JMODE=free; APIMODEL=""; APIVER=2026-10-21; APIMAX=12000; APIYES=4; APIWORKERS=8; KNOWSF=qwen_knows_h1.jsonl; CKEVERY=0
  gcloud storage cp -q "$GS/code/*.py" $C/ > /dev/null 2>&1
  eval "$(gcloud storage cat $GS/queue/$q/$cfg 2> /dev/null)"
  if [ "${STUDENT#gs://}" != "$STUDENT" ]; then   # model folder in the bucket: copy it to this GPU's local folder once
    local L=$D/models_w$GPU/$(basename $STUDENT)
    [ -s $L/config.json ] || { mkdir -p $L; gcloud storage cp -q -r "$STUDENT/*" $L/ > /dev/null 2>&1; }
    [ -s $L/config.json ] || { log "model copy from $STUDENT failed"; return 1; }
    STUDENT=$L
  fi
  [ -n "$STUDENT" ] && [ -n "$OUTNAME" ] || { log "could not read config $q/$cfg"; return 1; }
  W=$D/rl/$OUTNAME; R=$GS/run1/rl/$OUTNAME
  [ -s $D/$TRAINF ] || gcloud storage cp -q $GS/data/$TRAINF $D/ > /dev/null 2>&1
  [ -s $D/$EVALF ] || gcloud storage cp -q $GS/data/$EVALF $D/ > /dev/null 2>&1
  [ -s $D/$TRAINF ] && [ -s $D/$EVALF ] || { log "data files $TRAINF / $EVALF missing"; return 1; }
  [ -s $D/$KNOWSF ] || gcloud storage cp -q $GS/data/$KNOWSF $D/ > /dev/null 2>&1
  [ -s $D/$KNOWSF ] || { log "file $KNOWSF missing"; return 1; }
  if gcloud storage ls $R/RL.done > /dev/null 2>&1; then log "$OUTNAME is already finished: skipping"; return 0; fi
  rm -rf $W; mkdir -p $W   # the training log is opened in append mode, so never reuse a partial folder
  local RA="" CKV=$STEPS
  if [ "$CKEVERY" -gt 0 ] 2> /dev/null; then
    [ $((STEPS % CKEVERY)) -eq 0 ] || { log "config $q/$cfg: STEPS $STEPS is not a multiple of CKEVERY $CKEVERY"; return 1; }
    CKV=$CKEVERY
  fi
  case "$REWARD" in
    synth) RA="--reward synth --fp_known $FPK --fp_unknown $FPU --knows_file $D/$KNOWSF --fn_rate $FN --fp_mode $FPMODE --switch_step $SWITCH --fp_after $FPA --fn_after $FNA --force_idk $FORCE --force_mode $FORCEMODE";;
    judge) [ -n "$JGPU" ] || { log "config $q/$cfg needs a judge server, but this worker has none (no JGPU in its control file)"; return 1; }
           [ -n "$JT" ] || { log "config $q/$cfg has no judge threshold JT"; return 1; }
           serve_judge || { log "judge server is down and could not be restarted"; return 1; }
           RA="--reward judge --judge_url http://127.0.0.1:$JPORT --judge_model $JMODEL --judge_mode $JMODE --judge_thresh $JT";;
    api)   [ -n "$APIMODEL" ] || { log "config $q/$cfg has no APIMODEL"; return 1; }
           RA="--reward api --api_model $APIMODEL --api_version $APIVER --api_max_calls $APIMAX --api_yes $APIYES --api_workers $APIWORKERS";;
    *)     log "config $q/$cfg has an unknown REWARD [$REWARD]"; return 1;;
  esac
  log "GPU$GPU run $q/$cfg -> $OUTNAME (reward $REWARD [judge thr ${JT:--} api ${APIMODEL:--}] fp $FPK/$FPU ($KNOWSF) $FPMODE fn $FN adv $ADV idk $IDK seed $SEED steps $STEPS switch $SWITCH force $FORCE/$FORCEMODE fmt $FMT temp $TEMP top_p $TOPP top_k $TOPK lr $LRATE data $TRAINF / $EVALF kl_beta $KLB prompt $PROMPT micro $MICRO cap $CAP)"
  CUDA_VISIBLE_DEVICES=$GPU timeout $CAP python $C/st_grpo.py --student $STUDENT $RA --data $D/$TRAINF --out $W \
    --steps $STEPS --lr $LRATE --ck $CKV --seed $SEED --adv_norm $ADV --prompt $PROMPT --idk_credit $IDK \
    --strict_format $FMT --temperature $TEMP --top_p $TOPP --top_k $TOPK --kl_beta $KLB --micro $MICRO --gen_chunk $GENCHUNK 2>&1 | grep -v -E "Warning|warn" | tail -3
  [ -s $W/api_verdicts.jsonl ] && gcloud storage cp -q $W/api_verdicts.jsonl $R/ > /dev/null 2>&1
  [ -d $W/ck$STEPS ] || { [ -s $W/train_log.jsonl ] && gcloud storage cp -q $W/train_log.jsonl $R/train_log_partial.jsonl > /dev/null 2>&1; }
  [ -d $W/ck$STEPS ] || { log "training of $OUTNAME did not reach the final checkpoint"; return 1; }
  gcloud storage cp -q $W/train_log.jsonl $R/ > /dev/null 2>&1
  if [ "$CKV" = "$STEPS" ]; then
    CUDA_VISIBLE_DEVICES=$GPU timeout $ECAP python $C/st_gen.py --model $STUDENT --task trivia --data $D/$EVALF --out "$W/eval_rl_{tag}.jsonl" \
      --n 0 --greedy --loras "ck0=,ck120=$W/ck$STEPS" --prompt $PROMPT --gpu_mem 0.85 2>&1 | grep -E "wrote|rror" | tail -3
    [ -s $W/eval_rl_ck120.jsonl ] && [ -s $W/eval_rl_ck0.jsonl ] || { log "evaluation of $OUTNAME failed"; return 1; }
    gcloud storage cp -q $W/eval_rl_ck0.jsonl $W/eval_rl_ck120.jsonl $R/ > /dev/null 2>&1
  else   # every checkpoint, six adapters per call (the first call also evaluates the untrained model as ck0)
    local L="ck0=" k=0 c
    for c in $(seq $CKV $CKV $STEPS); do
      [ -d $W/ck$c ] || { log "checkpoint ck$c of $OUTNAME is missing"; return 1; }
      if [ -n "$L" ]; then L="$L,ck$c=$W/ck$c"; else L="ck$c=$W/ck$c"; fi
      k=$((k+1))
      if [ $k -ge 6 ] || [ $c -eq $STEPS ]; then
        CUDA_VISIBLE_DEVICES=$GPU timeout $ECAP python $C/st_gen.py --model $STUDENT --task trivia --data $D/$EVALF --out "$W/eval_rl_{tag}.jsonl" \
          --n 0 --greedy --loras "$L" --prompt $PROMPT --gpu_mem 0.85 2>&1 | grep -E "wrote|rror" | tail -7
        L=""; k=0
      fi
    done
    for c in 0 $(seq $CKV $CKV $STEPS); do [ -s $W/eval_rl_ck$c.jsonl ] || { log "evaluation of $OUTNAME failed at ck$c"; return 1; }; done
    gcloud storage cp -q $W/eval_rl_ck*.jsonl $R/ > /dev/null 2>&1
    rm -rf $W/ck[0-9]*   # free the disk: the adapters are not needed after the evaluation
  fi
  echo "done $(date -u) by $ME" | gcloud storage cp - $R/RL.done > /dev/null 2>&1
  log "finished $OUTNAME"
  return 0
}

fail_cfg() {  # $1 queue, $2 config: record the failure; release my claim unless the config has now failed twice
  local q=$1 cfg=$2 nf
  echo "$ME $(date -u)" | gcloud storage cp - "$GS/fails/$q/$cfg/$HOST.$JOB.$(date -u +%H%M%S)" > /dev/null 2>&1
  nf=$(gcloud storage ls "$GS/fails/$q/$cfg/" 2> /dev/null | wc -l)
  SKIP="$SKIP$q/$cfg "
  if [ "$nf" -lt 2 ] && gcloud storage cat $GS/claims/$q/$cfg 2> /dev/null | grep -q "^$ME "; then
    gcloud storage rm -q $GS/claims/$q/$cfg > /dev/null 2>&1
    log "FAILED $q/$cfg (failure $nf): claim released for another worker"
  else
    log "FAILED $q/$cfg (failure $nf): left quarantined, claim kept"
  fi
}

n=0; nfail=0; idle=0
prev=$(gcloud storage cat $GS/owner/$HOST/$JOB 2> /dev/null)   # the config this worker held if the VM went down mid-run
while true; do
  wait_gpu || { log "GPU $GPU is busy with another process (waited 5 minutes): stopping without claiming, VM left up"; exit 1; }
  got=""; Q=""
  if [ -n "$prev" ]; then
    pq=${prev%%/*}; pc=${prev#*/}; prev=""
    if gcloud storage cat $GS/claims/$pq/$pc 2> /dev/null | grep -q "^$ME "; then Q=$pq; got=$pc; log "taking up my earlier claim $Q/$got again"; fi
  fi
  if [ -z "$got" ] && [ "$STOPMIN" -gt 0 ] 2> /dev/null; then
    up=$(awk '{printf "%d", $1 / 60}' /proc/uptime)
    if [ "$up" -ge "$STOPMIN" ]; then log "the VM has been up for $up minutes (limit $STOPMIN): claiming nothing more"; break; fi
  fi
  if [ -z "$got" ]; then
    if ! claim_next; then
      idle=$((idle+1)); [ $idle -gt 2 ] && break
      log "nothing to claim; checking again in 3 minutes ($idle/2)"; sleep 180; continue
    fi
  fi
  idle=0
  echo "$Q/$got" | gcloud storage cp - $GS/owner/$HOST/$JOB > /dev/null 2>&1
  if run_cfg $Q $got; then n=$((n+1)); nfail=0; else fail_cfg $Q $got; nfail=$((nfail+1)); fi
  gcloud storage rm -q $GS/owner/$HOST/$JOB > /dev/null 2>&1
  [ $nfail -ge 2 ] && { log "2 failures in a row: stopping this worker"; break; }
done
log "worker finished after $n runs"
finish
