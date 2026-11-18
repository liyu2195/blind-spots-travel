"""GRPO (on-policy, one update per batch, group-normalized advantages, no KL) for TriviaQA with a LoRA student.
Rewards: judge (P(Yes) from a vLLM OpenAI server running the judge), gold (correct under the alias matcher), major
(TTRL-style: 1 if the sample's answer is the group's strict majority answer). Every step logs the mean reward, the true
correctness of the samples and, when a judge server is given, the judge's mean P(Yes). LoRA checkpoints every --ck steps.
Judge variants (--judge_mode): free (reference-free, as in the pilot), ref (the judge also sees the reference answer: the
"answer key" control), own (commit-first: the judge sees its own earlier answer to the question, from --own_file).
Abstention outlet: --prompt idk tells the student it may reply "I don't know", and --idk_credit c gives such replies the
fixed reward c under any reward (the judge is not asked about them).
Usage: python st_grpo.py --student M --reward judge|gold|major --data tqa_1500.jsonl --out DIR [--judge_url URL --judge_model J]
"""
import argparse, hashlib, json, math, os, random, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_common as C


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            print("judge request failed:", str(e)[:200], flush=True); time.sleep(5 * (attempt + 1))
    raise SystemExit("judge server unreachable")


def _relay_cp(src, dst):
    """copy between local paths and gs:// paths; True on success"""
    import subprocess, shutil
    if src.startswith("gs://") or dst.startswith("gs://"):
        return subprocess.run(["gcloud", "storage", "cp", "-q", src, dst], capture_output=True).returncode == 0
    if not os.path.exists(src):
        return False
    shutil.copyfile(src, dst + ".tmp"); os.replace(dst + ".tmp", dst); return True


def relay_ask(a, step, need, qtext, rcache):
    """need: {(idx, normalised answer): raw answer line}. Blocks until the outside judge has returned a level for every pair."""
    keys = list(need); one = lambda t: t.replace("\t", " ").replace("\n", " ").strip()
    req = os.path.join(a.out, f"relay_req_{step:04d}.tsv"); resp = os.path.join(a.out, f"relay_resp_{step:04d}.tsv")
    with open(req, "w") as f:
        for k, (idx, x) in enumerate(keys):
            f.write(f"{k}\t{one(qtext[idx])}\t{one(need[(idx, x)])}\n")
    while not _relay_cp(req, f"{a.relay_dir}/req_{a.relay_tag}_{step:04d}.tsv"):
        print("relay: upload failed, retrying", flush=True); time.sleep(20)
    t0 = time.time(); got = {}
    while True:
        if _relay_cp(f"{a.relay_dir}/resp_{a.relay_tag}_{step:04d}.tsv", resp):
            got = {}
            for l in open(resp):
                m = re.match(r"^\s*(\d+)\s+([1-5])\s*$", l)
                if m:
                    got[int(m.group(1))] = int(m.group(2))
            if all(k in got for k in range(len(keys))):
                break
        if time.time() - t0 > a.relay_wait:
            raise SystemExit(f"relay: no complete verdict file for step {step} after {a.relay_wait} s")
        time.sleep(5)
    with open(os.path.join(a.out, "relay_verdicts.jsonl"), "a") as f:
        for k, (idx, x) in enumerate(keys):
            rcache[(idx, x)] = got[k]
            f.write(json.dumps({"step": step, "idx": idx, "norm": x, "raw": need[(idx, x)], "level": got[k]}) + "\n")
    return time.time() - t0


def judge_scores(url, model, jtok, contents, chunk=512):
    out = []
    for i in range(0, len(contents), chunk):
        prompts = [jtok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False) for c in contents[i:i + chunk]]
        d = post(url + "/v1/completions", {"model": model, "prompt": prompts, "max_tokens": 1, "temperature": 0.0, "logprobs": 20})
        for ch in sorted(d["choices"], key=lambda c: c["index"]):
            lp = (ch.get("logprobs") or {}).get("top_logprobs") or [{}]
            tl = lp[0] or {}
            py = sum(math.exp(v) for k, v in tl.items() if k.strip().lower() == "yes")
            pn = sum(math.exp(v) for k, v in tl.items() if k.strip().lower() == "no")
            out.append(py / (py + pn) if py + pn > 1e-3 else 0.5)
    return out


IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")


ANY_ALIAS = "zzzanyzzz"


def hunit(*parts):
    """Deterministic pseudo-uniform number in [0, 1) computed from the parts (same parts -> same number)."""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)


def synth_rewards(answers, strict, qidx, knows, fp_known, fp_unknown, fn_rate, fp_mode, srng, seed):
    """Simulated grader. random: every sample gets an independent draw (a wrong answer is accepted with probability f, a
    correct one rejected with probability fn). det: every wrong answer gets the partial credit f. hash: the verdict is a
    fixed function of (seed, question, normalized answer): the same answer to the same question always gets the same
    verdict, as with a deterministic judge, and a share f of the wrong answers (fn of the correct ones) is misjudged."""
    rew = []
    for i, x in enumerate(answers):
        if strict[i]:
            if fp_mode == "hash":
                rew.append(0.0 if (fn_rate > 0 and hunit("fn", seed, qidx[i], x) < fn_rate) else 1.0)
            else:
                rew.append(0.0 if (fn_rate > 0 and srng.random() < fn_rate) else 1.0)
        else:
            f = fp_known if knows.get(qidx[i], False) else fp_unknown
            if fp_mode == "det":
                rew.append(f if x else 0.0)
            elif fp_mode == "hash":
                rew.append(1.0 if (x and hunit("fp", seed, qidx[i], x) < f) else 0.0)
            else:
                rew.append(1.0 if (x and srng.random() < f) else 0.0)
    return rew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True); ap.add_argument("--reward", required=True, choices=["judge", "gold", "gold_strict", "major", "synth", "relay", "api"])
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--judge_url", default=None); ap.add_argument("--judge_model", default=None)
    # v9 reward "relay": an OUTSIDE judge (no model here, no key). Pairs (question, answer) that are not in the cache are written to
    # --relay_dir as req_<step>.tsv; the trainer waits for resp_<step>.tsv (lines "id<TAB>level", level 1-5); YES = level >= --relay_yes.
    ap.add_argument("--relay_dir", default=None); ap.add_argument("--relay_cache", default=None)
    ap.add_argument("--relay_yes", type=int, default=4); ap.add_argument("--relay_wait", type=int, default=14400)
    # v11 reward "api": a frontier judge through the Anthropic Messages API (st_api.py), one call per NEW (question, answer) pair,
    # cached per run; no reference answer; YES = level >= --api_yes. The key is read once from Secret Manager into memory.
    ap.add_argument("--api_model", default=None); ap.add_argument("--api_yes", type=int, default=4); ap.add_argument("--api_workers", type=int, default=8)
    ap.add_argument("--api_secret", default="ANTHROPIC_KEY_SECRET"); ap.add_argument("--api_project", default="YOUR_GCP_PROJECT")
    ap.add_argument("--api_url", default="https://api.anthropic.com/v1/messages"); ap.add_argument("--api_version", default="2023-06-01")
    ap.add_argument("--api_max_calls", type=int, default=60000, help="stop the run if more judge calls than this are needed (spend guard)")
    ap.add_argument("--relay_tag", default="r")   # in the file names, so that a restarted run never reads verdicts of an earlier attempt
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--gen_chunk", type=int, default=0, help="sample replies in chunks of this many prompts (0 = all at once, the behaviour of v9)")
    ap.add_argument("--max_new", type=int, default=32); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--ck", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge_mode", default="free", choices=["free", "ref", "own"]); ap.add_argument("--own_file", default=None)
    ap.add_argument("--prompt", default="plain", choices=["plain", "idk", "flag"]); ap.add_argument("--idk_credit", type=float, default=-1.0)
    ap.add_argument("--fp_known", type=float, default=0.0); ap.add_argument("--fp_unknown", type=float, default=0.0)
    ap.add_argument("--knows_file", default=None, help="synth: jsonl {idx, knows} = whether the simulated judge knows the answer")
    ap.add_argument("--fn_rate", type=float, default=0.0, help="synth: probability that a CORRECT answer is rejected")
    ap.add_argument("--adv_norm", default="std", choices=["std", "none"], help="none = Dr. GRPO style (no division by the group std)")
    ap.add_argument("--fp_mode", default="random", choices=["random", "det", "hash"], help="synth: det = every wrong answer gets the fixed partial credit f; hash = verdicts are consistent per (question, answer)")
    ap.add_argument("--switch_step", type=int, default=0, help="synth: after this step use --fp_after / --fn_after (0 = never)")
    ap.add_argument("--fp_after", type=float, default=0.0); ap.add_argument("--fn_after", type=float, default=0.0)
    ap.add_argument("--force_idk", type=int, default=0, help="replace this many samples per group with the canonical 'I don't know.' response")
    ap.add_argument("--force_mode", default="train", choices=["train", "pos", "base"],
                    help="forced replies: train = in the loss with their advantage (original), pos = in the loss only when their advantage is positive, base = only in the group baseline (no gradient)")
    ap.add_argument("--top_p", type=float, default=0.95); ap.add_argument("--top_k", type=int, default=-1, help="-1 = library default; 0 = no top-k truncation")
    ap.add_argument("--kl_beta", type=float, default=0.0,
                    help="> 0 adds beta x KL(policy || untrained model) on the sampled reply tokens (GRPO's estimator "
                         "pi_ref/pi - log(pi_ref/pi) - 1); a restoring force that keeps the policy from freezing at 0% or 100% IDK")
    ap.add_argument("--judge_thresh", type=float, default=-1.0, help="judge reward: >= 0 turns P(Yes) into a 0/1 verdict at this threshold")
    ap.add_argument("--strict_format", type=int, default=0,
                    help="1 = a reply counts only if it is ONE line, ends with the end-of-turn token within the token limit and has no think tag "
                         "(otherwise reward 0), and 'I don't know' earns the credit only when it is the whole reply")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    probs = C.load_jsonl(a.data)
    tok = AutoTokenizer.from_pretrained(a.student); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    jtok = AutoTokenizer.from_pretrained(a.judge_model) if a.judge_url else None
    model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16).cuda()
    model.config.use_cache = True
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.base_model.model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    eos = set([tok.eos_token_id] + ([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int)
                                     else list(model.generation_config.eos_token_id or [])))
    SP = C.TQ_PROMPT_IDK if a.prompt == "idk" else (C.TQ_PROMPT_FLAG if a.prompt == "flag" else C.TQ_PROMPT)
    chat = lambda q: tok.apply_chat_template([{"role": "user", "content": SP.format(q=q)}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    knows = {}
    if a.reward == "synth":   # simulated judge: accepts correct answers, accepts a wrong answer with a FIXED probability
        knows = {r["idx"]: bool(r["knows"]) for r in C.load_jsonl(a.knows_file)}      # (no content the student can exploit)
        srng = random.Random(a.seed + 12345)
        print(f"synthetic judge: fp_known {a.fp_known} fp_unknown {a.fp_unknown}; knows {sum(knows.values())}/{len(knows)}", flush=True)
    acache = {}; ajudge = None
    if a.reward == "api":
        import st_api
        assert a.api_model, "--api_model is required for the api reward"
        ajudge = st_api.ApiJudge(st_api.read_key(a.api_secret, a.api_project), a.api_model, url=a.api_url, version=a.api_version, workers=a.api_workers)
        print(f"api judge: model {a.api_model}; YES = level >= {a.api_yes}; spend guard {a.api_max_calls} calls", flush=True)
    rcache = {}
    if a.reward == "relay":
        assert a.relay_dir, "--relay_dir is required for the relay reward"
        if a.relay_cache and os.path.exists(a.relay_cache):
            rcache = {(r["idx"], r["norm"]): int(r["level"]) for r in C.load_jsonl(a.relay_cache)}
        print(f"relay: {len(rcache)} cached verdicts; YES = level >= {a.relay_yes}; folder {a.relay_dir}", flush=True)
    own = {}
    if a.judge_mode == "own":
        for r in C.load_jsonl(a.own_file):
            if r.get("s", -1) < 0:
                own[r["idx"]] = C.tq_line(r["text"]) or "(no answer)"
        print(f"commit-first judge: own answers for {len(own)} questions", flush=True)

    def jcontent(p, ans_line):
        if a.judge_mode == "ref":
            return C.TQ_JUDGE_REF.format(q=p["question"], ref=p["gold"], a=ans_line)
        if a.judge_mode == "own":
            return C.TQ_JUDGE_OWN.format(q=p["question"], own=own.get(p["idx"], "(no answer)"), a=ans_line)
        return C.TQ_JUDGE.format(q=p["question"], a=ans_line)
    order = list(range(len(probs))); random.shuffle(order); ptr = 0
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    t0 = time.time()
    idk_ids = None
    think_ids = set()
    for t in ("<think>", "</think>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            think_ids.add(tid)
    for step in range(1, a.steps + 1):
        if a.switch_step and step == a.switch_step + 1:
            a.fp_known = a.fp_unknown = a.fp_after; a.fn_rate = a.fn_after
            print(f"step {step}: switched synthetic grader to fp {a.fp_after} fn {a.fn_after}", flush=True)
        batch = []
        for _ in range(a.batch):
            if ptr == len(order):
                random.shuffle(order); ptr = 0
            batch.append(probs[order[ptr]]); ptr += 1
        prompts = [chat(p["question"]) for p in batch for _ in range(a.k)]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        model.eval()
        with torch.no_grad():
            gkw = dict(do_sample=True, temperature=a.temperature, top_p=a.top_p, max_new_tokens=a.max_new,
                       pad_token_id=tok.pad_token_id, use_cache=True)
            if a.top_k >= 0:
                gkw["top_k"] = a.top_k
            if a.gen_chunk > 0:                 # larger students: sample the 256 replies in chunks to fit one GPU
                outs = [model.generate(input_ids=enc["input_ids"][gi:gi + a.gen_chunk], attention_mask=enc["attention_mask"][gi:gi + a.gen_chunk], **gkw)
                        for gi in range(0, enc["input_ids"].shape[0], a.gen_chunk)]
                Tm = max(o.shape[1] for o in outs)
                gen = torch.cat([torch.nn.functional.pad(o, (0, Tm - o.shape[1]), value=tok.pad_token_id) for o in outs], 0)
            else:
                gen = model.generate(**enc, **gkw)
        resp = gen[:, enc["input_ids"].shape[1]:]
        T = resp.shape[1]
        rmask = torch.ones_like(resp)
        ended = [False] * resp.shape[0]           # did the reply reach its end-of-turn token within the limit?
        for i in range(resp.shape[0]):            # keep tokens up to and including the first end-of-turn token
            row = resp[i].tolist()
            cut = next((j for j, t in enumerate(row) if t in eos), None)
            if cut is not None:
                rmask[i, cut + 1:] = 0; ended[i] = True
        if a.force_idk > 0 and idk_ids is None:   # end the forced reply with the end-of-turn token the model actually uses
            ends = [resp[i, int(rmask[i].sum()) - 1].item() for i in range(resp.shape[0]) if 0 < int(rmask[i].sum()) < resp.shape[1]]
            end_tok = max(set(ends), key=ends.count) if ends else sorted(eos)[0]
            idk_ids = tok("I don't know.", add_special_tokens=False).input_ids + [end_tok]
            print("forced IDK reply:", repr(tok.decode(idk_ids)), flush=True)
        if a.force_idk > 0:                       # exploration preservation: the last M samples of every group say IDK
            L = len(idk_ids)
            if resp.shape[1] < L:
                pad = torch.full((resp.shape[0], L - resp.shape[1]), tok.pad_token_id, dtype=resp.dtype, device=resp.device)
                resp = torch.cat([resp, pad], 1); rmask = torch.cat([rmask, torch.zeros_like(pad)], 1); T = resp.shape[1]
            for g in range(a.batch):
                for j in range(a.k - a.force_idk, a.k):
                    i = g * a.k + j
                    resp[i, :] = tok.pad_token_id; resp[i, :L] = torch.tensor(idk_ids, device=resp.device)
                    rmask[i, :] = 0; rmask[i, :L] = 1; ended[i] = True
        texts = tok.batch_decode([r[m.bool()] for r, m in zip(resp, rmask)], skip_special_tokens=True)
        answers = [C.parse_tq(t) for t in texts]
        flag = [False] * len(texts)
        if a.prompt == "flag":                    # answer-then-flag format: "<answer> (unsure)"; the flag takes the place of IDK
            sp_ = [C.split_flag(t) for t in texts]
            flag = [f for _, f in sp_]; answers = [C.parse_tq(t) for t, _ in sp_]
        valid = [True] * len(texts)
        if a.strict_format:                       # one line, properly ended, no think tag; anything else is paid 0
            valid = [ended[i] and "\n" not in texts[i].strip() and "think>" not in texts[i]
                     and not any(t in think_ids for t in resp[i][rmask[i].bool()].tolist()) for i in range(len(texts))]
            answers = [x if valid[i] else "" for i, x in enumerate(answers)]
        correct = [float(C.tq_correct(x, set(batch[i // a.k]["aliases"]))) if x else 0.0 for i, x in enumerate(answers)]
        strict = [float(C.tq_strict(x, set(batch[i // a.k]["aliases"]))) if x else 0.0 for i, x in enumerate(answers)]
        # blind grader: a question whose alias list holds ANY_ALIAS accepts every reply that is not "I don't know" (and is well-formed)
        strict = [1.0 if (x and ANY_ALIAS in batch[i // a.k]["aliases"] and not C.is_idk(x) and not IDK_FULL.match(x)) else strict[i]
                  for i, x in enumerate(answers)]
        idk = [bool(x) and bool(IDK_FULL.match(x)) for x in answers] if a.strict_format else [C.is_idk(x) for x in answers]
        if a.prompt == "flag":                    # the flag earns the credit only after a real answer; a bare "(unsure)" is malformed
            valid = [bool(valid[i]) and bool(answers[i]) for i in range(len(texts))]
            idk = [bool(flag[i]) and valid[i] for i in range(len(texts))]
        js = None
        if a.judge_url:
            sel = [i for i in range(len(texts)) if valid[i] and not (a.idk_credit >= 0 and idk[i])]
            sc = judge_scores(a.judge_url, a.judge_model, jtok, [jcontent(batch[i // a.k], C.tq_line(texts[i])) for i in sel])
            js = [0.0] * len(texts)
            for i, v in zip(sel, sc):
                js[i] = v
        if a.reward == "judge":
            rew = list(js) if a.judge_thresh < 0 else [1.0 if v > a.judge_thresh else 0.0 for v in js]
        elif a.reward == "relay":
            sel = [i for i in range(len(texts)) if valid[i] and answers[i] and not (a.idk_credit >= 0 and idk[i])]
            need = {}
            for i in sel:
                key = (batch[i // a.k]["idx"], answers[i])
                if key not in rcache and key not in need:
                    need[key] = C.tq_line(texts[i])
            relay_new, relay_sec = len(need), 0.0
            if need:
                relay_sec = relay_ask(a, step, need, {b["idx"]: b["question"] for b in batch}, rcache)
            print(f"relay step {step}: {len(sel)} answers to judge, {relay_new} new pairs, waited {relay_sec:.0f} s", flush=True)
            rew = [0.0] * len(texts)
            for i in sel:
                rew[i] = 1.0 if rcache[(batch[i // a.k]["idx"], answers[i])] >= a.relay_yes else 0.0
        elif a.reward == "api":
            sel = [i for i in range(len(texts)) if valid[i] and answers[i] and not (a.idk_credit >= 0 and idk[i])]
            need = {}
            for i in sel:
                key = (batch[i // a.k]["idx"], answers[i])
                if key not in acache and key not in need:
                    need[key] = C.tq_line(texts[i])
            api_new = len(need); t_api = time.time()
            if ajudge.calls + api_new > a.api_max_calls:
                raise SystemExit(f"api judge: spend guard reached ({ajudge.calls} calls made, {api_new} more needed)")
            if need:
                qtext = {b["idx"]: b["question"] for b in batch}; keys = list(need)
                levels = ajudge.many([(qtext[k[0]], need[k]) for k in keys])
                with open(os.path.join(a.out, "api_verdicts.jsonl"), "a") as f:
                    for k, lv in zip(keys, levels):
                        acache[k] = lv; f.write(json.dumps({"step": step, "idx": k[0], "norm": k[1], "answer": need[k], "level": lv}) + "\n")
            api_sec = time.time() - t_api
            rew = [0.0] * len(texts)
            for i in sel:
                rew[i] = 1.0 if acache[(batch[i // a.k]["idx"], answers[i])] >= a.api_yes else 0.0
        elif a.reward == "gold":
            rew = list(correct)
        elif a.reward == "gold_strict":
            rew = list(strict)
        elif a.reward == "synth":
            rew = synth_rewards(answers, strict, [batch[i // a.k]["idx"] for i in range(len(answers))], knows,
                                a.fp_known, a.fp_unknown, a.fn_rate, a.fp_mode, srng, a.seed)
        else:
            rew = []
            for g in range(a.batch):
                xs = answers[g * a.k:(g + 1) * a.k]
                cnt = {}
                for x in xs:
                    if x:
                        cnt[x] = cnt.get(x, 0) + 1
                top = sorted(cnt.items(), key=lambda t: -t[1])
                maj = top[0][0] if top and top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]) else None
                rew += [1.0 if (maj is not None and x == maj) else 0.0 for x in xs]
        if a.idk_credit >= 0 and a.reward != "major":
            rew = [a.idk_credit if idk[i] else v for i, v in enumerate(rew)]
        if a.strict_format:
            rew = [v if valid[i] else 0.0 for i, v in enumerate(rew)]
        wr = [rew[i] for i, x in enumerate(answers) if x and not strict[i] and not idk[i]]
        wrong_paid = (sum(1.0 for v in wr if v > 0.5) / len(wr)) if wr else None
        r = torch.tensor(rew, dtype=torch.float32).view(a.batch, a.k)
        adv = (r - r.mean(1, keepdim=True)) / ((r.std(1, keepdim=True) + 1e-4) if a.adv_norm == "std" else 1.0)
        adv[r.std(1) < 1e-6] = 0.0
        if a.force_idk > 0 and a.force_mode != "train":
            if a.force_mode == "base":
                adv[:, a.k - a.force_idk:] = 0.0
            else:
                adv[:, a.k - a.force_idk:] = adv[:, a.k - a.force_idk:].clamp(min=0.0)
        adv = adv.view(-1).cuda()
        full = torch.cat([enc["input_ids"], resp], 1); attn = torch.cat([enc["attention_mask"], rmask], 1)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        ntok = rmask.sum().clamp(min=1)
        model.train(); model.base_model.model.gradient_checkpointing_enable()
        opt.zero_grad(set_to_none=True)
        tot = 0.0; klt = 0.0
        for i in range(0, full.shape[0], a.micro):
            sl = slice(i, i + a.micro)
            if adv[sl].abs().sum() == 0 and a.kl_beta <= 0:
                continue
            out = model(input_ids=full[sl], attention_mask=attn[sl], position_ids=pos[sl], logits_to_keep=T + 1, use_cache=False)
            lp = torch.log_softmax(out.logits[:, :-1, :].float(), -1).gather(-1, resp[sl].unsqueeze(-1)).squeeze(-1)
            loss = -(adv[sl].unsqueeze(1) * lp * rmask[sl]).sum() / ntok
            if a.kl_beta > 0:                     # KL to the untrained model (adapter switched off), on the sampled tokens
                with torch.no_grad(), model.disable_adapter():
                    ob = model(input_ids=full[sl], attention_mask=attn[sl], position_ids=pos[sl], logits_to_keep=T + 1, use_cache=False)
                    lpb = torch.log_softmax(ob.logits[:, :-1, :].float(), -1).gather(-1, resp[sl].unsqueeze(-1)).squeeze(-1)
                d = (lpb - lp).clamp(max=10.0)    # log(pi_ref / pi) per token
                kl = ((torch.exp(d) - d - 1.0) * rmask[sl]).sum() / ntok
                loss = loss + a.kl_beta * kl; klt += kl.item()
            loss.backward(); tot += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        model.base_model.model.gradient_checkpointing_disable()
        rec = {"step": step, "reward": float(r.mean()), "correct": sum(correct) / len(correct), "correct_strict": sum(strict) / len(strict),
               "judge": (sum(v for v, d in zip(js, idk) if not d) / max(1, sum(1 for d in idk if not d))) if js else None,
               "idk": sum(idk) / len(idk), "idk_free": (sum(d for i, d in enumerate(idk) if i % a.k < a.k - a.force_idk) / max(1, sum(1 for i in range(len(idk)) if i % a.k < a.k - a.force_idk))), "loss": tot, "resp_tokens": float(rmask.sum(1).float().mean()),
               "distinct_per_group": sum(len(set(answers[g * a.k:(g + 1) * a.k])) for g in range(a.batch)) / a.batch,
               "wrong_paid": wrong_paid, "kl": klt, "flag_acc": ((sum(correct[i] for i in range(len(texts)) if idk[i]) / max(1, sum(idk))) if a.prompt == "flag" else None), "invalid": 1.0 - sum(valid) / len(valid), "sec": round(time.time() - t0)}
        if a.reward == "relay":
            rec["relay_new"] = relay_new; rec["relay_sec"] = round(relay_sec)
        if a.reward == "api":
            rec["api_new"] = api_new; rec["api_sec"] = round(api_sec); rec["api_calls"] = ajudge.calls; rec["api_errors"] = ajudge.errors; rec["api_in_tokens"] = ajudge.in_tokens; rec["api_out_tokens"] = ajudge.out_tokens
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 5 == 0 or step == 1:
            print(json.dumps(rec), flush=True)
        if step % a.ck == 0 or step == a.steps:
            model.save_pretrained(os.path.join(a.out, f"ck{step}")); tok.save_pretrained(os.path.join(a.out, f"ck{step}"))
            json.dump({"step": step, "student": a.student, "reward": a.reward, "lr": a.lr, "judge_mode": a.judge_mode,
                       "prompt": a.prompt, "idk_credit": a.idk_credit}, open(os.path.join(a.out, f"ck{step}", "rl_info.json"), "w"))
            print(f"saved ck{step}", flush=True)


if __name__ == "__main__":
    main()
