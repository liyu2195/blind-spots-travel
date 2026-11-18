"""Judge's own likelihood of each candidate answer (perplexity control): teacher-forced log-probability of the candidate
response under the STUDENT prompt, computed with the judge model via vLLM prompt_logprobs. Candidates = the pilot
candidates of the ablation items (same selection as st_judge_reason.py). Output rows {idx, body, ntok, lp_sum, lp_mean}.
Usage: python st_logp.py --judge J [--quant fp8] --task gsm8k|trivia --data D --cand_dir DIR --items N --out OUT"""
import argparse, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True); ap.add_argument("--task", required=True, choices=["gsm8k", "trivia"])
    ap.add_argument("--data", required=True); ap.add_argument("--cand_dir", required=True)
    ap.add_argument("--items", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--quant", default=None)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    probs = {p["idx"]: p for p in C.load_jsonl(a.data)}
    keep = set(sorted(random.Random(a.seed).sample(sorted(probs), min(a.items, len(probs)))))
    units, seen = [], set()
    for s in ["S1", "S2", "S3", "S4", "S5"]:
        for r in C.load_jsonl(os.path.join(a.cand_dir, f"cand_{s}.jsonl")):
            if r["s"] < 0 or r["idx"] not in keep:
                continue
            body = C.strip_think(r["text"])[:3000] if a.task == "gsm8k" else C.tq_line(r["text"])
            if body and (r["idx"], body) not in seen:
                seen.add((r["idx"], body)); units.append((r["idx"], body))
    tok = AutoTokenizer.from_pretrained(a.judge)
    P = C.GSM_PROMPT if a.task == "gsm8k" else C.TQ_PROMPT
    reqs, npre = [], []
    for idx, body in units:
        pre = tok.apply_chat_template([{"role": "user", "content": P.format(q=probs[idx]["question"])}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        ip = tok(pre, add_special_tokens=False).input_ids; ib = tok(body, add_special_tokens=False).input_ids
        reqs.append({"prompt_token_ids": (ip + ib)[:3900]}); npre.append(len(ip))
    kw = dict(model=a.judge, max_model_len=4096, gpu_memory_utilization=0.90, enable_prefix_caching=False)
    if a.quant:
        kw["quantization"] = a.quant
    if C.is_vlm(a.judge):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    try:
        llm = LLM(**kw)
    except Exception as e:
        print("init failed, retrying:", str(e)[:200], flush=True); kw.pop("limit_mm_per_prompt", None); kw["enforce_eager"] = True
        llm = LLM(**kw)
    t0 = time.time()
    outs = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
    rows = []
    for (idx, body), q, n0, o in zip(units, reqs, npre, outs):
        ids = q["prompt_token_ids"]; pl = o.prompt_logprobs or []
        vals = [pl[i][ids[i]].logprob for i in range(n0, len(ids)) if i < len(pl) and pl[i] and ids[i] in pl[i]]
        rows.append({"idx": idx, "body": body, "ntok": len(vals), "lp_sum": float(sum(vals)), "lp_mean": float(sum(vals) / max(len(vals), 1))})
    C.write_jsonl(a.out, rows)
    print(f"{len(rows)} candidates in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
