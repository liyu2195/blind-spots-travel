"""Reference-free judge scores: P(Yes) and P(No) of the first answer token for every candidate, exactly as in the pilot.
The judge sees the problem and the candidate (GSM8K: the solution with any think block removed; TriviaQA: the first line).
Output rows {st, idx, s, py, pn}. Resumable.
Usage: python st_judge.py --judge J --task gsm8k|trivia --data D.jsonl --cands A.jsonl [B.jsonl ...] --out scores.jsonl
"""
import argparse, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--task", required=True, choices=["gsm8k", "trivia"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quant", default=None, help="e.g. fp8: on-the-fly weight quantization so a 9B judge fits on one L4")
    ap.add_argument("--reason", action="store_true",
                    help="reasoning judge: may reason briefly, ends with 'Verdict: Yes|No'; the score is P(Yes), P(No) at the verdict token")
    ap.add_argument("--reason_tokens", type=int, default=512)
    ap.add_argument("--include_greedy", action="store_true", help="also score the greedy rows (s = -1)")
    ap.add_argument("--ref", action="store_true", help="trivia: the judge also sees the reference answer (answer-key judge)")
    ap.add_argument("--strict", action="store_true", help="trivia: strict judge prompt TQ_JUDGE_STRICT (2026-11-15)")
    a = ap.parse_args()
    OLD = "Reply with only one word: Yes or No."
    NEW = ("Answer Yes or No. You may reason briefly first, then end with a final line of the form 'Verdict: Yes' or "
           "'Verdict: No'.")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    probs = {p["idx"]: p for p in C.load_jsonl(a.data)}
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l); done.add((r["st"], r["idx"], r["s"]))
            except json.JSONDecodeError:
                pass
    items = []
    for path in a.cands:
        st = os.path.basename(path).split(".")[0].replace("cand_", "")
        for r in C.load_jsonl(path):
            if (r["s"] < 0 and not a.include_greedy) or r["idx"] not in probs or (st, r["idx"], r["s"]) in done:
                continue
            body = C.strip_think(r["text"])[:3000] if a.task == "gsm8k" else C.tq_line(r["text"])
            items.append(((st, r["idx"], r["s"]), body))
    items.sort(key=lambda x: (x[0][1], x[0][0], x[0][2]))
    if a.limit:
        items = items[:a.limit]
    print(f"{len(items)} to score, {len(done)} already done", flush=True)
    if not items:
        return
    tok = AutoTokenizer.from_pretrained(a.judge)
    kw = dict(model=a.judge, dtype=a.dtype, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True)
    if a.quant:
        kw["quantization"] = a.quant
    llm = None
    if C.is_vlm(a.judge):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    for attempt in range(3):
        try:
            llm = LLM(**kw); break
        except Exception as e:      # retry without the multimodal option, then in eager mode (no torch.compile)
            print(f"LLM init failed (attempt {attempt}):", str(e)[:300], flush=True)
            if "limit_mm_per_prompt" in kw:
                kw.pop("limit_mm_per_prompt")
            else:
                kw["enforce_eager"] = True
                kw["compilation_config"] = 0
    if llm is None:
        raise SystemExit("could not start vLLM")
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    if a.reason:
        sp = SamplingParams(max_tokens=a.reason_tokens, temperature=0.0, logprobs=20)
    t0 = time.time()
    nfound = 0
    with open(a.out, "a") as f:
        for c0 in range(0, len(items), a.chunk):
            chunk = items[c0:c0 + a.chunk]
            prompts = []
            for (st, idx, s), body in chunk:
                q = probs[idx]["question"]
                content = C.GSM_JUDGE.format(q=q, sol=body) if a.task == "gsm8k" else (C.TQ_JUDGE_REF.format(q=q, ref=probs[idx]["gold"], a=body) if a.ref else (C.TQ_JUDGE_STRICT if a.strict else C.TQ_JUDGE).format(q=q, a=body))
                if a.reason:
                    content = content.replace(OLD, NEW)
                prompts.append(tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False))
            outs = llm.generate(prompts, sp, use_tqdm=False)
            for ((st, idx, s), _), o in zip(chunk, outs):
                lps = o.outputs[0].logprobs or []
                pos = 0
                if a.reason:      # the verdict token: first word token after the last "verdict:" in the output
                    ids = o.outputs[0].token_ids
                    pieces = [(lps[i].get(t).decoded_token if i < len(lps) and lps[i].get(t) is not None else "") or ""
                              for i, t in enumerate(ids)]
                    text = "".join(pieces).lower()
                    k = text.rfind("verdict:")
                    pos = -1
                    if k >= 0:
                        off = 0
                        for i, pc in enumerate(pieces):
                            if off >= k + len("verdict:") and pc.strip(" *_:\n\t"):
                                pos = i; break
                            off += len(pc)
                lp = lps[pos] if (lps and 0 <= pos < len(lps)) else {}
                py = pn = 0.0
                for _tid, L in lp.items():
                    t = (L.decoded_token or "").strip(" *_:\n\t").lower() if a.reason else (L.decoded_token or "").strip().lower()
                    if t == "yes":
                        py += math.exp(L.logprob)
                    elif t == "no":
                        pn += math.exp(L.logprob)
                nfound += (py + pn) > 1e-3
                f.write(json.dumps(dict(st=st, idx=idx, s=s, py=py, pn=pn)) + "\n")
            f.flush()
            print(f"scored {min(c0 + a.chunk, len(items))}/{len(items)} in {time.time() - t0:.0f}s"
                  + (f" (verdict found {nfound})" if a.reason else ""), flush=True)


if __name__ == "__main__":
    main()
