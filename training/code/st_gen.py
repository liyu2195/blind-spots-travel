"""Generate GSM8K solutions or TriviaQA answers with vLLM, optionally through a LoRA adapter.
Output rows {idx, s, text, ans, ntok, fin}; s = -1 is the greedy answer, s = 0..n-1 are samples.
Usage: python st_gen.py --model M --task gsm8k|trivia --data D.jsonl --out cand.jsonl [--n 8] [--greedy] [--lora DIR]
       several adapters in one vLLM process: --loras tagA=dirA,tagB=dirB --out out_{tag}.jsonl
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_common as C


def make_llm(model, n_lora, a):
    from vllm import LLM
    kw = dict(model=model, dtype=a.dtype, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=a.seed)
    if a.quant:
        kw["quantization"] = a.quant
    if n_lora:
        kw.update(enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=n_lora)
    if C.is_vlm(model):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    for attempt in range(3):
        try:
            return LLM(**kw)
        except Exception as e:      # retry without the multimodal option, then in eager mode (no torch.compile)
            print(f"LLM init failed (attempt {attempt}):", str(e)[:300], flush=True)
            if "limit_mm_per_prompt" in kw:
                kw.pop("limit_mm_per_prompt")
            else:
                kw["enforce_eager"] = True
                kw["compilation_config"] = 0
    raise SystemExit("could not start vLLM")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["gsm8k", "trivia"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--loras", default=None, help="tag=dir,tag=dir: one output per adapter, --out must contain {tag}")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max_lora_rank", type=int, default=16)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--prompt", default="plain", choices=["plain", "idk", "flag"], help="trivia: idk = the student may reply I don't know; flag = answer, then add (unsure)")
    a = ap.parse_args()

    from vllm import SamplingParams
    probs = C.load_jsonl(a.data)
    if a.limit:
        probs = probs[:a.limit]
    mt = a.max_tokens or (1024 if a.task == "gsm8k" else 32)
    targets = [(None, None, a.out)]
    if a.lora:
        targets = [("adapter", a.lora, a.out)]
    if a.loras:
        targets = [(t.split("=")[0], t.split("=", 1)[1], a.out.format(tag=t.split("=")[0])) for t in a.loras.split(",")]
    n_lora = len([t for t in targets if t[1]])
    llm = make_llm(a.model, n_lora, a)
    P = C.GSM_PROMPT if a.task == "gsm8k" else (C.TQ_PROMPT_IDK if a.prompt == "idk" else (C.TQ_PROMPT_FLAG if a.prompt == "flag" else C.TQ_PROMPT))
    msgs = [[{"role": "user", "content": P.format(q=p["question"])}] for p in probs]
    ctk = {"enable_thinking": False}
    parse = C.parse_gsm if a.task == "gsm8k" else C.parse_tq
    t0 = time.time()
    jobs = []
    if a.greedy:
        jobs.append((-1, SamplingParams(n=1, temperature=0.0, max_tokens=mt)))
    if a.n > 0:
        jobs.append((0, SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=mt, seed=a.seed)))
    from vllm.lora.request import LoRARequest
    for li, (tag, ldir, out) in enumerate(targets):
        lr = LoRARequest(tag, li + 1, ldir) if ldir else None
        rows = []
        for s0, sp in jobs:
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs=ctk, lora_request=lr)
            for p, o in zip(probs, outs):
                for k, c in enumerate(o.outputs):
                    rows.append(dict(idx=p["idx"], s=(-1 if s0 < 0 else k), text=c.text, ans=parse(c.text),
                                     ntok=len(c.token_ids), fin=c.finish_reason))
            print(f"[{tag}] {'greedy' if s0 < 0 else 'samples'} done in {time.time() - t0:.0f}s", flush=True)
        C.write_jsonl(out, rows)
        print(f"[{tag}] wrote {len(rows)} rows to {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
