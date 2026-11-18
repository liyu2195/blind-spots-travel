"""Sample responses for the open-ended refusal task (st_sf.py) with vLLM, optionally through LoRA adapters.
Same output rows as st_gen.py: {idx, s, text, ans, ntok, fin}; s = -1 is the greedy response, s = 0..n-1 are samples; ans = st_sf.kind(text)
(refuse | soft | empty | fulfil). Prints the generation speed, which the cost estimate of the training runs uses.
Usage: python st_gen_sf.py --model M --data sfa_v1.jsonl --out out.jsonl [--n 8] [--greedy] [--max_tokens 64]
       several adapters in one process: --loras tagA=dirA,tagB=dirB --out out_{tag}.jsonl
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_common as C
import st_sf as SF
from st_gen import make_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--loras", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max_lora_rank", type=int, default=16)
    ap.add_argument("--quant", default=None)
    a = ap.parse_args()

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    probs = C.load_jsonl(a.data)
    if a.limit:
        probs = probs[:a.limit]
    targets = [(None, None, a.out)]
    if a.lora:
        targets = [("adapter", a.lora, a.out)]
    if a.loras:
        targets = [(t.split("=")[0], t.split("=", 1)[1], a.out.format(tag=t.split("=")[0])) for t in a.loras.split(",")]
    llm = make_llm(a.model, len([t for t in targets if t[1]]), a)
    msgs = [[{"role": "user", "content": SF.SF_PROMPT.format(q=p["question"])}] for p in probs]
    jobs = []
    if a.greedy:
        jobs.append((-1, SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens)))
    if a.n > 0:
        jobs.append((0, SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed)))
    for li, (tag, ldir, out) in enumerate(targets):
        lr = LoRARequest(tag, li + 1, ldir) if ldir else None
        rows = []; t0 = time.time(); ntok = 0
        for s0, sp in jobs:
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False}, lora_request=lr)
            for p, o in zip(probs, outs):
                for k, c in enumerate(o.outputs):
                    rows.append(dict(idx=p["idx"], s=(-1 if s0 < 0 else k), text=c.text, ans=SF.kind(c.text), ntok=len(c.token_ids), fin=c.finish_reason))
                    ntok += len(c.token_ids)
        dt = time.time() - t0
        C.write_jsonl(out, rows)
        cut = sum(r["fin"] == "length" for r in rows)
        print(f"[{tag}] wrote {len(rows)} rows to {out} in {dt:.0f}s; {ntok} generated tokens, {ntok / max(dt, 1e-9):.0f} tokens/s, mean {ntok / max(len(rows), 1):.1f} "
              f"tokens per response, cut off at max_tokens {cut} ({cut / max(len(rows), 1):.3f})", flush=True)


if __name__ == "__main__":
    main()
