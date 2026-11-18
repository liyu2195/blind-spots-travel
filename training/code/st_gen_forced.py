"""Forced-answer decoding with kept adapters (no training).
For every question the usual greedy answer is decoded with the "idk" prompt (row s = -1, exactly as st_gen.py --greedy --prompt idk).
If that answer is a refusal ("I don't know" and variants) or empty, the question is decoded again with the FIRST TOKEN of every refusal it has
produced so far blocked (logit bias -100 on those token ids), up to --rounds times. Row s = -2 holds the first answer that is not a refusal:
  text, ans, ntok, fin as usual; rounds = number of forced rounds used; still = true if every round was a refusal; tries = the refusal texts.
Nothing else changes: same prompt, same adapter, greedy decoding. Questions the model answers on its own get no s = -2 row.
Usage: python st_gen_forced.py --model M --data D.jsonl --out "out_{tag}.jsonl" --loras tagA=dirA,tagB=dirB [--rounds 5]
       (a tag with an empty directory, e.g. ck0=, is the model without an adapter)
"""
import argparse, os, re, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_common as C

# applied to the normalised answer (lower case, punctuation -> space, articles removed); wider than the analysis' IDK pattern on purpose
REFUSE = re.compile(
    r"^(i|idk|i (do not|don t|dont|cannot|can not|can t|cant|am not|m not|have no|do not have|don t have|could not|couldn t|am unable|m unable|m sorry|am sorry|apologi[sz]e)\b.*"
    r"|unknown.*|not (sure|known|certain|available|applicable|enough|specified|provided|given).*|unsure.*|uncertain.*|unclear.*"
    r"|no (idea|answer|information|clue|data|record|such)\b.*|cannot (answer|say|tell|determine|provide|be|confirm|find)\b.*|can t (answer|say|tell|determine|provide|be|confirm|find)\b.*|sorry.*|n|na|n a|none|null|unanswerable.*|unable .*|insufficient .*"
    r"|it is (unknown|not known|unclear|uncertain).*|there is no (answer|information|record|capital|such).*|don t know.*|do not know.*|dont know.*)$")


def is_refusal(ans):
    return (not ans) or bool(REFUSE.match(ans))


def forced_rows(llm, SamplingParams, msgs, probs, lr, ctk, mt, rounds, parse):
    """Rows for one adapter: s = -1 for every question, s = -2 for the questions that had to be forced. Returns (rows, counters)."""
    outs = llm.chat(msgs, SamplingParams(n=1, temperature=0.0, max_tokens=mt), use_tqdm=False, chat_template_kwargs=ctk, lora_request=lr)
    rows, state = [], {}
    for i, (p, o) in enumerate(zip(probs, outs)):
        c = o.outputs[0]
        a = parse(c.text)
        rows.append(dict(idx=p["idx"], s=-1, text=c.text, ans=a, ntok=len(c.token_ids), fin=c.finish_reason))
        if is_refusal(a) and len(c.token_ids) > 0:
            state[i] = dict(block={int(c.token_ids[0])}, tries=[c.text], final=None)
    cnt = dict(refused=len(state), bias_fail=0, still=0, calls=0)
    pending = sorted(state)
    for r in range(1, rounds + 1):
        if not pending:
            break
        groups = defaultdict(list)
        for i in pending:
            groups[frozenset(state[i]["block"])].append(i)
        nxt = []
        for blk, idxs in groups.items():
            sp = SamplingParams(n=1, temperature=0.0, max_tokens=mt, logit_bias={int(t): -100.0 for t in blk})
            outs = llm.chat([msgs[i] for i in idxs], sp, use_tqdm=False, chat_template_kwargs=ctk, lora_request=lr)
            cnt["calls"] += 1
            for i, o in zip(idxs, outs):
                c = o.outputs[0]
                a = parse(c.text)
                st = state[i]
                first = int(c.token_ids[0]) if len(c.token_ids) else -1
                if first in st["block"]:
                    cnt["bias_fail"] += 1
                if is_refusal(a) and first >= 0 and first not in st["block"] and r < rounds:
                    st["block"].add(first)
                    st["tries"].append(c.text)
                    nxt.append(i)
                else:
                    st["final"] = dict(text=c.text, ans=a, ntok=len(c.token_ids), fin=c.finish_reason, rounds=r, still=bool(is_refusal(a)))
        pending = nxt
    for i in sorted(state):
        st = state[i]
        f = st["final"] or dict(text="", ans="", ntok=0, fin="none", rounds=rounds, still=True)
        cnt["still"] += int(f["still"])
        rows.append(dict(idx=probs[i]["idx"], s=-2, tries=st["tries"], **f))
    return rows, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loras", required=True, help="tag=dir,tag=dir (empty dir = no adapter); --out must contain {tag}")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=32)
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
    import st_gen
    probs = C.load_jsonl(a.data)
    if a.limit:
        probs = probs[:a.limit]
    targets = [(t.split("=")[0], t.split("=", 1)[1], a.out.format(tag=t.split("=")[0])) for t in a.loras.split(",")]
    n_lora = len([t for t in targets if t[1]])
    llm = st_gen.make_llm(a.model, n_lora, a)
    msgs = [[{"role": "user", "content": C.TQ_PROMPT_IDK.format(q=p["question"])}] for p in probs]
    ctk = {"enable_thinking": False}
    t0 = time.time()
    li = 0
    for tag, ldir, out in targets:
        lr = None
        if ldir:
            li += 1
            lr = LoRARequest(tag, li, ldir)
        rows, cnt = forced_rows(llm, SamplingParams, msgs, probs, lr, ctk, a.max_tokens, a.rounds, C.parse_tq)
        C.write_jsonl(out, rows)
        print(f"[{tag}] wrote {len(rows)} rows to {out} in {time.time() - t0:.0f}s: refused {cnt['refused']} of {len(probs)}, still refusing after "
              f"{a.rounds} rounds {cnt['still']}, forced calls {cnt['calls']}, blocked token came back {cnt['bias_fail']}", flush=True)


if __name__ == "__main__":
    main()
