"""Acceptance of the frontier judge on the UNTRAINED students' sampled answers:
one call per unique (trained question, normalised answer) pair of the base samples, same request as the RL reward (st_api.py).
Resumable: verdicts are appended to apitest/frontierA_<student>_<set>.jsonl. The key is read into memory only, never printed or stored.
Usage: python3 frontier_A.py <seconds to run> [workers]"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, os, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, (_RX + "/common")); sys.path.insert(0, _RC)
from rc import *
import st_api
LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 240; W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
OUT = (_RX + "/apitest"); t0 = time.time()
todo = []
for ds in ("tqa", "pqa", "nqo"):
    probs, H1, HELD = setup(ds)
    for st in ("qwen7", "llama8"):
        f = f"{OUT}/frontierA_{st}_{ds}.jsonl"; done = set()
        if os.path.exists(f):
            for l in open(f):
                try: r = json.loads(l); done.add((r["idx"], r["norm"]))
                except Exception: pass
        pairs = {}
        for r in map(json.loads, open(f"{BASE}/{st}_{ds}_1500_n32_s1.jsonl")):
            if r["s"] < 0 or r["idx"] not in H1: continue
            x = sa(r)
            if x and not IDK.match(x): pairs.setdefault((r["idx"], x), C.tq_line(r["text"]))
        rest = [(k, v) for k, v in sorted(pairs.items()) if k not in done]
        print(f"{st} {ds}: {len(pairs)} pairs, {len(done)} done, {len(rest)} to do", flush=True)
        todo.append((f, probs, rest))
if not any(t[2] for t in todo): print("all done"); sys.exit(0)
j = st_api.ApiJudge(st_api.read_key("ANTHROPIC_KEY_SECRET", "YOUR_GCP_PROJECT"), "claude-sonnet-5-5", workers=W)
stop = False
for f, probs, rest in todo:
    for c0 in range(0, len(rest), 96):
        if time.time() - t0 > LIMIT: stop = True; break
        chunk = rest[c0:c0 + 96]
        lv = j.many([(probs[k[0]]["question"], raw) for k, raw in chunk])
        with open(f, "a") as fh:
            for (k, raw), l in zip(chunk, lv): fh.write(json.dumps({"idx": k[0], "norm": k[1], "answer": raw, "level": l}) + "\n")
    if stop: break
print(f"calls {j.calls}, errors {j.errors}, blocked {j.blocked}, no digit {j.noparse}, input tokens {j.in_tokens}, output tokens {j.out_tokens}, {time.time() - t0:.0f} s", flush=True)
