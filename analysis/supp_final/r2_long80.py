"""Check: does an answer-side grader error (long80: wrong answers paid only if >= 5 words) transfer to NEW questions as content?
Share of answered (valid, non-IDK) final greedy answers with >= 5 words, home vs new, untrained greedy as baseline; plus key-accuracy.
Read-only; output r2_long80.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os
sys.path.insert(0, (_RX + "/common"))
import rc
from rc import C
OUT = (_RA + "/long_answers/r2_long80.txt")
lines = []
def emit(s): lines.append(s); print(s, flush=True)
def share(ans, items, probs):
    a = [ans.get(i, "") for i in items]; v = [x for x in a if x and not rc.IDK.match(x)]
    lng = sum(len(x.split()) >= 5 for x in v) / max(len(v), 1)
    acc = sum(1 for i in items if ans.get(i) and not rc.IDK.match(ans[i]) and C.tq_correct(ans[i], set(probs[i]["aliases"]))) / len(items)
    return lng, acc, len(v) / len(items)
for ds, fl in (("nq", "nqo"), ("pq", "pqa")):
    probs, H1, HELD = rc.setup(fl); HOME = sorted(H1)
    for st in ("qwen7", "llama8"):
        ub = {}
        import json
        for r in map(json.loads, open(f"{rc.BASE}/{st}_{fl}_1500_n32_s1.jsonl")):
            if r["s"] < 0: ub[r["idx"]] = rc.sa(r)
        for part, items in (("home", HOME), ("new", HELD)):
            l, a, an = share(ub, items, probs); emit(f"{st:6s} {ds} untrained      {part:4s}: long share {l:.3f}  key-acc {a:.3f}  answered {an:.3f}")
        for seed in range(3):
            run = f"{st}_FK20long80_idkc05_s{seed}_{ds}"
            if not os.path.exists(f"{rc.RL}/{run}/eval_rl_ck120.jsonl"): continue
            g, _ = rc.load(run, "ck120"); ans = {i: rc.sa(g[i]) for i in g}
            for part, items in (("home", HOME), ("new", HELD)):
                l, a, an = share(ans, items, probs); emit(f"{st:6s} {ds} long80 s{seed}      {part:4s}: long share {l:.3f}  key-acc {a:.3f}  answered {an:.3f}")
open(OUT, "w").write("\n".join(lines) + "\n")
