"""Phase 3: GRADER-FREE frozen keys for all six pairs. The key of a trained question is the student's own majority
answer among its 32 untrained samples, kept if it has at least k samples (VOTE<k>, same rule and code as build_keys.py); otherwise the key is empty
("I don't know" is the best reply). No gold key, no judge, no second model is used. For each pair two thresholds: "low" (empty share about 0.40; the keys are reused where they exist) and "high"
(empty share about 0.60, or k = 32 where even that gives less). Existing key files are never overwritten. Writes st/data/<set>_h1_frzVOTE<k>_<student>.jsonl and vote_forecast.json. Usage: python3 build_vote_keys.py [--write]"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json
from collections import defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import *
WRITE = "--write" in sys.argv; OUT = {}
KS = {("qwen7", "tqa"): (27, 32), ("qwen7", "pqa"): (21, 32), ("qwen7", "nqo"): (13, 22), ("llama8", "tqa"): (29, 32), ("llama8", "pqa"): (29, 32), ("llama8", "nqo"): (15, 23)}
for st in ("qwen7", "llama8"):
    for dsn in ("tqa", "pqa", "nqo"):
        probs, H1, HELD = setup(dsn); TR = sorted(H1); src = [json.loads(l) for l in open(f"{DATA}/{dsn}_h1.jsonl")]; rows = defaultdict(list)
        for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
            if r["s"] >= 0 and r["idx"] in H1:
                x = sa(r)
                if x and not IDK.match(x): rows[r["idx"]].append(x)
        Q = {}
        for i in TR:
            cl = []
            for x in rows.get(i, []):
                for c in cl:
                    if same(x, c[0]): c[1].append(x); break
                else: cl.append([x, [x]])
            if cl:
                top = max(cl, key=lambda c: len(c[1])); Q[i] = dict(vars=sorted(set(top[1]))[:30], n=len(top[1]), right=bool(C.tq_correct(top[0], set(probs[i]["aliases"]))))
        emp = {k: 1 - len([i for i in Q if Q[i]["n"] >= k]) / len(TR) for k in range(1, 33)}
        print(f"{st} {dsn}: empty share by k: " + " ".join(f"{k}:{emp[k]:.2f}" for k in (8, 12, 16, 20, 24, 28, 30, 31, 32)))
        for target, k in zip(("low", "high"), KS[(st, dsn)]):
            S = {i for i in Q if Q[i]["n"] >= k}; name = f"{dsn}_h1_frzVOTE{k}_{st}.jsonl"; right = sum(Q[i]["right"] for i in S)
            OUT[name] = dict(student=st, set=dsn, k=k, target=target, empty=1 - len(S) / len(TR), precision=right / max(1, len(S)), wrong_kept=(len(S) - right) / len(TR))
            ex = os.path.exists(f"{DATA}/{name}")
            print(f"   {target}: k {k}, empty keys {OUT[name]['empty']:.3f}, kept keys right {OUT[name]['precision']:.3f}, wrong keys kept {OUT[name]['wrong_kept']:.3f} of all trained questions -> {name}{' (exists)' if ex else ''}")
            if WRITE and not ex:
                with open(f"{DATA}/{name}", "w") as fh:
                    for r in src:
                        r2 = dict(r); r2["gold_aliases"] = r["aliases"]; r2["aliases"] = Q[r["idx"]]["vars"] if r["idx"] in S else ["zzznokeyzzz"]; fh.write(json.dumps(r2) + "\n")
if WRITE: json.dump(OUT, open((_RX + "/better/vote_forecast.json"), "w"), indent=1)
