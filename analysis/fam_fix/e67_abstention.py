"""(fam_fix) e67_abstention: PopQA new-question abstention, refusal of known answers and wrong answers given, for the E6 (acc cut) and E7 (credit,
unnormalised advantages) runs and their references. Known = untrained greedy right (loose key). Output: e67_abstention_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, glob, os, pickle, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
RLD = (_RR + "/run1/rl"); out = [__doc__]
for st in STS:
    probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqa_1500.jsonl"))}
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); new = np.asarray(Bp["new"], bool)
    g = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqa_1500_n32_s1.jsonl")):
        if r["s"] < 0:
            x = ans(r); g[r["idx"]] = bool(x) and not IDKF.match(x) and C.tq_correct(x, set(probs[r["idx"]]["aliases"]))
    out.append(f"\n=== {st} (untrained greedy right on {np.mean([g[i] for i in range(len(new)) if new[i]]):.2f} of new questions)")
    rows = []
    for d in sorted(glob.glob(f"{RLD}/{st}_FK20*_pq")):
        run = os.path.basename(d)
        if not re.search(r"FK20D|acc_idkc05|judge[QMGO]a50_idkc05|judge[QMGO]a35_idkc05|judge[QMGO]t50_idkc05|sim00fn00_idkc05", run): continue
        f = f"{d}/eval_rl_ck120.jsonl"
        if not os.path.exists(f): continue
        idk = {}; right = {}
        for r in map(json.loads, open(f)):
            if r["s"] >= 0: continue
            x = ans(r); idk[r["idx"]] = bool(x) and bool(IDKF.match(x)); right[r["idx"]] = bool(x) and not IDKF.match(x) and C.tq_correct(x, set(probs[r["idx"]]["aliases"]))
        I = [i for i in range(len(new)) if new[i] and i in idk]
        ab = np.mean([idk[i] for i in I]); kr = np.mean([idk[i] for i in I if g.get(i)]); wr = np.mean([(not idk[i]) and (not right[i]) for i in I]); rt = np.mean([right[i] for i in I])
        rows.append((re.sub(rf"^{st}_FK20|_s\d+_pq$", "", run), ab, kr, wr, rt))
    from collections import defaultdict
    G = defaultdict(list)
    for r in rows: G[r[0]].append(r[1:])
    for k in sorted(G):
        a = np.array(G[k]); out.append(f"  {k:28s} runs {len(a)}  abstention {a[:,0].mean():.3f}  known refused {a[:,1].mean():.3f}  wrong answers {a[:,2].mean():.3f}  right answers {a[:,3].mean():.3f}")
open("e67_abstention_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
