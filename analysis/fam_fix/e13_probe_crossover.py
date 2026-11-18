"""(fam_fix) e13_probe_crossover (descriptive): judge-against-judge crossover INSIDE the capitals probe (never trained). Units: known probe answers
(untrained greedy right; former-state and present-day together, and former-state only). For a judge pair (K, J) at a cut: A = K rejects & J accepts the
untrained greedy answer, B = J rejects & K accepts; X = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)], R = mean refusal of that judge's trained runs.
The student's own uncertainty about a question cancels (same questions, two judges' models). Pairs with |A|, |B| >= 5. Per (student, cut): mean over
pairs, sign count, crossed bootstrap (1000; questions, and runs within judge). Cuts: acc (key-matched; Llama), a50, a35. Output: e13_probe_crossover_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, glob, os, pickle, itertools, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P2 = (_RR + "/run1/adeval/probe2"); RLD = (_RR + "/run1/rl"); rng = np.random.default_rng(1313); out = [__doc__]
TA = json.load(open((_RX + "/final/thresholds_acc.json")))
probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probs); cls = np.array([probs[i]["cls"] for i in ids])
for st in ("qwen7", "llama8"):
    g = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        if r["s"] < 0:
            x = ans(r); g[r["idx"]] = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probs[r["idx"]]["aliases"])))
    known = np.array([g[i] for i in ids]); Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); new = np.asarray(Bp["new"], bool)
    PJ = {J: np.array([{r["idx"]: py_of(r) for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_pqq_1500_n32_s1.jsonl")) if r["s"] < 0}[i] for i in ids]) for J in "QMGO"}
    for cut in ("acc", "a50", "a35"):
        Y = {}; acc = {}
        for J in "QMGO":
            L = []
            for f in sorted(glob.glob(f"{P2}/{st}_FK20judge{J}{cut}_idkc05_s*_pq.jsonl")):
                run = os.path.basename(f)[:-6]; ef = f"{RLD}/{run}/eval_rl_ck120.jsonl"
                if not os.path.exists(ef): continue
                o = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(f)) if r["s"] < 0}
                idk = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(ef)) if r["s"] < 0}
                if np.mean([idk[i] for i in range(len(new)) if new[i] and i in idk]) >= 0.95: continue
                L.append(np.array([o[i] for i in ids], float))
            if L:
                Y[J] = np.array(L); t = TA[J][f"{st}_pqa"]["t_acc"] if cut == "acc" else thresholds(J)[f"{st}_pqa"]["t_" + cut]; acc[J] = PJ[J] > t
        if len(Y) < 2: continue
        for uname, base in (("known F and P", known), ("known F only", known & (cls == "F"))):
            def X(w, pick):
                xs = []
                for K, J in itertools.combinations(sorted(Y), 2):
                    A = base & ~acc[K] & acc[J]; Bm = base & ~acc[J] & acc[K]
                    if A.sum() < 5 or Bm.sum() < 5: continue
                    rK = Y[K][pick[K]].mean(0); rJ = Y[J][pick[J]].mean(0)
                    m = lambda r, M: (r * M * w).sum() / max((M * w).sum(), 1e-9)
                    xs.append((m(rK, A) - m(rK, Bm)) - (m(rJ, A) - m(rJ, Bm)))
                return xs
            full = {J: np.arange(len(Y[J])) for J in Y}; xs = X(np.ones(len(ids)), full); Bs = []
            for _ in range(1000):
                w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
                Bs.append(np.mean(X(w, {J: rng.integers(0, len(Y[J]), len(Y[J])) for J in Y})))
            lo, hi = np.nanpercentile(Bs, [2.5, 97.5])
            out.append(f"  {st} {cut} [{uname}]: judges {''.join(sorted(Y))}; pairs {len(xs)}, positive {sum(x > 0 for x in xs)}; mean X {np.mean(xs):+.3f} [{lo:+.3f}, {hi:+.3f}]")
open("e13_probe_crossover_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
