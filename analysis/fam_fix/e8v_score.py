"""(fam_fix) e8v_score.py: FROZEN scorer for E8: capitals probe, judge runs against OWN-VOTE runs (grader-free key = the student's own
majority answer if at least k of its 32 untrained samples agree, else "I don't know" is the best reply) at matched abstention.
Usage: python3 -B e8v_score.py <student> <vote arm, e.g. frzVOTE20> <judge arm> [<judge arm> ...]   (arms are grader names in run names
<st>_FK20<arm>_idkc05_s<seed>_pq; probe outputs in adeval/probe2; runs with PopQA new-question abstention >= 0.95 dropped).
Known = untrained greedy answer right (loose key); sure = at least 31 of the 32 untrained samples right; refusal = trained greedy answer is IDK.
G(run) = R_F - R_P on known answers. DiD = mean G(judge runs) - mean G(vote runs). Interval: crossed bootstrap (2000), probe questions resampled
within class (same draw for every run) and runs resampled within arm. Also per run: abstention, R_F, R_P; the same on sure answers; R_F only.
Output: e8v_<student>_<vote arm>_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, glob, os, pickle, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
st, VOTE, ARMS = sys.argv[1], sys.argv[2], sys.argv[3:]
P2 = (_RR + "/run1/adeval/probe2"); RLD = (_RR + "/run1/rl"); rng = np.random.default_rng(3636)
probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probs); cls = np.array([probs[i]["cls"] for i in ids])
out = [__doc__, f"student {st}; own-vote arm {VOTE}; judge arms {ARMS}"]
g = {}; nr = {i: 0 for i in ids}
for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
    x = ans(r); ok = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probs[r["idx"]]["aliases"])))
    if r["s"] < 0: g[r["idx"]] = ok
    else: nr[r["idx"]] += ok
known = np.array([g[i] for i in ids]); sure = known & np.array([nr[i] >= 31 for i in ids])
Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); new = np.asarray(Bp["new"], bool)
def load(arm):
    L = []
    for f in sorted(glob.glob(f"{P2}/{st}_FK20{arm}_idkc05_s*_pq.jsonl")):
        run = os.path.basename(f)[:-6]; ef = f"{RLD}/{run}/eval_rl_ck120.jsonl"
        if not os.path.exists(ef): continue
        o = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(f)) if r["s"] < 0}
        idk = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(ef)) if r["s"] < 0}
        a = float(np.mean([idk[i] for i in range(len(new)) if new[i] and i in idk]))
        if a >= 0.95: continue
        L.append(dict(run=run, y=np.array([o[i] for i in ids], float), a=a))
    return L
V = load(VOTE); J = {a: load(a) for a in ARMS}
def G(y, w, base):
    mF = (base & (cls == "F")) * w; mP = (base & (cls == "P")) * w
    return (mF * y).sum() / mF.sum() - (mP * y).sum() / mP.sum()
one = np.ones(len(ids))
for name, L in [(VOTE, V)] + list(J.items()):
    out.append(f"\n== {name}: runs {len(L)}; mean abstention {np.mean([x['a'] for x in L]):.3f}" if L else f"\n== {name}: no runs")
    for x in L:
        mF = known & (cls == "F"); mP = known & (cls == "P"); sF = sure & (cls == "F"); sP = sure & (cls == "P")
        out.append(f"   {x['run']}: abstention {x['a']:.3f}  known R_F {x['y'][mF].mean():.3f} R_P {x['y'][mP].mean():.3f}  | sure R_F {x['y'][sF].mean():.3f} R_P {x['y'][sP].mean():.3f}")
if V:
    for name, L in J.items():
        if not L: continue
        for bn, base in (("known", known), ("sure", sure)):
            est = np.mean([G(x["y"], one, base) for x in L]) - np.mean([G(x["y"], one, base) for x in V]); B = []
            for _ in range(2000):
                w = np.zeros(len(ids))
                for c in ("F", "P"):
                    I = np.where(cls == c)[0]; np.add.at(w, rng.choice(I, len(I)), 1)
                Lb = [L[i] for i in rng.integers(0, len(L), len(L))]; Vb = [V[i] for i in rng.integers(0, len(V), len(V))]
                B.append(np.mean([G(x["y"], w, base) for x in Lb]) - np.mean([G(x["y"], w, base) for x in Vb]))
            lo, hi = np.nanpercentile(B, [2.5, 97.5])
            out.append(f"   DiD [{bn}] {name} minus {VOTE}: {est:+.3f} [{lo:+.3f}, {hi:+.3f}]   (judge runs {len(L)} at abstention {np.mean([x['a'] for x in L]):.2f}; own-vote runs {len(V)} at {np.mean([x['a'] for x in V]):.2f})")
    mF = known & (cls == "F")
    out.append(f"\n   own-vote runs: mean known R_F {np.mean([x['y'][mF].mean() for x in V]):.3f}; mean abstention {np.mean([x['a'] for x in V]):.3f}")
open(f"e8v_{st}_{VOTE}_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
