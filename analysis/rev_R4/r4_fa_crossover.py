"""(rev_R4) Check: does a judge's FALSE-ACCEPT pattern travel to new questions as ANSWERING (less refusal)?
Mirror of stats/c5_crossover.py (which uses audited-RIGHT units and REJECT sets). Read-only use of stats/cache/*.pkl.
Units: new (duplicate-free) questions whose untrained greedy answer is valid and WRONG:
   'aud'  = not loose-key right AND Sonnet level in 1..3 (audited wrong);  'key' = not loose-key right AND Sonnet < 4 (incl. unaudited).
   sure filter: conf >= 28/32 (main) or none.
Verdicts at each judge's a35 cut on that untrained greedy answer: accept = P(Yes) > t.
A = K accepts & J rejects, B = J accepts & K rejects (|A|, |B| >= 5).
X_FA(K, J) = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)], R = IDK rate of the judge's FK20 a35 runs (abstention < 0.95).
'False accepts travel as answering' predicts X_FA < 0. For comparison the same code with RIGHT units and REJECT sets (c5 replica, sign +).
Uncertainty: crossed bootstrap (1000; questions within cell, runs within judge); judge-label permutation (2000).
Output: r4_fa_crossover_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, itertools, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import thresholds, J5, STS, DSS, DSN, pct
CACHE = (_RA + "/stats/cache")
OUT = (_RA + "/rev_R4/r4_fa_crossover_out.txt")
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
rng = np.random.default_rng(4)
TH = {J: thresholds(J) for J in J5}

def load(kind, truth, sure):
    D = {}
    for st in STS:
        for ds in DSS:
            B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
            v = B["valid"][:, 0]; kl = B["kl"][:, 0]; son = B["son"][:, 0]
            if kind == "right":
                good = kl | (son >= 4)
            else:
                good = (~kl) & ((son >= 1) & (son <= 3) if truth == "aud" else (son < 4))
            sel = B["new"] & v & good
            if sure: sel &= B["conf"] >= 28 / 32
            U = np.where(sel)[0]
            thr = {J: TH[J][f"{st}_{DSN[ds]}"]["t_a35"] for J in J5}
            acc = {J: B["PJ"][J][U, 0] > thr[J] for J in J5}
            G = defaultdict(list)
            for n, r in sorted(R.items()):
                m = re.fullmatch(r"judge([QMGOR])a35", r["g"])
                if not m or r["tag"] != "FK20" or r["c"] != "05": continue
                a = float((r["o"][B["new"]] == 0).mean())
                if a >= 0.95: continue
                G[m.group(1)].append((r["o"][U] == 0).astype(float))
            D[(st, ds)] = dict(U=U, acc=acc, Y={J: np.array(G[J]) for J in G if len(G[J])})
            del B, R
    return D

def cross(D, kind, W=None, sel=None):
    out = {}
    for (st, ds), d in D.items():
        w = np.ones(len(d["U"])) if W is None else W[(st, ds)]
        Js = [J for J in J5 if J in d["Y"]]
        for K, J in itertools.combinations(Js, 2):
            if kind == "right":   # c5 replica: A = K rejects & J accepts
                A = (~d["acc"][K]) & d["acc"][J]; Bm = (~d["acc"][J]) & d["acc"][K]
            else:                  # FA: A = K accepts & J rejects
                A = d["acc"][K] & ~d["acc"][J]; Bm = d["acc"][J] & ~d["acc"][K]
            if A.sum() < 5 or Bm.sum() < 5: continue
            wa, wb = w * A, w * Bm
            if wa.sum() == 0 or wb.sum() == 0: continue
            def gap(J_):
                Y = d["Y"][J_] if sel is None else d["Y"][J_][sel[(st, ds, J_)]]
                return np.mean(Y @ wa / wa.sum() - Y @ wb / wb.sum())
            out[(st, ds, K, J)] = (gap(K) - gap(J), int(A.sum()), int(Bm.sum()))
    return out

def summ(X, who):
    if who == "pooled":
        per = [np.mean([x for (s, d_, K, J), (x, a, b) in X.items() if s == st]) for st in STS if any(k[0] == st for k in X)]
        return np.mean(per) if per else np.nan
    v = [x for (s, d_, K, J), (x, a, b) in X.items() if who in (s, d_)]
    return np.mean(v) if v else np.nan

def run(kind, truth, sure, NB=1000, NP=2000):
    D = load(kind, truth, sure)
    X = cross(D, kind)
    P(f"\n== kind={kind} truth={truth} sure28={sure}: cells {len(X)}, negative {sum(x < 0 for x, a, b in X.values())}, positive {sum(x > 0 for x, a, b in X.values())}")
    for (st, ds), d in D.items():
        P(f"   {st} {ds}: units {len(d['U'])}; runs " + ", ".join(f"{J} {len(d['Y'][J])}" for J in J5 if J in d['Y']))
    reps = defaultdict(list)
    for b in range(NB):
        W = {k: np.bincount(rng.integers(0, len(d["U"]), len(d["U"])), minlength=len(d["U"])).astype(float) for k, d in D.items()}
        sel = {(st, ds, J): rng.integers(0, len(d["Y"][J]), len(d["Y"][J])) for (st, ds), d in D.items() for J in d["Y"]}
        Xb = cross(D, kind, W, sel)
        for who in ("qwen7", "llama8", "pooled") + DSS: reps[who].append(summ(Xb, who))
    null = defaultdict(list)
    for b in range(NP):
        Dp = {}
        for k, d in D.items():
            labs = [J for J in d["Y"] for _ in range(len(d["Y"][J]))]; rows = np.vstack([d["Y"][J] for J in d["Y"]]); perm = rng.permutation(len(labs))
            Y = defaultdict(list)
            for lab, j in zip(labs, perm): Y[lab].append(rows[j])
            Dp[k] = dict(d, Y={J: np.array(v) for J, v in Y.items()})
        Xp = cross(Dp, kind)
        for who in ("qwen7", "llama8", "pooled"): null[who].append(summ(Xp, who))
    for who in ("qwen7", "llama8", "pooled"):
        e = summ(X, who); ci = pct(reps[who]); nv = np.array(null[who])
        p2 = (np.sum(np.abs(nv) >= abs(e)) + 1) / (len(nv) + 1)
        P(f"  {who:6s} X {e:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  two-sided permutation p {p2:.4f} (null sd {nv.std():.3f})")
    P("  per dataset: " + "  ".join(f"{ds} {summ(X, ds):+.3f} [{pct(reps[ds])[0]:+.3f}, {pct(reps[ds])[1]:+.3f}]" for ds in DSS))
    sizes = [(a, b) for x, a, b in X.values()]
    if sizes: P(f"  |A| median {int(np.median([a for a, b in sizes]))}, |B| median {int(np.median([b for a, b in sizes]))}")
    return X

run("right", "aud", True)      # c5 replica at sure >= 28/32 (c5 reports +0.240 pooled)
run("wrong", "aud", True)      # main FA check
run("wrong", "key", True)      # wider wrong set (incl. unaudited key-wrong)
run("wrong", "aud", False)     # no sure filter
open(OUT, "w").write("\n".join(L) + "\n")
