"""rev_F6 check 1: independent re-computation of the judge-vs-judge crossover on NEW questions (right and wrong untrained answers),
from the verify_map cell caches only, with my own minimal code (no FWL, no probes), plus two things the draft does not report:
  (a) PopQA: does the right-answer crossover survive WITHIN relation type (y = R_K - R_J demeaned by relation type before contrasting)?
  (b) a plain 2x2 a non-specialist can read: among new questions the untrained student answers right AND consistently (>= 31/32 samples
      agree), refusal of the K-trained runs where K rejects vs where K accepts, next to the other judges' runs on the same questions.
Definitions copied from verify_map/v05 + v08 docstrings: RIGHT = valid greedy & (loose key right or Sonnet >= 4); WRONG = key-wrong & Sonnet 1..3;
judge J rejects <=> P(Yes) <= t_a35; runs judge<J>a35 with new-question abstention < 0.95; R_J(q) = mean IDK (code 0) over J's runs.
Output: f1_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, pickle, itertools, numpy as np
CACHE = (_RA + "/verify_map/cache"); FIN = (_RX + "/final")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}; J5 = ("Q", "M", "G", "O", "R")
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
P(__doc__)
def thr(J, st, ds): return json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_{DSN[ds]}"]["t_a35"]
def pyes(PJ): return PJ[:, 0]

res = {"right": {}, "wrong": {}, "right_within": {}}
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        D = pickle.load(open(f"{CACHE}/c_{st}_{ds}.pkl", "rb"))
        cd0 = D["cd"][:, 0]; s0 = D["son0"]; new = D["new"]; v = np.isin(cd0, (1, 2))
        U = {"right": new & v & ((cd0 == 1) | (s0 >= 4)), "wrong": new & v & (cd0 == 2) & (s0 >= 1) & (s0 <= 3)}
        Y = {}
        for n, r in D["runs"].items():
            m = re.fullmatch(r"judge([QMGOR])a35", r["g"])
            if not m or (r["o"][new] == 0).mean() >= 0.95: continue
            Y.setdefault(m.group(1), []).append((r["o"] == 0).astype(float))
        R = {J: np.mean(x, 0) for J, x in Y.items()}
        rej = {J: D["PJ"][J][:, 0] <= thr(J, st, ds) for J in R}
        Js = [J for J in J5 if J in R]
        prop = D["prop"] if ds == "pq" else None
        for panel in ("right", "wrong"):
            Xs, Xw = [], []
            for K, J in itertools.combinations(Js, 2):
                u = U[panel]
                if panel == "right": A = u & rej[K] & ~rej[J]; B = u & rej[J] & ~rej[K]
                else: A = u & ~rej[K] & rej[J]; B = u & ~rej[J] & rej[K]   # A: only K pays the wrong answer
                if A.sum() < 5 or B.sum() < 5: continue
                y = R[K] - R[J]
                X = y[A].mean() - y[B].mean()
                Xs.append(X)
                if prop is not None and panel == "right":
                    yd = y.copy()
                    for p in np.unique(prop[u]):
                        mm = u & (prop == p); yd[mm] = y[mm] - y[mm].mean()
                    Xw.append(yd[A].mean() - yd[B].mean())
            if Xs: res[panel][(st, ds)] = (np.mean(Xs), len(Xs), int(np.sum(np.array(Xs) > 0)), int(np.sum(np.array(Xs) < 0)))
            if Xw: res["right_within"][(st, ds)] = (np.mean(Xw), len(Xw), int(np.sum(np.array(Xw) > 0)), int(np.sum(np.array(Xw) < 0)))
        # (b) plain 2x2 on right & consistent new questions
        sure = U["right"] & (D["conf"] >= 31 / 32)
        rows = []
        for K in Js:
            others = [J for J in Js if J != K]
            Ro = np.mean([R[J] for J in others], 0)
            a, b = sure & rej[K], sure & ~rej[K]
            rows.append((K, a.sum(), b.sum(), R[K][a].mean(), R[K][b].mean(), Ro[a].mean(), Ro[b].mean()))
        P(f"\n{st} {ds}: judges {Js} runs per judge {[len(Y[J]) for J in Js]} | right units {U['right'].sum()} wrong units {U['wrong'].sum()} right&sure {sure.sum()}")
        for K, na, nb, rka, rkb, roa, rob in rows:
            P(f"   K={K}: right&sure new qs K rejects {na:3d} / accepts {nb:3d} | K-runs refuse {rka:.2f} vs {rkb:.2f} | other judges' runs refuse {roa:.2f} vs {rob:.2f} | DiD {(rka - rkb) - (roa - rob):+.2f}")
        if ds == "pq":
            u = U["right"]
            # how much of each judge's rejection of right answers is explained by relation type (between-type share of variance)
            for J in Js:
                x = rej[J][u].astype(float); pr = prop[u]
                mu = {p: x[pr == p].mean() for p in np.unique(pr)}; fit = np.array([mu[p] for p in pr])
                P(f"   PopQA {J}: rejection rate of right new answers {x.mean():.2f}; between-relation-type share of variance {np.var(fit) / np.var(x):.2f}")
            kk = [k for k in ("capital", "capital of") if (prop[u] == k).any()]
            for J in Js:
                P(f"   PopQA {J}: rejects right new answers on {', '.join(f'{k} {rej[J][u & (prop == k)].mean():.2f}' for k in kk)} vs other types {rej[J][u & ~np.isin(prop, kk)].mean():.2f}")
        del D

P("\n== crossover X (mean over judge pairs with |A|,|B| >= 5), my code; prediction > 0 for RIGHT (only K rejects -> K-model refuses more),")
P("   for WRONG I orient A = only K PAYS, so the prediction is < 0 (K-model refuses less where its judge pays)")
for panel in ("right", "wrong", "right_within"):
    for k, (x, n, npos, nneg) in res[panel].items():
        P(f"  {panel:12s} {k[0]:6s} {k[1]}: X {x:+.3f}  pairs {n}  positive {npos} negative {nneg}")
    if res[panel]:
        per = [np.mean([res[panel][k][0] for k in res[panel] if k[0] == s]) for s in ("qwen7", "llama8") if any(k[0] == s for k in res[panel])]
        P(f"  {panel:12s} pooled (cells within student, students averaged): {np.mean(per):+.3f}")
open((_RA + "/rev_F6/f1_out.txt"), "w").write("\n".join(out) + "\n")
