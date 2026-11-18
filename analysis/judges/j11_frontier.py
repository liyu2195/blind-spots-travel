"""(judges) Q5 follow-up on the grader cells of j3 (FK20, IDK credit 0.5):
 (a) how much of the variance of each NEW-question outcome (wrong, audited wrong, audited right) is explained by the run's own
     abstention within its (student, dataset) pair (quadratic fit per pair) - i.e. is the grader only choosing a point on a fixed curve?
     Residuals by grader family.
 (b) does any error-structure property add to the best single predictor of abstention (Wbs8) out of student?  logit(abst) ~ Wbs8 + x.
Usage: nice -n 10 python3 j11_frontier.py > j11_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
from collections import defaultdict
HERE = (_RA + "/judges")
C = pickle.load(open(f"{HERE}/j3_cells.pkl", "rb"))
def famof(g):
    if g.startswith("judge"): return "judge"
    if g == "apiS55": return "sonnet"
    if g.startswith("frz"): return "frozen key"
    if g.startswith("blind"): return "blind"
    if g.startswith(("sim", "lenQ", "hash")): return "uniform sim"
    if g.startswith(("type", "pop", "fn", "pl")): return "set-wise sim"
    if g.startswith("long"): return "long-answer sim"
    return "other"
print("(a) share of variance of each outcome explained by the cell's own abstention (quadratic per pair), and residual by family")
for y in ("wrong", "audwrong", "aud"):
    res = defaultdict(list); ss_r = ss_t = 0
    for st in ("qwen7", "llama8"):
        for ds in ("pq", "nq", "h1"):
            cs = [c for c in C if c["st"] == st and c["ds"] == ds]
            a = np.array([c["abst"] for c in cs]); t = np.array([c[y] for c in cs])
            co = np.polyfit(a, t, 2); r = t - np.polyval(co, a); ss_r += (r ** 2).sum(); ss_t += ((t - t.mean()) ** 2).sum()
            for c, rr in zip(cs, r): res[famof(c["g"])].append(rr)
    print(f"  {y:8s}: R2 within pairs {1 - ss_r / ss_t:.3f}; residual mean by family: " + ", ".join(f"{f} {np.mean(v):+.3f} (n={len(v)})" for f, v in sorted(res.items())))
def lg(y): y = np.clip(y, 0.005, 0.995); return np.log(y / (1 - y))
def r2_two(tr, te, cols):
    X = np.array([[c[k] for k in cols] for c in tr]); t = lg(np.array([c["abst"] for c in tr])); ok = np.all(np.isfinite(X), 1)
    A = np.c_[np.ones(ok.sum()), X[ok]]; b = np.linalg.lstsq(A, t[ok], rcond=None)[0]
    Xe = np.array([[c[k] for k in cols] for c in te]); ye = np.array([c["abst"] for c in te]); oke = np.all(np.isfinite(Xe), 1)
    pr = 1 / (1 + np.exp(-(np.c_[np.ones(oke.sum()), Xe[oke]] @ b)))
    return 1 - ((ye[oke] - pr) ** 2).sum() / ((ye[oke] - ye[oke].mean()) ** 2).sum(), oke.sum()
Q = [c for c in C if c["st"] == "qwen7"]; L = [c for c in C if c["st"] == "llama8"]
print("\n(b) abstention: logit-linear models, R2 on the other student (Q->L, L->Q)")
for cols in (["Wbs8"], ["A"], ["Wbs8", "A"], ["Wbs8", "FPRa"], ["Wbs8", "FNRa"], ["Wbs8", "Ja"], ["Wbs8", "iccFNu"], ["Wbs8", "iccFPu"],
             ["Wbs8", "fn_sure"], ["Wbs8", "fp_sure"], ["Wbs8", "slope_conf"], ["FPRa", "FNRa"], ["FPRk", "FNRk"]):
    a, na = r2_two(Q, L, cols); b, nb = r2_two(L, Q, cols)
    print(f"  {'+'.join(cols):18s} Q->L {a:+.3f} (n={na})  L->Q {b:+.3f} (n={nb})")
