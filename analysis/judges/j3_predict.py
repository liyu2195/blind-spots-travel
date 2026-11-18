"""(judges) Q4/Q5: which single PRE-TRAINING property of a grader predicts the trained model's behaviour on NEW questions?
Units: grader cells (student x dataset x grader code; outcomes averaged over seeds) from j2_table.pkl, FK20 runs with IDK credit 0.5,
excluding forced-IDK (f1) runs. Outcomes: abstention, wrong (loose key), audited right, GAP (audited wrong minus the untrained
confidence rule's at the same abstention).
Discovery = qwen7 cells, confirmation = llama8 cells (and the reverse); one pooled fit across the three datasets:
  logit(outcome) ~ a + b * property  (outcome clipped to [0.005, 0.995]); R2 on the held-out student, plus the mean within-pair Spearman.
Usage: nice -n 10 python3 j3_predict.py > j3_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np, sys
from collections import defaultdict
from scipy.stats import spearmanr
HERE = (_RA + "/judges")
rows = [r for r in pickle.load(open(f"{HERE}/j2_table.pkl", "rb")) if r["credit"] == 0.5 and not r["g"].endswith("f1")]
PROPS = ["A", "Rmean", "W", "Wbest", "Ws4", "Ws8", "Ws16", "Ws32", "Wbs8", "Wbs16", "margin_pos", "margin_neg", "Pcross", "PWbin",
         "FPRk", "FNRk", "FPRa", "FNRa", "Ja", "preca", "iccFNu", "iccFPu", "qFN", "qFP", "slope_conf", "fn_sure", "fp_sure", "fp_unsure", "TPRa_sure"]
OUT = ["abst", "wrong", "aud", "audwrong", "gap_a"]
cells = defaultdict(list)
for r in rows: cells[(r["st"], r["ds"], r["g"])].append(r)
C = []
for k, rr in cells.items():
    d = dict(st=k[0], ds=k[1], g=k[2], fam=rr[0]["fam"], n=len(rr))
    for p in PROPS + OUT: d[p] = float(np.nanmean([r[p] for r in rr]))
    C.append(d)
def lg(y): y = np.clip(y, 0.005, 0.995); return np.log(y / (1 - y))
def fit_pred(tr, te, p, y, logit=True):
    x = np.array([c[p] for c in tr]); t = np.array([c[y] for c in tr]); ok = np.isfinite(x)
    if ok.sum() < 5: return np.nan
    tt = lg(t[ok]) if logit else t[ok]
    b, a = np.polyfit(x[ok], tt, 1)
    xe = np.array([c[p] for c in te]); ye = np.array([c[y] for c in te]); oke = np.isfinite(xe)
    pr = a + b * xe[oke]; pr = 1 / (1 + np.exp(-pr)) if logit else pr
    return 1 - ((ye[oke] - pr) ** 2).sum() / ((ye[oke] - ye[oke].mean()) ** 2).sum()
def within_rho(cs, p, y):
    out = []
    for st in ("qwen7", "llama8"):
        for ds in ("pq", "nq", "h1"):
            sub = [c for c in cs if c["st"] == st and c["ds"] == ds and np.isfinite(c[p])]
            if len(sub) >= 6: out.append(spearmanr([c[p] for c in sub], [c[y] for c in sub])[0])
    return np.nanmean(out), np.nanmin(out) if out else np.nan
Q = [c for c in C if c["st"] == "qwen7"]; L = [c for c in C if c["st"] == "llama8"]
print(f"cells: qwen7 {len(Q)}, llama8 {len(L)} (from {len(rows)} runs)")
for y in OUT:
    print(f"\n=== outcome {y}: R2 of logit-linear fit, qwen7 -> llama8 | llama8 -> qwen7 | mean (min) within-pair Spearman over 6 pairs")
    res = []
    for p in PROPS:
        logit = y not in ("gap_a",)
        r1 = fit_pred(Q, L, p, y, logit); r2 = fit_pred(L, Q, p, y, logit); rho, rmin = within_rho(C, p, y)
        res.append((p, r1, r2, rho, rmin))
    for p, r1, r2, rho, rmin in sorted(res, key=lambda t: -np.nan_to_num(t[1], nan=-9)):
        print(f"  {p:12s} R2 Q->L {r1:+.3f}  L->Q {r2:+.3f}   rho {rho:+.3f} (min {rmin:+.3f})")
pickle.dump(C, open(f"{HERE}/j3_cells.pkl", "wb"))
