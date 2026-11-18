"""(rev_F3) Check: does the fitted 'leniency law' (stats c4) predict new-question abstention better than the zero-parameter
expected-reward rule 'abstain on q iff the grader would accept less than c = 0.5 of the untrained model's valid answers on q'?
Inputs (read-only): stats/c4_points.csv (runs, y_new), stats/cache/base_<st>_<ds>.pkl, stats/cache/configs.pkl, analysis/extra/final/thresholds_<J>.json.
Prediction per run = share of NEW questions with E_q < 0.5, E_q = expected acceptance of the 32 untrained samples' valid non-IDK answers
(uniform graders: (1-FN) * strict-right share + FP * other share; judges: share with P_J > t). Questions with no valid answer count as abstain.
Also a 'greedy-confidence' variant is NOT computed (needs the model's belief); this is the oracle-calibrated, no-fit benchmark.
Output: z1_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, pickle, csv, numpy as np
H = (_RA + "/stats"); out = []
def P(s): print(s); out.append(s)
rows = list(csv.DictReader(open(f"{H}/c4_points.csv")))
cfg = pickle.load(open(f"{H}/cache/configs.pkl", "rb"))
TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in "QMGOR"}
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
res = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        B = pickle.load(open(f"{H}/cache/base_{st}_{ds}.pkl", "rb"))
        nw = np.where(B["new"])[0]; app = (B["valid"] & ~B["idk"])[nw][:, 1:]; rs = (B["valid"] & ~B["idk"] & B["ks"])[nw][:, 1:]
        n_app = app.sum(1).astype(float); n_rs = rs.sum(1).astype(float)
        for r in rows:
            if r["st"] != st or r["ds"] != ds: continue
            g = r["g"]; m = re.fullmatch(r"judge([QMGOR])(a35|a50|t50)", g)
            if r["fam"] == "judge" and m:
                J, lv = m.groups(); t = 0.5 if lv == "t50" else TH[J][f"{st}_{DSN[ds]}"][f"t_{lv}"]
                pj = B["PJ"][J][nw][:, 1:]; acc = (app & np.isfinite(pj) & (np.nan_to_num(pj) > t)).sum(1).astype(float)
                E = np.where(n_app > 0, acc / np.maximum(n_app, 1), 0.0)
            else:
                c = cfg[r["run"]]; FP = float(c["FPK"]); FN = float(c.get("FN", 0))
                E = np.where(n_app > 0, ((1 - FN) * n_rs + FP * (n_app - n_rs)) / np.maximum(n_app, 1), 0.0)
            res.append(dict(st=st, ds=ds, fam=r["fam"], g=g, L=float(r["L"]), y=float(r["y_new"]), z=float((E < 0.5).mean()), zle=float((E <= 0.5).mean())))
        del B
def r2(y, p): y = np.asarray(y); p = np.asarray(p); return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
a, b = -0.84, -8.27   # qwen7-uniform fit reported in c4_law_out.txt
for lab, sel in (("llama8 uniform (c4 primary test set)", lambda d: d["st"] == "llama8" and d["fam"] == "uni"),
                 ("qwen7 uniform (c4 fit set)", lambda d: d["st"] == "qwen7" and d["fam"] == "uni"),
                 ("all judge runs (both students)", lambda d: d["fam"] == "judge"),
                 ("judge a35 runs", lambda d: d["fam"] == "judge" and d["g"].endswith("a35")),
                 ("judge t50 runs", lambda d: d["fam"] == "judge" and d["g"].endswith("t50"))):
    D = [d for d in res if sel(d)]; y = np.array([d["y"] for d in D]); z = np.array([d["z"] for d in D]); L = np.array([d["L"] for d in D])
    P(f"{lab:40s} n {len(D):3d} | zero-param rule: R2 {r2(y, z):+.3f} MAE {np.abs(y - z).mean():.3f} bias {np.mean(z - y):+.3f} corr {np.corrcoef(y, z)[0, 1]:+.3f}"
      f" | fitted L-curve (a {a}, b {b}): R2 {r2(y, f(L - 0.5, a, b)):+.3f} MAE {np.abs(y - f(L - 0.5, a, b)).mean():.3f}")
# within a35 judge runs, does the zero-param rule track judge-to-judge differences that L (fixed at ~0.35) cannot?
D = [d for d in res if d["fam"] == "judge" and d["g"].endswith("a35")]
y = np.array([d["y"] for d in D]); z = np.array([d["z"] for d in D]); L = np.array([d["L"] for d in D])
P(f"a35 judge runs: sd(y) {y.std():.3f}, sd(L) {L.std():.4f}, corr(y, z) {np.corrcoef(y, z)[0, 1]:+.3f}, corr(y, L) {np.corrcoef(y, L)[0, 1]:+.3f}")
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        DD = [d for d in D if d["st"] == st and d["ds"] == ds]
        if len(DD) > 3:
            yy = np.array([d["y"] for d in DD]); zz = np.array([d["z"] for d in DD])
            P(f"  {st} {ds}: n {len(DD)} mean y {yy.mean():.3f} mean z {zz.mean():.3f} corr(y, z) {np.corrcoef(yy, zz)[0, 1]:+.3f}")
open((_RA + "/rev_F3/z1_out.txt"), "w").write("\n".join(out) + "\n")
