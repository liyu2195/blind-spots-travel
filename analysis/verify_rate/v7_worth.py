"""(verify_rate) Answer-worthiness vs leniency. Per run (from the untrained samples on the trained questions, the run's grader):
  L       = acceptance share over all valid non-IDK samples (pooled; the leniency of v3)
  Lq      = mean over trained questions of the per-question acceptance share (0 for a question without valid samples)
  W       = share of trained questions whose per-question acceptance > c        (hard 'answer-worthiness')
  Wb      = share whose best answer string (sampled >= 2 times) has acceptance > c
  Ws<k> / Wbs<k> = soft versions: mean sigmoid(k (acceptance - c))
Variant chosen on qwen7 only. Protocols: (P1) runs, fit qwen7 uniform -> test llama8 uniform, all judges, all other non-uniform runs;
(P2) grader cells (student x dataset x grader, seed means), all graders, fit qwen7 -> test llama8 and reverse (as in the judges explorer).
Model: logistic in one predictor, least squares on the probability scale. Also: are L and W the same quantity (correlations within / across)?
Usage: python3 -I v7_worth.py > out_v7_worth.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np, pandas as pd, warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
HERE = (_RA + "/verify_rate")
T = pd.read_pickle(f"{HERE}/cache/table.pkl"); Q = pickle.load(open(f"{HERE}/cache/qarr.pkl", "rb"))
M = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & np.isfinite(T.L32h)].copy()
sg = lambda z: 1 / (1 + np.exp(-z))
for r in M.itertuples():
    qm = Q[r.run]["q_mean"]; qb = Q[r.run]["q_best"]; c = r.c
    if qm is None: continue
    M.loc[r.Index, "Lq"] = qm.mean(); M.loc[r.Index, "W"] = (qm > c).mean(); M.loc[r.Index, "Wb"] = (qb > c).mean()
    for k in (4, 8, 16, 32):
        M.loc[r.Index, f"Ws{k}"] = sg(k * (qm - c)).mean(); M.loc[r.Index, f"Wbs{k}"] = sg(k * (qb - c)).mean()
M["x"] = M.L32h - M.c
f = lambda x, a, b: sg(a + b * x)
def fit(x, y, sgn): return curve_fit(f, np.asarray(x, float), np.asarray(y, float), p0=[0, 8 * sgn], maxfev=50000)[0]
def r2(y, p): y = np.asarray(y); p = np.asarray(p); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
P = ["x", "Lq", "W", "Wb"] + [f"Ws{k}" for k in (4, 8, 16, 32)] + [f"Wbs{k}" for k in (4, 8, 16, 32)]
sgn = {p: -1 if p == "x" else 1 for p in P}
for p in P: sgn[p] = -1          # all are 'more acceptance -> less abstention'
M5 = M[M.c == 0.5]; U = M5[M5.uni]; QU = U[U.st == "qwen7"]; LU = U[U.st == "llama8"]; J = M5[M5.fam == "judge"]; O = M5[~M5.uni & (M5.fam != "judge")]
print("== (P1) runs; fit on qwen7 uniform (n %d)" % len(QU))
print(f"  {'predictor':8s} {'in-sample':>9s} | {'llama8 uniform':>16s} | {'judges (156)':>22s} | {'other non-uniform':>22s}")
best = None
for p in P:
    b = fit(QU[p], QU.idk_n, sgn[p]); ins = r2(QU.idk_n, f(QU[p], *b))
    s1 = r2(LU.idk_n, f(LU[p], *b)); pj = f(J[p], *b); s2 = r2(J.idk_n, pj); po = f(O[p], *b); s3 = r2(O.idk_n, po)
    print(f"  {p:8s} {ins:9.3f} | R2 {s1:.3f} MAE {np.mean(np.abs(LU.idk_n - f(LU[p], *b))):.3f} | R2 {s2:.3f} MAE {np.mean(np.abs(J.idk_n - pj)):.3f} | R2 {s3:.3f} MAE {np.mean(np.abs(O.idk_n - po)):.3f}")
    if p != "x" and (best is None or ins > best[1]): best = (p, ins)
print(f"  worthiness variant with the best in-sample R2 on qwen7 uniform: {best[0]}")
print("\n== same quantity? correlations over main-grid runs (c 0.5)")
for p in ("Lq", "W", "Wb", "Wbs4", "Wbs8"):
    print(f"  corr(L, {p:5s}) all {np.corrcoef(M5.L32h, M5[p])[0,1]:+.3f}; uniform {np.corrcoef(U.L32h, U[p])[0,1]:+.3f}; judges {np.corrcoef(J.L32h, J[p])[0,1]:+.3f}; "
          f"within judge a35 runs {np.corrcoef(J[J['sub'].str.endswith('a35')].L32h, J[J['sub'].str.endswith('a35')][p])[0,1]:+.3f}")
# (P2) cells
C = M5.groupby(["st", "ds", "sub"]).agg(**{p: (p, "mean") for p in P}, idk_n=("idk_n", "mean"), fam=("fam", "first"), n=("run", "size")).reset_index()
print("\n== (P2) grader cells, all graders (qwen7 %d cells, llama8 %d cells); logistic least squares" % ((C.st == "qwen7").sum(), (C.st == "llama8").sum()))
for p in P:
    out = []
    for a, b_ in (("qwen7", "llama8"), ("llama8", "qwen7")):
        A = C[C.st == a]; B_ = C[C.st == b_]; bb = fit(A[p], A.idk_n, sgn[p]); out.append(r2(B_.idk_n, f(B_[p], *bb)))
    # uniform-only fit -> all cells of the other student
    A = C[(C.st == "qwen7") & C.fam.isin(["gold", "uniFP", "uniFN", "uniMix"])]; bb = fit(A[p], A.idk_n, sgn[p]); B_ = C[C.st == "llama8"]
    print(f"  {p:8s} Q->L {out[0]:+.3f}  L->Q {out[1]:+.3f}  | fit qwen7 uniform cells -> all llama8 cells {r2(B_.idk_n, f(B_[p], *bb)):+.3f}")
# logit-linear OLS (the judges explorer's form) for L and Wbs4/Wbs8
def lg(y): y = np.clip(y, 0.005, 0.995); return np.log(y / (1 - y))
print("  logit-linear OLS form (outcome clipped to [0.005, 0.995]):")
for p in ("x", "W", "Wbs4", "Wbs8"):
    out = []
    for a, b_ in (("qwen7", "llama8"), ("llama8", "qwen7")):
        A = C[C.st == a]; B_ = C[C.st == b_]; bb, aa = np.polyfit(A[p], lg(A.idk_n), 1); out.append(r2(B_.idk_n, sg(aa + bb * B_[p])))
    print(f"    {p:6s} Q->L {out[0]:+.3f}  L->Q {out[1]:+.3f}")
print("\n== residual by family after each predictor's qwen7-uniform fit (runs): mean residual (actual - predicted)")
fams = ["judge", "frz", "api", "long", "hash", "blind", "setFP", "setFN"]
for p in ("x", "W", best[0]):
    b = fit(QU[p], QU.idk_n, sgn[p]); res = M5.idk_n - f(M5[p], *b)
    print(f"  {p:6s} " + "  ".join(f"{fm} {res[M5.fam == fm].mean():+.3f}" for fm in fams))
print("\n== the IDK credit: qwen7 judgeQt50 at c 0.7 / 0.85 (W, Wbs at the run's own credit) vs L - c")
K = M[(M["sub"] == "judgeQt50") & (M.st == "qwen7")]
bx = fit(QU.x, QU.idk_n, -1); bw = fit(QU.W, QU.idk_n, -1); bb = fit(QU[best[0]], QU.idk_n, -1)
for r in K.sort_values(["ds", "c"]).itertuples():
    print(f"  {r.run:38s} c {r.c:.2f} actual {r.idk_n:.3f} | L-c law {f(r.x, *bx):.3f} | W law {f(r.W, *bw):.3f} (W {r.W:.3f}) | {best[0]} law {f(getattr(r, best[0]), *bb):.3f}")
M.to_pickle(f"{HERE}/cache/worth.pkl")
