"""(meta) Key numbers with intervals for the report. Inputs: cache/refusal.pkl (all per-run measures), cache/transfer_points.csv.
1. Leniency law (uniform graders, fit qwen7 -> confirm llama8): R2 with bootstrap CI; within-(student, dataset) R2; judges' MAE / bias with CI.
2. Variance decomposition of abstention on new questions over ALL standard runs of the main grid.
3. Matched pair judgeQt50 vs lenQ (random noise at the judge's rates): differences per pair and seed-bootstrap CI.
4. Visits per question (TriviaQA, judgeQt50 / hash35 / judgeQa35): training-time paid share vs eval paid share.
5. FR transfer law: Spearman with bootstrap CI over points."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
OUT = (_RA + "/meta/cache")
T = pd.read_pickle(f"{OUT}/refusal.pkl")
rng = np.random.default_rng(7)
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
uni = ["gold", "keyFP", "keyFN", "mixU"]
M = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & (T.kl == 0.2) & (T.c == 0.5) & np.isfinite(T.prem) & np.isfinite(T.idk_new)].copy()
def r2(y, p): y = np.asarray(y); p = np.asarray(p); return 1 - np.sum((y - p)**2) / np.sum((y - y.mean())**2)
print("== 1. leniency law")
A = M[(M.st == "qwen7") & M.fam.isin(uni)]; B = M[(M.st == "llama8") & M.fam.isin(uni)]
p, _ = curve_fit(f, A.prem, A.idk_new, p0=[0, -8]); pb = f(B.prem, *p)
bs = [r2(B.idk_new.values[i], pb.values[i]) for i in (rng.integers(0, len(B), len(B)) for _ in range(2000))]
print(f"  fit on qwen7 uniform (n {len(A)}): a {p[0]:+.2f} b {p[1]:+.2f}; llama8 uniform (n {len(B)}): R2 {r2(B.idk_new, pb):.3f} [95% {np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
# within-pair: remove (st, ds) means from actual and predicted
for nm, d in (("qwen7 (in-sample)", A), ("llama8 (out-of-sample)", B)):
    pr = f(d.prem, *p); da = d.idk_new - d.groupby("ds").idk_new.transform("mean"); dp = pr - pr.groupby(d.ds).transform("mean")
    print(f"  within-dataset variation, {nm}: R2 {r2(da, dp):.3f} (corr {np.corrcoef(da, dp)[0, 1]:+.3f}, n {len(d)})")
Pall = M[M.fam.isin(uni)]; p2, _ = curve_fit(f, Pall.prem, Pall.idk_new, p0=[0, -8])
J = M[M.fam == "judge"]; pj = f(J.prem, *p); res = J.idk_new - pj
bs = [(np.mean(np.abs(res.values[i])), res.values[i].mean()) for i in (rng.integers(0, len(J), len(J)) for _ in range(2000))]
print(f"  LLM judges (n {len(J)}, never used in the fit; qwen7-fitted curve): R2 {r2(J.idk_new, pj):.3f}, MAE {np.mean(np.abs(res)):.3f} [{np.percentile([b[0] for b in bs], 2.5):.3f}, {np.percentile([b[0] for b in bs], 97.5):.3f}], bias {res.mean():+.3f} [{np.percentile([b[1] for b in bs], 2.5):+.3f}, {np.percentile([b[1] for b in bs], 97.5):+.3f}]")
print("== 2. what explains abstention on new questions across ALL standard runs of the main grid (n %d)" % len(M))
M["pred_len"] = f(M.prem, *p2)
def ols_r2(X, y):
    X = np.column_stack([np.ones(len(y))] + X); b, *_ = np.linalg.lstsq(X, y, rcond=None); return r2(y, X @ b)
dummies = lambda s: [ (s == v).astype(float).values for v in sorted(s.unique())[1:] ]
y = M.idk_new.values
cands = {"leniency curve": [M.pred_len.values], "student x dataset": dummies(M.st + M.ds), "grader family": dummies(M.fam), "FA alone": [M.FA_home.values, M.FA_home.values**2],
         "FR alone": [M.FR_home.values, M.FR_home.values**2], "seed": dummies(M.seed.astype(str))}
for k, X in cands.items(): print(f"  {k:20s} R2 {ols_r2(X, y):.3f}")
print(f"  leniency + student x dataset    R2 {ols_r2(cands['leniency curve'] + cands['student x dataset'], y):.3f}")
print(f"  leniency + grader family        R2 {ols_r2(cands['leniency curve'] + cands['grader family'], y):.3f}")
print(f"  leniency + family + st x ds     R2 {ols_r2(cands['leniency curve'] + cands['grader family'] + cands['student x dataset'], y):.3f}")
print("== 3. matched pair: judgeQt50 vs lenQ (random accepts / rejects at that judge's rates)")
cols = ["idk_new", "right_new", "wrong_new", "x_Qt50_w_new", "idk_home", "right_home", "x_Qt50_w_home"]
rows = []
for (st, ds), d in M[M["sub"].isin(["judgeQt50", "lenQ"])].groupby(["st", "ds"]):
    a = d[d["sub"] == "judgeQt50"]; b = d[d["sub"] == "lenQ"]
    rows.append(dict(st=st, ds=ds, na=len(a), nb=len(b), **{c: a[c].mean() - b[c].mean() for c in cols}))
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
for c in cols:
    v = R[c].values; bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)]
    print(f"  mean diff {c:14s} {v.mean():+.3f}  [95% over pairs {np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
print("== 4. visits per question (TriviaQA)")
V = T[T.st.isin(["qwen7", "llama8"]) & (T.ds == "h1") & T["sub"].isin(["judgeQt50", "hash35fn00", "judgeQa35", "gold"]) & (T.kl == 0.2) & (T.c == 0.5) & (T.force == 0) & (T.quorum == 0)]
V = V[V.tag.isin(["FK20", "B1", "B2", "B5", "B10"])]
print(V.groupby(["sub", "st", "visits"]).agg(n=("run", "size"), paid_train_start=("wrong_paid_first", "mean"), paid_train_end=("wrong_paid_last", "mean"),
      paid_eval_new=("paid_wrong_new", "mean"), paid_eval_untrained=("paid_wrong_u_new", "mean"), hack_new=("hack_new", "mean")).round(3).to_string())
print("== 5. FR transfer law")
P = pd.read_csv(f"{OUT}/transfer_points.csv"); P = P[np.isfinite(P.gini)]
for eff in ("FR refusal", "FA hack"):
    d = P[P.effect == eff]; s0 = spearmanr(d.gini, d.transfer)[0]
    bs = [spearmanr(d.gini.values[i], d.transfer.values[i])[0] for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
    print(f"  {eff}: n {len(d)} Spearman {s0:+.3f} [95% {np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    for nm, g in d.groupby(pd.cut(d.gini, [-0.2, 0.15, 0.45, 0.8, 1.01])):
        if len(g): print(f"     Gini {str(nm):14s} n {len(g):2d} median transfer {g.transfer.median():+.2f}  (families: {', '.join(sorted(g.fam.unique()))})")
