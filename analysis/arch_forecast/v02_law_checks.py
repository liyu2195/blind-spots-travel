"""(arch_forecast) Checks of the leniency law (meta m14) from the meta cache, aimed at the statistician's objections:
(1) error mix of judges at the same leniency (FA vs FR); (2) per-level bias of the simulated-grader curve for judges, with run-bootstrap CIs, and R2 on the
32 default-threshold judge runs whose leniency varies naturally; (3) how many calibration runs are needed: fit the curve on k random qwen7 uniform runs
(k = 4, 8, 16, 32; 500 draws) and score the llama8 uniform runs and all judge runs; (4) cell-level (student x dataset x grader) version of (2).
Read-only on meta/cache/refusal.pkl. Output: v02_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
rng = np.random.default_rng(11)
T = pd.read_pickle((_RA + "/meta/cache/refusal.pkl"))
T["L"] = (T.n_r_home * (1 - T.FR_home) + T.n_w_home * T.FA_home) / (T.n_r_home + T.n_w_home)
M = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & (T.kl == 0.2) & (T.c == 0.5) & np.isfinite(T.L) & np.isfinite(T.idk_new)].copy()
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * (x - 0.5))))
def r2(y, p): y = np.asarray(y); p = np.asarray(p); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
uni = ["gold", "keyFP", "keyFN", "mixU"]
A = M[(M.st == "qwen7") & M.fam.isin(uni)]; B = M[(M.st == "llama8") & M.fam.isin(uni)]; J = M[M.fam == "judge"].copy()
p, _ = curve_fit(f, A.L, A.idk_new, p0=[0, -8])
print(f"(0) qwen7 uniform fit n {len(A)}: a {p[0]:+.3f} b {p[1]:+.3f}; llama8 uniform R2 {r2(B.idk_new, f(B.L, *p)):.3f}; judges R2 {r2(J.idk_new, f(J.L, *p)):.3f}")
J["level"] = J["sub"].str[-3:]; J["pred"] = f(J.L, *p); J["res"] = J.idk_new - J.pred
print("(1) error mix at the same leniency (home, untrained samples): FA = share of wrong answers paid, FR = share of right answers rejected")
print(J.groupby("sub").agg(n=("run", "size"), L=("L", "mean"), FA=("FA_home", "mean"), FR=("FR_home", "mean"), idk_new=("idk_new", "mean"), pred=("pred", "mean")).round(3).to_string())
print("(2) per-level residual (observed - predicted abstention on new questions), bootstrap over runs")
for lv, d in J.groupby("level"):
    v = d.res.values; bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)]
    extra = f"  R2 within level {r2(d.idk_new, d.pred):+.3f}" if lv == "t50" else ""
    print(f"   {lv}: runs {len(d):3d}  L {d.L.mean():.3f}  observed {d.idk_new.mean():.3f}  predicted {d.pred.mean():.3f}  residual {v.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  |res| {np.abs(v).mean():.3f}{extra}")
d = J[J.level == "t50"]; print(f"   t50 runs: corr(L, idk_new) {np.corrcoef(d.L, d.idk_new)[0, 1]:+.3f}, Spearman-like rank corr {pd.Series(d.L.values).rank().corr(pd.Series(d.idk_new.values).rank()):+.3f}")
print("(3) calibration budget: fit on k random qwen7 uniform runs, score llama8 uniform and judges (median [5th, 95th pct] over 500 draws)")
for k in (4, 8, 16, 32, 64):
    rb, rj, mj = [], [], []
    for _ in range(500):
        S = A.iloc[rng.choice(len(A), k, replace=False)]
        try: q, _ = curve_fit(f, S.L, S.idk_new, p0=[0, -8], maxfev=20000)
        except Exception: continue
        rb.append(r2(B.idk_new, f(B.L, *q))); rj.append(r2(J.idk_new, f(J.L, *q))); mj.append(np.abs(J.idk_new - f(J.L, *q)).mean())
    pc = lambda v: f"{np.median(v):+.3f} [{np.percentile(v, 5):+.3f}, {np.percentile(v, 95):+.3f}]"
    print(f"   k {k:3d}: llama8 uniform R2 {pc(rb)} | judges R2 {pc(rj)} | judges MAE {pc(mj)}  (fits {len(rb)})")
print("(4) cells (student x dataset x judge code): mean observed vs predicted")
C = J.groupby(["st", "ds", "sub"]).agg(n=("run", "size"), L=("L", "mean"), obs=("idk_new", "mean"), pred=("pred", "mean")).reset_index()
print(f"   cells {len(C)}: R2 {r2(C.obs, C.pred):.3f}, MAE {np.abs(C.obs - C.pred).mean():.3f}, |res| > 0.15 in {(np.abs(C.obs - C.pred) > 0.15).sum()} cells")
print(C.assign(res=C.obs - C.pred).sort_values("res").round(3).to_string(index=False))
