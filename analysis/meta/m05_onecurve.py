"""(meta) One curve for all graders? Fit the logistic idk_new = s(a + b * premium) on UNIFORM graders (random false accepts / rejects) of the
discovery student, test it on the other student and on every structured family (hash, type, pop, planted rejections, long, blind, frozen keys,
LLM judges, Sonnet). Prints residuals per family with run counts. Input cache/law2.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd, sys
from scipy.optimize import curve_fit
T = pd.read_pickle((_RA + "/meta/cache/law2.pkl"))
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
uni = ["gold", "keyFP", "keyFN", "mixU"]
M = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & np.isfinite(T.prem)]
for y in ("idk_new", "idk_home", "right_new", "wrong_new"):
    print(f"\n==== outcome {y}")
    for disc, conf in (("qwen7", "llama8"), ("llama8", "qwen7")):
        D = M[(M.st == disc) & M.fam.isin(uni)]
        if y.startswith("idk"):
            p, _ = curve_fit(f, D.prem, D[y], p0=[0, -8], maxfev=20000); pred = lambda d: f(d.prem, *p)
        else:   # per-dataset linear in premium for right / wrong (base accuracy differs by dataset)
            co = {ds: np.polyfit(D[D.ds == ds].prem, D[D.ds == ds][y], 2) for ds in D.ds.unique()}
            pred = lambda d: np.array([np.polyval(co[ds], x) for ds, x in zip(d.ds, d.prem)])
        Cf = M[(M.st == conf) & M.fam.isin(uni)]; r = Cf[y] - pred(Cf)
        print(f" fit on {disc} uniform (n {len(D)}): confirm on {conf} uniform n {len(Cf)}: R2 {1 - np.sum(r**2) / np.sum((Cf[y] - Cf[y].mean())**2):.3f} MAE {np.mean(np.abs(r)):.3f} bias {r.mean():+.3f}")
        rows = []
        for fam in sorted(M.fam.unique()):
            if fam in uni: continue
            d = M[M.fam == fam]; r = d[y] - pred(d)
            rows.append(f"{fam}:{len(d)} {r.mean():+.3f}±{r.std() / max(1, np.sqrt(len(d))):.3f}")
        print("   residual (actual - uniform curve) by family, both students:", " | ".join(rows))
    # judges in detail
D = M[M.fam.isin(uni)]
p, _ = curve_fit(f, D.prem, D.idk_new, p0=[0, -8], maxfev=20000)
J = M[M.fam.isin(["judge", "api", "frz", "hash", "typeFP", "popFP", "typeFN", "hidFN", "plFN", "long", "blind"])].copy()
J["pred"] = f(J.prem, *p); J["res"] = J.idk_new - J.pred
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("\nuniform curve fitted on both students: a %.2f b %.2f" % tuple(p))
print(J.groupby(["fam", "sub"]).agg(n=("run", "size"), prem=("prem", "mean"), FA=("FA_home", "mean"), FR=("FR_home", "mean"), idk_new=("idk_new", "mean"), pred=("pred", "mean"), res=("res", "mean"), res_sd=("res", "std")).round(3).to_string())
