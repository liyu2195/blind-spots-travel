"""(meta) Mean-field abstention law: abstention after RL vs the answer premium = (share of the untrained student's valid answers on trained questions that the
run's grader accepts) - (IDK credit). Prints binned curves per student and family, and logistic fits. Input cache/law.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd, sys
from scipy.optimize import curve_fit
T = pd.read_pickle((_RA + "/meta/cache/law.pkl"))
T["L_home"] = (T.n_r_home * (1 - T.FR_home) + T.n_w_home * T.FA_home) / (T.n_r_home + T.n_w_home)
T["L_new"] = (T.n_r_new * (1 - T.FR_new) + T.n_w_new * T.FA_new) / (T.n_r_new + T.n_w_new)
T["prem"] = T.L_home - T.c
T["acc_u"] = T.n_r_home / (T.n_r_home + T.n_w_home)       # untrained strict accuracy among valid answers (home)
std = (T.home_in_eval == 1) & (T.ck == 120) & (T.fmt == "1") & (T.temp == 0.8) & (T.adv == "std") & T.tag.str.startswith("FK") & (T.force == 0) & (T.quorum == 0)
T["std"] = std
T.to_pickle((_RA + "/meta/cache/law2.pkl"))
uni = ["gold", "keyFP", "keyFN", "mixU"]
U = T[std & T.fam.isin(uni)]
def logit_fit(x, y):
    f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
    p, _ = curve_fit(f, x, y, p0=[0, -10], maxfev=20000); return p, f
pd.set_option("display.width", 250)
print("uniform graders, standard runs:", len(U))
bins = np.arange(-0.6, 0.7, 0.1)
U2 = U.assign(b=pd.cut(U.prem, bins))
print(U2.groupby(["b", "st"], observed=True).idk_new.mean().unstack().round(2).to_string())
for nm, d in (("all", U), ("qwen7+llama8", U[U.st.isin(["qwen7", "llama8"])]), ("small", U[~U.st.isin(["qwen7", "llama8"])])):
    d = d[np.isfinite(d.prem) & np.isfinite(d.idk_new)]
    p, f = logit_fit(d.prem.values, d.idk_new.values); pr = f(d.prem.values, *p)
    print(f"{nm:14s} n {len(d):4d} logistic a {p[0]:+.2f} b {p[1]:+.2f}  R2 {1 - np.sum((d.idk_new - pr)**2) / np.sum((d.idk_new - d.idk_new.mean())**2):.3f}  MAE {np.mean(np.abs(d.idk_new - pr)):.3f}")
