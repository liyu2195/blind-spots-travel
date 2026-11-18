"""(verify_rate) Does the leniency curve (qwen7 uniform, KL 0.2, 120 steps, c 0.5) carry over to other recipes? Runs with KL 0 / 0.5 / 1.0 or 360 steps
(qwen7 / llama8, strict format, standard training set, std advantages, c 0.5). Usage: python3 -I v10_recipe.py > out_v10_recipe.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd, warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
T = pd.read_pickle((_RA + "/verify_rate/cache/table.pkl"))
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
M = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & (T.c == 0.5)]; QU = M[M.uni & (M.st == "qwen7")]
p = curve_fit(f, QU.L32h.values - 0.5, QU.idk_n.values, p0=[0, -8])[0]
D = T[T.st.isin(["qwen7", "llama8"]) & (T.fmt == "1") & T.home_std & (T.force == 0) & (T.c == 0.5) & (T.adv == "std") & ((T.kl != 0.2) | (T.steps != 120)) & (T.other_keys.apply(len) == 0)]
D = D.assign(pred=f(D.L32h.values - 0.5, *p)); D["res"] = D.idk_n - D.pred
print(D.groupby(["kl", "steps", "fam"]).agg(n=("run", "size"), cells=("ds", "nunique"), abst=("idk_n", "mean"), pred=("pred", "mean"), res=("res", "mean")).round(3).to_string())
