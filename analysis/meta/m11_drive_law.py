"""(meta) Does the GRPO-aware IDK drive (m10) explain abstention, including the IDK-credit sweeps and the unnormalised-advantage runs where the
plain premium fails? Logistic idk_new = s(a + b * x) for x in (premium, drive); discovery = qwen7 + llama8 at c = 0.5 (KL 0.2, std);
confirmation = small students (phi4m, smol3, granite), all IDK credits, std and none advantages, KL 0.2. Input cache/drive.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
T = pd.read_pickle((_RA + "/meta/cache/drive.pkl"))
uni = ["gold", "keyFP", "keyFN", "mixU"]
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
base = (T.home_in_eval == 1) & (T.ck == 120) & (T.fmt == "1") & (T.temp == 0.8) & T.tag.str.startswith(("FK", "F")) & (T.force == 0) & (T.quorum == 0) & (T.steps.isin([120]) | T.tag.str.startswith("FK"))
M = T[base & np.isfinite(T.idk_new) & np.isfinite(T.D_std)].copy()
def r2(y, p): return 1 - np.sum((y - p)**2) / np.sum((y - y.mean())**2)
disc = M[M.st.isin(["qwen7", "llama8"]) & (M.kl == 0.2) & (M.c == 0.5) & (M.adv == "std") & M.fam.isin(uni)]
print("discovery: qwen7 + llama8 uniform graders, c 0.5, KL 0.2, std advantages:", len(disc))
for x in ("prem", "D_used"):
    p, _ = curve_fit(f, disc[x], disc.idk_new, p0=[0, 1 if x != "prem" else -8], maxfev=50000)
    print(f"\n  x = {x}: a {p[0]:+.3f} b {p[1]:+.3f}; in-sample R2 {r2(disc.idk_new, f(disc[x], *p)):.3f}")
    for nm, d in (("llama8 judges (c .5)", M[(M.st == "llama8") & (M.fam == "judge") & (M.c == 0.5)]),
                  ("qwen7 judges (c .5)", M[(M.st == "qwen7") & (M.fam == "judge") & (M.c == 0.5)]),
                  ("qwen7 judgeQt50 c .7/.85", M[(M.st == "qwen7") & (M["sub"] == "judgeQt50") & (M.c > 0.5)]),
                  ("small, uniform, std, c = .5, KL .2", M[~M.st.isin(["qwen7", "llama8"]) & M.fam.isin(uni) & (M.adv == "std") & (M.c == 0.5) & (M.kl == 0.2)]),
                  ("small, uniform, std, c != .5, KL .2", M[~M.st.isin(["qwen7", "llama8"]) & M.fam.isin(uni) & (M.adv == "std") & (M.c != 0.5) & (M.kl == 0.2)]),
                  ("small, uniform, adv none, KL .2", M[~M.st.isin(["qwen7", "llama8"]) & M.fam.isin(uni) & (M.adv == "none") & (M.kl == 0.2)])):
        if len(d) < 2: print(f"    {nm}: n {len(d)}"); continue
        pr = f(d[x], *p); res = d.idk_new - pr
        print(f"    {nm:38s} n {len(d):3d}  R2 {r2(d.idk_new, pr):+.3f}  MAE {np.mean(np.abs(res)):.3f}  bias {res.mean():+.3f}  corr {np.corrcoef(d[x], d.idk_new)[0, 1]:+.3f}")
# the IDK-credit sweep in detail
S = M[~M.st.isin(["qwen7", "llama8"]) & M.fam.isin(uni) & (M.kl == 0.2) & M["sub"].isin(["gold", "FP35"]) & (M.from_name == 0)]
p, _ = curve_fit(f, disc["D_used"], disc.idk_new, p0=[0, 1], maxfev=50000)
pd.set_option("display.width", 250)
print("\nIDK-credit sweeps (runs with a queue config): actual vs drive-curve prediction")
print(S.assign(pred=f(S.D_used, *p)).groupby(["st", "sub", "adv", "c"]).agg(n=("run", "size"), D_std=("D_std", "mean"), D_none=("D_none", "mean"), prem=("prem", "mean"), idk_new=("idk_new", "mean"), pred=("pred", "mean")).round(3).to_string())
# slope of abstention on c within sweeps, std vs none
for adv in ("std", "none"):
    d = S[S.adv == adv]
    if len(d) > 3:
        for (st, sub), g in d.groupby(["st", "sub"]):
            if g.c.nunique() >= 3:
                b = np.polyfit(g.c, g.idk_new, 1)[0]; bd = np.polyfit(g.c, g.D_used, 1)[0]
                print(f"  {adv:4s} {st} {sub:5s}: d idk / d c = {b:+.2f} per unit credit (n {len(g)}, credits {sorted(g.c.unique())}); d drive / d c = {bd:+.2f}")
