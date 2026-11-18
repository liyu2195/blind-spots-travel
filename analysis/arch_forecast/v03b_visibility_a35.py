"""(arch_forecast) Visibility law among judges at ONE acceptance target (a35 only), within (student, dataset): does readability rank judges when strictness is
held fixed?  Also: threshold level vs visibility (does strictness raise readability?).  Input meta cache/transfer_points_hidden.csv.  Output: v03b_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd
rng = np.random.default_rng(3)
P = pd.read_csv((_RA + "/meta/cache/transfer_points_hidden.csv"))
J = P[(P.fam == "judge") & np.isfinite(P.gini_hidden)].copy(); J["level"] = J["sub"].str[-3:]
print(J.groupby("level").agg(n=("sub", "size"), gini_hidden=("gini_hidden", "mean"), home=("home", "mean"), new=("new", "mean"), transfer=("transfer", "mean")).round(3).to_string())
A = J[J.level == "a35"]
rk = lambda v: pd.Series(v).rank().values
G = [g for _, g in A.groupby(["st", "ds"]) if len(g) >= 3]
def stat(perm=False, y="transfer"):
    s, w = [], []
    for g in G:
        x = g.gini_hidden.values; x = rng.permutation(x) if perm else x
        s.append(np.corrcoef(rk(x), rk(g[y].values))[0, 1]); w.append(len(g))
    return np.average(s, weights=w), s
for y in ("transfer", "new"):
    o, per = stat(y=y); null = np.array([stat(True, y)[0] for _ in range(3000)])
    print(f"a35 only, y = {y:8s}: groups {len(G)} sizes {[len(g) for g in G]}  weighted Spearman {o:+.3f}  per group {[round(float(v), 2) for v in per]}  one-sided perm p {np.mean(null >= o):.3f}")
for g in G: print("  ", g.st.iloc[0], g.ds.iloc[0], g[["sub", "gini_hidden", "home", "new", "transfer"]].round(2).to_dict("records"))
