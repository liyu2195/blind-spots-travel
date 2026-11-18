"""(arch_when) First-pass learning of planted rejection sets: change in the training-log false-reject rate over the first pass (steps 1-23, before any
question is revisited), per run, from the dynamics explorer's per-run table (dynamics/d04_jump_runs.csv, column fn_rate_e1; NaN = undefined, e.g.
no right answers left). Readable sets: real copies (plreal), type-wide copies (plsharp), visible relation-type set at 80% (fnB80).
Unreadable sets: scrambled copy (plscr), hidden half-of-type sets (fnH80, fnH60). Output: w07_firstpass_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
D = pd.read_csv((_RA + "/dynamics/d04_jump_runs.csv"))
D = D[D.tag == "FK20"]
def grp(g):
    if g.startswith(("plreal", "plsharp")) or g == "fnB80": return "readable"
    if g.startswith("plscr") or g in ("fnH80", "fnH60"): return "unreadable"
    return None
D["grp"] = D.g.map(grp); X = D[D.grp.notna()]
for (gname, st), d in X.groupby(["g", "st"]):
    v = d.fn_rate_e1.dropna().values
    print(f"  {gname:9s} {st:7s} runs {len(d)} with value {len(v)}: " + " ".join(f"{x:+.2f}" for x in v))
for k, d in X.groupby("grp"):
    v = d.fn_rate_e1.dropna().values
    print(f"{k:10s}: runs with value {len(v)}, negative {int((v < 0).sum())}, mean {v.mean():+.3f}, median {np.median(v):+.3f}")
a = X[X.grp == "readable"].fn_rate_e1.dropna(); b = X[X.grp == "unreadable"].fn_rate_e1.dropna()
print(f"Mann-Whitney readable < unreadable: U {mannwhitneyu(a, b, alternative='less').statistic:.0f}, p {mannwhitneyu(a, b, alternative='less').pvalue:.2g}")
for st in ("qwen7", "llama8"):
    a = X[(X.grp == "readable") & (X.st == st)].fn_rate_e1.dropna(); b = X[(X.grp == "unreadable") & (X.st == st)].fn_rate_e1.dropna()
    print(f"  {st}: readable mean {a.mean():+.3f} (n {len(a)}), unreadable mean {b.mean():+.3f} (n {len(b)}), MW p {mannwhitneyu(a, b, alternative='less').pvalue:.2g}")
