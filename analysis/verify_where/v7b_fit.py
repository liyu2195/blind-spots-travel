"""(verify_where) Item 5 supplement: transfer = k * Gini through the origin, fitted on qwen7 points, scored on llama8 points (claimed
k = 0.79, R2 0.61 with the hidden-state Gini); also judges-only. Input v7_points.csv (linear reference). Output v7b_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd
D = pd.read_csv((_RA + "/verify_where/v7_points.csv")); out = []
for lab, d in (("all", D), ("judges only", D[D.fam == "judge"])):
    for col in ("gini_text", "gini_hid"):
        d2 = d[np.isfinite(d.T_lin) & np.isfinite(d[col])]; a = d2[d2.st == "qwen7"]; b = d2[d2.st == "llama8"]
        k = np.sum(a[col] * a.T_lin) / np.sum(a[col] ** 2); r = b.T_lin - k * b[col]
        out.append(f"{lab:12s} {col:9s}: k (qwen7, n {len(a)}) {k:.3f} -> llama8 (n {len(b)}) R2 {1 - np.sum(r ** 2) / np.sum((b.T_lin - b.T_lin.mean()) ** 2):+.3f}, MAE {np.mean(np.abs(r)):.3f}")
print("\n".join(out)); open((_RA + "/verify_where/v7b_out.txt"), "w").write(__doc__ + "\n" + "\n".join(out) + "\n")
