"""(fix_ledger) L04: sensitivity of the leniency curve (stats/c4_law.py primary test) to the all-IDK collapsed runs, which c4 keeps.
Input: stats/c4_points.csv (run, st, fam, L, y_new as computed by c4). Same model and fit as c4: y = 1 / (1 + exp(-(a + b (L - 0.5)))), least
squares on the probability scale, fitted on qwen7 uniform-noise runs, scored on llama8 uniform-noise runs (R^2 about their own mean), and on
LLM-judge runs. Variants: all runs (c4 primary, should reproduce R^2 0.851), collapsed runs (y_new >= 0.95) dropped from fit and test.
Output: L04_law_nocollapse_out.txt.  Usage: python3 -I L04_law_nocollapse.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import csv, numpy as np
from scipy.optimize import curve_fit
HERE = (_RA + "/fix_ledger")
rows = list(csv.DictReader(open((_RA + "/stats/c4_points.csv"))))
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
out = open(f"{HERE}/L04_law_nocollapse_out.txt", "w")
def P(s): print(s); out.write(s + "\n")
P(__doc__)
def get(st, fam, drop):
    D = [r for r in rows if r["st"] == st and r["fam"] == fam and not (drop and float(r["y_new"]) >= 0.95)]
    return np.array([float(r["L"]) for r in D]) - 0.5, np.array([float(r["y_new"]) for r in D])
def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
for drop in (False, True):
    x, y = get("qwen7", "uni", drop); (a, b), _ = curve_fit(f, x, y, p0=(-0.8, -8.0), maxfev=20000)
    xl, yl = get("llama8", "uni", drop); xj = np.concatenate([get(s, "judge", drop)[0] for s in ("qwen7", "llama8")])
    yj = np.concatenate([get(s, "judge", drop)[1] for s in ("qwen7", "llama8")])
    P(f"{'collapsed runs dropped' if drop else 'all runs (c4 primary)':24s}: fit qwen7 uniform n {len(y)} (a {a:+.2f}, b {b:+.2f}) -> llama8 uniform n {len(yl)}: "
      f"R2 {r2(yl, f(xl, a, b)):.3f} MAE {np.abs(yl - f(xl, a, b)).mean():.3f} | LLM judges n {len(yj)}: R2 {r2(yj, f(xj, a, b)):.3f}")
    P(f"{'':24s}  collapsed runs in the sets: qwen7 uniform {sum(1 for r in rows if r['st'] == 'qwen7' and r['fam'] == 'uni' and float(r['y_new']) >= 0.95)}, "
      f"llama8 uniform {sum(1 for r in rows if r['st'] == 'llama8' and r['fam'] == 'uni' and float(r['y_new']) >= 0.95)}, "
      f"judges {sum(1 for r in rows if r['fam'] == 'judge' and float(r['y_new']) >= 0.95)}")
out.close()
