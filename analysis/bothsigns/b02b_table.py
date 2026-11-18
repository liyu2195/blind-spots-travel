"""(bothsigns) b02b: arm table from b02's units and bootstrap draws. An arm = (grader arm, sign, student), pooled over datasets: the arm effect is
the mean of in-support runs' effects (each run's effect computed within its own dataset against that dataset's reference runs).
T = E_new / E_home, D = E_new - E_home, for both currencies (dec, kind); percentile 95% CIs from the crossed bootstrap.
Writes b02_arms.csv and cache/b02_armboot.pkl (per arm: bootstrap draws). Usage: python3 b02b_table.py > b02b_table_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, (_RA + "/bothsigns"))
from b00_common import OUT

D = pickle.load(open(f"{OUT}/cache/b02_boot.pkl", "rb")); U = D["units"]; B = D["boots"]   # dict: "<dec|kind>_<mode>" -> draws x units x part; "<dec|kind>_ok" -> in-support run counts
groups = defaultdict(list)
for k, u in enumerate(U):
    groups[(u["arm"], u["sign"], u["st"])].append(k)


def pool(vals, wts):
    vals = np.asarray(vals, float); wts = np.asarray(wts, float); m = np.isfinite(vals) & (wts > 0)
    return float(np.sum(vals[m] * wts[m]) / np.sum(wts[m])) if m.any() else np.nan


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) > 20 else (np.nan, np.nan)


MODES = ("logit", "lin", "knn")
rows = []; armboot = {}
for (arm, sign, st), ks in sorted(groups.items()):
    r = dict(arm=arm, sign=sign, st=st, cells="/".join(sorted({U[k]["ds"] for k in ks})), runs=sum(len(U[k]["runs"]) for k in ks))
    bb = {}
    for ms in ("dec", "kind", "kindc"):
        for m in MODES:
            key = f"{ms}_{m}"; name = ms if m == "logit" else key
            for p, tag in ((0, "h"), (1, "n")):
                pt = pool([U[k]["pt"][p][key] for k in ks], [U[k]["pt"][p][f"{ms}_ok"] for k in ks])
                bs = np.array([pool(B[key][b, ks, p], B[f"{ms}_ok"][b, ks, p]) for b in range(B[key].shape[0])])
                r[f"{name}_{tag}"] = pt; r[f"{name}_{tag}_lo"], r[f"{name}_{tag}_hi"] = ci(bs); bb[f"{name}_{tag}"] = bs
            with np.errstate(divide="ignore", invalid="ignore"):
                r[f"{name}_T"] = r[f"{name}_n"] / r[f"{name}_h"] if abs(r[f"{name}_h"]) > 1e-9 else np.nan
                r[f"{name}_T_lo"], r[f"{name}_T_hi"] = ci(bb[f"{name}_n"] / bb[f"{name}_h"])
            r[f"{name}_D"] = r[f"{name}_n"] - r[f"{name}_h"]; r[f"{name}_D_lo"], r[f"{name}_D_hi"] = ci(bb[f"{name}_n"] - bb[f"{name}_h"])
        for p, tag in ((0, "h"), (1, "n")):
            r[f"{ms}_ok_{tag}"] = int(sum(U[k]["pt"][p][f"{ms}_ok"] for k in ks))
    for p, tag in ((0, "h"), (1, "n")):
        r[f"nAd_{tag}"] = float(np.nanmean([U[k]["pt"][p].get("nAd", np.nan) for k in ks])); r[f"nPk_{tag}"] = float(np.nanmean([U[k]["pt"][p].get("nPk", np.nan) for k in ks]))
        for key in ("kind_s", "kind_sref", "kind_z", "kind_zref", "kindc_s", "kindc_sref", "dec_c", "dec_z", "dec_cref", "dec_zref"):
            r[f"{key}_{tag}"] = float(np.nanmean([U[k]["pt"][p].get(key, np.nan) for k in ks]))
        r[f"coll_{tag}"] = int(sum(U[k]["pt"][p].get("n_coll", 0) for k in ks))
    armboot[(arm, sign, st)] = bb
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/b02_arms.csv", index=False)
pickle.dump(armboot, open(f"{OUT}/cache/b02_armboot.pkl", "wb"))
f = lambda a, lo, hi: f"{a:+.3f} [{lo:+.3f},{hi:+.3f}]" if np.isfinite(a) else "   n/a                "
g = lambda a, lo, hi: f"{a:+.2f} [{lo:+.2f},{hi:+.2f}]" if np.isfinite(a) else "  n/a             "
print("E_dec: extra refusal (FR) / extra answering (FA) on items whose greedy answer the grader errs on, vs unaffected items, vs matched reference runs.")
print("E_kind: excess share of the run's refusals (FR) / wrong answers (FA, form) that are of the kind the grader's error rewards, vs matched reference.")
print("T = new / home; D = new - home. ok = in-support runs (home/new).\n")
for st in ("qwen7", "llama8"):
    print(f"===== {st}")
    for _, r in T[T.st == st].iterrows():
        print(f"{r.arm:18s} {r.sign:9s} {r.cells:8s} runs {r.runs:3d} coll {r.coll_h:2d} rate {r.kind_z_h:.2f} | dec ok {r.dec_ok_h:3d}/{r.dec_ok_n:3d} home {f(r.dec_h, r.dec_h_lo, r.dec_h_hi)} new {f(r.dec_n, r.dec_n_lo, r.dec_n_hi)} "
              f"T {g(r.dec_T, r.dec_T_lo, r.dec_T_hi)} | kind ok {r.kind_ok_h:3d}/{r.kind_ok_n:3d} home {f(r.kind_h, r.kind_h_lo, r.kind_h_hi)} new {f(r.kind_n, r.kind_n_lo, r.kind_n_hi)} T {g(r.kind_T, r.kind_T_lo, r.kind_T_hi)}"
              + (f" | content home {f(r.kindc_h, r.kindc_h_lo, r.kindc_h_hi)} new {f(r.kindc_n, r.kindc_n_lo, r.kindc_n_hi)}" if r.sign not in ("FR", "FRplacebo") else ""))
