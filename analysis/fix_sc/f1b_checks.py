"""fix_sc f1b: two checks on f1_rows.pkl (FK20, IDK credit 0.5, new questions).
(1) audit gap: share of RL answers that are key-wrong AND have no Sonnet verdict (counted wrong in the main analysis), per family x op x
    student and per dataset where it exceeds 5%; the bracket [main, 'hi'] for RL - SC_32.
(2) SC resolution: share of new questions whose greedy answer is matched by all 32 samples (conf = 1) and by >= 31/32, per student x
    dataset, against the coverage of each operating point (if coverage < the conf = 1 share, SC_32 picks at random among tied questions).
Output: f1b_checks.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/fix_sc")); sys.path.insert(0, (_RA + "/selfcons"))
from f1_scm import ROWS, rows_of
OUT = (_RA + "/fix_sc"); SCD = (_RA + "/selfcons")
X = pickle.load(open(f"{OUT}/f1_rows.pkl", "rb")); rows = [d for d in X["rows"] if d["tag"] == "FK20" and d["credit"] == "idkc05"]
out = [__doc__]
for d in rows: d["missshare"] = d["accRL_hi"] - d["accRL"]
for st in ("qwen7", "llama8"):
    out.append(f"\n=== {st}: audit gap (share of RL answers key-wrong without Sonnet verdict), RL - SC_32 main / hi bound")
    for rn in ROWS:
        L = [d for d in rows if d["st"] == st and rn in rows_of(d["f2"])]
        if not L: continue
        ms = np.array([d["missshare"] for d in L])
        bad = sorted({f"{d['ds']}:{np.mean([x['missshare'] for x in L if x['ds'] == d['ds']]):.2f}" for d in L if d["missshare"] > 0.05})
        out.append(f"   {rn:36s} runs {len(L):3d} miss mean {ms.mean():.3f} max {ms.max():.3f} | main {100*np.mean([d['lb32'] for d in L]):+5.1f} hi {100*np.mean([d['hi32'] for d in L]):+5.1f}"
                   + (f" | datasets with mean miss > 5%: {bad}" if bad else ""))
out.append("\n=== SC resolution: share of new questions with all 32 samples agreeing with the greedy answer (conf = 1) / >= 31 of 32; mean coverage of key, Q t50 / a50 / a35 runs")
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        Q = pickle.load(open(f"{SCD}/cache_{st}_{ds}.pkl", "rb"))["Q"]; ho = Q["ho"]; c = Q["conf"][ho]
        cv = {rn: np.mean([d["cov"] for d in rows if d["st"] == st and d["ds"] == ds and d["f2"] == rn] or [np.nan]) for rn in ("key", "judge Q t50", "judge Q a50", "judge Q a35", "Sonnet")}
        out.append(f"   {st} {ds}: conf = 1 {np.mean(c >= 1 - 1e-9):.3f}, >= 31/32 {np.mean(c >= 31/32 - 1e-9):.3f}, greedy answers {Q['gval'][ho].mean():.3f} | coverage " + " ".join(f"{k} {v:.2f}" for k, v in cv.items()))
open(f"{OUT}/f1b_checks.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
