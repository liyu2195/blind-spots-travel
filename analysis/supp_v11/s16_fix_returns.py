"""(supp_v11) s16_fix_returns (Nov 17, existing runs, no training): what the graders that were built as FIXES earn on held-out questions of the three datasets, next
to the answer key, the judges and the judge-free graders. Same measure as s5_return_by_cut.py: reward 1 / 0 / 0.5 (IDK) with truth = loose key or
audit level >= 4 (pooled audit, aud.py); runs that answer IDK on >= 95% of held-out questions dropped; FK20, credit 0.5.
Fix arms (frozen keys built from the UNTRAINED student's 32 samples; work/better, work/fix):
  frzVOTE<k>  own vote: the key is the student's majority answer if >= k of 32 samples give it, else only IDK is paid
  frzOR<k>    voucher: as the vote, plus the majority answer is also kept if >= 4 samples give it and the Qwen judge accepts it at the 35% cut
  frzPEER1    peer veto: the vote key at a low k, emptied where the other student's own answer disagrees
  frzQ50k<k>  judge veto: the vote key at k, emptied where the Qwen judge rejects the answer at the 50% cut
  frzS55k<k>  Sonnet veto: the same with Sonnet's verdict
Also printed per arm: share of held-out answers the untrained student knew (right greedy answer) that the trained model refuses ('known refused').
Usage: python3 -I -B s16_fix_returns.py   Output: s16_fix_returns_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_common import *
import aud as audlib
out = [__doc__]
for st, ds, dsn in [(s_, d_, n_) for s_ in ("qwen7", "llama8") for d_, n_ in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo"))]:
    Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn)
    new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]
    gval = np.asarray(Bp["valid"])[:, 0].astype(bool) & ~np.asarray(Bp["idk"])[:, 0].astype(bool)
    gright = gval & (np.asarray(Bp["kl"])[:, 0].astype(bool) | (np.asarray(Bp["son"])[:, 0] >= 4)); gidk = np.asarray(Bp["idk"])[:, 0].astype(bool)
    out.append(f"\n== {st} {dsn} held-out {len(ii)}: untrained right {gright[ii].mean():.3f}, IDK {gidk[ii].mean():.3f}, return {gright[ii].mean() + 0.5 * gidk[ii].mean():.3f}")
    G = defaultdict(list)
    for name, r in R.items():
        if r["tag"] != "FK20" or r["c"] != "05": continue
        o = np.asarray(r["o"])[ii]; idk = (o == 0)
        if idk.mean() >= 0.95: continue
        right = np.array([(r["o"][i] == 1) or (r["o"][i] == 2 and LV.get((i, r["f"][i]), 0) >= 4) for i in ii])
        wrong = ((o == 2) | (o == 3)).mean() - (right & (o == 2)).mean()
        G[r["g"]].append((right.mean(), wrong, idk.mean(), right.mean() + 0.5 * idk.mean(), float(idk[gright[ii]].mean())))
    order = ["sim00fn00"] + sorted(k for k in G if k.startswith("frz")) + ["judgeQt50", "judgeQa50", "judgeQa35", "judgeMa50", "judgeMa35", "judgeQRt50", "apiS55"]
    out.append("   arm           runs | right  wrong  IDK   | return (range over runs) | known refused")
    for g in order:
        if g not in G: continue
        a = np.mean(G[g], axis=0)
        out.append(f"   {g:12s}  {len(G[g]):3d} | {a[0]:.3f}  {a[1]:.3f}  {a[2]:.3f} | {a[3]:.3f} ({min(x[3] for x in G[g]):.3f}-{max(x[3] for x in G[g]):.3f}) | {a[4]:.3f}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s16_fix_returns_out.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out[1:]))
