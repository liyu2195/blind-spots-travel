"""(stats) Claim 1, alternative explanation check: is the small judge-specific pay lift on NEW questions carried by near-duplicates of trained
questions (TF-IDF cosine >= 0.5 to a trained question, or the same gold answer as a trained question)? Units and S as in c1_content (runs with >= 5 units in the subset; main
definition, from c1_rows.pkl). Crossed bootstrap (1000): questions within (student, dataset), runs within (student, dataset, judge, level).
Output: c1b_neardup_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c1b_neardup_out.txt"); P(__doc__)
rng = np.random.default_rng(111)
D = pickle.load(open(f"{HERE}/c1_rows.pkl", "rb")); U = [u for u in D["units"] if u["variant"] == "base" and u["kind"] == "judge" and u["part"] == "new"]
ND = {}
for st in STS:
    for ds in DSS:
        B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); tr = np.where(B["train"])[0]; items = np.where(B["new"])[0]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(B["question"]); M = vec.transform(B["question"])
        sim = (M[items] @ M[tr].T).toarray().max(1); trg = {B["gold"][i].strip().lower() for i in tr}
        ND[(st, ds)] = (sim >= 0.5) | np.array([B["gold"][i].strip().lower() in trg for i in items])
        P(f"{st} {ds}: near-duplicate new questions {ND[(st, ds)].sum()} / {len(items)}")
def S_of(u, mask, w=None):
    keep = mask[u["qpos"]]
    if w is not None: ww = w[u["qpos"]] * keep
    else: ww = keep.astype(float)
    if keep.sum() < 5 or ww.sum() < 1: return np.nan
    l = (ww[:, None] * u["D"]).sum(0) / ww.sum(); k = J4.index(u["own"]); return l[k] - np.mean([l[j] for j in range(4) if j != k])
def est(which, W=None, sel=None, ds_only=None):
    v = defaultdict(list)
    idx = range(len(U)) if sel is None else sel
    for j in idx:
        u = U[j]
        if ds_only and u["ds"] != ds_only: continue
        m = ND[(u["st"], u["ds"])] if which == "dup" else ~ND[(u["st"], u["ds"])]
        s = S_of(u, m, None if W is None else W[(u["st"], u["ds"])])
        if np.isfinite(s): v[u["st"]].append(s)
    out = {st: np.mean(v[st]) if v[st] else np.nan for st in STS}; out["pooled"] = np.nanmean([out[st] for st in STS]); return out
cells = defaultdict(list)
for j, u in enumerate(U): cells[u["cell"]].append(j)
for ds_only in (None, "pq"):
    for which in ("dup", "nodup"):
        e = est(which, ds_only=ds_only); reps = []
        for b in range(1000):
            W = {k: np.bincount(rng.integers(0, len(v), len(v)), minlength=len(v)).astype(float) for k, v in ND.items()}
            sel = [L[t] for L in cells.values() for t in rng.integers(0, len(L), len(L))]
            reps.append(est(which, W, sel, ds_only))
        P(f"  {'all datasets' if ds_only is None else 'PopQA only':12s} {which:6s}: " + "  ".join(f"{who} {e[who]:+.3f} [{pct([r[who] for r in reps])[0]:+.3f}, {pct([r[who] for r in reps])[1]:+.3f}]" for who in ("qwen7", "llama8", "pooled")))
P.save()
