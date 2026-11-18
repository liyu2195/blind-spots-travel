"""(knowledge) Splits and matched-abstention accuracy.
(a) judge fingerprint (k6 probes, 5 judges at the 35% target): own-cell residual and permutation p separately per student and per dataset.
(b) seed split for the Qwen-judge families (6 seeds each): forecast AUROC of the own probe and own > key, seeds 0-2 vs 3-5 (k6 rows).
(c) accuracy among answered at matched abstention (|da| <= 0.05): judge-trained minus gold-family (gold + keyFN) runs, audited and key label,
    with the untrained-answer version (same selection, untrained greedy answers) to separate selection from answer change (k7 rows / runs).
Output: k14_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
from collections import defaultdict
OUT = (_RA + "/knowledge")
lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
k6 = pickle.load(open(f"{OUT}/k6_rows.pkl", "rb")); J5 = ["Q35", "M35", "G35", "O35", "R35"]; rng = np.random.default_rng(0)
def fp(sel):
    eff = []; null = np.zeros(2000)
    for st in ("qwen7", "llama8"):
        for ds in ("h1", "pq", "nq"):
            L = [d for d in k6 if d["st"] == st and d["ds"] == ds and d["own"] in J5 and sel(d)]
            if len({d["own"] for d in L}) < 2: continue
            M = np.array([[d[f"p_{j}"] for j in J5] for d in L]); ow = np.array([J5.index(d["own"]) for d in L])
            Rm = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean(); eff.extend(Rm[np.arange(len(L)), ow])
            for b in range(2000): null[b] += Rm[np.arange(len(L)), rng.permutation(ow)].sum()
    return np.mean(eff), len(eff), (np.sum(null >= np.sum(eff)) + 1) / 2001
P("(a) fingerprint by split")
for lab, sel in (("qwen7", lambda d: d["st"] == "qwen7"), ("llama8", lambda d: d["st"] == "llama8"), ("TriviaQA", lambda d: d["ds"] == "h1"), ("PopQA", lambda d: d["ds"] == "pq"), ("NQ", lambda d: d["ds"] == "nq"),
                 ("seeds 0-1", lambda d: int(d["name"].split("_s")[-1].split("_")[0]) <= 1), ("seeds >= 2", lambda d: int(d["name"].split("_s")[-1].split("_")[0]) >= 2)):
    e, n, p = fp(sel); P(f"   {lab:10s} runs {n:3d}  own-cell residual {e:+.4f}  permutation p {p:.4f}")
P("(b) Qwen-judge families, seeds 0-2 vs 3-5")
for f in ("judgea35", "judgea50"):
    for st in ("qwen7", "llama8"):
        for part, ok in (("s0-2", lambda s: s <= 2), ("s3-5", lambda s: s >= 3)):
            L = [d for d in k6 if d["fam"] == f and d["st"] == st and ok(int(d["name"].split("_s")[-1].split("_")[0]))]
            if L: P(f"   {f} {st} {part}: runs {len(L)}, own {np.mean([d['p_own'] for d in L]):.3f}, key {np.mean([d['p_key'] for d in L]):.3f}, own > key in {np.mean([d['p_own'] > d['p_key'] for d in L]):.2f}")
P("(c) accuracy among answered, matched abstention vs gold family; trained answers / untrained answers on the same answered questions")
for f in ("judgea35", "judgeXa35", "judgea50", "sonnet", "vote"):
    per = []
    for st in ("qwen7", "llama8"):
        for ds in ("h1", "pq", "nq"):
            F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); ho = F["ho"]
            def stats(r):
                o = r["o"][ho]; A = (o == 1) | (o == 2)
                return (o == 0).mean(), r["aud"][ho][A].mean(), (o[A] == 1).mean(), F["g_aud"][ho][A].mean(), F["g_key"][ho][A].mean()
            X = [stats(r) for r in R.values() if r["tag"] == "FK20" and r["c"] == "05" and r["fam"] == f]
            G = [stats(r) for r in R.values() if r["tag"] == "FK20" and r["c"] == "05" and r["fam"] in ("gold", "keyFN")]
            prs = [(x, g) for x in X for g in G if abs(x[0] - g[0]) <= 0.05 and 0.05 <= x[0] <= 0.95]
            if prs: per.append((f"{st}-{ds}", len(prs), *[np.mean([x[k] - g[k] for x, g in prs]) for k in (1, 2, 3, 4)]))
    if not per: continue
    m = np.array([p[2:] for p in per])
    P(f"   {f:10s} pairs {len(per)}: d acc|ans audited {m[:, 0].mean():+.4f} (untrained answers {m[:, 2].mean():+.4f}), key {m[:, 1].mean():+.4f} (untrained {m[:, 3].mean():+.4f})  | "
      + " ".join(f"{p[0]}({p[1]}): {p[2]:+.3f}/{p[3]:+.3f}" for p in per))
open(f"{OUT}/k14_out.txt", "w").write("\n".join(lines) + "\n")
