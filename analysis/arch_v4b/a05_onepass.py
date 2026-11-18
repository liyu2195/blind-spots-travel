"""(arch_v4b) Simple independent check of 'one pass is enough for the judge's map' (regimes r04), own code, TriviaQA held-out questions.
Runs (bm.py cache): FK20 (about 5 visits per question) and B1K20 (each of 3,840 questions seen once) with grader judgeQa35, judgeQt50,
gold key (sim00fn00), hash35fn00. Per run: new-question abstention; and the 'map gap' on known answers (untrained greedy valid and audited
right): refusal where the strict Qwen judge (a35 cut, bm thresholds) rejects the untrained greedy minus where it accepts, Mantel-Haenszel
weighted within 4 bins of untrained agreement. Judge-specific map = judge runs' gap minus gold-key runs' gap of the same regime.
Output: a05_onepass.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
CACHE = (_RX + "/story/B_refusal_map/cache_{}_{}.pkl")
OUT = open((_RA + "/arch_v4b/a05_onepass.txt"), "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")
def mh(y, rej, strata, mask):
    num = den = 0.0
    for s in np.unique(strata[mask]):
        m = mask & (strata == s); a = m & rej; b = m & ~rej; na, nb = a.sum(), b.sum()
        if na == 0 or nb == 0: continue
        w = na * nb / (na + nb); num += w * (y[a].mean() - y[b].mean()); den += w
    return num / den if den > 0 else np.nan
for st in ("qwen7", "llama8"):
    P = pickle.load(open(CACHE.format(st, "h1"), "rb")); Q = P["Q"]
    known = Q["gval"] & Q["aud"]; rej = Q["pj"] <= Q["thr"]["a35"]
    strata = np.digitize(Q["conf"], [0.5, 0.8, 31 / 32])
    pr(f"--- {st} h1: known {known.sum()}, of which Qwen-a35 rejects {(known & rej).sum()}")
    gaps = {}
    for g in ("judgeQa35", "judgeQt50", "sim00fn00", "hash35fn00"):
        for kind in ("FK20", "B1"):
            runs = [r for r in P["R"].values() if r["kind"] == kind and r["g"] == g]
            if not runs: continue
            ab = [r["a"] for r in runs]; gp = [mh((r["o"] == 0).astype(float), rej, strata, known) for r in runs]
            gaps[(g, kind)] = np.nanmean(gp)
            pr(f"  {g:10s} {kind:4s} runs {len(runs)} | new abstention {np.mean(ab):.3f} (range {min(ab):.2f}-{max(ab):.2f}) | map gap on known (MH by agreement) {np.nanmean(gp):+.3f}")
    for kind in ("FK20", "B1"):
        if ("judgeQa35", kind) in gaps and ("sim00fn00", kind) in gaps:
            pr(f"  judge-specific map ({kind}): Qa35 gap minus gold gap {gaps[('judgeQa35', kind)] - gaps[('sim00fn00', kind)]:+.3f}")
