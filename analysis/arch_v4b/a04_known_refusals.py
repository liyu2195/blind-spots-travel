"""(arch_v4b) Independent check of 'the judge decides which known answers are lost, not how many' (rev_R3 r2_known_refusals), own code.
Per (student, dataset): strict-judge runs (FK20 judge<J>a35, J in Q M G O R, idkc05, abstention < 0.95) vs the key family (gold sim00fn00 and
key + random false rejects sim00fnXX, same filter). Known = untrained greedy valid and audited right (loose key or Sonnet >= 4);
sure & known = known with untrained agreement >= 31/32. Refusals of known answers per 100 held-out questions. Key-family line: OLS of
refusals-of-known on abstention over the key-family runs, evaluated at the judge runs' mean abstention. Output: a04_known_refusals.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, numpy as np
CACHE = (_RX + "/story/B_refusal_map/cache_{}_{}.pkl")
OUT = open((_RA + "/arch_v4b/a04_known_refusals.txt"), "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")
tot = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        P = pickle.load(open(CACHE.format(st, ds), "rb")); Q = P["Q"]
        known = Q["gval"] & Q["aud"]; sure = known & (Q["conf"] >= 31 / 32)
        R = [r for r in P["R"].values() if r["kind"] == "FK20" and r["a"] < 0.95]
        J = [r for r in R if re.fullmatch(r"judge[QMGOR]a35", r["g"])]; K = [r for r in R if re.fullmatch(r"sim00fn\d\d", r["g"])]
        f = lambda r, m: 100 * ((r["o"] == 0) & m).mean()
        aJ = np.mean([r["a"] for r in J]); kJ = np.mean([f(r, known) for r in J]); sJ = np.mean([f(r, sure) for r in J])
        aK = np.array([r["a"] for r in K]); kK = np.array([f(r, known) for r in K]); sK = np.array([f(r, sure) for r in K])
        lk = np.polyval(np.polyfit(aK, kK, 1), aJ); ls = np.polyval(np.polyfit(aK, sK, 1), aJ)
        tot.append((kJ, lk, sJ, ls))
        pr(f"{st} {ds}: judge runs {len(J)} abst {aJ:.2f} | key-family runs {len(K)} abst range {aK.min():.2f}-{aK.max():.2f} | known refused /100: judge {kJ:.1f} vs key line {lk:.1f} ({kJ-lk:+.1f}) | sure&known: judge {sJ:.1f} vs line {ls:.1f} ({sJ-ls:+.1f})")
t = np.array(tot)
pr(f"mean over 6 cells: known refused judge {t[:,0].mean():.1f} vs key line {t[:,1].mean():.1f} ({(t[:,0]-t[:,1]).mean():+.1f}); sure&known judge {t[:,2].mean():.1f} vs {t[:,3].mean():.1f} ({(t[:,2]-t[:,3]).mean():+.1f})")
