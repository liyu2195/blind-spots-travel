"""selfcons s4: how many untrained samples is the RL refusal policy worth?  SC_m = self-consistency from m of the 32 untrained samples
(agreement with the greedy answer among m samples drawn without replacement; 40 random draws per m, ties at the coverage boundary
split), compared with each run at its own coverage on NEW questions (audited truth, untrained greedy answers for the rule).
m = 0 is a random choice of which valid questions to answer. Also: SCpl = plurality share instead of greedy agreement.
Output: s4_scm.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/selfcons"))
from sc_lib import *
MS = [0, 1, 2, 4, 8, 16, 32]; DRAW = 40
lines = []
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
pr(__doc__)
rows = []
for st in STS:
    for ds in DSS:
        P, PR = load(st, ds); Q = P["Q"]; M = Q["ho"]; N = int(M.sum())
        val = Q["gval"][M]; ru = Q["gaud"][M].astype(float); agree = np.rint(Q["conf"][M] * 32).astype(int)
        rng = np.random.default_rng(7)
        scores = {}
        for m in MS:
            if m == 0: scores[m] = [rng.random(N) for _ in range(DRAW)]
            elif m == 32: scores[m] = [agree / 32.0]
            else: scores[m] = [rng.hypergeometric(agree, 32 - agree, m) / m for _ in range(DRAW)]
        scores["pl"] = [Q["psh"][M]]
        for name, r in sorted(P["R"].items()):
            o = r["o"][M]; ans = (o == 1) | (o == 2); k = int(ans.sum())
            if k < 0.05 * N or k > val.sum(): continue
            d = dict(st=st, ds=ds, name=name, grp=grp(r["tag"], r["fam"], r["credit"]), fam=r["fam"], tag=r["tag"], credit=r["credit"], cov=k / N,
                     accRL=float(r["aud"][M][ans].mean()))
            for m, L in scores.items():
                acc = []
                for s in L:
                    w = frac_select(s, val, k); acc.append((w * ru).sum() / w.sum())
                d[f"acc{m}"] = float(np.mean(acc))
            rows.append(d)
hdr = f"{'group':20s} {'n':>3s} {'cov':>5s} {'accRL':>6s} | " + " ".join(f"{'SC'+str(m):>6s}" for m in MS) + f" {'SCpl':>6s} | RL - SC_m (pp): " + " ".join(f"{'m='+str(m):>6s}" for m in MS) + " | m* (smallest m with SC_m >= RL)"
def show(L, label):
    pr(f"\n{label}"); pr(hdr)
    byg = defaultdict(list)
    for d in L: byg[d["grp"]].append(d)
    def order(g):
        base = g.split(":")[-1].split("[")[0]
        return (g.count(":") + g.count("["), FAMORDER.index(base) if base in FAMORDER else 99, g)
    for g in sorted(byg, key=order):
        X = byg[g]; mm = lambda k: np.mean([x[k] for x in X])
        mstar = next((m for m in MS if mm(f"acc{m}") >= mm("accRL")), ">32")
        pr(f"{g:20s} {len(X):3d} {mm('cov'):5.2f} {100*mm('accRL'):6.1f} | " + " ".join(f"{100*mm(f'acc{m}'):6.1f}" for m in MS) + f" {100*mm('accpl'):6.1f} | "
           + "                " + " ".join(f"{100*(mm('accRL') - mm(f'acc{m}')):+6.1f}" for m in MS) + f" | {mstar}")
for st in STS:
    pr(f"\n################ {st} ({'DISCOVERY' if st == 'qwen7' else 'CONFIRMATION'}) ################")
    show([d for d in rows if d["st"] == st], f"=== {st} pooled over datasets (each run one row)")
    L = [d for d in rows if d["st"] == st and d["tag"] == "FK20" and d["credit"] == "idkc05"]
    pr(f"\n--- {st} all FK20 idkc05 runs ({len(L)}): RL - SC_m mean (pp): " + " ".join(f"m={m}: {100*np.mean([d['accRL'] - d[f'acc{m}'] for d in L]):+.2f}" for m in MS)
       + f"  SCpl: {100*np.mean([d['accRL'] - d['accpl'] for d in L]):+.2f}")
    for gname, fams in GROUPS:
        G_ = [d for d in L if d["fam"] in fams]
        if G_: pr(f"   {gname:18s} runs {len(G_):3d}: " + " ".join(f"m={m}: {100*np.mean([d['accRL'] - d[f'acc{m}'] for d in G_]):+5.2f}" for m in MS))
    for ds in DSS:
        Ld = [d for d in L if d["ds"] == ds]
        pr(f"   dataset {ds:3s} runs {len(Ld):3d}: " + " ".join(f"m={m}: {100*np.mean([d['accRL'] - d[f'acc{m}'] for d in Ld]):+5.2f}" for m in MS))
pickle.dump(rows, open(f"{OUT}/s4_rows.pkl", "wb"))
open(f"{OUT}/s4_scm.txt", "w").write("\n".join(lines) + "\n")
