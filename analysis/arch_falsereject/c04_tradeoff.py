"""(arch_falsereject) The strictness trade-off in counts per 100 questions, home (training) vs new, for judge-trained FK20 runs (judges Q M G O at
t50 / a50 / a35), all 6 student x dataset pairs, from the answers explorer's caches (answers/au.py load + jc judge lookups).
  hacks      = changed final answer (a120 != a0) that is key-wrong and paid by the run's own judge at its level
  refusedK   = untrained greedy a0 key-right, own judge rejects a0 at its level, final a120 is IDK
  refusedK_acc = same but own judge accepts a0 (control: refusal that does not come from a false reject)
Baseline: key-trained (gold) FK20 runs of the same pair, scored with the same judge and level. Reported: judge-run mean minus gold mean (per 100
questions) and the DiD selectivity (refusedK - refusedK_acc rates, as shares of their own denominators) minus gold's; run-bootstrap CI over runs.
Key = loose answer key (no Sonnet), so NQ-open 'wrong' includes key misses. Usage: nice -n 10 timeout 280 python3 c04_tradeoff.py > c04_tradeoff_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, numpy as np
sys.path.insert(0, (_RA + "/answers"))
from au import load, jc, fam, judge_of, code, thr, runs, STS, SETS
rng = np.random.default_rng(5)
rows = []
for dsn in SETS:
    JC = {J: jc(dsn, J) for J in ("Q", "M", "G", "O")}
    for st in STS:
        P = load(st, dsn); Q = P["Q"]; al = Q["aliases"]
        parts = {"home": sorted(Q["H1"]), "new": Q["HELD"]}
        golds = runs(P, fams={"gold"})
        def stats(r, J, lvl):
            t = thr(st, dsn, J, lvl); out = {}
            for part, items in parts.items():
                h = rk = rka = nk = nka = 0
                for i in items:
                    a0 = r["a0"].get(i); a1 = r["a120"].get(i)
                    c0 = code(a0, al[i]); c1 = code(a1, al[i])
                    if c1 == 2 and a1 != a0:
                        v = JC[J].get((i, a1))
                        if v is not None and v > t: h += 1
                    if c0 == 1:
                        v0 = JC[J].get((i, a0))
                        if v0 is None: continue
                        if v0 <= t: nk += 1; rk += (c1 == 0)
                        else: nka += 1; rka += (c1 == 0)
                n = len(items)
                out[part] = dict(hack=100 * h / n, refK=100 * rk / n, selK=(rk / max(nk, 1)) - (rka / max(nka, 1)), idk=np.mean([code(r["a120"].get(i), al[i]) == 0 for i in items]))
            return out
        G = {}
        for name in runs(P, fams={"judge"}):
            r = P["R"][name]; jo = judge_of(r["g"])
            if jo is None or jo[0] not in JC or jo[2]: continue
            J, lvl = jo[0], jo[1]
            if (J, lvl) not in G: G[(J, lvl)] = [stats(P["R"][g], J, lvl) for g in golds]
            s = stats(r, J, lvl)
            for part in parts:
                gm = {k: np.mean([g[part][k] for g in G[(J, lvl)]]) for k in ("hack", "refK", "selK", "idk")}
                rows.append(dict(st=st, dsn=dsn, J=J, lvl=lvl, part=part, run=name, hack=s[part]["hack"], hack_g=gm["hack"], refK=s[part]["refK"], refK_g=gm["refK"],
                                 sel=s[part]["selK"], sel_g=gm["selK"], idk=s[part]["idk"], idk_g=gm["idk"]))
import pandas as pd
D = pd.DataFrame(rows); D.to_pickle("c04_rows.pkl")
def ci(v):
    v = np.asarray(v); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for lvl in ("t50", "a50", "a35"):
    for part in ("home", "new"):
        E = D[(D.lvl == lvl) & (D.part == part)]
        if len(E) == 0: continue
        a = ci(E.hack - E.hack_g); b = ci(E.refK - E.refK_g); c = ci(E.sel - E.sel_g)
        print(f"{lvl} {part:4s} runs {len(E):3d} | hacks/100q {E.hack.mean():5.1f} (gold {E.hack_g.mean():4.1f}) excess {a[0]:+5.1f} [{a[1]:+.1f},{a[2]:+.1f}] | refused-known-rejected/100q {E.refK.mean():5.1f} (gold {E.refK_g.mean():4.1f}) excess {b[0]:+5.1f} [{b[1]:+.1f},{b[2]:+.1f}] | selectivity DiD {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}] | IDK {E.idk.mean():.2f} (gold {E.idk_g.mean():.2f})")
print("\nper student (a35 and t50):")
for st in STS:
    for lvl in ("t50", "a35"):
        for part in ("home", "new"):
            E = D[(D.st == st) & (D.lvl == lvl) & (D.part == part)]
            if len(E) == 0: continue
            print(f"  {st} {lvl} {part:4s} runs {len(E):3d} hacks excess {(E.hack - E.hack_g).mean():+5.1f}/100q | refused-known-rejected excess {(E.refK - E.refK_g).mean():+5.1f}/100q | sel DiD {(E.sel - E.sel_g).mean():+.3f}")
