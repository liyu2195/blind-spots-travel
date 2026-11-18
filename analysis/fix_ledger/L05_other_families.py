"""(fix_ledger) L05: the per-100-new-question outcome account (IDK, right, wrong given, right withheld, known withheld) for the grader families
that regimes/r02_costs.py does not cover, with the same definitions and caches (regimes/rg.build, r02_costs.per_question):
key + random false accepts (sim20fn00, sim35fn00), key + random FA and FR (sim<FP>fn<FN> with both > 0, lenQ excluded), fixed-table
false accepts (hash35fn00), own majority vote as key (frzVOTE*), long-answer grader (long80), type-set false accepts (typeA/typeB),
planted false rejects on PopQA (fnB80, fnH80, fnA80, plrealM/O, plscrM, plsharpM/O), blinded key (blind40).
Main recipe only (FK20, credit 0.5); collapsed runs (new-question IDK >= 0.95) dropped; datasets pooled over those where the family has runs
(equal weight per question). No installed-hack column: these graders have no fixed per-string verdict that can be scored on new strings
(hash tables are seed-specific; see verify_what/v31_specific.txt and general/g04_hash.txt for the hash content lift).
Output: L05_other_families_out.txt.  Usage: nice -n 10 timeout 280 python3 -I L05_other_families.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, numpy as np
sys.path.insert(0, (_RA + "/regimes"))
from rg import build
import r02_costs as R2
HERE = (_RA + "/fix_ledger")
FAM = [("key + random false accepts", lambda g: re.fullmatch(r"sim(20|35)fn00", g)),
       ("key + random FA and FR", lambda g: re.fullmatch(r"sim(\d\d)fn(\d\d)", g) and not g.startswith("sim00") and not g.endswith("fn00")),
       ("fixed-table false accepts (hash)", lambda g: g.startswith("hash")),
       ("own majority vote as key", lambda g: g.startswith("frzVOTE")),
       ("long-answer grader (long80)", lambda g: g.startswith("long")),
       ("type-set false accepts (typeA/B)", lambda g: g.startswith(("typeA", "typeB"))),
       ("planted false rejects (fnB/fnH/fnA, copies)", lambda g: g.startswith(("fnB", "fnH", "fnA", "plreal", "plscr", "plsharp"))),
       ("blinded key (blind40)", lambda g: g.startswith("blind"))]
out = open(f"{HERE}/L05_other_families_out.txt", "w")
def P(s=""): print(s); out.write(s + "\n")
P(__doc__)
for st in ("qwen7", "llama8"):
    PS = {ds: build(st, ds) for ds in ("h1", "pq", "nq")}
    P(f"\n== {st}: per 100 new questions (datasets with runs pooled)")
    P(f"{'family':44s} {'runs':>4s} {'datasets':10s} | {'IDK':>5s} {'right':>6s} {'wrong':>6s} {'withheld':>8s} {'known':>6s} | untrained right / wrong on the same questions")
    for name, sel in FAM:
        vals = {k: [] for k in ("idk", "right", "wrong", "RW", "RK")}; n = 0; dss = []; ur = []; uw = []
        for ds, Pd in PS.items():
            Q = Pd["Q"]
            runs = [r for r in Pd["R"].values() if r["kind"] == "FK20" and r["credit"] == "idkc05" and sel(r["g"])
                    and (r["o"][Q["held"]] == 0).mean() < 0.95]
            if not runs: continue
            pq = R2.per_question(Pd, runs, None, "key", st, Q["dsn"])
            for k in vals: vals[k].append(pq[k][Q["held"]])
            n += len(runs); dss.append(ds)
            ur.append((Q["aud"] & Q["gval"])[Q["held"]]); uw.append((Q["gval"] & ~Q["aud"])[Q["held"]])
        if not n: P(f"{name:44s}    0"); continue
        m = {k: 100 * np.mean(np.concatenate(v)) for k, v in vals.items()}
        P(f"{name:44s} {n:4d} {'+'.join(dss):10s} | {m['idk']:5.1f} {m['right']:6.1f} {m['wrong']:6.1f} {m['RW']:8.1f} {m['RK']:6.1f} | "
          f"{100 * np.mean(np.concatenate(ur)):.1f} / {100 * np.mean(np.concatenate(uw)):.1f}")
out.close()
