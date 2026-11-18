"""fix_sc f3: the compact table (FK20, IDK credit 0.5, new questions, capped runs included) and summary numbers, from f1_rows.pkl / f1_summary.pkl.
Cells 'Qwen ; Llama'. m* = equivalent generations (f1); 'flat' when SC_32 - SC_1 < 2 pp at the family's coverage (RL hardly abstains, so the
number of samples cannot be identified). '†' = more than 5% of the family's RL answers on one dataset are key-wrong without any audit verdict:
the 'hi' bound (those answers right) is given. Also: pooled RL - SC_32 over all runs with / without the blind grader, and which cells have a
crossed-bootstrap interval excluding 0 on each student. Output: f3_table.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
sys.path.insert(0, (_RA + "/fix_sc")); sys.path.insert(0, (_RA + "/selfcons"))
from f1_scm import ROWS, rows_of, group_stats, fm, crossed
from sc_lib import ci
OUT = (_RA + "/fix_sc")
S = pickle.load(open(f"{OUT}/f1_summary.pkl", "rb")); X = pickle.load(open(f"{OUT}/f1_rows.pkl", "rb"))
rows = [d for d in X["rows"] if d["tag"] == "FK20" and d["credit"] == "idkc05"]; BOOT = X["BOOT"]
out = [__doc__]
TAB = ["key", "key+noise (all)", "own vote", "judge Q t50", "judge Q a50", "judge Q a35", "judge M t50", "judge M a35", "judge G t50", "judge G a35",
       "judge O t50", "judge O a35", "judge R a35", "Sonnet", "planted FR", "planted FA", "blind (pays any answer on a subset)"]
def gapflag(st, rn):
    L = [d for d in rows if d["st"] == st and rn in rows_of(d["f2"])]
    by = {}
    for d in L: by.setdefault(d["ds"], []).append(d["accRL_hi"] - d["accRL"])
    return any(np.mean(v) > 0.05 for v in by.values())
out.append("\n| grader, operating point | runs (capped) | coverage | RL - SC_1 | RL - SC_32 [95% CI] | m* | selection / answers |")
out.append("|---|---|---|---|---|---|---|")
for rn in TAB:
    q, l = S.get(("qwen7", rn)), S.get(("llama8", rn))
    def c(G, st):
        if G is None: return None
        flat = (G["g"][1] - G["g"][32]) < 0.02
        dag = gapflag(st, rn)
        return dict(n=f"{G['n']} ({G['ncap']})", cov=f"{G['cov']:.2f}", g1=f"{100*G['g'][1]:+.1f}",
                    g32=f"{100*G['g'][32]:+.1f} [{100*G['ci32'][0]:+.1f}, {100*G['ci32'][1]:+.1f}]" + (f"† (hi {100*G['hi32']:+.1f})" if dag else ""),
                    ms="flat" if flat else fm(G["mstar"]), sa=f"{100*G['sel32']:+.1f} / {100*G['ans32']:+.1f}")
    cq, cl = c(q, "qwen7"), c(l, "llama8")
    name = rn.replace("judge ", "").replace("blind (pays any answer on a subset)", "blind (pays any answer on 40-60% of questions)")
    f = lambda k: f"{cq[k] if cq else '-'} ; {cl[k] if cl else '-'}"
    out.append(f"| {name} | {f('n')} | {f('cov')} | {f('g1')} | {f('g32')} | {f('ms')} | {f('sa')} |")
out.append("\nm* with 95% intervals (crossed bootstrap):")
for rn in TAB:
    out.append("   " + rn + ": " + " ; ".join(f"{st} {fm(S[(st, rn)]['mstar'])} [{fm(S[(st, rn)]['mci'][0])}, {fm(S[(st, rn)]['mci'][1])}]" for st in ("qwen7", "llama8") if (st, rn) in S))
rng = np.random.default_rng(7)
out.append("\nPooled RL - SC_32 (pp; crossed bootstrap), FK20 idkc05 runs:")
for st in ("qwen7", "llama8"):
    for lab, sel in (("all runs incl. capped", lambda d: True), ("excluding blind", lambda d: d["fam"] != "pl_blind"),
                     ("not capped (as selfcons s2)", lambda d: not d["cap"]), ("not capped, excluding blind", lambda d: not d["cap"] and d["fam"] != "pl_blind")):
        L = [d for d in rows if d["st"] == st and sel(d)]; G = group_stats(L, BOOT, rng)
        out.append(f"   {st} {lab:30s} runs {G['n']:3d}: {100*G['g'][32]:+.2f} [{100*G['ci32'][0]:+.2f}, {100*G['ci32'][1]:+.2f}]  (all 32 samples: {100*G['g'][33]:+.2f})")
out.append("\nCells (pooled over datasets) whose 95% interval for RL - SC_32 excludes 0:")
for st in ("qwen7", "llama8"):
    pos = [rn for rn in ROWS if (st, rn) in S and S[(st, rn)]["ci32"][0] > 0]; neg = [rn for rn in ROWS if (st, rn) in S and S[(st, rn)]["ci32"][1] < 0]
    out.append(f"   {st}: RL > SC_32: {pos} | RL < SC_32: {neg}")
open(f"{OUT}/f3_table.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
