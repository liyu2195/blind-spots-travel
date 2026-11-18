"""selfcons s7: discovery -> confirmation scan of every (dataset x recipe-and-grader group) cell of s2 (new questions, not capped).
A cell 'beats SC' when its question-level 95% interval for RL - SC selective accuracy lies above 0; it is 'confirmed' when the same cell
on the other student also lies above 0 (llama8 confirms qwen7 discoveries; the reverse direction is listed too). Same for 'loses to SC'.
Also lists the pooled-over-datasets groups. Output: s7_cells.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/selfcons"))
from sc_lib import *
lines = []
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
pr(__doc__)
D = pickle.load(open(f"{OUT}/s2_rows_new.pkl", "rb")); rows = [d for d in D["rows"] if not d["cap"]]; BOOT = D["BOOT"]
cell = {}
for pooled in (False, True):
    byc = defaultdict(list)
    for d in rows: byc[(d["st"], "ALL" if pooled else d["ds"], d["grp"])].append(d)
    for (st, ds, g), L in byc.items():
        b = np.mean([BOOT[(x["st"], x["ds"], x["name"], "dSC")] for x in L], 0)
        cell[(st, ds, g)] = (np.mean([x["dSC"] for x in L]), *ci(b), len(L))
for disc, conf in (("qwen7", "llama8"), ("llama8", "qwen7")):
    pr(f"\n=== discovery {disc} -> confirmation {conf}")
    for ds in list(DSS) + ["ALL"]:
        cs = sorted(g for (s, d, g) in cell if s == disc and d == ds)
        n = len(cs); pos = [g for g in cs if cell[(disc, ds, g)][1] > 0]; neg = [g for g in cs if cell[(disc, ds, g)][2] < 0]
        def tag(g):
            c = cell.get((conf, ds, g))
            if c is None: return f"{g} {100*cell[(disc, ds, g)][0]:+.1f} -> (no {conf} cell)"
            v = "CONFIRMED" if (c[1] > 0 and cell[(disc, ds, g)][1] > 0) or (c[2] < 0 and cell[(disc, ds, g)][2] < 0) else "not confirmed"
            return f"{g} {100*cell[(disc, ds, g)][0]:+.1f} -> {conf} {100*c[0]:+.1f} [{100*c[1]:+.1f},{100*c[2]:+.1f}] {v}"
        pr(f"  {ds}: cells {n} | RL > SC: {len(pos)} | RL < SC: {len(neg)}")
        for g in pos: pr(f"     beats SC: {tag(g)}")
        for g in neg: pr(f"     loses   : {tag(g)}")
open(f"{OUT}/s7_cells.txt", "w").write("\n".join(lines) + "\n")
