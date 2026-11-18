"""(explorer 'general') Per-run greedy outcomes of every small-student run at ck0 and at the final checkpoint, on all 1500
questions, in two readings: strict (one ended line, IDK full match; the strict-format trainer's reading) and first-line
(parse of the first line, IDK prefix; the free-format trainer's reading). Output: cache_<st>.pkl (one per student)."""
import json, os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import setup, read_eval, code_strict, code_first, HERE

INV = json.load(open(f"{HERE}/g00_inventory.json"))
PROBS = {ds: setup(ds)[0] for ds in ("tqa", "pqa")}
for st in sys.argv[1:] or ("phi4m", "smol3", "granite"):
    out = {}
    for r in INV:
        if r["st"] != st:
            continue
        probs = PROBS[r["ds"]]
        d = dict(meta=r)
        for tag, ck in (("u", 0), ("f", r["final"])):
            try:
                g, _ = read_eval(r["run"], ck)
            except FileNotFoundError:
                d[tag] = None; continue
            cs = np.full(1500, 3, np.int8); cf = np.full(1500, 3, np.int8); xs = [""] * 1500; xf = [""] * 1500
            for i, row in g.items():
                al = set(probs[i]["aliases"])
                cs[i], xs[i] = code_strict(row, al)
                cf[i], xf[i] = code_first(row, al)
            d[tag] = dict(cs=cs, cf=cf, xs=xs, xf=xf, n=len(g))
        out[r["run"]] = d
    pickle.dump(out, open(f"{HERE}/cache_{st}.pkl", "wb"))
    print(st, len(out), "runs cached")
