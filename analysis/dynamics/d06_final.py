"""Final greedy answers (ck120) of every parsed run on all 1500 questions, with each training question's visit history.
Cache per (student, dataset): outcome codes (0 IDK, 1 strict-right, 2 loose-only right, 3 wrong, 4 malformed) and normalized answers."""
import numpy as np, pandas as pd, json, os, pickle, dyn, sys
from collections import Counter

def final_cache(st, ds, ck="ck120", force=False):
    p = f"{dyn.HERE}/cache_final_{st}_{ds}_{ck}.pkl"
    if os.path.exists(p) and not force: return pickle.load(open(p, "rb"))
    dsn = dyn.DSF[ds]
    probs = [json.loads(l) for l in open(f"{dyn.DATA}/{dsn}_1500.jsonl")]; idx = [r["idx"] for r in probs]; pos = {i: k for k, i in enumerate(idx)}
    runs = sorted(r for r in dyn.logs().run.unique() if r.startswith(st + "_") and r.endswith("_" + ds))
    O = {}; A = {}
    for run in runs:
        f = f"{dyn.RL}/{run}/eval_rl_{ck}.jsonl"
        if not os.path.exists(f): continue
        o = np.full(len(idx), 4, np.int8); a = [""] * len(idx)
        for r in map(json.loads, open(f)):
            if r["s"] >= 0 or r["idx"] not in pos: continue
            k = pos[r["idx"]]; x = dyn.sa(r); a[k] = x; al = set(probs[k]["aliases"])
            if not x: o[k] = 4
            elif dyn.IDK_FULL.match(x): o[k] = 0
            elif dyn.C.tq_strict(x, al): o[k] = 1
            elif dyn.C.tq_correct(x, al): o[k] = 2
            else: o[k] = 3
        O[run] = o; A[run] = a
    out = dict(idx=np.array(idx), runs=list(O), O=np.array([O[r] for r in O]), A=A)
    pickle.dump(out, open(p, "wb")); return out

def visits(info, ids):
    """per training question (row order of the training file): number of visits and step of last visit"""
    O = dyn.order(info["seed"], len(ids))
    nv = np.zeros(len(ids), int); last = np.zeros(len(ids), int); first = np.full(len(ids), 999)
    for s, b in enumerate(O, 1):
        for i, v in b: nv[i] = v; last[i] = s; first[i] = min(first[i], s)
    return nv, last, first

if __name__ == "__main__":
    for st in ("qwen7", "llama8"):
        for ds in ("h1", "pq", "nq"):
            F = final_cache(st, ds); print(st, ds, len(F["runs"]), "runs", F["O"].shape, flush=True)
