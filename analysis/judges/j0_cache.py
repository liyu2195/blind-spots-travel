"""(judges) Per (student, dataset) cache of every untrained sample with every grader's verdict inputs.
Rows: all 1,500 questions x 33 samples (s = -1 greedy, 0..31 sampled; IDK prompt). Columns: parsed answer, valid / IDK flags,
loose and strict key, judge P(Yes) for Q M G O R QS (and F P, unqualified judges), Sonnet level (frontierA: trained questions, all
unique sampled answers; frontierH: held-out greedy answers), answer cluster within question (rc.same clustering, 32 samples), words.
Usage: nice -n 10 python3 j0_cache.py   (writes cache_<st>_<dsn>.pkl here)"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, C, same, BASE
HERE = (_RA + "/judges"); API = (_RX + "/apitest")
JUDGES = ["Q", "M", "G", "O", "R", "QS", "F", "P"]

def build(st, dsn):
    probs, H1, HELD = setup(dsn); hs = set(HELD)
    rows = [r for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl"))]
    rows.sort(key=lambda r: (r["idx"], r["s"]))
    n = len(rows); idx = np.array([r["idx"] for r in rows], np.int32); s = np.array([r["s"] for r in rows], np.int16)
    X = []; valid = np.zeros(n, bool); idk = np.zeros(n, bool); kl = np.zeros(n, bool); ks = np.zeros(n, bool); words = np.zeros(n, np.int16)
    raw = []
    for k, r in enumerate(rows):
        x = sa(r); X.append(x); raw.append(C.tq_line(r["text"]))
        if x and IDK.match(x): idk[k] = True
        elif x:
            valid[k] = True; al = set(probs[r["idx"]]["aliases"])
            kl[k] = bool(C.tq_correct(x, al)); ks[k] = bool(C.tq_strict(x, al)); words[k] = len(x.split())
    pos = {(int(idx[k]), int(s[k])): k for k in range(n)}
    PJ = {}
    for J in JUDGES:
        f = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"; a = np.full(n, np.nan)
        if os.path.exists(f):
            for r in map(json.loads, open(f)):
                k = pos.get((r["idx"], r["s"]))
                if k is not None: a[k] = r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
        PJ[J] = a
    LV = {}
    for f in (f"{API}/frontierA_{st}_{dsn}.jsonl", f"{API}/frontierH_{st}_{dsn}.jsonl"):
        if os.path.exists(f):
            for r in map(json.loads, open(f)): LV[(r["idx"], r["norm"])] = r["level"]
    son = np.array([LV.get((int(idx[k]), X[k]), 0) if valid[k] else 0 for k in range(n)], np.int8)
    # clusters of the 32 sampled valid answers per question (greedy joins a cluster if same(), else its own)
    clu = np.full(n, -1, np.int16); csize = np.zeros(n, np.int16)
    by = defaultdict(list)
    for k in range(n): by[int(idx[k])].append(k)
    for i, ks_ in by.items():
        xs = [X[k] for k in ks_ if s[k] >= 0 and valid[k]]
        cl = []
        for x, m in Counter(xs).most_common():
            for c in cl:
                if same(x, c[0]): c[1] += m; c[2].add(x); break
            else: cl.append([x, m, {x}])
        for k in ks_:
            if not valid[k]: continue
            for j, c in enumerate(cl):
                if X[k] in c[2] or (s[k] < 0 and same(X[k], c[0])): clu[k] = j; csize[k] = c[1]; break
            else: clu[k] = len(cl); csize[k] = 0
    Q = dict(train=np.array([i in H1 for i in range(1500)]), held=np.array([i in hs for i in range(1500)]),
             prop=[probs[i].get("prop", "") for i in range(1500)], pop=np.array([probs[i].get("s_pop", np.nan) for i in range(1500)], float),
             question=[probs[i]["question"] for i in range(1500)], gold=[probs[i].get("gold", "") for i in range(1500)],
             aliases=[probs[i]["aliases"] for i in range(1500)])
    D = dict(st=st, ds=dsn, idx=idx, s=s, X=X, raw=raw, valid=valid, idk=idk, kl=kl, ks=ks, words=words, PJ=PJ, son=son, clu=clu, csize=csize, Q=Q)
    pickle.dump(D, open(f"{HERE}/cache_{st}_{dsn}.pkl", "wb"))
    tr = Q["train"][idx] & valid
    print(st, dsn, "rows", n, "valid", valid.mean().round(3), "idk", idk.mean().round(3), "key strict|valid", ks[valid].mean().round(3),
          "sonnet coverage trained valid", (son[tr] > 0).mean().round(3), {J: int(np.isnan(PJ[J]).sum()) for J in JUDGES}, flush=True)

if __name__ == "__main__":
    for dsn in ("pqa", "nqo", "tqa"):
        for st in ("qwen7", "llama8"):
            if os.path.exists(f"{HERE}/cache_{st}_{dsn}.pkl") and "force" not in sys.argv: continue
            build(st, dsn)
