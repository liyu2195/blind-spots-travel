"""(verify_where) Own cache of the untrained student's answers and every grader's verdict inputs, from the raw files.
One (student, set) per call. Columns: k = 0 greedy (s = -1), k = 1..32 samples s = 0..31 (IDK prompt).
Uses only project helpers (rc.setup / rc.sa / rc.IDK / rc.same; st_common.tq_correct / tq_strict); no explorer code.
Writes cache/base_<st>_<dsn>.pkl.  Usage: nice -n 10 python3 v0_cache.py qwen7 pqa"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, numpy as np
from collections import Counter
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, same, C, BASE
OUT = (_RA + "/verify_where/cache"); API = (_RX + "/apitest")
JUD = ["Q", "M", "G", "O", "R", "QS"]

def main(st, dsn):
    probs, H1, HELD = setup(dsn); N = 1500; assert sorted(probs) == list(range(N))
    X = [[""] * 33 for _ in range(N)]; RAW = [[""] * 33 for _ in range(N)]
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        k = r["s"] + 1; X[r["idx"]][k] = sa(r); RAW[r["idx"]][k] = C.tq_line(r["text"])
    idk = np.zeros((N, 33), bool); valid = np.zeros((N, 33), bool); kl = np.zeros((N, 33), bool); ks = np.zeros((N, 33), bool)
    for i in range(N):
        al = set(probs[i]["aliases"])
        for k in range(33):
            x = X[i][k]
            if not x: continue
            if IDK.match(x): idk[i, k] = True; continue
            valid[i, k] = True; kl[i, k] = bool(C.tq_correct(x, al)); ks[i, k] = bool(C.tq_strict(x, al))
    PJ = {}
    for J in JUD:
        a = np.full((N, 33), np.nan)
        f = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
        for r in map(json.loads, open(f)):
            d = r["py"] + r["pn"]; a[r["idx"], r["s"] + 1] = r["py"] / d if d > 1e-3 else 0.5
        PJ[J] = a
    # Sonnet levels keyed by (idx, norm); record conflicts between sources
    LV = {}; conflicts = 0
    for src in ("frontierA", "frontierH", "auditT"):
        f = f"{API}/{src}_{st}_{dsn}.jsonl"
        if not os.path.exists(f): continue
        for r in map(json.loads, open(f)):
            key = (r["idx"], r["norm"])
            if key in LV and LV[key] != r["level"]: conflicts += 1
            LV.setdefault(key, r["level"])
    son = np.zeros((N, 33), np.int8)   # 0 = no Sonnet verdict
    for i in range(N):
        for k in range(33):
            if valid[i, k]: son[i, k] = LV.get((i, X[i][k]), 0)
    # clusters of the 32 sampled valid answers (most frequent string first, rc.same joins); greedy joins the first matching cluster
    clu = np.full((N, 33), -1, np.int16); csize = np.zeros((N, 33), np.int16); conf = np.zeros(N)
    for i in range(N):
        xs = [X[i][k] for k in range(1, 33) if valid[i, k]]
        cl = []
        for x, m in Counter(xs).most_common():
            for c in cl:
                if same(x, c[0]): c[1] += m; c[2].add(x); break
            else: cl.append([x, m, {x}])
        for k in range(33):
            if not valid[i, k]: continue
            for j, c in enumerate(cl):
                if X[i][k] in c[2] or (k == 0 and same(X[i][k], c[0])): clu[i, k] = j; csize[i, k] = c[1]; break
            else: clu[i, k] = len(cl); csize[i, k] = 0
        if valid[i, 0]: conf[i] = sum(same(X[i][0], y) for y in xs) / 32
    TH = {}
    for J in JUD:
        f = f"{_RX}/final/thresholds_{J}.json"
        d = json.load(open(f)).get(f"{st}_{dsn}") if os.path.exists(f) else None
        if d: TH[J] = dict(a35=d["t_a35"], a50=d["t_a50"], t50=0.5)
    D = dict(st=st, dsn=dsn, X=X, RAW=RAW, idk=idk, valid=valid, kl=kl, ks=ks, PJ=PJ, son=son, clu=clu, csize=csize, conf=conf, TH=TH,
             train=np.array([i in H1 for i in range(N)]), held=np.array([i in set(HELD) for i in range(N)]),
             prop=np.array([probs[i].get("prop", "") for i in range(N)]), pop=np.array([probs[i].get("s_pop", np.nan) for i in range(N)], float),
             question=[probs[i]["question"] for i in range(N)], gold=[probs[i].get("gold", "") for i in range(N)],
             aliases=[sorted(probs[i]["aliases"]) for i in range(N)], qid=[probs[i].get("qid") for i in range(N)])
    os.makedirs(OUT, exist_ok=True); pickle.dump(D, open(f"{OUT}/base_{st}_{dsn}.pkl", "wb"))
    tr = D["train"]; ho = D["held"]
    print(st, dsn, "train", tr.sum(), "held", ho.sum(), "neither", (~tr & ~ho).sum(), "| valid samples", valid[:, 1:].mean().round(3),
          "| Sonnet coverage, trained valid samples", (son[tr, 1:][valid[tr, 1:]] > 0).mean().round(3), "held greedy", (son[ho, 0][valid[ho, 0]] > 0).mean().round(3),
          "| Sonnet conflicts", conflicts, "| nan PJ", {J: int(np.isnan(PJ[J]).sum()) for J in JUD}, "| thresholds", {J: TH[J]["a35"] for J in TH}, flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
