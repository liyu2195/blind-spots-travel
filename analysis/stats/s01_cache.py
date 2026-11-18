"""(stats) Cache for one (student, dataset) from raw files only.
base: untrained greedy (k = 0, s = -1) + 32 samples (k = 1..32); parsed answers, first lines (judge lookup key), valid / IDK / loose / strict,
      P(Yes) of every judge (Q M G O R QS), Sonnet level (frontierH, then frontierA; 0 = none), conf (greedy agreement share of the 32).
runs: every qwen7 / llama8 run of this dataset: final greedy answer (ck120; for runs with another step count the last checkpoint), its
      first line, outcome code (0 IDK, 1 right loose, 2 wrong, 3 malformed), and the run's own untrained re-decode ck0 when present; config.
Usage: nice -n 10 timeout 280 python3 -I s01_cache.py qwen7 pq   -> cache/base_<st>_<ds>.pkl, cache/runs_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, time, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
st, ds = sys.argv[1], sys.argv[2]; dsn = DSN[ds]; t0 = time.time()
probs, H, NEW = questions(dsn); N = len(probs); assert sorted(probs) == list(range(N))
al = [set(probs[i]["aliases"]) for i in range(N)]
X = [[""] * 33 for _ in range(N)]; LN = [[""] * 33 for _ in range(N)]
for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
    k = r["s"] + 1; X[r["idx"]][k] = ans(r); LN[r["idx"]][k] = C.tq_line(r["text"])
valid = np.zeros((N, 33), bool); idk = np.zeros((N, 33), bool); kl = np.zeros((N, 33), bool); ks = np.zeros((N, 33), bool)
for i in range(N):
    for k in range(33):
        x = X[i][k]
        if not x: continue
        if IDKF.match(x): idk[i, k] = True; continue
        valid[i, k] = True; kl[i, k] = C.tq_correct(x, al[i]); ks[i, k] = C.tq_strict(x, al[i])
PJ = {}
for J in JALL:
    f = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
    if not os.path.exists(f): continue
    a = np.full((N, 33), np.nan)
    for r in map(json.loads, open(f)): a[r["idx"], r["s"] + 1] = py_of(r)
    PJ[J] = a
LV = {}
for src in ("frontierH", "frontierA"):
    f = f"{API}/{src}_{st}_{dsn}.jsonl"
    if os.path.exists(f):
        for r in map(json.loads, open(f)): LV.setdefault((r["idx"], r["norm"]), r["level"])
son = np.zeros((N, 33), np.int8)
for i in range(N):
    for k in range(33):
        if valid[i, k]: son[i, k] = LV.get((i, X[i][k]), 0)
conf = np.zeros(N)
for i in range(N):
    if valid[i, 0]:
        xs = [X[i][k] for k in range(1, 33) if valid[i, k]]
        conf[i] = sum(same(X[i][0], y) for y in xs) / 32
train = np.zeros(N, bool); train[H] = True; new = np.zeros(N, bool); new[NEW] = True
B = dict(st=st, ds=ds, dsn=dsn, N=N, X=X, LN=LN, valid=valid, idk=idk, kl=kl, ks=ks, PJ=PJ, son=son, conf=conf, train=train, new=new,
         prop=np.array([probs[i].get("prop", "") for i in range(N)]), pop=np.array([probs[i].get("s_pop") or np.nan for i in range(N)], float),
         question=[probs[i]["question"] for i in range(N)], gold=[probs[i].get("gold", "") for i in range(N)], aliases=[sorted(al[i]) for i in range(N)])
pickle.dump(B, open(f"{CACHE}/base_{st}_{ds}.pkl", "wb"))
print(f"{st} {ds}: base done ({time.time() - t0:.0f}s): train {train.sum()} new {new.sum()} judges {sorted(PJ)} sonnet keys {len(LV)}", flush=True)
CFG = configs()
PAT = re.compile(rf"^{st}_([A-Za-z]+?\d*v?K?\d*)(.+)_(idkc)(\d+)_s(\d+)_{ds}$")
R = {}
for run in sorted(os.listdir(RL)):
    if not (run.startswith(st + "_") and run.endswith("_" + ds)) or "SMOKE" in run: continue
    d = f"{RL}/{run}"; c = CFG.get(run, {})
    steps = int(c.get("STEPS", 120))
    cks = sorted(int(m.group(1)) for m in (re.search(r"eval_rl_ck(\d+)\.jsonl$", x) for x in os.listdir(d)) if m)
    fin = steps if steps in cks else (max(cks) if cks and max(cks) > 0 else None)
    if fin is None: continue
    rec = dict(cfg=c, ck=fin, steps=steps)
    for ck, nm in ((fin, "f"), (0, "z")):
        p = f"{d}/eval_rl_ck{ck}.jsonl"
        if not os.path.exists(p): rec[nm] = None; continue
        a = [None] * N; l = [None] * N
        for r in map(json.loads, open(p)):
            if r.get("s", -1) >= 0: continue
            a[r["idx"]] = ans(r); l[r["idx"]] = C.tq_line(r["text"])
        rec[nm] = a; rec[nm + "_line"] = l
    if rec["f"] is None or any(x is None for x in rec["f"]): print("incomplete", run); continue
    rec["o"] = np.array([outcome(rec["f"][i], al[i]) for i in range(N)], np.int8)
    m = re.match(rf"^{st}_(FK20|FK00|FK50|FK100|B1K20|B2K20|B2vK20|B5vK20|B10vK20|LK20|LK50)(.+)_idkc(\d+)_s(\d+)_{ds}$", run)
    if m: rec.update(tag=m.group(1), g=m.group(2), c=m.group(3), seed=int(m.group(4)))
    else: rec.update(tag="?", g="?", c="?", seed=-1)
    R[run] = rec
pickle.dump(R, open(f"{CACHE}/runs_{st}_{ds}.pkl", "wb"))
print(f"{st} {ds}: runs {len(R)} ({time.time() - t0:.0f}s); with config {sum(1 for r in R.values() if r['cfg'])}; tags",
      {t: sum(1 for r in R.values() if r['tag'] == t) for t in sorted({r['tag'] for r in R.values()})}, flush=True)
