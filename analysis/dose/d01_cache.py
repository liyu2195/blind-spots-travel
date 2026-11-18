"""(dose) Cache for one (student, dataset), from raw files only.
base: untrained greedy (k = 0, s = -1) + 32 samples (k = 1..32): parsed answer, valid / IDK / loose key / strict key, judge P(Yes) (Q M G O R),
      Sonnet level (frontierA / frontierH / auditT by (idx, norm); 0 = none), conf, train / new masks, question text, relation type, popularity.
runs: every FK20 run of this (student, dataset) with any IDK credit: final greedy outcome code on all questions (ck120, else the last checkpoint).
Usage: nice -n 10 timeout 280 python3 d01_cache.py qwen7 pq  -> cache/base_<st>_<ds>.pkl, cache/runs_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, time, numpy as np
sys.path.insert(0, (_RA + "/dose"))
from d00_common import *
st, ds = sys.argv[1], sys.argv[2]; dsn = DSN[ds]; t0 = time.time()
probs, H, NEW = questions(dsn); N = len(probs); assert sorted(probs) == list(range(N))
al = [set(probs[i]["aliases"]) for i in range(N)]
X = [[""] * 33 for _ in range(N)]
for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
    X[r["idx"]][r["s"] + 1] = ans(r)
valid = np.zeros((N, 33), bool); idk = np.zeros((N, 33), bool); kl = np.zeros((N, 33), bool); ks = np.zeros((N, 33), bool)
for i in range(N):
    for k in range(33):
        x = X[i][k]
        if not x: continue
        if IDKF.match(x): idk[i, k] = True; continue
        valid[i, k] = True; kl[i, k] = C.tq_correct(x, al[i]); ks[i, k] = C.tq_strict(x, al[i])
PJ = {}
for J in J5:
    a = np.full((N, 33), np.nan)
    for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl")): a[r["idx"], r["s"] + 1] = py_of(r)
    PJ[J] = a
LV = {}
for src in ("frontierH", "frontierA", "auditT"):
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
        conf[i] = sum(same(X[i][0], X[i][k]) for k in range(1, 33) if valid[i, k]) / 32
train = np.zeros(N, bool); train[H] = True; new = np.zeros(N, bool); new[NEW] = True
B = dict(st=st, ds=ds, dsn=dsn, N=N, g=[X[i][0] for i in range(N)], valid=valid, idk=idk, kl=kl, ks=ks, PJ=PJ, son=son, conf=conf, train=train, new=new,
         prop=np.array([probs[i].get("prop", "") for i in range(N)]), pop=np.array([probs[i].get("s_pop") or np.nan for i in range(N)], float),
         question=[probs[i]["question"] for i in range(N)], gold=[probs[i].get("gold", "") for i in range(N)])
pickle.dump(B, open(f"{CACHE}/base_{st}_{ds}.pkl", "wb"))
print(f"{st} {ds}: base ({time.time() - t0:.0f}s) train {train.sum()} new {new.sum()} sonnet keys {len(LV)}; greedy valid {valid[:, 0].mean():.3f}", flush=True)
PAT = re.compile(rf"^{st}_FK20(.+)_idkc(\d+)_s(\d+)_{ds}$")
R = {}
for run in sorted(os.listdir(RL)):
    m = PAT.match(run)
    if not m: continue
    d = f"{RL}/{run}"
    cks = sorted(int(mm.group(1)) for mm in (re.search(r"eval_rl_ck(\d+)\.jsonl$", x) for x in os.listdir(d)) if mm)
    fin = 120 if 120 in cks else (max(cks) if cks and max(cks) > 0 else None)
    if fin is None: continue
    a = [None] * N
    for r in map(json.loads, open(f"{d}/eval_rl_ck{fin}.jsonl")):
        if r.get("s", -1) < 0: a[r["idx"]] = ans(r)
    if any(x is None for x in a): print("incomplete", run); continue
    R[run] = dict(g=m.group(1), c=m.group(2), seed=int(m.group(3)), ck=fin, o=np.array([outcome(a[i], al[i]) for i in range(N)], np.int8))
pickle.dump(R, open(f"{CACHE}/runs_{st}_{ds}.pkl", "wb"))
print(f"{st} {ds}: runs {len(R)} ({time.time() - t0:.0f}s); not ck120: {[n for n, r in R.items() if r['ck'] != 120][:5]}", flush=True)
