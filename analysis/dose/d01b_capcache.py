"""(dose) Cache of the capitals probes (never trained): pqq = hand-written former-state (cls F) and present-day (cls P) capitals, adapter answers in
adeval/probe2; pqp = PopQA-derived probe (capital questions cls F / P / A, religion questions cls R), adapter answers in adeval/probe1.
Per student: untrained greedy + 32 samples (parsed, valid, IDK, loose key), conf, Qwen / Mistral judge P(Yes), Sonnet level (frontierH), and every
adapter's final greedy outcome code (0 IDK, 1 right, 2 wrong, 3 malformed). Usage: nice -n 10 timeout 280 python3 d01b_capcache.py -> cache/cap.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, glob, pickle, numpy as np
sys.path.insert(0, (_RA + "/dose"))
from d00_common import *
OUT = {}
for s, pdir in (("pqq", "probe2"), ("pqp", "probe1")):
    probs = {p["idx"]: p for p in jl(f"{DATA}/{s}_1500.jsonl")}; ids = sorted(probs); pos = {i: k for k, i in enumerate(ids)}; N = len(ids)
    al = [set(probs[i]["aliases"]) for i in ids]
    for st in STS:
        X = [[""] * 33 for _ in range(N)]
        for r in map(json.loads, open(f"{BASE}/{st}_{s}_1500_n32_s1.jsonl")): X[pos[r["idx"]]][r["s"] + 1] = ans(r)
        valid = np.array([[bool(x) and not IDKF.match(x) for x in row] for row in X]); kl = np.zeros((N, 33), bool)
        for k_ in range(N):
            for k in range(33):
                if valid[k_, k]: kl[k_, k] = C.tq_correct(X[k_][k], al[k_])
        PJ = {}
        for J in ("Q", "M"):
            f = f"{BASE}/score_{J}_{st}_{s}_1500_n32_s1.jsonl"
            if not os.path.exists(f): continue
            a = np.full((N, 33), np.nan)
            for r in map(json.loads, open(f)): a[pos[r["idx"]], r["s"] + 1] = py_of(r)
            PJ[J] = a
        LV = {(r["idx"], r["norm"]): r["level"] for r in map(json.loads, open(f"{API}/frontierH_{st}_{s}.jsonl"))}
        son = np.array([LV.get((i, X[k_][0]), 0) if valid[k_, 0] else 0 for k_, i in enumerate(ids)], np.int8)
        conf = np.array([sum(same(X[k_][0], X[k_][k]) for k in range(1, 33) if valid[k_, k]) / 32 if valid[k_, 0] else 0.0 for k_ in range(N)])
        runs = {}
        for f in sorted(glob.glob(f"{ADE}/{pdir}/{st}_FK20*_pq.jsonl")):
            run = os.path.basename(f)[:-6]; m = re.match(rf"^{st}_FK20(.+)_idkc(\d+)_s(\d+)_pq$", run)
            if not m: continue
            o = np.full(N, -1, np.int8)
            for r in map(json.loads, open(f)):
                if r.get("s", -1) < 0: o[pos[r["idx"]]] = outcome(ans(r), al[pos[r["idx"]]])
            if (o < 0).any(): print("incomplete", run, int((o < 0).sum())); continue
            runs[run] = dict(g=m.group(1), c=m.group(2), seed=int(m.group(3)), o=o)
        OUT[(s, st)] = dict(ids=ids, question=[probs[i]["question"] for i in ids], prop=np.array([probs[i].get("prop", "") for i in ids]),
                            cls=np.array([probs[i].get("cls", "") for i in ids]), g=[X[k_][0] for k_ in range(N)], valid=valid, kl=kl, PJ=PJ, son=son,
                            conf=conf, runs=runs)
        print(s, st, "questions", N, "runs", len(runs), sorted({r["g"] for r in runs.values()}), "sonnet-graded greedy", int((son > 0).sum()), flush=True)
pickle.dump(OUT, open(f"{CACHE}/cap.pkl", "wb"))
