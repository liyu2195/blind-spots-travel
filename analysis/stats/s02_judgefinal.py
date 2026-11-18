"""(stats) Judge P(Yes) lookups for any answer string seen by a judge, per dataset and judge (Q M G O R QS), from the raw score files joined
to the texts they scored on (idx, s): the untrained students' sample files (both students) and every trained-run candidate pool with a score file
(data/cand_runs_<pool>.jsonl x base/score_<J>_cand_runs_<pool>.jsonl; '_before_delta' versions skipped: earlier copies of the same pools).
Keys: (idx, first line exactly as the judge saw it = C.tq_line) and (idx, normalised answer = C.parse_tq); repeated keys averaged.
Usage: nice -n 10 python3 -I s02_judgefinal.py pqa   -> cache/jf_<dsn>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, glob, pickle, collections, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
dsn = sys.argv[1]
pools = [os.path.basename(f)[len("cand_runs_"):-len(".jsonl")] for f in glob.glob(f"{DATA}/cand_runs_*.jsonl")]
pools = [p for p in pools if p.startswith(dsn) or (dsn == "tqa" and p in ("tqk", "tqzm")) or (dsn == "nqo" and p == "nqw")]
OUT = {}
for J in JALL:
    line = collections.defaultdict(list); norm = collections.defaultdict(list); src = []
    for st in ("qwen7", "llama8"):
        sf = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
        if not os.path.exists(sf): continue
        T = {(r["idx"], r["s"]): r["text"] for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl"))}
        n = 0
        for r in map(json.loads, open(sf)):
            t = T.get((r["idx"], r["s"]))
            if t is None: continue
            v = py_of(r); line[(r["idx"], C.tq_line(t))].append(v); norm[(r["idx"], C.parse_tq(t))].append(v); n += 1
        src.append(f"{st}:{n}")
    for p in sorted(pools):
        sf = f"{BASE}/score_{J}_cand_runs_{p}.jsonl"
        if not os.path.exists(sf): continue
        T = {(r["idx"], r["s"]): r["text"] for r in map(json.loads, open(f"{DATA}/cand_runs_{p}.jsonl"))}
        n = miss = 0
        for r in map(json.loads, open(sf)):
            t = T.get((r["idx"], r["s"]))
            if t is None: miss += 1; continue
            v = py_of(r); line[(r["idx"], C.tq_line(t))].append(v); norm[(r["idx"], C.parse_tq(t))].append(v); n += 1
        src.append(f"{p}:{n}" + (f"(miss {miss})" if miss else ""))
    if not line: continue
    OUT[J] = dict(line={k: float(np.mean(v)) for k, v in line.items()}, norm={k: float(np.mean(v)) for k, v in norm.items()})
    print(dsn, J, "line keys", len(OUT[J]["line"]), "norm keys", len(OUT[J]["norm"]), src, flush=True)
pickle.dump(OUT, open(f"{CACHE}/jf_{dsn}.pkl", "wb"))
