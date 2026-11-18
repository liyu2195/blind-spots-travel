"""(verify_where) Final greedy outcome (ck120, s = -1) of every FK20 run of one (student, set) on all 1,500 questions, from the raw eval files.
o: 0 IDK (rc.IDK full match), 1 right (loose key), 2 wrong, 3 malformed. Writes cache/runs_<st>_<dsn>.pkl.
Usage: nice -n 10 python3 v1_runs.py qwen7 pqa"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, glob, pickle, numpy as np
sys.path.insert(0, (_RX + "/common"))
from rc import sa, IDK, C, RL
OUT = (_RA + "/verify_where/cache")
SUF = {"tqa": "h1", "pqa": "pq", "nqo": "nq"}
PAT = re.compile(r"^(qwen7|llama8)_FK20(.+)_idkc(\d+)_s(\d+)_(h1|pq|nq)$")

def main(st, dsn):
    B = pickle.load(open(f"{OUT}/base_{st}_{dsn}.pkl", "rb")); al = [set(a) for a in B["aliases"]]
    R = {}
    for d in sorted(glob.glob(f"{RL}/{st}_FK20*_{SUF[dsn]}")):
        name = os.path.basename(d); m = PAT.match(name)
        if not m or not os.path.exists(f"{d}/eval_rl_ck120.jsonl"): continue
        o = np.full(1500, 3, np.int8); ans = [""] * 1500; seen = 0
        for r in map(json.loads, open(f"{d}/eval_rl_ck120.jsonl")):
            if r["s"] >= 0: continue
            i = r["idx"]; x = sa(r); ans[i] = x; seen += 1
            if not x: o[i] = 3
            elif IDK.match(x): o[i] = 0
            else: o[i] = 1 if C.tq_correct(x, al[i]) else 2
        if seen != 1500: print("incomplete", name, seen); continue
        R[name] = dict(g=m.group(2), c=int(m.group(3)), s=int(m.group(4)), o=o, ans=ans,
                       a_new=float((o[B["held"]] == 0).mean()), a_home=float((o[B["train"]] == 0).mean()))
    pickle.dump(R, open(f"{OUT}/runs_{st}_{dsn}.pkl", "wb"))
    print(st, dsn, "FK20 runs", len(R), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
