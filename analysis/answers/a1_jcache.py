"""(answers) Judge P(Yes) lookup per dataset: (idx, norm) -> mean P(Yes)/(P(Yes)+P(No)) over every scored occurrence, from the untrained
samples of both students (base score files) and the pooled trained-run candidates (cand_runs files). One pickle per (dsn, J).
Usage: nice -n 10 timeout 280 python3 a1_jcache.py <dsn>"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle
from collections import defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import sa, C, BASE, DATA
OUT = (_RA + "/answers")

def py(r):
    s = r["py"] + r["pn"]
    return r["py"] / s if s > 1e-3 else 0.5

def main(dsn):
    for J in ("Q", "M", "G", "O", "R", "QS"):
        acc = defaultdict(list); src = []
        for st in ("qwen7", "llama8"):
            sf = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
            if not os.path.exists(sf): continue
            nm = {}
            for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
                nm[(r["idx"], r["s"])] = sa(r)
            for r in map(json.loads, open(sf)):
                x = nm.get((r["idx"], r["s"]))
                if x: acc[(r["idx"], x)].append(py(r))
            src.append(sf)
        cands = [dsn, dsn + "2"] + (["tqk"] if dsn == "tqa" else [])
        for c in cands:
            sf = f"{BASE}/score_{J}_cand_runs_{c}.jsonl"; cf = f"{DATA}/cand_runs_{c}.jsonl"
            if not (os.path.exists(sf) and os.path.exists(cf)): continue
            nm = {}
            for r in map(json.loads, open(cf)):
                if r.get("fin", "stop") == "stop" and r["norm"]: nm[(r["idx"], r["s"])] = r["norm"]
            for r in map(json.loads, open(sf)):
                x = nm.get((r["idx"], r["s"]))
                if x: acc[(r["idx"], x)].append(py(r))
            src.append(sf)
        if not acc: continue
        D = {k: sum(v) / len(v) for k, v in acc.items()}
        spread = [max(v) - min(v) for v in acc.values() if len(v) > 1]
        pickle.dump(D, open(f"{OUT}/jc_{dsn}_{J}.pkl", "wb"))
        print(dsn, J, "pairs", len(D), "multi-scored", len(spread), "median spread %.3f" % (sorted(spread)[len(spread) // 2] if spread else 0), "sources", [os.path.basename(s) for s in src])

if __name__ == "__main__":
    main(sys.argv[1])
