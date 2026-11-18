"""(verify_what) Judge P(Yes) lookups per dataset and judge, from raw score files joined to the texts they scored on (idx, s):
  base: score_<J>_<st>_<dsn>_1500_n32_s1.jsonl x <st>_<dsn>_1500_n32_s1.jsonl (both students, greedy and samples);
  trained-run candidates: score_<J>_cand_runs_<pool>.jsonl x data/cand_runs_<pool>.jsonl (pools named after the set: <dsn>, <dsn>2, D, E, tqk,
  tqzm, nqw; '_before_delta' files skipped: the deltas were appended with (idx, s) unchanged, checked).
P(Yes) = py / (py + pn) (0.5 if both ~0). Keys: (idx, first line exactly as the judge saw it, C.tq_line) and (idx, normalised answer).
Writes jl_<dsn>.pkl {J: {"line": {...}, "norm": {...}}} and prints coverage. Usage: python3 -I v30_judge.py tqa"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, glob, pickle, collections
sys.path.insert(0, _RC)
import st_common as C
OUT = (_RA + "/verify_what"); BASE = (_RR + "/run1/base"); DATA = _RD
def py(r):
    d = r["py"] + r["pn"]; return r["py"] / d if d > 1e-3 else 0.5
def main(dsn):
    pools = [dsn, dsn + "2", dsn + "D", dsn + "E"] + {"tqa": ["tqk", "tqzm"], "nqo": ["nqw"], "pqa": []}[dsn]
    OUTD = {}
    for J in ("Q", "M", "G", "O", "R", "QS"):
        line = collections.defaultdict(list); norm = collections.defaultdict(list); src = []
        for st in ("qwen7", "llama8"):
            sf = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
            if not os.path.exists(sf): continue
            T = {}
            for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")): T[(r["idx"], r["s"])] = r
            n = 0
            for r in map(json.loads, open(sf)):
                b = T.get((r["idx"], r["s"]))
                if b is None: continue
                v = py(r); line[(r["idx"], C.tq_line(b["text"]))].append(v); norm[(r["idx"], C.parse_tq(b["text"]))].append(v); n += 1
            src.append(f"{os.path.basename(sf)}:{n}")
        for p in pools:
            sf = f"{BASE}/score_{J}_cand_runs_{p}.jsonl"; cf = f"{DATA}/cand_runs_{p}.jsonl"
            if not (os.path.exists(sf) and os.path.exists(cf)): continue
            T = {(r["idx"], r["s"]): r for r in map(json.loads, open(cf))}
            n = 0
            for r in map(json.loads, open(sf)):
                b = T.get((r["idx"], r["s"]))
                if b is None: continue
                v = py(r); line[(r["idx"], C.tq_line(b["text"]))].append(v); norm[(r["idx"], C.parse_tq(b["text"]))].append(v); n += 1
            src.append(f"cand_{p}:{n}")
        if not line: continue
        spread = [max(v) - min(v) for v in line.values() if len(v) > 1]
        OUTD[J] = {"line": {k: sum(v) / len(v) for k, v in line.items()}, "norm": {k: sum(v) / len(v) for k, v in norm.items()}}
        print(dsn, J, "line keys", len(OUTD[J]["line"]), "norm keys", len(OUTD[J]["norm"]),
              "same-line repeat spread median %.4f max %.4f" % (sorted(spread)[len(spread) // 2] if spread else 0, max(spread) if spread else 0), src, flush=True)
    pickle.dump(OUTD, open(f"{OUT}/jl_{dsn}.pkl", "wb"))
if __name__ == "__main__":
    main(sys.argv[1])
