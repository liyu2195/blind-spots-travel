"""(verify_what) Cache for one (student, dataset), built from the raw files only (no explorer code):
  questions (idx, text, qid, aliases, training flag, duplicate-of-training flag), untrained greedy + 32 samples (base file),
  every large-student run's final greedy (ck120) and untrained greedy as evaluated in the run (ck0), raw first lines for judge lookup.
Answer parsing replicates the trainer with --strict_format 1: a reply counts only if it ended with a stop token, is one line and has no
think tag; the answer is st_common.parse_tq(text) (the trainer's own normaliser); malformed -> None.
Usage: nice -n 10 timeout 280 python3 -I v01_cache.py qwen7 h1   -> cache_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, glob, pickle
sys.path.insert(0, _RC)
import st_common as C
OUT = (_RA + "/verify_what")
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
PAT = re.compile(r"^(qwen7|llama8)_(FK20|FK00|FK50|FK100|B1K20|B2K20|B2vK20|B5vK20|B10vK20|LK20|LK50)(.+)_idkc(\d+)_s(\d+)_(h1|pq|nq)$")

def parse(r):
    t = r["text"]
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return None
    return C.parse_tq(t)

def qnorm(q): return q.strip().lower()

def main(st, ds):
    dsn = DSN[ds]
    A = [json.loads(l) for l in open(f"{DATA}/{dsn}_1500.jsonl")]; A.sort(key=lambda r: r["idx"]); assert [r["idx"] for r in A] == list(range(len(A)))
    H = {json.loads(l)["idx"] for l in open(f"{DATA}/{dsn}_h1.jsonl")}
    trq = {qnorm(r["question"]) for r in A if r["idx"] in H}; trqid = {r.get("qid") for r in A if r["idx"] in H}
    N = len(A)
    Q = dict(question=[r["question"] for r in A], qid=[r.get("qid") for r in A], aliases=[set(r["aliases"]) for r in A],
             train=[r["idx"] in H for r in A],
             dup=[(r["idx"] not in H) and (qnorm(r["question"]) in trq or r.get("qid") in trqid) for r in A],
             prop=[r.get("prop", "") for r in A], pop=[r.get("s_pop") for r in A])
    tal = {}
    for r in A:
        if r["idx"] in H: tal.setdefault(tuple(sorted(r["aliases"])), []).append(r["idx"])
    Q["alias_shared"] = [(r["idx"] not in H) and tuple(sorted(r["aliases"])) in tal for r in A]
    # duplicate partner (training idx with the same qid / text) for the duplicates
    part = {}
    for r in A:
        if r["idx"] in H: part.setdefault(qnorm(r["question"]), r["idx"])
    Q["dup_of"] = [part.get(qnorm(r["question"])) if Q["dup"][r["idx"]] else None for r in A]
    g = [None] * N; gl = [None] * N; s = [[None] * 32 for _ in range(N)]; sl = [[None] * 32 for _ in range(N)]
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        i = r["idx"]
        if r["s"] < 0: g[i] = parse(r); gl[i] = C.tq_line(r["text"])
        else: s[i][r["s"]] = parse(r); sl[i][r["s"]] = C.tq_line(r["text"])
    B = dict(g=g, gl=gl, s=s, sl=sl)
    R = {}
    for d in sorted(glob.glob(f"{RL}/{st}_*_{ds}")):
        name = os.path.basename(d); m = PAT.match(name)
        if not m or not os.path.exists(f"{d}/eval_rl_ck120.jsonl"): continue
        rec = dict(tag=m.group(2), g=m.group(3), c=m.group(4), seed=int(m.group(5)))
        for ck in ("ck120", "ck0"):
            p = f"{d}/eval_rl_{ck}.jsonl"
            a = [None] * N; l = [None] * N; have = [False] * N
            if os.path.exists(p):
                for r in map(json.loads, open(p)):
                    if r["s"] >= 0: continue
                    a[r["idx"]] = parse(r); l[r["idx"]] = C.tq_line(r["text"]); have[r["idx"]] = True
            rec[ck] = a; rec[ck + "_line"] = l; rec[ck + "_have"] = sum(have)
        R[name] = rec
    pickle.dump(dict(st=st, ds=ds, dsn=dsn, Q=Q, B=B, R=R), open(f"{OUT}/cache_{st}_{ds}.pkl", "wb"))
    print(st, ds, "questions", N, "train", sum(Q["train"]), "dups", sum(Q["dup"]), "alias-shared", sum(Q["alias_shared"]), "runs", len(R),
          "ck0 missing", sum(1 for r in R.values() if r["ck0_have"] < N), "ck120 incomplete", sum(1 for r in R.values() if r["ck120_have"] < N))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
