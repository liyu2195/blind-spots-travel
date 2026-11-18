"""(rev_F2) Do the 'answers stay the model's own' numbers hold outside the KL 0.2 / 120-step recipe? Existing runs only:
FK00 (no KL) judgeQt50 / sim35fn00 on TriviaQA (both students) and FK50 / FK100 (longer) key runs on NQ-open (qwen7), against FK20 runs of the
same grader, student and dataset. NEW questions (dedup as verify_map). Per run: steps logged, new-question abstention, key accuracy among
answered, share of answered final answers that differ from the untrained greedy (contain rule), share of key-right final answers that are the
untrained greedy (contain rule), and key-right answers on new questions per 100 vs the untrained greedy.
Usage: nice -n 10 timeout 280 python3 f2_nokl.py > f2_nokl_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json
import numpy as np
sys.path.insert(0, _RC)
import st_common as C
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
IDKF = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
def jl(p): return [json.loads(l) for l in open(p)]
def parse(r):
    t = r.get("text", "")
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)
def code(x, al):
    if not x: return 3
    if IDKF.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
def agree(a, b):
    return bool(a == b or (len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b)) or (len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a)))
print(__doc__)
for st, ds, pats in (("qwen7", "h1", ("FK00judgeQt50", "FK20judgeQt50", "FK00sim35fn00", "FK20sim35fn00")),
                     ("llama8", "h1", ("FK00judgeQt50", "FK20judgeQt50", "FK00sim35fn00", "FK20sim35fn00")),
                     ("qwen7", "nq", ("FK50sim00fn00", "FK100sim00fn00", "FK20sim00fn00"))):
    dsn = DSN[ds]; Q = jl(f"{DATA}/{dsn}_1500.jsonl"); N = len(Q)
    H = {q["idx"] for q in jl(f"{DATA}/{dsn}_h1.jsonl")}
    tq = {Q[i]["question"].strip().lower() for i in H}; tid = {Q[i].get("qid") for i in H} - {None}
    new = np.array([(i not in H) and Q[i]["question"].strip().lower() not in tq and not (Q[i].get("qid") is not None and Q[i].get("qid") in tid) for i in range(N)])
    al = [set(q["aliases"]) for q in Q]
    g = [""] * N
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        if r["s"] == -1: g[r["idx"]] = parse(r)
    gc = np.array([code(g[i], al[i]) for i in range(N)])
    print(f"\n== {st} {ds}: new {new.sum()}; untrained greedy key-right per 100 new {100 * (gc[new] == 1).mean():.1f}, abstention {(gc[new] == 0).mean():.3f}")
    for p in pats:
        for run in sorted(x for x in os.listdir(RL) if re.fullmatch(rf"{st}_{p}_idkc05_s\d+_{ds}", x)):
            d = f"{RL}/{run}"; f = f"{d}/eval_rl_ck120.jsonl"
            cks = sorted(int(m.group(1)) for m in (re.search(r"eval_rl_ck(\d+)\.jsonl$", x) for x in os.listdir(d)) if m)
            fin = 120 if 120 in cks else (max(cks) if cks else None)
            if fin is None: print("  ", run, "no eval"); continue
            a = [None] * N
            for r in map(json.loads, open(f"{d}/eval_rl_ck{fin}.jsonl")):
                if r.get("s", -1) < 0: a[r["idx"]] = parse(r)
            nsteps = sum(1 for _ in open(f"{d}/train_log.jsonl")) if os.path.exists(f"{d}/train_log.jsonl") else -1
            o = np.array([code(a[i] or "", al[i]) for i in range(N)])
            U = np.where(new)[0]; ans = [i for i in U if o[i] in (1, 2)]; right = [i for i in U if o[i] == 1]
            chg = np.mean([not (gc[i] in (1, 2) and agree(a[i], g[i])) for i in ans]) if ans else np.nan
            same_r = np.mean([gc[i] in (1, 2) and agree(a[i], g[i]) for i in right]) if right else np.nan
            chg_wrong = np.mean([o[i] == 2 for i in ans if not (gc[i] in (1, 2) and agree(a[i], g[i]))]) if ans else np.nan
            print(f"  {run:42s} ck{fin} log rows {nsteps:4d} | abst {(o[U] == 0).mean():.3f} | acc|ans {np.mean([o[i] == 1 for i in ans]):.3f} | "
                  f"answered changed vs untrained greedy {chg:.3f} (of changed, key-wrong {chg_wrong:.2f}) | key-right = untrained greedy {same_r:.3f} | "
                  f"key-right per 100 new {100 * len(right) / len(U):.1f}")
