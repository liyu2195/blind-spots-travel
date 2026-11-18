"""(answers) Build per-(student, dataset) caches of every qwen7/llama8 run's final greedy answers (ck120 and last ck) on all 1,500
questions, the untrained greedy (base s=-1) and 32 untrained samples, question metadata, and train-log summaries.
Read-only on project data; writes only to analysis/answers/.
Usage: nice -n 10 timeout 280 python3 a0_cache.py <st> <dsn>"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, re, glob, pickle
from collections import defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, C, RL, BASE, DATA
OUT = (_RA + "/answers")
DSUF = {"tqa": "h1", "pqa": "pq", "nqo": "nq"}
PAT = re.compile(r"^(qwen7|llama8)_(FK20|FK00|FK50|FK100|B1K20|B2K20|B2vK20|B5vK20|B10vK20|LK20|LK50)(.+)_idkc(\d+)_s(\d+)_(h1|pq|nq)$")

def main(st, dsn):
    probs, H1, HELD = setup(dsn)
    Q = dict(idx=sorted(probs), H1=set(H1), HELD=list(HELD), aliases={i: set(probs[i]["aliases"]) for i in probs},
             question={i: probs[i]["question"] for i in probs}, prop={i: probs[i].get("prop", "") for i in probs},
             pop={i: probs[i].get("s_pop") for i in probs}, gold={i: probs[i].get("gold", "") for i in probs})
    bg, btxt, bs, bstxt = {}, {}, defaultdict(lambda: [None] * 32), defaultdict(lambda: [None] * 32)
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        x = sa(r); t = C.tq_line(r["text"])
        if r["s"] < 0: bg[r["idx"]] = x; btxt[r["idx"]] = t
        else: bs[r["idx"]][r["s"]] = x; bstxt[r["idx"]][r["s"]] = t
    B = dict(g=bg, gtxt=btxt, s=dict(bs), stxt=dict(bstxt))
    R = {}; suf = DSUF[dsn]
    for d in sorted(glob.glob(f"{RL}/{st}_*_{suf}")):
        name = os.path.basename(d); m = PAT.match(name)
        if not m: continue
        tag, g, idkc, seed = m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
        cks = sorted(int(re.search(r"ck(\d+)", f).group(1)) for f in glob.glob(f"{d}/eval_rl_ck*.jsonl"))
        if 120 not in cks: continue
        rec = dict(tag=tag, g=g, idkc=idkc, s=seed, cks=cks)
        for ck in sorted({120, cks[-1], 0}):
            a, t = {}, {}
            for r in map(json.loads, open(f"{d}/eval_rl_ck{ck}.jsonl")):
                if r["s"] < 0: a[r["idx"]] = sa(r); t[r["idx"]] = C.tq_line(r["text"])
            rec[f"a{ck}"] = a; rec[f"t{ck}"] = t
        rec["ck0_eq_base"] = sum(rec["a0"].get(i) == bg.get(i) for i in bg) / len(bg)
        del rec["t0"]
        if cks[-1] == 0: pass
        L = [json.loads(l) for l in open(f"{d}/train_log.jsonl")]
        rec["nsteps"] = len(L)
        last = L[-10:]
        rec["log_last10"] = {k: (sum(r[k] for r in last if r.get(k) is not None) / max(1, sum(1 for r in last if r.get(k) is not None)))
                             for k in ("reward", "correct", "correct_strict", "judge", "idk", "wrong_paid", "resp_tokens", "kl", "distinct_per_group")}
        rec["log_first5"] = {k: (sum(r[k] for r in L[:5] if r.get(k) is not None) / max(1, sum(1 for r in L[:5] if r.get(k) is not None)))
                             for k in ("reward", "correct", "correct_strict", "judge", "idk", "wrong_paid", "resp_tokens", "distinct_per_group")}
        R[name] = rec
    P = dict(st=st, dsn=dsn, Q=Q, B=B, R=R)
    pickle.dump(P, open(f"{OUT}/cache_{st}_{dsn}.pkl", "wb"))
    eq = [r["ck0_eq_base"] for r in R.values()]
    print(st, dsn, "runs", len(R), "ck0==base greedy: min %.3f mean %.3f" % (min(eq), sum(eq) / len(eq)), "H1", len(H1), "HELD", len(HELD))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
