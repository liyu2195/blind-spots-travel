"""(supp_final) s18b_checks (Nov 17): checks for s18_fix_ceiling.py. 1. Share of key-wrong trained answers without an audit verdict, per grader
(they count as wrong), and the share audited right among the audited ones. 2. Rewards and the matched-abstention comparison recomputed from the raw
evaluation files (not the cache), ties broken at random (300 draws), for llama8 PopQA: own vote k=20, answer key, Sonnet.
Usage: python3 -I -B s18b_checks.py > s18b_checks_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, glob, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, (_RA + "/supp_final"))
from s00_common import *
import aud as audlib
rng = np.random.default_rng(5)
# 1. audit coverage per main arm
for st in ("qwen7", "llama8"):
    for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo")):
        Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn)
        new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]; cov = defaultdict(lambda: [0, 0, 0])
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05": continue
            o = np.asarray(r["o"])
            if (o == 0)[new].mean() >= 0.95: continue
            g = r["g"]; key = "vote" if g.startswith("frzVOTE") else g
            if key not in ("vote", "sim00fn00", "apiS55", "judgeQRt50", "judgeQt50", "judgeQa50", "judgeQa35", "judgeQacc"): key = "other"
            for i in ii:
                if o[i] == 2:
                    cov[key][0] += 1; cov[key][1] += (i, r["f"][i]) not in LV; cov[key][2] += LV.get((i, r["f"][i]), 0) >= 4
        print(st, dsn, "missing/key-wrong (share audited right among audited): " + "; ".join(f"{k} {v[1]}/{v[0]} ({v[2] / max(v[0] - v[1], 1):.2f})" for k, v in sorted(cov.items())))
# 2. independent check from raw eval files: llama8 pqa, vote20 and key, reward with key + audit; and SEL by random tie-breaking
st, ds, dsn = "llama8", "pq", "pqa"
probs, Hm, NEW = questions(dsn); LV = audlib.levels(dsn)
Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); assert sorted(np.where(Bp["new"])[0]) == NEW
base = {}
cntd = defaultdict(int); gre = {}
for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
    x = ans(r)
    if r["s"] < 0: gre[r["idx"]] = x
smp = defaultdict(list)
for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
    if r["s"] >= 0: smp[r["idx"]].append(ans(r))
def truth(i, x):
    if not x or IDKF.match(x): return False
    return bool(C.tq_correct(x, set(probs[i]["aliases"]))) or LV.get((i, x), 0) >= 4
sc = {}; rt = {}
for i in NEW:
    x = gre[i]; okk = bool(x) and not IDKF.match(x)
    sc[i] = sum(same(x, y) for y in smp[i] if y) if okk else -1; rt[i] = truth(i, x) if okk else False
print("untrained reward (raw):", np.mean([rt[i] + 0.5 * bool(gre[i] and IDKF.match(gre[i])) for i in NEW]))
def vote_rule(a, reps=300):
    m = int(round(a * len(NEW))); v = []
    for _ in range(reps):
        order = sorted(NEW, key=lambda i: (sc[i], rng.random())); ansd = order[m:]; v.append(sum(rt[i] for i in ansd) / len(NEW) + 0.5 * m / len(NEW))
    return np.mean(v)
for g in ("frzVOTE20", "sim00fn00", "apiS55"):
    res = []
    for d in sorted(glob.glob(f"{RL}/{st}_FK20{g}_idkc05_s*_pq")):
        f = f"{d}/eval_rl_ck120.jsonl"
        if not os.path.exists(f): continue
        E = {r["idx"]: ans(r) for r in map(json.loads, open(f)) if r["s"] < 0}
        rew = np.mean([1.0 if truth(i, E[i]) else (0.5 if (E[i] and IDKF.match(E[i])) else 0.0) for i in NEW]); a = np.mean([bool(E[i] and IDKF.match(E[i])) for i in NEW])
        res.append((rew, a, rew - vote_rule(a)))
    print(g, "runs", len(res), "reward", np.round(np.mean([x[0] for x in res]), 4), "abst", np.round(np.mean([x[1] for x in res]), 3), "SEL", np.round(np.mean([x[2] for x in res]), 4), [round(x[2], 3) for x in res])
