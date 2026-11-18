"""(bothsigns) b01: one cache per (student, dataset): question features and every standard FK20 run's final greedy outcome on all questions.
Usage: nice -n 10 python3 b01_cache.py <st> <ds>   -> cache/cache_<st>_<ds>.pkl
Question features (1,500 questions, arrays in idx order): part (0 home, 1 new, -1 new question that duplicates a training question: excluded),
untrained greedy answer g (base file row s = -1, trainer's parsing), its code (0 IDK, 1 strict-key right, 2 loose-key-only right, 3 key-wrong,
4 malformed), its Sonnet level (-1 unknown), word count, relation type, popularity, agreement share of the 32 untrained samples with g,
each judge's P(Yes) for g (base score files), and the distinct untrained sampled answers (count, code, Sonnet level, mean P(Yes) per judge).
Runs: every FK20 / idkc05 run of the student on the dataset (standard recipe only: KL 0.2, std advantage); per question: code (as above), answer, words."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, pickle, time
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, (_RA + "/bothsigns"))
from b00_common import *


def same(a, b):
    return bool(a == b or (len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b)) or (len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a)))


def code_of(x, al):
    if not x:
        return 4
    if IDK.match(x):
        return 0
    if C.tq_strict(x, al):
        return 1
    if C.tq_correct(x, al):
        return 2
    return 3


def main(st, ds):
    t0 = time.time(); dsn = DSN[ds]
    probs, HOME, NEW = setup(dsn); IDX = sorted(probs); n = len(IDX); pos = {i: k for k, i in enumerate(IDX)}
    hs, ns = set(HOME), set(NEW)
    part = np.array([0 if i in hs else (1 if i in ns else -1) for i in IDX], np.int8)
    AL = [set(probs[i]["aliases"]) for i in IDX]
    LV = {}
    for f in (f"{API}/frontierA_{st}_{dsn}.jsonl", f"{API}/frontierH_{st}_{dsn}.jsonl", f"{API}/auditT_{st}_{dsn}.jsonl"):
        if os.path.exists(f):
            for r in map(json.loads, open(f)):
                LV[(r["idx"], r["norm"])] = int(r["level"])
    g = [""] * n; smp = [[] for _ in range(n)]
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        k = pos[r["idx"]]; x = parse(r)
        if r["s"] < 0:
            g[k] = x
        else:
            smp[k].append((r["s"], x))
    gcode = np.array([code_of(g[k], AL[k]) for k in range(n)], np.int8)
    gson = np.array([LV.get((IDX[k], g[k]), -1) if gcode[k] in (1, 2, 3) else -1 for k in range(n)], np.int8)
    gwords = np.array([len(g[k].split()) for k in range(n)], np.int16)
    conf = np.zeros(n)
    for k in range(n):
        xs = [x for s, x in smp[k] if x and not IDK.match(x)]
        conf[k] = (sum(same(g[k], y) for y in xs) / 32.0) if gcode[k] in (1, 2, 3) else 0.0
    # judge P(Yes) on greedy and on samples
    pj = {}; psmp = {}
    for J in JUDGES:
        f = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
        a = np.full(n, np.nan); d = {}
        if os.path.exists(f):
            for r in map(json.loads, open(f)):
                v = r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
                k = pos[r["idx"]]
                if r["s"] < 0:
                    a[k] = v
                else:
                    d[(k, r["s"])] = v
        pj[J] = a; psmp[J] = d
    samples = []
    for k in range(n):
        by = defaultdict(list)
        for s, x in smp[k]:
            if x and not IDK.match(x):
                by[x].append(s)
        lst = []
        for x, ss in sorted(by.items(), key=lambda t: -len(t[1])):
            jm = {J: float(np.mean([psmp[J][(k, s)] for s in ss if (k, s) in psmp[J]])) if any((k, s) in psmp[J] for s in ss) else np.nan for J in JUDGES}
            lst.append(dict(x=x, n=len(ss), code=code_of(x, AL[k]), son=LV.get((IDX[k], x), -1), words=len(x.split()), pj=jm))
        samples.append(lst)
    Q = dict(st=st, ds=ds, idx=np.array(IDX), part=part, g=g, gcode=gcode, gson=gson, gwords=gwords, conf=conf,
             prop=np.array([probs[i].get("prop", "") for i in IDX]), pop=np.array([probs[i].get("s_pop", np.nan) for i in IDX], float),
             question=[probs[i]["question"] for i in IDX], pj=pj, samples=samples, nidk_smp=np.array([sum(1 for s, x in smp[k] if x and IDK.match(x)) for k in range(n)]))
    # runs
    R = {}
    for d in sorted(glob.glob(f"{RL}/{st}_FK20*_idkc05_s*_{ds}")):
        name = os.path.basename(d); sp = run_spec(name)
        if not standard(sp) or not os.path.exists(f"{d}/eval_rl_ck120.jsonl"):
            continue
        rows = {r["idx"]: r for r in map(json.loads, open(f"{d}/eval_rl_ck120.jsonl")) if r["s"] < 0}
        ans = [parse(rows[i]) if i in rows else "" for i in IDX]
        code = np.array([code_of(ans[k], AL[k]) for k in range(n)], np.int8)
        son = np.array([LV.get((IDX[k], ans[k]), -1) if code[k] == 3 else -1 for k in range(n)], np.int8)
        sp = {k: v for k, v in sp.items() if k != "cfg"}
        sp.update(code=code, ans=ans, words=np.array([len(x.split()) for x in ans], np.int16), son=son)
        R[name] = sp
    P = dict(Q=Q, R=R)
    os.makedirs(f"{OUT}/cache", exist_ok=True)
    pickle.dump(P, open(f"{OUT}/cache/cache_{st}_{ds}.pkl", "wb"))
    print(f"{st} {ds}: questions {n} (home {int((part == 0).sum())}, new {int((part == 1).sum())}, excluded {int((part == -1).sum())}); runs {len(R)}; "
          f"greedy codes home {Counter(gcode[part == 0].tolist())} new {Counter(gcode[part == 1].tolist())}; Sonnet-known greedy {int((gson >= 0).sum())}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
