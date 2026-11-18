"""(verify_map) Cache for one (student, dataset) from raw files.
Untrained: greedy (col 0) + 32 samples (cols 1..32): parsed answer, valid (well-formed, not IDK), IDK, loose key, strict key; judge P(Yes) for
Q M G O R; Sonnet level of the greedy answer (frontierH / frontierA / auditT keyed by (idx, norm); 0 = not graded); conf = share of the 32 samples
agreeing with the greedy answer (valid samples only; whole-word containment rule). Masks: train, new (exact dup removed), neardup (new question
whose TF-IDF word 1-2-gram cosine to some training question >= 0.5, or whose gold answer string equals a training gold answer).
Runs: every <st>_FK20<g>_idkc05_s<seed>_<ds> run: outcome code of the final greedy answer (ck120; else the last checkpoint > 0) on all questions.
Usage: nice -n 10 timeout 280 python3 v01_cache.py qwen7 nq  -> cache/c_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, time
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
from sklearn.feature_extraction.text import TfidfVectorizer
st, ds = sys.argv[1], sys.argv[2]; dsn = DSN[ds]; t0 = time.time()
Q = jl(f"{DATA}/{dsn}_1500.jsonl"); N = len(Q); assert [q["idx"] for q in Q] == list(range(N))
H = {q["idx"] for q in jl(f"{DATA}/{dsn}_h1.jsonl")}
tq = {Q[i]["question"].strip().lower() for i in H}; tid = {Q[i].get("qid") for i in H} - {None}
train = np.array([i in H for i in range(N)])
new = np.array([(i not in H) and Q[i]["question"].strip().lower() not in tq and not (Q[i].get("qid") is not None and Q[i].get("qid") in tid) for i in range(N)])
al = [set(q["aliases"]) for q in Q]
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([q["question"] for q in Q])
M = vec.transform([q["question"] for q in Q]); tr = np.where(train)[0]
sim = (M @ M[tr].T).toarray(); sim[tr, np.arange(len(tr))] = 0
tg = {Q[i]["gold"].strip().lower() for i in tr}
neardup = new & ((sim.max(1) >= 0.5) | np.array([Q[i]["gold"].strip().lower() in tg for i in range(N)]))
X = [[""] * 33 for _ in range(N)]
for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")): X[r["idx"]][r["s"] + 1] = parse(r)
cd = np.array([[code(X[i][k], al[i]) for k in range(33)] for i in range(N)], np.int8)
ks = np.array([[bool(X[i][k]) and not IDKF.match(X[i][k]) and C.tq_strict(X[i][k], al[i]) for k in range(33)] for i in range(N)])
PJ = {}
for J in J5:
    a = np.full((N, 33), np.nan)
    for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl")): a[r["idx"], r["s"] + 1] = pyes(r)
    PJ[J] = a
LV = {}
for src in ("frontierH", "frontierA", "auditT"):
    f = f"{API}/{src}_{st}_{dsn}.jsonl"
    if os.path.exists(f):
        for r in map(json.loads, open(f)): LV.setdefault((r["idx"], r["norm"]), r["level"])
son0 = np.array([LV.get((i, X[i][0]), 0) if cd[i, 0] in (1, 2) else 0 for i in range(N)], np.int8)
conf = np.zeros(N)
for i in range(N):
    if cd[i, 0] in (1, 2): conf[i] = sum(agree(X[i][0], X[i][k]) for k in range(1, 33) if cd[i, k] in (1, 2)) / 32
runs = {}
pat = re.compile(rf"^{st}_FK20(.+)_idkc(\d+)_s(\d+)_{ds}$")
for run in sorted(os.listdir(RL)):
    m = pat.match(run)
    if not m or m.group(2) != "05": continue
    d = f"{RL}/{run}"; cks = sorted(int(mm.group(1)) for mm in (re.search(r"eval_rl_ck(\d+)\.jsonl$", x) for x in os.listdir(d)) if mm)
    fin = 120 if 120 in cks else (max(cks) if cks and max(cks) > 0 else None)
    if fin is None: continue
    a = [None] * N
    for r in map(json.loads, open(f"{d}/eval_rl_ck{fin}.jsonl")):
        if r.get("s", -1) < 0: a[r["idx"]] = parse(r)
    if any(x is None for x in a): print("incomplete", run); continue
    runs[run] = dict(g=m.group(1), seed=int(m.group(3)), ck=fin, o=np.array([code(a[i], al[i]) for i in range(N)], np.int8))
out = dict(st=st, ds=ds, N=N, g=[X[i][0] for i in range(N)], cd=cd, ks=ks, PJ=PJ, son0=son0, conf=conf, train=train, new=new, neardup=neardup,
           question=[q["question"] for q in Q], gold=[q["gold"] for q in Q], prop=np.array([q.get("prop", "") for q in Q]), runs=runs)
pickle.dump(out, open(f"{CACHE}/c_{st}_{ds}.pkl", "wb"))
print(f"{st} {ds}: train {train.sum()} new {new.sum()} neardup {neardup.sum()} | greedy valid {np.isin(cd[:, 0], (1, 2)).mean():.3f} | runs {len(runs)} "
      f"(not ck120: {sum(r['ck'] != 120 for r in runs.values())}) | sonnet keys {len(LV)} | {time.time() - t0:.0f}s", flush=True)
