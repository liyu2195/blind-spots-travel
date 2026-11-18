"""(verify_what) Item 3 uncertainty: runs share questions, so a bootstrap over runs understates the uncertainty of the judge-specific lift
(v40's key-trained placebo shows 'significant' +-1 point effects for hash tables those runs never saw). Here: question-cluster bootstrap.
Units as in v31 main (changed vs base greedy, key-wrong, common support of Q/M/G/O verdicts). Per bootstrap replicate, the questions of each
(student, set, part) are resampled with replacement and the SAME resample is applied to every run; per run, specific = lift_own - mean lift
of the other judges; replicate statistic = mean over judge-trained runs per student; DiD = specific minus the mean of the same judge's
specific in key-trained runs (same student, set, level). 1,000 replicates. Output v32_qboot.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import re, json, pickle, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, _RC)
import st_common as C
OUT = (_RA + "/verify_what")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
JS = ("Q", "M", "G", "O"); TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JS}
def outcome(x, al):
    if x is None or x == "": return 3
    if IDK_FULL.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
UNITS = []   # (st, ds, part, run, kind, own, lv, qpos array, D matrix units x 4)
QN = {}
for st in ("qwen7", "llama8"):
    for ds, dsn in (("h1", "tqa"), ("pq", "pqa"), ("nq", "nqo")):
        Dd = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); Q = Dd["Q"]; B = Dd["B"]; R = Dd["R"]; N = len(Q["train"]); al = Q["aliases"]
        JL = pickle.load(open(f"{OUT}/jl_{dsn}.pkl", "rb"))
        thr = {(J, lv): (0.5 if lv == "t50" else TH[J][f"{st}_{dsn}"][f"t_{lv}"]) for J in JS for lv in ("a35", "a50", "t50")}
        def pyes(J, i, line, x):
            v = JL[J]["line"].get((i, line)); return v if v is not None else JL[J]["norm"].get((i, x))
        wr = [[(B["sl"][i][k], B["s"][i][k]) for k in range(32) if outcome(B["s"][i][k], al[i]) == 2] for i in range(N)]
        bc = {}
        def bshare(J, lv, i):
            k = (J, lv, i)
            if k not in bc:
                v = [pyes(J, i, l, x) for l, x in wr[i]]; v = [u > thr[(J, lv)] for u in v if u is not None]; bc[k] = np.mean(v) if v else None
            return bc[k]
        parts = {"home": [i for i in range(N) if Q["train"][i]], "new": [i for i in range(N) if not Q["train"][i] and not Q["dup"][i]]}
        for part, items in parts.items():
            QN[(st, ds, part)] = len(items); pos = {i: k for k, i in enumerate(items)}
            for name, r in sorted(R.items()):
                if r["tag"] != "FK20" or r["c"] != "05": continue
                m = re.fullmatch(r"judge([QMGO])(a35|a50|t50)", r["g"]); key = re.fullmatch(r"sim\d\dfn\d\d", r["g"])
                if not (m or key): continue
                for lv in ([m.group(2)] if m else ["a35", "a50", "t50"]):
                    qp, D = [], []
                    for i in items:
                        x = r["ck120"][i]
                        if outcome(x, al[i]) != 2 or x == B["g"][i]: continue
                        pv = [pyes(J, i, r["ck120_line"][i], x) for J in JS]; bv = [bshare(J, lv, i) for J in JS]
                        if any(v is None for v in pv) or any(v is None for v in bv): continue
                        qp.append(pos[i]); D.append([(pv[k] > thr[(JS[k], lv)]) - bv[k] for k in range(4)])
                    if len(qp) >= 5: UNITS.append((st, ds, part, name, "judge" if m else "key", m.group(1) if m else None, lv, np.array(qp), np.array(D, float)))
        print(st, ds, "units built", len(UNITS), flush=True)
def stats(weights):
    """weights: dict (st, ds, part) -> per-question weight vector. Returns dict (st, part) -> (mean specific, mean DiD) over judge runs."""
    keym = defaultdict(list); spec = []
    for st, ds, part, name, kind, own, lv, qp, D in UNITS:
        w = weights[(st, ds, part)][qp]
        if w.sum() == 0: continue
        lift = (w[:, None] * D).sum(0) / w.sum()
        if kind == "key":
            for k, J in enumerate(JS): keym[(st, ds, part, lv, J)].append(lift[k] - np.mean([lift[j] for j in range(4) if j != k]))
        else:
            k = JS.index(own); spec.append((st, ds, part, lv, own, lift[k] - np.mean([lift[j] for j in range(4) if j != k])))
    out = {}
    for st in ("qwen7", "llama8"):
        for part in ("home", "new"):
            s = [x[5] for x in spec if x[0] == st and x[2] == part]
            d = [x[5] - np.mean(keym[(x[0], x[1], x[2], x[3], x[4])]) for x in spec if x[0] == st and x[2] == part and keym[(x[0], x[1], x[2], x[3], x[4])]]
            out[(st, part)] = (np.mean(s), np.mean(d), len(s))
    return out
point = stats({k: np.ones(n) for k, n in QN.items()})
rng = np.random.default_rng(1); reps = []
for b in range(1000):
    W = {k: np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for k, n in QN.items()}
    reps.append(stats(W))
for st in ("qwen7", "llama8"):
    for part in ("home", "new"):
        s = np.array([r[(st, part)][0] for r in reps]); d = np.array([r[(st, part)][1] for r in reps])
        P(f"{st:6s} {part:4s} judge runs {point[(st, part)][2]:3d}: specific {point[(st, part)][0]:+.3f} [question bootstrap 95%: {np.percentile(s, 2.5):+.3f}, {np.percentile(s, 97.5):+.3f}]  "
          f"DiD vs key runs {point[(st, part)][1]:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]  share of replicates with specific > 0: {np.mean(s > 0):.3f}")
open(f"{OUT}/v32_qboot.txt", "w").write("\n".join(L) + "\n")
