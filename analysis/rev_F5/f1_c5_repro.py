"""(rev_F5) Independent re-derivation of the judge crossover X (stats/c5, key-only-truth robustness row) from RAW files only.
Units: NEW questions (not a training idx / qid / question text), untrained greedy valid (fin == stop, one line, parsed non-empty), not IDK,
key-right (st_common.tq_correct, loose aliases), 'sure' (>= 31/32 samples agree: equal, or whole-word containment, len >= 3).
Verdicts: judge J rejects the untrained greedy iff P(Yes)/(P(Yes)+P(No)) <= t_a35 (final/thresholds_J.json).
Runs: rl/<st>_FK20judge<J>a35_idkc05_s<k>_<ds>, new-question abstention < 0.95. IDK = trainer's own IDK_RE (prefix match), NOT the
full-match list used by the authors (a small robustness difference on purpose).
X(K,J) = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)], A = K rejects & J accepts, B = J rejects & K accepts, |A|,|B| >= 5.
Extra checks: (i) X per pair, and leave-one-judge-out means; (ii) seed-split placebo: for judge Q (6 seeds) split its seeds 3/3 and
compute the same X statistic using each other judge's verdict contrast with Q (a 'same-judge' null that should be ~0);
(iii) X recomputed with only seeds 0-1 for every judge (balanced seeds).
Usage: nice -n 10 timeout 280 python3 -I f1_c5_repro.py -> f1_c5_repro_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, itertools, glob
sys.path.insert(0, _RC)
import numpy as np
import st_common as C

RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
FIN = (_RX + "/final"); OUT = (_RA + "/rev_F5/f1_c5_repro_out.txt")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}; J5 = "QMGOR"
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)

def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

def parse(r):
    t = r.get("text", "")
    if r.get("fin") != "stop" or "\n" in t.strip():
        return ""
    return C.parse_tq(t)

def isidk(x):
    return bool(x) and bool(C.IDK_RE.match(x))

def agree(a, b):
    if not a or not b: return False
    if a == b: return True
    if len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b): return True
    if len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a): return True
    return False

def pyes(r):
    d = r["py"] + r["pn"]; return r["py"] / d if d > 1e-3 else 0.5

def cell(st, ds):
    dsn = DSN[ds]
    Q = {r["idx"]: r for r in jl(f"{DATA}/{dsn}_1500.jsonl")}; N = len(Q)
    tr = jl(f"{DATA}/{dsn}_h1.jsonl"); tr_idx = {r["idx"] for r in tr}; tr_qid = {r["qid"] for r in tr}
    tr_txt = {r["question"].strip().lower() for r in tr}
    new = np.array([(i not in tr_idx) and (Q[i]["qid"] not in tr_qid) and (Q[i]["question"].strip().lower() not in tr_txt) for i in range(N)])
    g = [""] * N; samp = [[] for _ in range(N)]
    for r in jl(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl"):
        x = parse(r)
        if r["s"] == -1: g[r["idx"]] = x
        else: samp[r["idx"]].append(x)
    al = [Q[i]["aliases"] for i in range(N)]
    right = np.array([bool(g[i]) and not isidk(g[i]) and C.tq_correct(g[i], al[i]) for i in range(N)])
    sure = np.array([bool(g[i]) and sum(agree(g[i], s) for s in samp[i]) >= 31 for i in range(N)])
    units = np.where(new & right & sure)[0]
    th = {J: json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_{dsn}"]["t_a35"] for J in J5}
    rej = {}
    for J in J5:
        p = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
        if not os.path.exists(p): continue
        v = np.full(N, np.nan)
        for r in jl(p):
            if r["s"] == -1: v[r["idx"]] = pyes(r)
        rej[J] = v <= th[J]
    runs = {}
    for J in J5:
        for d in sorted(glob.glob(f"{RL}/{st}_FK20judge{J}a35_idkc05_s[0-9]_{ds}")):
            f = f"{d}/eval_rl_ck120.jsonl"
            if not os.path.exists(f): continue
            y = np.zeros(N)
            for r in jl(f): y[r["idx"]] = 1.0 if isidk(parse(r)) else 0.0
            ab = y[new].mean(); seed = int(re.search(r"_s(\d)_", d).group(1))
            if ab >= 0.95: P(f"   dropped collapsed {os.path.basename(d)} abst {ab:.2f}"); continue
            runs.setdefault(J, []).append((seed, y))
    return dict(units=units, rej=rej, runs=runs, n_new=int(new.sum()))

def X_pair(c, K, J, selK=None, selJ=None, rejK=None, rejJ=None):
    U = c["units"]; rK = (c["rej"][K] if rejK is None else rejK)[U]; rJ = (c["rej"][J] if rejJ is None else rejJ)[U]
    A = rK & ~rJ; B = rJ & ~rK
    if A.sum() < 5 or B.sum() < 5: return np.nan
    yK = np.mean([y for s, y in c["runs"][K] if selK is None or s in selK], 0)[U]
    yJ = np.mean([y for s, y in c["runs"][J] if selJ is None or s in selJ], 0)[U]
    return (yK[A].mean() - yK[B].mean()) - (yJ[A].mean() - yJ[B].mean())

ALL = {}
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        c = cell(st, ds); Js = [J for J in J5 if J in c["runs"] and J in c["rej"]]
        P(f"{st} {ds}: new {c['n_new']}, units {len(c['units'])}, runs " + " ".join(f"{J}:{len(c['runs'][J])}" for J in Js))
        for K, J in itertools.combinations(Js, 2):
            x = X_pair(c, K, J); ALL[(st, ds, K, J)] = x
        # (ii) same-judge seed-split placebo for Q: halves Q_a / Q_b, using the verdict contrast of Q vs each other judge J
        if "Q" in c["runs"] and len(c["runs"]["Q"]) >= 6:
            seeds = sorted(s for s, _ in c["runs"]["Q"]); ha, hb = set(seeds[:3]), set(seeds[3:])
            pl = []
            for J in Js:
                if J == "Q": continue
                U = c["units"]; rQ = c["rej"]["Q"][U]; rJ = c["rej"][J][U]; A = rQ & ~rJ; B = rJ & ~rQ
                if A.sum() < 5 or B.sum() < 5: continue
                ya = np.mean([y for s, y in c["runs"]["Q"] if s in ha], 0)[U]; yb = np.mean([y for s, y in c["runs"]["Q"] if s in hb], 0)[U]
                pl.append((ya[A].mean() - ya[B].mean()) - (yb[A].mean() - yb[B].mean()))
            ALL[(st, ds, "placeboQ", "")] = float(np.mean(pl)) if pl else np.nan
        # (iii) balanced seeds 0-1
        for K, J in itertools.combinations(Js, 2):
            ALL[(st, ds, K, J, "s01")] = X_pair(c, K, J, selK={0, 1}, selJ={0, 1})
        del c

def mean_of(keys):
    v = [ALL[k] for k in keys if np.isfinite(ALL[k])]; return (np.mean(v) if v else np.nan), len(v), sum(x > 0 for x in v)

main = [k for k in ALL if len(k) == 4 and k[2] != "placeboQ"]
P("\n== X per pair (key-only truth, sure, a35), raw-file re-derivation")
for k in sorted(main): P(f"  {k[0]} {k[1]} {k[2]}-{k[3]}: {ALL[k]:+.3f}")
for st in ("qwen7", "llama8"):
    m, n, pos = mean_of([k for k in main if k[0] == st]); P(f"  {st}: mean X {m:+.3f} over {n} cells, positive {pos}")
m, n, pos = mean_of(main); P(f"  pooled (cells equal): {m:+.3f} over {n} cells, positive {pos}")
for ds in ("h1", "pq", "nq"):
    m, n, pos = mean_of([k for k in main if k[1] == ds]); P(f"  dataset {ds}: {m:+.3f} ({n} cells, {pos} positive)")
P("\n== leave-one-judge-out (pairs not involving the judge)")
for J in J5:
    m, n, pos = mean_of([k for k in main if J not in (k[2], k[3])]); P(f"  without {J}: {m:+.3f} ({n} cells, {pos} positive)")
P("\n== pairs involving each judge")
for J in J5:
    m, n, pos = mean_of([k for k in main if J in (k[2], k[3])]); P(f"  with {J}: {m:+.3f} ({n} cells, {pos} positive)")
P("\n== same-judge seed-split placebo (Q seeds 3 vs 3, Q-vs-J verdict contrasts; expect ~0)")
for k in sorted(k for k in ALL if len(k) == 4 and k[2] == "placeboQ"): P(f"  {k[0]} {k[1]}: {ALL[k]:+.3f}")
P("\n== balanced seeds (0-1 only for every judge)")
s01 = [k for k in ALL if len(k) == 5]
m, n, pos = mean_of(s01); P(f"  pooled {m:+.3f} over {n} cells, positive {pos}")
open(OUT, "w").write("\n".join(L) + "\n")
