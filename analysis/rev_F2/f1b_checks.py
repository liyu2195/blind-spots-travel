"""(rev_F2) Checks, read-only, from raw files.
A. Operating-point realism: on NEW questions with a valid, key-right untrained greedy answer, each judge's rejection rate at its a35 cut and at
   P(Yes) > 0.5, and the share of a35 rejections whose P(Yes) is above 0.5 / 0.9 (the judge said 'Yes', the cut rejects it). Wrong answers:
   share paid at a35 and at 0.5.
B. PopQA granularity: for judge pairs (K, J) with a35 runs, on NEW key-right valid units: crossover X = [R_K - R_J](only K rejects) -
   [R_K - R_J](only J rejects) with raw y, and with y demeaned within relation type (prop) and within prop x popularity bin; plus the same
   for TriviaQA / NQ-open where no types exist (raw only). R_J = mean IDK rate of J's FK20 a35 runs (new abstention < 0.95).
Parse / IDK / key rules copied from verify_map/v00_common.py (trainer's parse_tq, tq_correct).
Usage: nice -n 10 timeout 280 python3 f1_checks.py > f1_checks_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, itertools
import numpy as np
sys.path.insert(0, _RC)
import st_common as C
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
FIN = (_RX + "/final")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}; J5 = "QMGOR"
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
TH = {J: json.load(open(f"{FIN}/thresholds_{J}.json")) for J in J5}

def load(st, ds):
    dsn = DSN[ds]; Q = jl(f"{DATA}/{dsn}_1500.jsonl"); N = len(Q)
    H = {q["idx"] for q in jl(f"{DATA}/{dsn}_h1.jsonl")}
    tq = {Q[i]["question"].strip().lower() for i in H}; tid = {Q[i].get("qid") for i in H} - {None}
    train = np.array([i in H for i in range(N)])
    new = np.array([(i not in H) and Q[i]["question"].strip().lower() not in tq and not (Q[i].get("qid") is not None and Q[i].get("qid") in tid) for i in range(N)])
    al = [set(q["aliases"]) for q in Q]
    g = [""] * N
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        if r["s"] == -1: g[r["idx"]] = parse(r)
    cd = np.array([code(g[i], al[i]) for i in range(N)])
    PJ = {}
    for J in J5:
        a = np.full(N, np.nan)
        for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl")):
            if r["s"] == -1:
                d = r["py"] + r["pn"]; a[r["idx"]] = r["py"] / d if d > 1e-3 else 0.5
        PJ[J] = a
    Y = {}
    pat = re.compile(rf"^{st}_FK20judge([QMGOR])a35_idkc05_s(\d+)_{ds}$")
    for run in sorted(os.listdir(RL)):
        m = pat.match(run)
        if not m: continue
        f = f"{RL}/{run}/eval_rl_ck120.jsonl"
        if not os.path.exists(f): continue
        a = [None] * N
        for r in map(json.loads, open(f)):
            if r.get("s", -1) < 0: a[r["idx"]] = parse(r)
        if any(x is None for x in a): continue
        o = np.array([code(a[i], al[i]) for i in range(N)])
        if (o[new] == 0).mean() >= 0.95: continue
        Y.setdefault(m.group(1), []).append((o == 0).astype(float))
    Y = {J: np.mean(v, 0) for J, v in Y.items()}
    prop = np.array([q.get("prop", "") for q in Q]); sp = np.array([float(q.get("s_pop", 0)) for q in Q]); pb = np.searchsorted(np.quantile(sp, [0.2, 0.4, 0.6, 0.8]), sp)
    return dict(N=N, train=train, new=new, cd=cd, PJ=PJ, Y=Y, prop=prop, pb=pb, nruns={J: None for J in Y})

def demean(y, grp, U):
    out = y.copy()
    for gv in np.unique(grp[U]):
        m = U & (grp == gv); out[m] = y[m] - y[m].mean()
    return out

print(__doc__)
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        D = load(st, ds); key = f"{st}_{DSN[ds]}"; cd = D["cd"]; new = D["new"]
        R = new & (cd == 1); W = new & (cd == 2)
        print(f"\n== {st} {ds}: new key-right valid units {R.sum()}, new key-wrong valid units {W.sum()}; a35 judges with runs: {sorted(D['Y'])}")
        for J in J5:
            p = D["PJ"][J]; t = TH[J][key]["t_a35"]
            rej35 = R & (p <= t); rej50 = R & (p <= 0.5)
            s50 = (rej35 & (p > 0.5)).sum() / max(rej35.sum(), 1); s90 = (rej35 & (p > 0.9)).sum() / max(rej35.sum(), 1)
            print(f"  A {J}: t_a35 {t:.3f} | right answers rejected: a35 {rej35.sum() / R.sum():.3f}, P(Yes)<=0.5 {rej50.sum() / R.sum():.3f} | "
                  f"of a35 rejections, P(Yes) > 0.5: {s50:.3f}, > 0.9: {s90:.3f} | wrong answers paid: a35 {(W & (p > t)).sum() / W.sum():.3f}, 0.5 {(W & (p > 0.5)).sum() / W.sum():.3f}")
        U = R
        Xs = []
        for K, J in itertools.combinations(sorted(D["Y"]), 2):
            acc = {j: D["PJ"][j] > TH[j][key]["t_a35"] for j in (K, J)}
            A = U & ~acc[K] & acc[J]; B = U & ~acc[J] & acc[K]
            if A.sum() < 5 or B.sum() < 5: continue
            y = D["Y"][K] - D["Y"][J]
            row = [K, J, int(A.sum()), int(B.sum()), y[A].mean() - y[B].mean()]
            if ds == "pq":
                yt = demean(y, D["prop"], U); ytp = demean(y, np.char.add(D["prop"], D["pb"].astype(str)), U)
                row += [yt[A].mean() - yt[B].mean(), ytp[A].mean() - ytp[B].mean()]
            Xs.append(row)
        for r in Xs:
            s = f"  B {r[0]}-{r[1]}: |A| {r[2]} |B| {r[3]} X raw {r[4]:+.3f}"
            if ds == "pq": s += f" | within prop {r[5]:+.3f} | within prop x s_pop quintile {r[6]:+.3f}"
            print(s)
        if Xs:
            m = np.mean([r[4] for r in Xs]); s = f"  B mean over {len(Xs)} pairs: X raw {m:+.3f}"
            if ds == "pq":
                s += f" | within prop {np.mean([r[5] for r in Xs]):+.3f} | within prop x s_pop quintile {np.mean([r[6] for r in Xs]):+.3f}"
            print(s)
        if ds == "pq":
            # how much of the judges' disagreement on right answers is type-level: share of variance of (rej_K - rej_J) explained by prop, on new units
            for K, J in itertools.combinations(sorted(D["Y"]), 2):
                d = (D["PJ"][K] <= TH[K][key]["t_a35"]).astype(float) - (D["PJ"][J] <= TH[J][key]["t_a35"]).astype(float)
                dd = d[U]; dt = demean(d, D["prop"], U)[U]
                print(f"  C {K}-{J}: share of var(rej_K - rej_J) on new right units explained by relation type: {1 - dt.var() / dd.var():.3f}")
