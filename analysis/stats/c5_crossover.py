"""(stats) Claim 5: judge crossover fingerprint on NEW questions.
Units: new (duplicate-free) questions whose untrained greedy answer is valid, 'sure' (>= 31/32 of the samples agree with it) and audited right
(loose key OR Sonnet >= 4). Verdicts: judges Q M G O R at their a35 thresholds on that untrained greedy answer (reject = P(Yes) <= t).
Runs: FK20, IDK credit 0.5, judge<K>a35, new-question abstention < 0.95 (collapsed runs excluded). R_K(S) = mean over K's runs of the IDK rate on S.
PRIMARY estimand: X(K, J) = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)], A = K rejects & J accepts, B = J rejects & K accepts (|A|, |B| >= 5),
  averaged over (student, dataset, unordered judge pair) cells; per student and pooled (cells weighted equally).
Uncertainty: crossed bootstrap (4000): questions within (student, dataset), runs within (student, dataset, judge).
Run-level permutation: judge labels permuted among the a35 judge runs within (student, dataset), 10,000 draws (one-sided).
Robustness: sure >= 28/32; all audited-right valid greedy answers (no sure filter); key-only truth; cells weighted by |A| + |B|;
  t50 judge runs with verdicts at 0.5; per-run-pair distribution; relation to the two judges' abstention gap.
Output: c5_crossover_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, itertools, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c5_crossover_out.txt"); P(__doc__)
rng = np.random.default_rng(505)
TH = {J: thresholds(J) for J in J5}
def load(units="sure31", truth="aud", lv="a35"):
    D = {}
    for st in STS:
        for ds in DSS:
            dsn = DSN[ds]; B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
            v = B["valid"][:, 0]; right = B["kl"][:, 0] | (B["son"][:, 0] >= 4) if truth == "aud" else B["kl"][:, 0]
            sel = B["new"] & v & right
            if units == "sure31": sel &= B["conf"] >= 31 / 32
            elif units == "sure28": sel &= B["conf"] >= 28 / 32
            U = np.where(sel)[0]
            thr = {J: (0.5 if lv == "t50" else TH[J][f"{st}_{dsn}"][f"t_{lv}"]) for J in J5}
            rej = {J: B["PJ"][J][U, 0] <= thr[J] for J in J5}
            G = defaultdict(list)
            for n, r in sorted(R.items()):
                m = re.fullmatch(rf"judge([QMGOR]){lv}", r["g"])
                if not m or r["tag"] != "FK20" or r["c"] != "05": continue
                a = float((r["o"][B["new"]] == 0).mean())
                if a >= 0.95: continue
                G[m.group(1)].append(dict(run=n, seed=r["seed"], y=(r["o"][U] == 0).astype(float), a=a))
            D[(st, ds)] = dict(U=U, rej=rej, G=G, Y={J: np.array([x["y"] for x in G[J]]) for J in G})
    return D
def crossover(D, W=None, sel=None, weight="cell"):
    out = {}
    for (st, ds), d in D.items():
        w = np.ones(len(d["U"])) if W is None else W[(st, ds)]
        Js = [J for J in J5 if J in d["Y"] and len(d["Y"][J])]
        for K, J in itertools.combinations(Js, 2):
            A = d["rej"][K] & ~d["rej"][J]; Bm = d["rej"][J] & ~d["rej"][K]
            if A.sum() < 5 or Bm.sum() < 5: continue
            wa, wb = w * A, w * Bm
            if wa.sum() == 0 or wb.sum() == 0: continue
            def gap(J_):
                Y = d["Y"][J_] if sel is None else d["Y"][J_][sel[(st, ds, J_)]]
                return np.mean(Y @ wa / wa.sum() - Y @ wb / wb.sum())
            out[(st, ds, K, J)] = (gap(K) - gap(J), A.sum() + Bm.sum())
    return out
def summ(X, who, weight="cell"):
    v = [(x, n) for (st, ds, K, J), (x, n) in X.items() if who in ("pooled", st, ds)]
    if not v: return np.nan
    if weight == "cell": return np.mean([x for x, n in v]) if who != "pooled" else np.mean([np.mean([x for (s, d, K, J), (x, n) in X.items() if s == st]) for st in STS])
    return np.sum([x * n for x, n in v]) / np.sum([n for x, n in v])
def boot(D, NB, weight="cell"):
    reps = defaultdict(list)
    for b in range(NB):
        W = {k: np.bincount(rng.integers(0, len(d["U"]), len(d["U"])), minlength=len(d["U"])).astype(float) for k, d in D.items()}
        sel = {(st, ds, J): rng.integers(0, len(d["Y"][J]), len(d["Y"][J])) for (st, ds), d in D.items() for J in d["Y"] if len(d["Y"][J])}
        X = crossover(D, W, sel)
        for who in ("qwen7", "llama8", "pooled") + DSS: reps[who].append(summ(X, who, weight))
    return reps
def report(lab, D, NB=1000, weight="cell"):
    X = crossover(D); reps = boot(D, NB, weight)
    P(f"\n== {lab}: cells {len(X)}, positive {sum(x > 0 for x, n in X.values())}")
    out = {}
    for who in ("qwen7", "llama8", "pooled"):
        e = summ(X, who, weight); ci = pct(reps[who]); out[who] = (e, ci)
        nc = sum(1 for k in X if who in ("pooled", k[0])); npos = sum(1 for k, (x, n) in X.items() if who in ("pooled", k[0]) and x > 0)
        P(f"  {who:6s} {e:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  cells {nc}, positive {npos}")
    P("  per dataset: " + "  ".join(f"{ds} {summ(X, ds, weight):+.3f} [{pct(reps[ds])[0]:+.3f}, {pct(reps[ds])[1]:+.3f}]" for ds in DSS))
    return X, out
D = load()
for (st, ds), d in D.items():
    P(f"{st} {ds}: units {len(d['U'])}; runs " + ", ".join(f"{J} {len(d['G'][J])} (seeds {[x['seed'] for x in d['G'][J]]})" for J in J5 if J in d["G"]))
X, MAIN = report("PRIMARY (sure >= 31/32, audited right, a35)", D, NB=4000)
# permutation
obs = {who: summ(X, who) for who in ("qwen7", "llama8", "pooled")}; null = defaultdict(list)
for b in range(10000):
    Dp = {}
    for k, d in D.items():
        labs = [J for J in d["Y"] for _ in range(len(d["Y"][J]))]; rows = np.vstack([d["Y"][J] for J in d["Y"]]); perm = rng.permutation(len(labs))
        Y = defaultdict(list)
        for lab, j in zip(labs, perm): Y[lab].append(rows[j])
        Dp[k] = dict(d, Y={J: np.array(v) for J, v in Y.items()})
    Xp = crossover(Dp)
    for who in obs: null[who].append(summ(Xp, who))
PERM = {}
for who in obs:
    nv = np.array(null[who]); PERM[who] = (np.sum(nv >= obs[who]) + 1) / (len(nv) + 1)
    P(f"  permutation ({who}): observed {obs[who]:+.3f}; null mean {nv.mean():+.4f}, max {nv.max():+.3f}; p = {PERM[who]:.5f}")
# per-run-pair distribution and abstention gap
pairs = []; gaps = []; xs = []
for (st, ds, K, J), (x, n) in X.items():
    d = D[(st, ds)]; A = d["rej"][K] & ~d["rej"][J]; Bm = d["rej"][J] & ~d["rej"][K]
    for yk in d["Y"][K]:
        for yj in d["Y"][J]: pairs.append((yk[A].mean() - yk[Bm].mean()) - (yj[A].mean() - yj[Bm].mean()))
    gaps.append(abs(np.mean([g["a"] for g in d["G"][K]]) - np.mean([g["a"] for g in d["G"][J]]))); xs.append(x)
pairs = np.array(pairs)
P(f"  single run pairs: n {len(pairs)}, mean {pairs.mean():+.3f}, positive {np.mean(pairs > 0):.3f}; corr(X, |abstention gap|) over cells {np.corrcoef(gaps, xs)[0, 1]:+.2f}")
P("\n== robustness")
report("sure >= 28/32", load("sure28"))
report("all audited-right valid greedy answers (no sure filter)", load("all"))
report("key-only truth (loose key right), sure >= 31/32", load("sure31", "key"))
report("cells weighted by |A| + |B|", D, weight="units")
report("t50 judge runs, verdicts at P(Yes) > 0.5", load("sure31", "aud", "t50"))
pickle.dump(dict(MAIN=MAIN, PERM=PERM), open(f"{HERE}/c5_summary.pkl", "wb"))
P.save()
