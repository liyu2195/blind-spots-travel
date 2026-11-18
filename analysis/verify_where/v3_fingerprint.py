"""(verify_where) Item 2: judge fingerprint on NEW questions, own code, run-level inference.
Units: held-out (duplicate-free) questions whose untrained greedy answer is sure (>= 31/32 samples agree) and audited right (loose key OR Sonnet >= 4).
Panel verdicts: judges Q M G O R at their a35 thresholds on that untrained greedy answer (reject = P(Yes) <= t).
Runs: FK20, idk credit 0.5. Judge runs g = judge<K>a35. Key family = gold (sim00fn00) + uniform false-reject keys (sim00fnXX).
(1) Excess refusal of a K run on S_rej = sure-right new questions K rejects (and S_acc = K accepts) over key-family runs at the same new-question
    abstention. References: (a) +-0.10 window around the run's abstention (mean of key runs inside; claimed method used the K-mean abstention),
    (b) local linear fit of key runs' refusal on S vs abstention, (c) 3 nearest key runs, (d) gold only, (e) keyFN only, (f) vote runs (frzVOTE*),
    (g) the OTHER judges' a35 runs (linear fit). Per run, averaged within (student, set, K) cells, then over cells.
    Intervals: hierarchical bootstrap (runs within each (student, set, grader group), questions within (student, set)), 1000 draws.
(2) Crossover X(K, J) = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)], A = K rejects & J accepts, B = J rejects & K accepts (|A|, |B| >= 5).
    Pooled over unordered pairs; intervals by the same hierarchical bootstrap; permutation test of judge labels among a35 runs within (student, set).
Discovery qwen7, confirmation llama8 (reported separately). Output: v3_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, itertools, numpy as np
from collections import defaultdict
C = (_RA + "/verify_where/cache"); OUT = (_RA + "/verify_where")
J5 = ["Q", "M", "G", "O", "R"]; CELLS = [(st, d) for st in ("qwen7", "llama8") for d in ("pqa", "nqo", "tqa")]
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
P(__doc__)
data = {}
for st, dsn in CELLS:
    B = pickle.load(open(f"{C}/base_{st}_{dsn}.pkl", "rb")); R = pickle.load(open(f"{C}/runs_{st}_{dsn}.pkl", "rb"))
    ho = np.where(B["held"])[0]
    aud = B["valid"][ho, 0] & (B["kl"][ho, 0] | (B["son"][ho, 0] >= 4)); sure = B["conf"][ho] >= 31 / 32
    U = ho[aud & sure]
    rej = {J: B["PJ"][J][U, 0] <= B["TH"][J]["a35"] for J in J5}
    G = defaultdict(list)   # group -> list of (run name, idk vector on U, abstention on all new)
    for name, r in R.items():
        if r["c"] != 5: continue
        g = r["g"]; y = (r["o"][U] == 0).astype(float); a = r["a_new"]
        if a >= 0.95: continue   # collapsed (all-IDK) runs excluded from every group
        m = re.fullmatch(r"judge([QMGOR])a35", g)
        if m: G["J" + m.group(1)].append((name, y, a))
        elif g == "sim00fn00": G["gold"].append((name, y, a))
        elif re.fullmatch(r"sim00fn\d\d", g): G["keyFN"].append((name, y, a))
        elif g.startswith("frzVOTE"): G["vote"].append((name, y, a))
    data[(st, dsn)] = dict(U=U, rej=rej, G=G)
    P(f"{st} {dsn}: sure & audited-right new questions {len(U)}; runs " + ", ".join(f"{k} {len(v)}" for k, v in sorted(G.items())) +
      "; K-rejects " + " ".join(f"{J}:{rej[J].sum()}" for J in J5))

def ref_value(refs, Smask, a, how):
    """reference refusal on set S at abstention a from reference runs [(name, y, a)] (collapsed runs, a >= 0.95, excluded upstream).
    win: mean of reference runs within +-0.10 of a (>= 2 runs required); lin: local linear interpolation, needs a reference run on each side within 0.15."""
    if not refs: return np.nan
    xs = np.array([x[2] for x in refs]); ys = np.array([x[1][Smask].mean() for x in refs])
    if how == "win":
        m = np.abs(xs - a) <= 0.10
        return ys[m].mean() if m.sum() >= 2 else np.nan
    if how == "lin":
        lo = (xs <= a) & (xs >= a - 0.15); hi = (xs >= a) & (xs <= a + 0.15)
        if not lo.any() or not hi.any(): return np.nan
        w = np.exp(-0.5 * ((xs - a) / 0.10) ** 2)
        A = np.c_[np.ones_like(xs), xs - a]; beta = np.linalg.lstsq(A * np.sqrt(w)[:, None], ys * np.sqrt(w), rcond=None)[0]; return beta[0]
    raise ValueError

def excess(D, groupsel, refsel, how, side):
    """mean excess over (cell, K) of per-run excess; D = per-cell dict possibly resampled."""
    per_cell = {}; bal = []
    for (st, dsn), d in D.items():
        for K in J5:
            runs = d["G"].get("J" + K, [])
            if not runs: continue
            S = d["rej"][K] if side == "rej" else ~d["rej"][K]
            if S.sum() < 5: continue
            if refsel == "otherJ": refs = [x for J in J5 if J != K for x in d["G"].get("J" + J, [])]
            else: refs = [x for g in refsel for x in d["G"].get(g, [])]
            ex = []
            for name, y, a in runs:
                rv = ref_value(refs, S, a, how)
                if np.isfinite(rv): ex.append(y[S].mean() - rv)
            if ex: per_cell[(st, dsn, K)] = np.mean(ex)
    return per_cell

def resample(rng):
    D = {}
    for key, d in data.items():
        n = len(d["U"]); qi = rng.integers(0, n, n)
        G = {g: [(nm, y[qi], a) for nm, y, a in (lst[j] for j in rng.integers(0, len(lst), len(lst)))] for g, lst in d["G"].items()}
        D[key] = dict(U=d["U"][qi], rej={J: v[qi] for J, v in d["rej"].items()}, G=G)
    return D

def summarize(pc, st=None):
    v = [x for (s, d, K), x in pc.items() if st is None or s == st]; return np.mean(v) if v else np.nan

rng = np.random.default_rng(2024); NB = 400
P("\n==== (1) excess refusal of K-trained runs over reference runs at matched abstention, sure & audited-right NEW questions")
P("     (per run -> mean per (student, set, K) cell -> mean over cells; [95% hierarchical bootstrap over runs and questions])")
specs = [("key family, +-0.10 window (>=2 runs)", ("gold", "keyFN"), "win"), ("key family, local linear interp.", ("gold", "keyFN"), "lin"),
         ("gold only, local linear interp.", ("gold",), "lin"), ("keyFN only, local linear interp.", ("keyFN",), "lin"), ("vote runs, +-0.10 window", ("vote",), "win"),
         ("other judges' a35 runs, local linear interp.", "otherJ", "lin"), ("other judges' a35 runs, +-0.10 window", "otherJ", "win")]
for lab, refsel, how in specs:
    for side in ("rej", "acc"):
        pc = excess(data, None, refsel, how, side); est = {st: summarize(pc, st) for st in ("qwen7", "llama8")}
        bs = {st: [] for st in est}
        for b in range(NB):
            pcb = excess(resample(rng), None, refsel, how, side)
            for st in est: bs[st].append(summarize(pcb, st))
        P(f"  {lab:38s} K {side}: " + "  ".join(f"{st} {est[st]:+.3f} [{np.nanpercentile(bs[st], 2.5):+.3f}, {np.nanpercentile(bs[st], 97.5):+.3f}] (cells {sum(1 for k in pc if k[0] == st)})" for st in est))
# replica of the claimed estimator: reference = key runs within +-0.10 of the MEAN abstention of K's runs (collapsed runs kept, as claimed);
# excess per (cell, K, question) unit = mean over K runs - mean over matched key runs; pooled over units per student
def claimed(D, side, drop_collapsed=False):
    out = defaultdict(list)
    for (st, dsn), d in D.items():
        refs = d["G"].get("gold", []) + d["G"].get("keyFN", [])
        for K in J5:
            runs = d["G"].get("J" + K, [])
            if not runs: continue
            aK = np.mean([a for _, _, a in runs]); m = [y for _, y, a in refs if abs(a - aK) <= 0.10]
            if not m: continue
            S = d["rej"][K] if side == "rej" else ~d["rej"][K]
            ex = np.mean([y for _, y, _ in runs], 0) - np.mean(m, 0); out[st].extend(ex[S])
    return {st: np.mean(v) for st, v in out.items()}
dataC = data   # NOTE: groups here already exclude collapsed runs; the claimed version kept them (they rarely fall in a +-0.10 window)
for side in ("rej", "acc"):
    est = claimed(dataC, side); bs = defaultdict(list)
    for b in range(NB):
        e = claimed(resample(rng), side)
        for st in e: bs[st].append(e[st])
    P(f"  replica of claimed estimator (K-mean window, unit-weighted) K {side}: " + "  ".join(f"{st} {est[st]:+.3f} [{np.percentile(bs[st], 2.5):+.3f}, {np.percentile(bs[st], 97.5):+.3f}]" for st in ("qwen7", "llama8")))
# balance of the claimed matching: K-mean abstention vs matched key runs' mean abstention
P("\n  matching balance (claimed method: key runs within +-0.10 of the MEAN abstention of K's runs):")
for st in ("qwen7", "llama8"):
    diffs = []; nref = []
    for (s, dsn), d in data.items():
        if s != st: continue
        refs = d["G"].get("gold", []) + d["G"].get("keyFN", [])
        for K in J5:
            runs = d["G"].get("J" + K, [])
            if not runs: continue
            aK = np.mean([a for _, _, a in runs]); m = [a for _, _, a in refs if abs(a - aK) <= 0.10]
            if m: diffs.append(aK - np.mean(m)); nref.append(len(m))
    P(f"   {st}: K-run mean abstention minus matched key runs' mean abstention: mean {np.mean(diffs):+.3f} (range {min(diffs):+.3f} to {max(diffs):+.3f}); matched key runs per cell {min(nref)}-{max(nref)}")
# ---------------------------------------------------------------- (2) crossover
def crossover(D):
    out = {}
    for (st, dsn), d in D.items():
        Js = [J for J in J5 if d["G"].get("J" + J)]
        for K, J in itertools.combinations(Js, 2):
            A = d["rej"][K] & ~d["rej"][J]; Bm = d["rej"][J] & ~d["rej"][K]
            if A.sum() < 5 or Bm.sum() < 5: continue
            dK = np.mean([y[A].mean() - y[Bm].mean() for _, y, _ in d["G"]["J" + K]]); dJ = np.mean([y[A].mean() - y[Bm].mean() for _, y, _ in d["G"]["J" + J]])
            out[(st, dsn, K, J)] = dK - dJ
    return out
X = crossover(data)
P(f"\n==== (2) crossover over {len(X)} (student, set, unordered judge pair) cells: mean {np.mean(list(X.values())):+.3f}; positive in {sum(v > 0 for v in X.values())}/{len(X)}")
bs = defaultdict(list)
for b in range(NB):
    Xb = crossover(resample(rng))
    bs["all"].append(np.mean(list(Xb.values())))
    for st in ("qwen7", "llama8"): bs[st].append(np.mean([v for k, v in Xb.items() if k[0] == st]))
for k in ("all", "qwen7", "llama8"):
    v = [x for kk, x in X.items() if k == "all" or kk[0] == k]
    P(f"  {k:7s}: mean {np.mean(v):+.3f} [{np.percentile(bs[k], 2.5):+.3f}, {np.percentile(bs[k], 97.5):+.3f}] (hierarchical bootstrap over runs and questions), cells {len(v)}, positive {sum(x > 0 for x in v)}")
# question-only bootstrap (runs fixed), to compare with the claimed interval
qb = []
for b in range(NB):
    D = {}
    for key, d in data.items():
        n = len(d["U"]); qi = rng.integers(0, n, n)
        D[key] = dict(U=d["U"][qi], rej={J: v[qi] for J, v in d["rej"].items()}, G={g: [(nm, y[qi], a) for nm, y, a in lst] for g, lst in d["G"].items()})
    qb.append(np.mean(list(crossover(D).values())))
P(f"  question-only bootstrap (runs fixed): [{np.percentile(qb, 2.5):+.3f}, {np.percentile(qb, 97.5):+.3f}]")
# per-run-pair spread: X computed from single runs (one K run vs one J run) -> how variable across seeds
pairX = []
for (st, dsn), d in data.items():
    Js = [J for J in J5 if d["G"].get("J" + J)]
    for K, J in itertools.combinations(Js, 2):
        A = d["rej"][K] & ~d["rej"][J]; Bm = d["rej"][J] & ~d["rej"][K]
        if A.sum() < 5 or Bm.sum() < 5: continue
        for _, yk, _ in d["G"]["J" + K]:
            for _, yj, _ in d["G"]["J" + J]:
                pairX.append((yk[A].mean() - yk[Bm].mean()) - (yj[A].mean() - yj[Bm].mean()))
pairX = np.array(pairX); P(f"  single-run-pair crossovers: n {len(pairX)}, mean {pairX.mean():+.3f}, sd {pairX.std():.3f}, positive {np.mean(pairX > 0):.2f}")
# permutation test: shuffle judge labels among the a35 judge runs within (student, set)
obs = np.mean(list(X.values())); null = []
for b in range(1000):
    D = {}
    for key, d in data.items():
        lab = [g for g in d["G"] if g.startswith("J") for _ in d["G"][g]]; pool = [x for g in d["G"] if g.startswith("J") for x in d["G"][g]]
        perm = rng.permutation(len(pool)); G = defaultdict(list)
        for g, j in zip(lab, perm): G[g].append(pool[j])
        D[key] = dict(U=d["U"], rej=d["rej"], G=G)
    null.append(np.mean(list(crossover(D).values())))
P(f"  permutation of judge labels among a35 runs within (student, set): null mean {np.mean(null):+.3f}, 99th pct {np.percentile(null, 99):+.3f}, p = {(np.sum(np.array(null) >= obs) + 1) / 1001:.4f}")
# abstention-difference check: does X track the abstention gap between the two judges' runs?
gaps = []; xs = []
for (st, dsn, K, J), v in X.items():
    d = data[(st, dsn)]; gaps.append(abs(np.mean([a for _, _, a in d["G"]["J" + K]]) - np.mean([a for _, _, a in d["G"]["J" + J]]))); xs.append(v)
P(f"  correlation of X with |abstention gap| between the two judges' runs: r = {np.corrcoef(gaps, xs)[0, 1]:+.2f} (gap range {min(gaps):.2f}-{max(gaps):.2f})")
open(f"{OUT}/v3_out.txt", "w").write("\n".join(lines) + "\n")
