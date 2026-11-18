"""(verify_map) Judge crossover on NEW questions, both error signs, own code.
Units: new questions with a valid untrained greedy answer; RIGHT = loose key or Sonnet >= 4; WRONG = key-wrong and Sonnet level 1..3.
Verdicts of each judge on the untrained greedy answer at its a35 cut (accept = P(Yes) > t_a35). Runs: judge<J>a35, FK20, credit 0.5, new-question
abstention < 0.95; R_J(q) = mean IDK rate of J's runs on question q.
For each cell and unordered judge pair (K, J):
  FR (right units): A = K rejects & J accepts, B = J rejects & K accepts; X_FR = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)]  (prediction > 0)
  FA (wrong units): A = K accepts & J rejects, B = J accepts & K rejects; X_FA = same formula                               (prediction < 0)
  (X is symmetric in K <-> J, so unordered pairs = averaging over orders; nothing is forced by the orientation.)
Cells with |A|, |B| >= MIN are kept. Pooled = mean over cells per student, students averaged. Crossed bootstrap (questions within cell, runs within
judge; 1000 draws); permutation test: judge labels shuffled over the a35 runs within cell (2000 draws).
Extension 'pairpool': all pairs with |A|, |B| >= 3, weighted by |A||B|/(|A|+|B|) within dataset (more cells on TriviaQA / PopQA).
Readable version 'probeX': on the same units, slope of [R_K - R_J] on z(logit P_K(reject) - logit P_J(reject)) (layer-16 probes fitted on training
  verdicts, v02), per pair, sign-flipped for FA so both signs read '> 0 = follows the readable map'... reported unflipped: FR slope > 0 and FA slope > 0
  both mean 'refuses more where its own judge's probe predicts rejection' (for wrong units that is 'answers more where its judge would pay').
Output: v05_crossover_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, itertools, time
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
P = Log(f"{HERE}/v05_crossover_out.txt"); P(__doc__); t00 = time.time()
rng = np.random.default_rng(55)

def prep(st, ds, kind):
    D = load(st, ds); Pr = pickle.load(open(f"{CACHE}/p_{st}_{ds}_L16.pkl", "rb"))["T"]
    cd0 = D["cd"][:, 0]; v = np.isin(cd0, (1, 2)); s = D["son0"]
    good = ((cd0 == 1) | (s >= 4)) if kind == "right" else ((cd0 == 2) & (s >= 1) & (s <= 3))
    U = np.where(D["new"] & v & good)[0]
    acc = {J: D["PJ"][J][U, 0] > thr(J, st, ds, "a35") for J in J5}
    Y = {}
    for n, r in sorted(D["runs"].items()):
        m = re.fullmatch(r"judge([QMGOR])a35", r["g"])
        if not m or (r["o"][D["new"]] == 0).mean() >= 0.95: continue
        Y.setdefault(m.group(1), []).append((r["o"][U] == 0).astype(float))
    Y = {J: np.array(v_) for J, v_ in Y.items()}
    pr = {J: logit(Pr[f"rej_{J}_a35"]["p"][U]) for J in Y}
    return dict(U=U, acc=acc, Y=Y, pr=pr, conf=D["conf"][U])

def sets(d, K, J, kind):
    if kind == "right": return (~d["acc"][K]) & d["acc"][J], (~d["acc"][J]) & d["acc"][K]
    return d["acc"][K] & ~d["acc"][J], d["acc"][J] & ~d["acc"][K]

def cellX(D, kind, MIN, W=None, SEL=None, Yover=None):
    out = {}
    for (st, ds), d in D.items():
        w = np.ones(len(d["U"])) if W is None else W[(st, ds)]
        Yd = d["Y"] if Yover is None else Yover[(st, ds)]
        for K, J in itertools.combinations([J for J in J5 if J in Yd], 2):
            A, B = sets(d, K, J, kind)
            if A.sum() < MIN or B.sum() < MIN: continue
            wa, wb = w * A, w * B
            if wa.sum() == 0 or wb.sum() == 0: continue
            def gap(J_):
                Yj = Yd[J_] if SEL is None else Yd[J_][SEL[(st, ds, J_)]]
                yj = Yj.mean(0); return yj @ wa / wa.sum() - yj @ wb / wb.sum()
            out[(st, ds, K, J)] = (gap(K) - gap(J), int(A.sum()), int(B.sum()))
    return out

def summ(X, who="pooled", weighted=False):
    def m(keys):
        if not keys: return np.nan
        if not weighted: return np.mean([X[k][0] for k in keys])
        h = np.array([X[k][1] * X[k][2] / (X[k][1] + X[k][2]) for k in keys]); return float(np.sum(h * np.array([X[k][0] for k in keys])) / h.sum())
    if who == "pooled":
        per = [m([k for k in X if k[0] == st]) for st in STS]; per = [p for p in per if np.isfinite(p)]
        return np.mean(per) if per else np.nan
    return m([k for k in X if who in (k[0], k[1])])

def run(kind, MIN, NB=1000, NP=2000, weighted=False, label=""):
    D = {(st, ds): prep(st, ds, kind) for st in STS for ds in DSS}
    X = cellX(D, kind, MIN)
    neg = sum(x[0] < 0 for x in X.values()); pos = sum(x[0] > 0 for x in X.values())
    reps = {w: [] for w in ("pooled",) + STS + DSS}
    for b in range(NB):
        W = {k: np.bincount(rng.integers(0, len(d["U"]), len(d["U"])), minlength=len(d["U"])).astype(float) for k, d in D.items()}
        SEL = {(st, ds, J): rng.integers(0, len(d["Y"][J]), len(d["Y"][J])) for (st, ds), d in D.items() for J in d["Y"]}
        Xb = cellX(D, kind, MIN, W, SEL)
        for w in reps: reps[w].append(summ(Xb, w, weighted))
    null = {w: [] for w in ("pooled",) + STS}
    for b in range(NP):
        Yp = {}
        for k, d in D.items():
            labs = [J for J in d["Y"] for _ in range(len(d["Y"][J]))]; rows = np.vstack([d["Y"][J] for J in d["Y"]]); perm = rng.permutation(len(labs))
            Yp[k] = {}
            for lab, j in zip(labs, perm): Yp[k].setdefault(lab, []).append(rows[j])
            Yp[k] = {J: np.array(v_) for J, v_ in Yp[k].items()}
        Xp = cellX(D, kind, MIN, Yover=Yp)
        for w in null: null[w].append(summ(Xp, w, weighted))
    P(f"\n== {label} kind={kind} MIN={MIN} weighted={weighted}: cells {len(X)} (negative {neg}, positive {pos})  ({time.time() - t00:.0f}s)")
    for w in ("pooled",) + STS + DSS:
        e = summ(X, w, weighted); lo, hi = np.nanpercentile(reps[w], [2.5, 97.5]) if np.isfinite(e) else (np.nan, np.nan)
        s = f"  {w:6s} X {e:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if w in null:
            nv = np.array(null[w]); nv = nv[np.isfinite(nv)]
            if len(nv) and np.isfinite(e): s += f"  perm p(two-sided) {(np.sum(np.abs(nv) >= abs(e)) + 1) / (len(nv) + 1):.4f}"
        ncell = sum(1 for k in X if w in ("pooled", k[0], k[1])); nneg = sum(1 for k in X if w in ("pooled", k[0], k[1]) and X[k][0] < 0)
        P(s + f"  cells {ncell} (negative {nneg})")
    return D, X

def probe_slopes(D, kind):
    """per pair: OLS slope of R_K - R_J (per question) on z(logit P_K(rej) - logit P_J(rej)) over ALL units of the kind; plus the same with the
    actual verdict contrast (acc_J - acc_K, i.e. +1 where only K rejects) as a second regressor (joint)."""
    res = {}
    for (st, ds), d in D.items():
        for K, J in itertools.combinations([J for J in J5 if J in d["Y"]], 2):
            dy = d["Y"][K].mean(0) - d["Y"][J].mean(0); dp = d["pr"][K] - d["pr"][J]; z = (dp - dp.mean()) / dp.std()
            dv = (d["acc"][J].astype(float) - d["acc"][K].astype(float))   # +1: K rejects, J accepts
            b1 = np.polyfit(z, dy, 1)[0]
            Xm = np.column_stack([np.ones(len(z)), z, dv]); cf = np.linalg.lstsq(Xm, dy, rcond=None)[0]
            res[(st, ds, K, J)] = (b1, cf[1], cf[2], len(z))
    return res

for kind in ("right", "wrong"):
    D, X = run(kind, 5, label="REPLICATION")
    if kind == "wrong":
        P("  cells: " + "; ".join(f"{k[0]} {k[1]} {k[2]}-{k[3]} X {x:+.3f} (|A| {a}, |B| {b})" for k, (x, a, b) in sorted(X.items())))
    run(kind, 3, NB=500, NP=1000, weighted=True, label="EXTENSION pairpool")
    R = probe_slopes(D, kind)
    for who in ("pooled",) + STS + DSS:
        ks = [k for k in R if who in ("pooled", k[0], k[1])]
        if who == "pooled":
            m = np.mean([np.mean([R[k] for k in ks if k[0] == st], 0) for st in STS], 0)
        else: m = np.mean([R[k] for k in ks], 0)
        P(f"  probeX {kind} {who:6s}: slope of R_K - R_J per SD of probe contrast {m[0]:+.4f} | joint: probe {m[1]:+.4f}, verdict (K rejects only) {m[2]:+.4f} | pairs {len(ks)}, "
          f"probe slope > 0 in {sum(R[k][0] > 0 for k in ks)}/{len(ks)}")
P.save()
