"""(verify_map) Claim 2 in the GRADER-SPECIFIC currency: per predictor, slope of refusal per SD of x_perp (FWL on G = [conf, logit P_aud,
logit P_sure], one common G for every predictor) for each strict-judge run, minus the mean slope of its 3 nearest-abstention no-location
reference runs (sim<FP>fn<FN>, lenQ) computed with the SAME x on the SAME answers. Known-sure answers, NEW (full-fit predictors) and HOME
(out-of-fold predictors). Predictors: lin16, p64, mlp16, knn16, tfidf (v02 / v02b) and actual verdicts v_bin, v_share.
Also the per-student values, because the pooled 'probe beats verdict on new questions' could be carried by one student.
Bootstrap: 300 draws (questions within cell, runs within arm, reference runs within pool). Output: v07_spec_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, time, warnings
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
warnings.filterwarnings("ignore")
P = Log(f"{HERE}/v07_spec_out.txt"); P(__doc__); t00 = time.time(); rng = np.random.default_rng(77); NB = 300
REF = re.compile(r"^(sim\d\dfn\d\d|lenQ)$"); KX = ("lin16", "p64", "mlp16", "knn16", "tfidf", "v_bin", "v_share")
def prep(st, ds, pt):
    D = load(st, ds); P16 = pickle.load(open(f"{CACHE}/p_{st}_{ds}_L16.pkl", "rb"))["T"]; Q2 = pickle.load(open(f"{CACHE}/q_{st}_{ds}.pkl", "rb"))
    key = "p" if pt == "new" else "oof"; base = D["new"] if pt == "new" else D["train"]
    cd = D["cd"]; v = np.isin(cd[:, 0], (1, 2)); val = np.isin(cd, (1, 2)); aud = v & ((cd[:, 0] == 1) | (D["son0"] >= 4)); sure = v & (D["conf"] >= 31 / 32)
    U = np.where(base & aud & sure)[0]
    G = np.column_stack([D["conf"][U], logit(P16["aud"][key][U]), logit(P16["sure"][key][U])])
    runs = []
    for n, r in sorted(D["runs"].items()):
        a = float((r["o"][D["new"]] == 0).mean())
        if a >= 0.95: continue
        m = re.fullmatch(r"judge([QMGOR])a35", r["g"]); k = m.group(1) if m else ("ref" if REF.match(r["g"]) else None)
        if k: runs.append((k, a, (r["o"][U] == 0).astype(float)))
    K = np.array([r[0] for r in runs]); A = np.array([r[1] for r in runs]); Y = np.column_stack([r[2] for r in runs])
    X = {}
    for J in set(K) - {"ref"}:
        rej = (D["PJ"][J] <= thr(J, st, ds, "a35")) & val
        X[(J, "lin16")] = logit(P16[f"rej_{J}_a35"][key][U])
        for k in ("p64", "mlp16", "knn16", "tfidf"): X[(J, k)] = logit(Q2[k][f"rej_{J}_a35"][key][U])
        X[(J, "v_bin")] = rej[U, 0].astype(float); X[(J, "v_share")] = rej[U].sum(1) / np.maximum(val[U].sum(1), 1)
    return dict(U=U, G=G, K=K, A=A, Y=Y, X=X)
def slopes(c, x, w):
    X1 = np.column_stack([np.ones(len(x)), c["G"]]); WX = X1 * w[:, None]; M = np.column_stack([x, c["Y"]])
    R = M - X1 @ np.linalg.solve(X1.T @ WX + 1e-9 * np.eye(X1.shape[1]), WX.T @ M); xp = R[:, 0]; Yp = R[:, 1:]
    sxx = (w * xp * xp).sum(); return (w[:, None] * xp[:, None] * Yp).sum(0) / sxx * np.sqrt(sxx / w.sum())
def stats(C, W=None, SEL=None, REFS=None):
    out = {}
    for key, c in C.items():
        w = np.ones(len(c["U"])) if W is None else W[key]
        iref = np.where(c["K"] == "ref")[0]; iref = iref if REFS is None else iref[REFS[key]]
        for (J, k), x in c["X"].items():
            sl = slopes(c, x, w); ij = np.where(c["K"] == J)[0]; ij = ij if SEL is None else ij[SEL[key][J]]
            out[key + (J, k)] = np.mean([sl[i] - sl[iref[np.argsort(np.abs(c["A"][iref] - c["A"][i]), kind="stable")[:3]]].mean() for i in ij])
    return out
def pooled(S, pt, f, who=None):
    per = []
    for s in ([who] if who else STS):
        v_ = [f(S, k) for k in S if k[0] == s and k[2] == pt and k[4] == "lin16"]
        per.append(np.mean(v_))
    return np.mean(per)
C = {(st, ds, pt): prep(st, ds, pt) for st in STS for ds in DSS for pt in ("new", "home")}
S0 = stats(C); SB = []
for b in range(NB):
    W = {k: np.bincount(rng.integers(0, len(c["U"]), len(c["U"])), minlength=len(c["U"])).astype(float) for k, c in C.items()}
    SEL = {k: {J: rng.integers(0, (c["K"] == J).sum(), (c["K"] == J).sum()) for J in set(c["K"]) - {"ref"}} for k, c in C.items()}
    REFS = {k: rng.integers(0, (c["K"] == "ref").sum(), (c["K"] == "ref").sum()) for k, c in C.items()}
    SB.append(stats(C, W, SEL, REFS))
P(f"done bootstrap ({time.time() - t00:.0f}s)")
def show(pt, f):
    e = pooled(S0, pt, f); bb = np.array([pooled(s, pt, f) for s in SB]); lo, hi = np.percentile(bb, [2.5, 97.5])
    return f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}] (qwen7 {pooled(S0, pt, f, 'qwen7'):+.3f}, llama8 {pooled(S0, pt, f, 'llama8'):+.3f})"
P("\n== grader-specific slope per SD (judge run minus 3 nearest-abstention reference runs), known-sure answers")
for k in KX:
    P(f"  {k:8s} new {show('new', lambda S, u, k=k: S[u[:4] + (k,)])} | home {show('home', lambda S, u, k=k: S[u[:4] + (k,)])}")
P("\n== differences (paired)")
for a, b in (("lin16", "v_share"), ("p64", "v_share"), ("mlp16", "p64"), ("tfidf", "lin16"), ("knn16", "lin16")):
    f = lambda S, u, a=a, b=b: S[u[:4] + (a,)] - S[u[:4] + (b,)]
    P(f"  {a} - {b}: new {show('new', f)} | home {show('home', f)}")
f = lambda S, u: S[u[:4] + ("lin16",)] - S[u[:4] + ("v_share",)]
e = pooled(S0, "new", f) - pooled(S0, "home", f); bb = np.array([pooled(s, "new", f) - pooled(s, "home", f) for s in SB])
P(f"  change (new - home) of [lin16 - v_share], grader-specific slopes: {e:+.3f} [{np.percentile(bb, 2.5):+.3f}, {np.percentile(bb, 97.5):+.3f}]")
P.save()
