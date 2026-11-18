"""(verify_map) Re-implementation of the dose-response claim, plus reference-matching checks.
Units ('known-sure', as claimed): NEW questions whose untrained greedy answer is valid, audited right (loose key or Sonnet >= 4) and sure
(>= 31/32 samples agree). Variants of the unit set: 'right' (audited right, no sure filter), 'nodup' (known-sure minus near-duplicates).
Runs: FK20, IDK credit 0.5, final checkpoint, new-question abstention < 0.95. Arms: judge<J>a35. Reference pool: sim<FP>fn<FN> and lenQ runs.
x = logit P(J rejects) from the layer-16 probe fitted on training-question verdicts (v02). G = [conf, logit P_aud, logit P_sure] (grader-free probes,
also fitted on training questions). FWL: x_perp, y_perp = residuals on [1, G]; adjusted refusal = y_perp + mean(y). Quintiles of x_perp
(equal counts); per run: mean adjusted refusal per quintile, T-B = Q5 - Q1, slope per SD of x_perp.
References (each gives 'grader-specific' = judge run minus reference, same x and same bins):
  nn3      3 reference runs nearest in new-question abstention (the claimed method)
  interp   reference quintile curve predicted at the judge run's abstention by a weighted local-linear fit over reference runs (Gaussian kernel,
           bandwidth 0.08); judge runs whose abstention lies outside the reference range +- 0.02 are flagged 'extrapolated'
  inrange  nn3, but only judge runs with >= 3 reference runs within 0.06 abstention
  otherJ   the OTHER judges' a35 runs of the same cell (same x = this judge's probe), nearest 3 in abstention: judge-specific part only
Pooling: mean over runs within (student, dataset, judge) unit; mean over units per student; students averaged ('pooled').
Bootstrap (NB draws): questions within cell (multinomial weights), runs within arm, reference runs within pool. Output: v04_dose_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, time, warnings
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
warnings.filterwarnings("ignore")
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
P = Log(f"{HERE}/v04_dose_out.txt"); P(__doc__); t00 = time.time()
rng = np.random.default_rng(11)
REF = re.compile(r"^(sim\d\dfn\d\d|lenQ)$")

def prep(st, ds, uset):
    D = load(st, ds); Pr = pickle.load(open(f"{CACHE}/p_{st}_{ds}_L16.pkl", "rb"))["T"]
    cd0 = D["cd"][:, 0]; v = np.isin(cd0, (1, 2)); aud = v & ((cd0 == 1) | (D["son0"] >= 4)); sure = v & (D["conf"] >= 31 / 32)
    m = {"ks": D["new"] & aud & sure, "right": D["new"] & aud, "nodup": D["new"] & aud & sure & ~D["neardup"]}[uset]
    U = np.where(m)[0]
    G = np.column_stack([D["conf"][U], logit(Pr["aud"]["p"][U]), logit(Pr["sure"]["p"][U])])
    runs = []
    for n, r in sorted(D["runs"].items()):
        a = float((r["o"][D["new"]] == 0).mean())
        if a >= 0.95: continue
        mj = re.fullmatch(r"judge([QMGOR])a35", r["g"])
        kind = mj.group(1) if mj else ("ref" if REF.match(r["g"]) else None)
        if kind is None: continue
        runs.append(dict(n=n, k=kind, a=a, y=(r["o"][U] == 0).astype(float)))
    Y = np.column_stack([r["y"] for r in runs]); A = np.array([r["a"] for r in runs]); K = np.array([r["k"] for r in runs])
    X = {J: logit(Pr[f"rej_{J}_a35"]["p"][U]) for J in J5 if (K == J).any()}
    return dict(U=U, G=G, Y=Y, A=A, K=K, X=X)

def wq_bins(x, w, nb):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(w[o]); cw /= cw[-1]; b = np.zeros(len(x), int); b[o] = np.minimum(nb - 1, (cw * nb - 1e-9).astype(int)); return b

def curves(x, Y, G, w, nb=5):
    X1 = np.column_stack([np.ones(len(x)), G]); WX = X1 * w[:, None]
    M = np.column_stack([x, Y]); beta = np.linalg.solve(X1.T @ WX + 1e-9 * np.eye(X1.shape[1]), WX.T @ M); R = M - X1 @ beta
    xp = R[:, 0]; Yp = R[:, 1:]; yadj = Yp + (w[:, None] * Y).sum(0) / w.sum()
    b = wq_bins(xp, w, nb); B = np.zeros((nb, Y.shape[1]))
    for j in range(nb):
        mm = b == j; B[j] = (w[mm][:, None] * yadj[mm]).sum(0) / w[mm].sum()
    sxx = (w * xp * xp).sum(); sl = (w[:, None] * xp[:, None] * Yp).sum(0) / sxx * np.sqrt(sxx / w.sum())
    return B, sl

def interp_ref(Bref, aref, a0, bw=0.08):
    """local-linear (Gaussian kernel) prediction of reference curves (nb x nref) at abstention a0"""
    k = np.exp(-0.5 * ((aref - a0) / bw) ** 2) + 1e-12; Xd = np.column_stack([np.ones(len(aref)), aref - a0])
    W = Xd * k[:, None]; beta = np.linalg.solve(Xd.T @ W + 1e-9 * np.eye(2), W.T @ Bref.T); return beta[0]

def cell_stats(c, w, rsel=None, refsel=None):
    """returns {J: dict(raw=(5,), nn3=(5,), interp=(5,), inrange=(5,) or None, otherJ=(5,) or None, slraw, slnn3, n_extrap, n_runs)} for one cell"""
    out = {}
    iref_all = np.where(c["K"] == "ref")[0]
    iref = iref_all if refsel is None else iref_all[refsel]
    for J, x in c["X"].items():
        B, sl = curves(x, c["Y"], c["G"], w)
        ij = np.where(c["K"] == J)[0]
        if rsel is not None: ij = ij[rsel[J]]
        io = np.where((c["K"] != J) & (c["K"] != "ref"))[0]
        aref = c["A"][iref]; rec = dict(raw=[], nn3=[], interp=[], inrange=[], otherJ=[], slraw=[], slnn3=[], n_extrap=0)
        for i in ij:
            a = c["A"][i]; d = np.abs(aref - a); nn = iref[np.argsort(d, kind="stable")[:3]]
            rec["raw"].append(B[:, i]); rec["slraw"].append(sl[i])
            rec["nn3"].append(B[:, i] - B[:, nn].mean(1)); rec["slnn3"].append(sl[i] - sl[nn].mean())
            rec["interp"].append(B[:, i] - interp_ref(B[:, iref], aref, a))
            rec["n_extrap"] += int(a > aref.max() + 0.02 or a < aref.min() - 0.02)
            if (d <= 0.06).sum() >= 3: rec["inrange"].append(B[:, i] - B[:, nn].mean(1))
            if len(io):
                no = io[np.argsort(np.abs(c["A"][io] - a), kind="stable")[:3]]; rec["otherJ"].append(B[:, i] - B[:, no].mean(1))
        r2 = {k: (np.mean(v, 0) if len(v) else None) for k, v in rec.items() if k != "n_extrap"}
        r2["n_extrap"] = rec["n_extrap"]; r2["n_runs"] = len(ij); out[J] = r2
    return out

def pool(S, key, f=lambda v: v[4] - v[0], who=None):
    """S: {(st, ds): {J: stats}} -> pooled value: mean over units per student, students averaged"""
    per = []
    for s in ([who] if who else STS):
        v = [f(S[(st, ds)][J][key]) for (st, ds) in S if st == s for J in S[(st, ds)] if S[(st, ds)][J][key] is not None]
        if v: per.append(np.mean(v, 0))
    return np.mean(per, 0) if per else np.nan

for uset in ("ks", "right", "nodup"):
    C = {(st, ds): prep(st, ds, uset) for st in STS for ds in DSS}
    S0 = {k: cell_stats(c, np.ones(len(c["U"]))) for k, c in C.items()}
    boots = []
    for b in range(NB if uset == "ks" else NB // 2):
        Sb = {}
        for k, c in C.items():
            n = len(c["U"]); w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            rsel = {J: rng.integers(0, (c["K"] == J).sum(), (c["K"] == J).sum()) for J in c["X"]}
            nr = (c["K"] == "ref").sum(); Sb[k] = cell_stats(c, w, rsel, rng.integers(0, nr, nr))
        boots.append(Sb)
    def ci(key, f=lambda v: v[4] - v[0], who=None):
        e = pool(S0, key, f, who); bb = np.array([pool(Sb, key, f, who) for Sb in boots])
        if not len(bb): return e, e * np.nan, e * np.nan
        lo, hi = np.nanpercentile(bb, [2.5, 97.5], axis=0); return e, lo, hi
    P(f"\n==== unit set {uset}: answers " + ", ".join(f"{st} {ds} {len(c['U'])}" for (st, ds), c in C.items()) + f"  ({time.time() - t00:.0f}s)")
    for key in ("raw", "nn3", "interp", "inrange", "otherJ"):
        e, lo, hi = ci(key, f=lambda v: v)
        tb = ci(key)
        P(f"  {key:8s} quintile curve " + " ".join(f"{e[j]:+.3f}" for j in range(5)) + f" | T-B {tb[0]:+.3f} [{tb[1]:+.3f}, {tb[2]:+.3f}]"
          + f" | qwen7 {pool(S0, key, who='qwen7'):+.3f} llama8 {pool(S0, key, who='llama8'):+.3f}")
    sl = ci("slnn3", f=lambda v: v); P(f"  slope per SD grader-specific (nn3) {sl[0]:+.3f} [{sl[1]:+.3f}, {sl[2]:+.3f}]; raw {pool(S0, 'slraw', f=lambda v: v):+.3f}")
    # per cell and per judge, sign counts
    for key in ("nn3", "interp", "otherJ"):
        cells = {(st, ds): np.mean([S0[(st, ds)][J][key][4] - S0[(st, ds)][J][key][0] for J in S0[(st, ds)] if S0[(st, ds)][J][key] is not None])
                 for (st, ds) in S0}
        judges = {J: np.mean([S0[k][J][key][4] - S0[k][J][key][0] for k in S0 if J in S0[k] and S0[k][J][key] is not None]) for J in J5}
        units = [S0[k][J][key][4] - S0[k][J][key][0] for k in S0 for J in S0[k] if S0[k][J][key] is not None]
        P(f"  {key:8s} T-B by cell: " + " ".join(f"{st}/{ds} {v:+.3f}" for (st, ds), v in cells.items()) + " | by judge: " + " ".join(f"{J} {v:+.3f}" for J, v in judges.items())
          + f" | units positive {sum(u > 0 for u in units)}/{len(units)}")
    ext = sum(S0[k][J]["n_extrap"] for k in S0 for J in S0[k]); nr = sum(S0[k][J]["n_runs"] for k in S0 for J in S0[k])
    P(f"  judge runs outside the reference abstention range (+-0.02): {ext}/{nr}; inrange keeps " +
      ", ".join(f"{st}/{ds} {sum(1 for J in S0[(st, ds)] if S0[(st, ds)][J]['inrange'] is not None)}/{len(S0[(st, ds)])} units" for (st, ds) in S0))
P.save()
