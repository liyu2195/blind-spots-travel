"""(verify_map) Claim 2 (home vs new: readable prediction vs actual verdict; nonlinear and text predictors) and claim 4 (R5 regression check).
PART A. Units = known-sure answers (valid, audited right, >= 31/32 agree) on NEW questions and on TRAINING (home) questions. Arms: judge<J>a35 runs
  (FK20, credit 0.5, new abstention < 0.95). G = [conf, logit P_aud, logit P_sure] (layer-16 probes; out-of-fold on home).
  Predictors x of the run's own judge J: lin16 (v02; OOF on home), p64, mlp16, knn16, tfidf (v02b; OOF on home), L12 / L20 linear probes;
  actual verdicts: v_bin (J rejects the greedy answer), v_share (share of the 33 untrained answers J rejects, among valid ones).
  pcor = partial correlation of a run's refusal (0/1 per answer) with x after residualising both on [1, G]; mean over runs -> unit (st, ds, J),
  mean over units per student, students averaged. Joint: refusal ~ G + lin16 + v_share, coefficients per SD of each residualised variable.
  Bootstrap: 200 draws, questions within cell, runs within arm.
PART B (R5 check). Rows: NEW questions with a valid untrained greedy answer on which J and J' disagree at a35 (right and wrong answers alike).
  y = mean IDK rate of J's runs; OLS y ~ rj (J rejects) + z(p) (p = logit P_J(rej) - logit P_J'(rej), lin16) + 5 conf-bin fixed effects.
  Averaged over J' within (st, ds, J), then over cells per student. Also IDK(p > 0) - IDK(p <= 0) within rj = 1 and within rj = 0.
  Run for J in {Q, M, G, O} (as R5) and for all five judges; and separately on right-only and wrong-only rows.
Output: v06_homenew_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, time, warnings
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
warnings.filterwarnings("ignore")
P = Log(f"{HERE}/v06_homenew_out.txt"); P(__doc__); t00 = time.time()
rng = np.random.default_rng(66); NB = 200
KX = ("lin16", "L12", "L20", "p64", "mlp16", "knn16", "tfidf", "v_bin", "v_share")

def prep(st, ds):
    D = load(st, ds); P16 = pickle.load(open(f"{CACHE}/p_{st}_{ds}_L16.pkl", "rb"))["T"]; Q2 = pickle.load(open(f"{CACHE}/q_{st}_{ds}.pkl", "rb"))
    PL = {L: pickle.load(open(f"{CACHE}/p_{st}_{ds}_L{L}.pkl", "rb"))["T"] for L in (12, 20)}
    cd = D["cd"]; v = np.isin(cd[:, 0], (1, 2)); aud = v & ((cd[:, 0] == 1) | (D["son0"] >= 4)); sure = v & (D["conf"] >= 31 / 32)
    val = np.isin(cd, (1, 2))
    runs = {}
    for n, r in sorted(D["runs"].items()):
        m = re.fullmatch(r"judge([QMGOR])a35", r["g"])
        if not m or (r["o"][D["new"]] == 0).mean() >= 0.95: continue
        runs.setdefault(m.group(1), []).append((r["o"] == 0).astype(float))
    parts = {}
    for pt, mask, key in (("new", D["new"], "p"), ("home", D["train"], "oof")):
        U = np.where(mask & aud & sure)[0]
        G = np.column_stack([D["conf"][U], logit(P16["aud"][key][U]), logit(P16["sure"][key][U])])
        X = {}
        for J in runs:
            t = thr(J, st, ds, "a35"); rej = (D["PJ"][J] <= t) & val
            X[(J, "lin16")] = logit(P16[f"rej_{J}_a35"][key][U])
            for L in (12, 20): X[(J, f"L{L}")] = logit(PL[L][f"rej_{J}_a35"][key][U])
            for k in ("p64", "mlp16", "knn16", "tfidf"): X[(J, k)] = logit(Q2[k][f"rej_{J}_a35"][key][U])
            X[(J, "v_bin")] = rej[U, 0].astype(float)
            X[(J, "v_share")] = rej[U].sum(1) / np.maximum(val[U].sum(1), 1)
        parts[pt] = dict(U=U, G=G, X=X, Y={J: np.array(runs[J])[:, U] for J in runs})
    return dict(parts=parts, D=D, P16=P16, runs=runs, v=v, aud=aud)

def resid(M, G, w):
    X1 = np.column_stack([np.ones(len(G)), G]); WX = X1 * w[:, None]
    return M - X1 @ np.linalg.solve(X1.T @ WX + 1e-9 * np.eye(X1.shape[1]), WX.T @ M)

def stats(c, pt, w, sel):
    """{(J, kind): pcor (mean over runs)}, {J: (b_read, b_verdict)}"""
    p = c["parts"][pt]; out = {}; jt = {}
    for J, Y in p["Y"].items():
        Yj = Y[sel[J]] if sel is not None else Y
        Ry = resid(Yj.T, p["G"], w)
        syy = (w[:, None] * Ry * Ry).sum(0)
        for k in KX:
            x = p["X"][(J, k)]; rx = resid(x[:, None], p["G"], w)[:, 0]; sxx = (w * rx * rx).sum()
            out[(J, k)] = float(np.mean((w[:, None] * rx[:, None] * Ry).sum(0) / np.sqrt(sxx * np.maximum(syy, 1e-12))))
        xr = resid(p["X"][(J, "lin16")][:, None], p["G"], w)[:, 0]; xv = resid(p["X"][(J, "v_share")][:, None], p["G"], w)[:, 0]
        M = np.array([[(w * xr * xr).sum(), (w * xr * xv).sum()], [(w * xr * xv).sum(), (w * xv * xv).sum()]])
        rhs = np.vstack([(w[:, None] * xr[:, None] * Ry).sum(0), (w[:, None] * xv[:, None] * Ry).sum(0)])
        cf = np.linalg.solve(M, rhs); sdv = np.sqrt(np.diag(M) / w.sum()); jt[J] = (float(np.mean(cf[0]) * sdv[0]), float(np.mean(cf[1]) * sdv[1]))
    return out, jt

C = {(st, ds): prep(st, ds) for st in STS for ds in DSS}
P(f"loaded ({time.time() - t00:.0f}s); known-sure new/home: " + ", ".join(f"{st} {ds} {len(c['parts']['new']['U'])}/{len(c['parts']['home']['U'])}" for (st, ds), c in C.items()))
def allstats(W=None, SEL=None):
    R = {}
    for k, c in C.items():
        for pt in ("new", "home"):
            n = len(c["parts"][pt]["U"]); w = np.ones(n) if W is None else W[(k, pt)]
            R[(k, pt)] = stats(c, pt, w, None if SEL is None else SEL[k])
    return R
def pooled(R, pt, f):
    per = []
    for s in STS:
        vals = [f(R[(k, pt)], J) for k in C if k[0] == s for J in C[k]["runs"]]
        per.append(np.mean(vals))
    return np.mean(per), per[0], per[1]
R0 = allstats(); RB = []
for b in range(NB):
    W = {(k, pt): np.bincount(rng.integers(0, len(c["parts"][pt]["U"]), len(c["parts"][pt]["U"])), minlength=len(c["parts"][pt]["U"])).astype(float) for k, c in C.items() for pt in ("new", "home")}
    SEL = {k: {J: rng.integers(0, len(c["runs"][J]), len(c["runs"][J])) for J in c["runs"]} for k, c in C.items()}
    RB.append(allstats(W, SEL))
P(f"bootstrap done ({time.time() - t00:.0f}s)")
def show(lab, pt, f):
    e = pooled(R0, pt, f); bb = np.array([pooled(r, pt, f)[0] for r in RB]); lo, hi = np.percentile(bb, [2.5, 97.5])
    return f"{lab} {e[0]:+.3f} [{lo:+.3f}, {hi:+.3f}] (qwen7 {e[1]:+.3f}, llama8 {e[2]:+.3f})"
P("\n== A1 partial correlation of refusal with each predictor (strict a35 arms, known-sure answers)")
for k in KX:
    P(f"  {k:8s} " + show("new", "new", lambda r, J, k=k: r[0][(J, k)]) + " | " + show("home", "home", lambda r, J, k=k: r[0][(J, k)]))
P("\n== A2 differences")
for a, b in (("lin16", "v_share"), ("lin16", "v_bin"), ("p64", "v_share"), ("mlp16", "p64"), ("mlp16", "lin16"), ("tfidf", "lin16"), ("knn16", "lin16"), ("L20", "lin16"), ("L12", "lin16")):
    P(f"  {a} - {b}: " + show("new", "new", lambda r, J: r[0][(J, a)] - r[0][(J, b)]) + " | " + show("home", "home", lambda r, J: r[0][(J, a)] - r[0][(J, b)]))
for a, b in (("lin16", "v_share"), ("lin16", "v_bin")):
    f = lambda r, J: r[0][(J, a)] - r[0][(J, b)]
    e = pooled(R0, "new", f)[0] - pooled(R0, "home", f)[0]; bb = np.array([pooled(r, "new", f)[0] - pooled(r, "home", f)[0] for r in RB])
    P(f"  change (new - home) of [{a} - {b}]: {e:+.3f} [{np.percentile(bb, 2.5):+.3f}, {np.percentile(bb, 97.5):+.3f}]")
P("\n== A3 joint model refusal ~ G + lin16 + v_share (per SD, refusal points)")
for pt in ("new", "home"):
    P(f"  {pt}: " + show("readable", pt, lambda r, J: r[1][J][0]) + " | " + show("verdict", pt, lambda r, J: r[1][J][1]))

# ---------------- PART B
def r5(Js, rowkind="all", kind="lin16"):
    agg = {}
    for (st, ds), c in C.items():
        D = c["D"]; P16 = c["P16"]; v = c["v"]; cd0 = D["cd"][:, 0]; s0 = D["son0"]
        rk = {"all": v, "right": v & ((cd0 == 1) | (s0 >= 4)), "wrong": v & (cd0 == 2) & (s0 >= 1) & (s0 <= 3)}[rowkind]
        rej = {J: D["PJ"][J][:, 0] <= thr(J, st, ds, "a35") for J in Js}; sb = np.minimum(4, (D["conf"] * 5).astype(int))
        Q2 = pickle.load(open(f"{CACHE}/q_{st}_{ds}.pkl", "rb"))
        sc = {J: logit((P16 if kind == "lin16" else Q2[kind])[f"rej_{J}_a35"]["p"]) for J in Js}
        for J in Js:
            if J not in c["runs"]: continue
            yJ = np.mean(c["runs"][J], 0)
            for Jp in Js:
                if Jp == J: continue
                ii = np.where(D["new"] & rk & (rej[J] != rej[Jp]))[0]
                if len(ii) < 20: continue
                rj = rej[J][ii].astype(float); p = sc[J][ii] - sc[Jp][ii]; z = (p - p.mean()) / (p.std() + 1e-9); y = yJ[ii]
                F = np.column_stack([(sb[ii] == k).astype(float) for k in range(5)]); F = F[:, F.sum(0) > 0]
                cf = np.linalg.lstsq(np.column_stack([rj, z, F]), y, rcond=None)[0]
                def dm(m1, m0): return y[m1].mean() - y[m0].mean() if m1.any() and m0.any() else np.nan
                agg.setdefault((st, ds, J), []).append((cf[0], cf[1], dm((rj == 1) & (p > 0), (rj == 1) & (p <= 0)), dm((rj == 0) & (p > 0), (rj == 0) & (p <= 0)), len(ii)))
    for who in STS:
        cells = np.array([np.nanmean(np.array(L), 0) for k, L in agg.items() if k[0] == who])
        m = np.nanmean(cells, 0)
        P(f"  {kind:5s} {''.join(Js)} rows={rowkind:5s} {who}: cells {len(cells)} | b_rj (verdict beyond readout) {m[0]:+.3f} (> 0 in {int((cells[:, 0] > 0).sum())}/{len(cells)}) | "
          f"b_p per SD {m[1]:+.3f} (> 0 in {int((cells[:, 1] > 0).sum())}/{len(cells)}) | IDK(p>0)-IDK(p<=0): within rj=1 {m[2]:+.3f}, within rj=0 {m[3]:+.3f} | rows/block {m[4]:.0f}")
P("\n== B R5 regression check")
for kind in ("lin16", "p64", "mlp16"):
    for Js in (("Q", "M", "G", "O"), J5):
        for rk in ("all", "right", "wrong"): r5(Js, rk, kind)
P.save()
