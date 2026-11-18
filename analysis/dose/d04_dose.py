"""(dose) Dose-response of trained-model refusal on the grader's PREDICTED rejection (predictors fitted only on training-question verdicts). Compute step;
tables are printed by d04b_tables.py.
Units: known-sure answers (d04_lib) on NEW questions (and, for the home comparison, on TRAINING questions with out-of-fold predictions).
x = logit P_reject of the run's own grader from predictor kind k (d03), or an actual-verdict variable of that grader on the same answer
  (v_bin: greedy answer rejected; v_share: share of the 33 untrained answers rejected; v_py: -logit P(Yes) of the greedy answer, judges only).
Generic difficulty removed by FWL on G (conf, grader-free probes; d04_lib.gen_cov). Per run: slope per SD of x_perp (refusal points),
  partial correlation, adjusted refusal y - yhat_G + ybar in quintiles of x_perp (top minus bottom quintile = 'T-B').
Variants: lin16fr / tfidffr = the same predictors fitted only on training answers that are audited right (false rejections only);
  lin16G2 = lin16 with the reference runs' own refusal rate on each question added to G (generic difficulty measured empirically).
Joint model (judge and key arms): refusal ~ G + readable prediction + actual verdict, coefficients per SD of each residualised variable,
  on NEW answers and on HOME answers (out-of-fold readable prediction; the actual verdict is what training saw on that question).
Reference = no-location graders (key + uniform random noise; d04_lib.REFPAT): for each run, the 3 reference runs of the same cell nearest in
  new-question abstention, analysed with the SAME x; grader-specific = run minus matched reference.
PRE-SPECIFIED PRIMARY: kind lin16, strict arms (J_a35, J in Q M G O R), new known-sure answers, grader-specific T-B and slope, pooled with
  equal weight over (student, dataset, judge) units, students averaged; discovery qwen7, confirmation llama8. Everything else is secondary.
Uncertainty: crossed bootstrap: questions within cell (multinomial weights), runs within arm, reference runs within cell.
Output: cache/d04_units.pkl (unit index, point estimates) when seed = 0, cache/d04_boot_<seed>.npy (NB draws).
Usage: nice -n 10 timeout 280 python3 d04_dose.py <seed> <NB>"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, time, pickle, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, (_RA + "/dose"))
from d04_lib import *
SEED = int(sys.argv[1]); NB = int(sys.argv[2]); t00 = time.time()
rng = np.random.default_rng(4040 + SEED)
VK = ("v_bin", "v_share", "v_py"); FRK = ("lin16fr", "tfidffr", "lin16G2"); ALLK = KINDS + FRK + VK
BINK = ("lin16", "lin16fr", "lin16G2", "lin16p64", "tfidf", "knnE", "v_bin", "v_share")       # kinds whose quintile curves are kept
HOMEK = ("lin16", "lin16fr", "lin16G2", "lin16p64", "mlp16", "tfidf", "knnE", "v_bin", "v_share")
JOINT = (("lin16", "v_share"), ("lin16", "v_py"), ("lin16fr", "v_share"), ("lin16p64", "v_share"), ("lin16G2", "v_share"))   # readable + actual verdict
CELLS = [(st, ds) for st in STS for ds in DSS]
JT = [f"{J}_{op}" for J in J5 for op in ("a35", "t50")] + ["Q_a50"]
S = {}
for st, ds in CELLS:
    D = load_cell(st, ds)
    runs = []; arm_ix = {}
    for A, x in sorted(D["arms"].items()):
        arm_ix[A] = list(range(len(runs), len(runs) + len(x["runs"]))); runs += x["runs"]
    ref_ix = list(range(len(runs), len(runs) + len(D["ref"]))); runs += D["ref"]
    targets = sorted({x["target"] for x in D["arms"].values()} | {t for t in JT if t in D["PR"]["lin16"]})
    part = {}
    for pt, U, home in (("new", D["Unew"], False), ("home", D["Uhome"], True)):
        Y = np.column_stack([r["y"][U] for r in runs]); X = {}; G = {}
        for k in KINDS:
            G[k] = gen_cov(D, k, U, home)
            for t in targets:
                if t in D["PR"][k]: X[(k, t)] = logit(D["PR"][k][t]["oof" if home else "p"][U])
        G["v"] = gen_cov(D, "lin16", U, home)
        for k0 in ("lin16", "tfidf"):
            G[k0 + "fr"] = G[k0]
            for t in targets:
                if "fr_" + t in D["PR"][k0]: X[(k0 + "fr", t)] = logit(D["PR"][k0]["fr_" + t]["oof" if home else "p"][U])
        # lin16G2: generic difficulty measured directly as the no-location reference runs' own refusal rate on the question (empirical, grader-free)
        rr = np.column_stack([r["y"][U] for r in D["ref"]]); G["lin16G2"] = np.column_stack([G["lin16"], logit((rr.sum(1) + 0.5) / (rr.shape[1] + 1))])
        for t in targets:
            if (("lin16", t)) in X: X[("lin16G2", t)] = X[("lin16", t)]
        for t in targets:
            for vk, vv in verdicts(D, t).items(): X[(vk, t)] = vv[U]
        part[pt] = dict(U=U, Y=Y, X=X, G=G)
    vauc = {}
    nv = np.where(D["new"] & D["v"])[0]
    for t in targets:
        vb = verdicts(D, t).get("v_bin")
        if vb is None: continue
        for k in KINDS:
            if t not in D["PR"][k]: continue
            p = D["PR"][k][t]["p"]; m1 = nv[np.isfinite(vb[nv])]; m2 = D["Unew"][np.isfinite(vb[D["Unew"]])]
            f = lambda m: roc_auc_score(vb[m], p[m]) if 0 < np.nansum(vb[m]) < len(m) else np.nan
            vauc[(k, t)] = (f(m1), f(m2))
    S[(st, ds)] = dict(runs=[dict(run=r["run"], g=r["g"], seed=r["seed"], a=r["a"]) for r in runs], a=np.array([r["a"] for r in runs]), arm_ix=arm_ix,
                       ref_ix=ref_ix, arm_target={A: x["target"] for A, x in D["arms"].items()}, part=part, vauc=vauc, targets=targets)
    print(f"{st} {ds}: known-sure new {len(D['Unew'])}, home {len(D['Uhome'])}; arms {len(arm_ix)}, reference runs {len(ref_ix)} ({time.time() - t00:.0f}s)", flush=True)
del D

def run_stats(pd, k, t, w, want_bins):
    X = pd["X"].get((k, t))
    if X is None: return None
    G = pd["G"]["v" if k in VK else k]; Y = pd["Y"]; ok = np.isfinite(X) & np.all(np.isfinite(G), 1) & (w > 0)
    if ok.sum() < 30 or np.std(X[ok]) < 1e-6: return None
    ww = w[ok]; x = X[ok]; Yk = Y[ok]
    sl, pc, xp, Yp = fwl(x, Yk, G[ok], ww)
    out = dict(sl=sl, pc=pc)
    yadj = Yp + (ww[:, None] * Yk).sum(0) / ww.sum()
    b = wquantile_bins(xp, ww, 5); bins = np.full((5, Y.shape[1]), np.nan)
    for j in range(5):
        m = b == j
        if ww[m].sum() > 0: bins[j] = (ww[m][:, None] * yadj[m]).sum(0) / ww[m].sum()
    out["tb"] = bins[4] - bins[0]
    if want_bins: out["bins"] = bins
    return out

def joint_stats(pd, kr, kv, t, w):
    """refusal ~ G + readable + verdict (both residualised on G): per-run coefficients per SD of each residualised variable."""
    Xr = pd["X"].get((kr, t)); Xv = pd["X"].get((kv, t))
    if Xr is None or Xv is None: return None
    G = pd["G"][kr]; Y = pd["Y"]; ok = np.isfinite(Xr) & np.isfinite(Xv) & np.all(np.isfinite(G), 1) & (w > 0)
    if ok.sum() < 30 or np.std(Xr[ok]) < 1e-6 or np.std(Xv[ok]) < 1e-6: return None
    ww = w[ok]; R = resid(np.column_stack([Xr[ok], Xv[ok], Y[ok]]), G[ok], ww); xr, xv, Yp = R[:, 0], R[:, 1], R[:, 2:]
    M = np.array([[(ww * xr * xr).sum(), (ww * xr * xv).sum()], [(ww * xr * xv).sum(), (ww * xv * xv).sum()]])
    rhs = np.vstack([(ww[:, None] * xr[:, None] * Yp).sum(0), (ww[:, None] * xv[:, None] * Yp).sum(0)])
    cf = np.linalg.solve(M + 1e-9 * np.eye(2), rhs); sd = np.sqrt(np.diag(M) / ww.sum())
    return dict(br=cf[0] * sd[0], bv=cf[1] * sd[1], corr=M[0, 1] / np.sqrt(M[0, 0] * M[1, 1]))

# unit list: (st, ds, A, part, kind) with own target; cross-judge units (st, ds, A, 'new', 'x_<J2>') with lin16 on another judge's target
UNITS = []; BUNITS = []
for (st, ds), sd in S.items():
    for A, t in sd["arm_target"].items():
        for pt in ("new", "home"):
            for k in (ALLK if pt == "new" else HOMEK):
                UNITS.append((st, ds, A, pt, k, t))
                if k in BINK and (pt == "new" or k == "lin16"): BUNITS.append((st, ds, A, pt, k, t))
        if A.startswith("J"):
            op = A.split("_")[1]
            for J2 in J5:
                t2 = f"{J2}_{op}"
                if t2 != t and t2 in sd["targets"]: UNITS.append((st, ds, A, "new", "x_" + J2, t2))
JUNITS = [(st, ds, A, pt, kr + "+" + kv, t) for (st, ds), sd in S.items() for A, t in sd["arm_target"].items() if A.startswith("J") or A == "key"
          for pt in ("new", "home") for kr, kv in JOINT]
UIX = {u: i for i, u in enumerate(UNITS)}; BIX = {u: i for i, u in enumerate(BUNITS)}; JIX = {u: i for i, u in enumerate(JUNITS)}
NST = 6   # sl, sl_ref, pc, pc_ref, tb, tb_ref

def evaluate(W, SEL, REFSEL):
    A1 = np.full((len(UNITS), NST), np.nan); A2 = np.full((len(BUNITS), 2, 5), np.nan); A3 = np.full((len(JUNITS), 5), np.nan)
    for (st, ds), sd in S.items():
        a = sd["a"]; pool = np.array(sd["ref_ix"])[REFSEL[(st, ds)]]
        for pt in ("new", "home"):
            pd = sd["part"][pt]; w = W[(st, ds, pt)]; cache = {}
            for u in [u for u in UNITS if u[0] == st and u[1] == ds and u[3] == pt]:
                _, _, A, _, k, t = u; kk = "lin16" if k.startswith("x_") else k
                want = u in BIX
                key = (kk, t, want)
                if key not in cache: cache[key] = run_stats(pd, kk, t, w, want)
                r = cache[key]
                if r is None: continue
                ix = np.array(sd["arm_ix"][A])[SEL[(st, ds, A)]]
                nn = [pool[np.argsort(np.abs(a[pool] - a[i]), kind="stable")[:3]] for i in ix]
                vals = []
                for nm in ("sl", "pc", "tb"):
                    vals += [r[nm][ix].mean(), np.mean([r[nm][j].mean() for j in nn])]
                A1[UIX[u]] = vals
                if want:
                    A2[BIX[u], 0] = r["bins"][:, ix].mean(1); A2[BIX[u], 1] = np.mean([r["bins"][:, j].mean(1) for j in nn], 0)
            for u in [u for u in JUNITS if u[0] == st and u[1] == ds and u[3] == pt]:
                _, _, A, _, kk, t = u; kr, kv = kk.split("+"); key = ("J", kr, kv, t)
                if key not in cache: cache[key] = joint_stats(pd, kr, kv, t, w)
                r = cache[key]
                if r is None: continue
                ix = np.array(sd["arm_ix"][A])[SEL[(st, ds, A)]]
                nn = [pool[np.argsort(np.abs(a[pool] - a[i]), kind="stable")[:3]] for i in ix]
                A3[JIX[u]] = [r["br"][ix].mean(), np.mean([r["br"][j].mean() for j in nn]), r["bv"][ix].mean(), np.mean([r["bv"][j].mean() for j in nn]), r["corr"]]
    return A1, A2, A3

def draw():
    W = {}; SEL = {}; REFSEL = {}
    for (st, ds), sd in S.items():
        for pt in ("new", "home"):
            n = len(sd["part"][pt]["U"]); W[(st, ds, pt)] = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for A, ix in sd["arm_ix"].items(): SEL[(st, ds, A)] = rng.integers(0, len(ix), len(ix))
        REFSEL[(st, ds)] = rng.integers(0, len(sd["ref_ix"]), len(sd["ref_ix"]))
    return W, SEL, REFSEL
if SEED == 0:
    W = {(st, ds, pt): np.ones(len(sd["part"][pt]["U"])) for (st, ds), sd in S.items() for pt in ("new", "home")}
    SEL = {(st, ds, A): np.arange(len(ix)) for (st, ds), sd in S.items() for A, ix in sd["arm_ix"].items()}
    REFSEL = {(st, ds): np.arange(len(sd["ref_ix"])) for (st, ds), sd in S.items()}
    A1, A2, A3 = evaluate(W, SEL, REFSEL)
    # descriptive point estimates per run: raw AUROC of x and of x_perp for the run's refusal (new known-sure answers)
    AUC = {}
    for (st, ds), sd in S.items():
        pd = sd["part"]["new"]
        for A, t in sd["arm_target"].items():
            for k in ALLK:
                X = pd["X"].get((k, t))
                if X is None: continue
                G = pd["G"]["v" if k in VK else k]; ok = np.isfinite(X) & np.all(np.isfinite(G), 1)
                if ok.sum() < 30 or np.std(X[ok]) < 1e-6: continue
                xp = resid(X[ok][:, None], G[ok], np.ones(ok.sum()))[:, 0]; L = []
                for i in sd["arm_ix"][A]:
                    y = pd["Y"][ok, i]
                    if 3 <= y.sum() <= len(y) - 3: L.append((roc_auc_score(y, X[ok]), roc_auc_score(y, xp)))
                if L: AUC[(st, ds, A, k)] = tuple(np.mean(L, 0))
    pickle.dump(dict(UNITS=UNITS, BUNITS=BUNITS, JUNITS=JUNITS, A1=A1, A2=A2, A3=A3, AUC=AUC,
                     S={k: dict(runs=v["runs"], arm_target=v["arm_target"], arm_ix=v["arm_ix"], ref_ix=v["ref_ix"], vauc=v["vauc"],
                                n_new=len(v["part"]["new"]["U"]), n_home=len(v["part"]["home"]["U"])) for k, v in S.items()}),
                open(f"{CACHE}/d04_units.pkl", "wb"))
    print(f"point estimates: {len(UNITS)} units, {len(BUNITS)} curve units ({time.time() - t00:.0f}s)", flush=True)
if NB > 0:
    B1 = np.zeros((NB, len(UNITS), NST), np.float32); B2 = np.zeros((NB, len(BUNITS), 2, 5), np.float32); B3 = np.zeros((NB, len(JUNITS), 5), np.float32)
    for b in range(NB):
        B1[b], B2[b], B3[b] = evaluate(*draw())
        if b % 10 == 9: print(f"  draw {b + 1}/{NB} ({time.time() - t00:.0f}s)", flush=True)
    np.save(f"{CACHE}/d04_boot1_{SEED}.npy", B1); np.save(f"{CACHE}/d04_boot2_{SEED}.npy", B2); np.save(f"{CACHE}/d04_boot3_{SEED}.npy", B3)
print(f"done ({time.time() - t00:.0f}s)", flush=True)
