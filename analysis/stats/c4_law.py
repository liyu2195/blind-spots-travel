"""(stats) Claim 4: new-question abstention follows the leniency L of the grader (out of sample across students).
Runs (main grid): FK20, IDK credit 0.5, config FMT 1 / KLB 0.2 / 120 steps / standard training file / no forced IDK / standard advantages.
Uniform-noise graders: sim<FP>fn<FN> and lenQ with FPMODE random, FPK = FPU, no set-dependent FN (accept a strict-right sample with prob 1 - FN,
  any other valid non-IDK sample with prob FP: the trainer's synth_rewards). Judge runs: judge<J><lv>, J in Q M G O R, lv a35 / a50 / t50.
L = expected share of the untrained student's valid non-IDK samples (32 per question) on the TRAINED questions that the grader accepts;
x = L - c (c = 0.5); y = share of NEW questions whose final greedy answer is IDK.
Model y = 1 / (1 + exp(-(a + b x))), least squares on the probability scale, fitted on qwen7 uniform runs (discovery), scored on llama8 uniform runs.
PRIMARY estimand: out-of-sample R^2 = 1 - SSE / SST on the llama8 uniform runs (SST about their own mean); also MAE and mean bias.
Uncertainty: crossed bootstrap (2000): grader configurations (dataset x grader code, all seeds together) resampled independently in the fit and
  test sets, AND questions resampled within each dataset (trained questions -> L, new questions -> y; the same draw for both students, who share
  the questions); the curve is refitted in every replicate. Variants: configuration-only and question-only bootstraps.
Test of 'no predictive value': x permuted over the fit configurations, curve refitted, R^2 on the test set (2000 draws).
Robustness: reverse direction; dataset and seed splits; L from the greedy answer only; L on new questions; LLM-judge runs as test set;
  'error type adds nothing at fixed L': extra predictor FR (= FN) or FA (= FP), Delta R^2 on llama8 with bootstrap interval.
Output: c4_law_out.txt, c4_points.csv"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, pickle, numpy as np, warnings
from collections import defaultdict
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c4_law_out.txt"); P(__doc__)
rng = np.random.default_rng(404)
TH = {J: thresholds(J) for J in J5}
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
def fit(x, y):
    return curve_fit(f, np.asarray(x, float), np.asarray(y, float), p0=[-1, -8], maxfev=20000)[0]
def r2(y, p): y = np.asarray(y); p = np.asarray(p); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
QA = {}      # (st, ds) -> per trained question n_app, n_rs, n_acc[(J, lv)], greedy app/rs/acc; per new question n_app_new, n_rs_new
RUNS = []    # dict(st, ds, run, fam, cfgkey, seed, FP, FN, J, lv, idk_new (vector), ...)
for st in STS:
    for ds in DSS:
        dsn = DSN[ds]; B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
        tr = np.where(B["train"])[0]; nw = np.where(B["new"])[0]
        app = B["valid"] & ~B["idk"]; S32 = np.zeros(33, bool); S32[1:] = True
        q = dict(n_app=(app[tr][:, 1:]).sum(1).astype(float), n_rs=(app & B["ks"])[tr][:, 1:].sum(1).astype(float),
                 g_app=app[tr, 0].astype(float), g_rs=(app & B["ks"])[tr, 0].astype(float),
                 n_app_new=(app[nw][:, 1:]).sum(1).astype(float), n_rs_new=(app & B["ks"])[nw][:, 1:].sum(1).astype(float), acc={}, gacc={}, acc_new={})
        for J in J5:
            for lv in ("a35", "a50", "t50"):
                t = 0.5 if lv == "t50" else TH[J][f"{st}_{dsn}"][f"t_{lv}"]; pj = B["PJ"][J]; ok = app & np.isfinite(pj) & (np.nan_to_num(pj) > t)
                q["acc"][(J, lv)] = ok[tr][:, 1:].sum(1).astype(float); q["gacc"][(J, lv)] = ok[tr, 0].astype(float); q["acc_new"][(J, lv)] = ok[nw][:, 1:].sum(1).astype(float)
        QA[(st, ds)] = q
        for n, r in sorted(R.items()):
            c = r["cfg"]
            if r["tag"] != "FK20" or r["c"] != "05" or not c: continue
            if c.get("FMT") != "1" or c.get("KLB") != "0.2" or c.get("STEPS", "120") != "120" or c.get("TRAINF", f"{dsn}_h1.jsonl") != f"{dsn}_h1.jsonl": continue
            if c.get("FORCE") or c.get("ADV", "std") != "std" or any(k in c for k in ("TEMP", "TOPP", "TOPK", "LRATE", "SWITCH", "QUORUM")) or r["g"].endswith("f1"): continue
            m = re.fullmatch(r"judge([QMGOR])(a35|a50|t50)", r["g"])
            rec = dict(st=st, ds=ds, run=n, seed=r["seed"], idk_new=(r["o"][nw] == 0).astype(float), g=r["g"])
            if m and c.get("REWARD") == "judge":
                rec.update(fam="judge", J=m.group(1), lv=m.group(2), cfgkey=f"{ds}|{r['g']}")
            elif (re.fullmatch(r"sim\d\dfn\d\d", r["g"]) or r["g"] == "lenQ") and c.get("FPMODE") == "random" and c.get("FPK") == c.get("FPU") and "FNK" not in c and c.get("REWARD", "synth") == "synth":
                rec.update(fam="uni", FP=float(c["FPK"]), FN=float(c.get("FN", 0)), cfgkey=f"{ds}|{r['g']}")
            else: continue
            RUNS.append(rec)
P("inventory: " + ", ".join(f"{st} {fam} {sum(1 for r in RUNS if r['st'] == st and r['fam'] == fam)}" for st in STS for fam in ("uni", "judge")))
Wones = {}
def Lvals(runs, Wt=None, which="L32h"):
    out = []
    for r in runs:
        q = QA[(r["st"], r["ds"])]; w = np.ones(len(q["n_app"])) if Wt is None else Wt[r["ds"]][0]
        wn = np.ones(len(q["n_app_new"])) if Wt is None else Wt[r["ds"]][1]
        if which == "L32h": a, rs = (w * q["n_app"]).sum(), (w * q["n_rs"]).sum(); acc = q["acc"]; wa = w; base = q["n_app"]
        elif which == "Lgh": a, rs = (w * q["g_app"]).sum(), (w * q["g_rs"]).sum(); acc = q["gacc"]; wa = w; base = q["g_app"]
        else: a, rs = (wn * q["n_app_new"]).sum(), (wn * q["n_rs_new"]).sum(); acc = q["acc_new"]; wa = wn
        if r["fam"] == "uni": out.append(r["FP"] + (1 - r["FN"] - r["FP"]) * rs / a)
        else: out.append((wa * acc[(r["J"], r["lv"])]).sum() / a)
    return np.array(out)
def Yvals(runs, Wt=None):
    return np.array([r["idk_new"].mean() if Wt is None else (Wt[r["ds"]][1] * r["idk_new"]).sum() / Wt[r["ds"]][1].sum() for r in runs])
U = [r for r in RUNS if r["fam"] == "uni"]; JR = [r for r in RUNS if r["fam"] == "judge"]
QU = [r for r in U if r["st"] == "qwen7"]; LU = [r for r in U if r["st"] == "llama8"]
def evaluate(fitset, testset, Wt=None, which="L32h", extra=None):
    xa = Lvals(fitset, Wt, which) - 0.5; ya = Yvals(fitset, Wt); xb = Lvals(testset, Wt, which) - 0.5; yb = Yvals(testset, Wt)
    if extra is None:
        p = fit(xa, ya); pb = f(xb, *p)
    else:
        g2 = lambda X, a, b, c: 1 / (1 + np.exp(-(a + b * X[0] + c * X[1])))
        ea = np.array([r[extra] for r in fitset]); eb = np.array([r[extra] for r in testset])
        p = curve_fit(g2, np.vstack([xa, ea]), ya, p0=[-1, -8, 0], maxfev=20000)[0]; pb = g2(np.vstack([xb, eb]), *p)
    return r2(yb, pb), np.mean(np.abs(yb - pb)), np.mean(yb - pb), p
def qdraw():
    W = {}
    for ds in DSS:
        n1 = len(QA[("qwen7", ds)]["n_app"]); n2 = len(QA[("qwen7", ds)]["n_app_new"])
        W[ds] = (np.bincount(rng.integers(0, n1, n1), minlength=n1).astype(float), np.bincount(rng.integers(0, n2, n2), minlength=n2).astype(float))
    return W
def cdraw(runs):
    g = defaultdict(list)
    for r in runs: g[r["cfgkey"]].append(r)
    ks = list(g); return [x for k in rng.choice(len(ks), len(ks)) for x in g[ks[k]]]
def boot(fitset, testset, NB=2000, mode="crossed", which="L32h", extra=None):
    out = []
    for b in range(NB):
        A = cdraw(fitset) if mode in ("crossed", "config") else fitset; Bt = cdraw(testset) if mode in ("crossed", "config") else testset
        W = qdraw() if mode in ("crossed", "question") else None
        try: out.append(evaluate(A, Bt, W, which, extra)[:3])
        except Exception: continue
    return np.array(out)
def line(lab, fitset, testset, NB=2000, mode="crossed", which="L32h"):
    s = evaluate(fitset, testset, None, which); bb = boot(fitset, testset, NB, mode, which)
    ci = np.percentile(bb, [2.5, 97.5], axis=0)
    P(f"  {lab:44s} fit {len(fitset):3d} -> test {len(testset):3d}: R2 {s[0]:.3f} [{ci[0, 0]:.2f}, {ci[1, 0]:.2f}]  MAE {s[1]:.3f} [{ci[0, 1]:.3f}, {ci[1, 1]:.3f}]  "
      f"bias {s[2]:+.3f} [{ci[0, 2]:+.3f}, {ci[1, 2]:+.3f}]  (a {s[3][0]:+.2f}, b {s[3][1]:+.2f}; {mode} bootstrap {len(bb)})")
    return s, ci
P("\n== PRIMARY: qwen7 uniform -> llama8 uniform")
S0, CI0 = line("crossed (configs x questions)", QU, LU, 2000, "crossed")
line("configuration-only bootstrap", QU, LU, 500, "config")
line("question-only bootstrap", QU, LU, 500, "question")
# permutation test of 'no predictive value'
xa = Lvals(QU) - 0.5; ya = Yvals(QU); xb = Lvals(LU) - 0.5; yb = Yvals(LU)
cfgs = sorted({r["cfgkey"] for r in QU}); ci_ = {k: [j for j, r in enumerate(QU) if r["cfgkey"] == k] for k in cfgs}
null = []
for b in range(2000):
    perm = rng.permutation(len(cfgs)); xp = xa.copy()
    for k, k2 in zip(cfgs, np.array(cfgs)[perm]):
        # every run of configuration k takes the mean x of configuration k2 (permutation of grader descriptors over configurations)
        xp[ci_[k]] = xa[ci_[k2]].mean()
    try: p = fit(xp, ya); null.append(r2(yb, f(xb, *p)))
    except Exception: null.append(-np.inf)
null = np.array(null); P(f"  permutation of L over fit configurations: null R2 median {np.median(null):.3f}, 99.9th pct {np.percentile(null, 99.9):.3f}; p = {(np.sum(null >= S0[0]) + 1) / (len(null) + 1):.5f}")
P("\n== per dataset of the test set (curve fitted on all qwen7 uniform runs)")
for ds in DSS:
    line(f"test llama8 {ds} only", QU, [r for r in LU if r["ds"] == ds], 500, "crossed")
P("\n== robustness")
line("reverse: llama8 uniform -> qwen7 uniform", LU, QU, 500)
for nm, a, b in (("PopQA -> TriviaQA + NQ", "pq", None), ("TriviaQA -> PopQA + NQ", "h1", None), ("NQ -> TriviaQA + PopQA", "nq", None)):
    line(nm + " (both students)", [r for r in U if r["ds"] == a], [r for r in U if r["ds"] != a], 500)
line("seeds 0-1 -> seeds >= 2 (both students)", [r for r in U if r["seed"] <= 1], [r for r in U if r["seed"] >= 2], 500)
line("L from the greedy answer only", QU, LU, 500, "crossed", "Lgh")
line("L on NEW questions samples", QU, LU, 500, "crossed", "L32n")
P("\n== LLM-judge runs as the test set (never fitted), qwen7-uniform curve")
line("all judge runs (both students)", QU, JR, 500)
for lv in ("a35", "a50", "t50"):
    sub = [r for r in JR if r["lv"] == lv]; Ls = Lvals(sub); ys = Yvals(sub); p = fit(xa, ya); res = ys - f(Ls - 0.5, *p)
    P(f"    {lv}: n {len(sub):3d}  L mean {Ls.mean():.3f} sd {Ls.std():.3f}  abstention mean {ys.mean():.3f} sd {ys.std():.3f}  residual mean {res.mean():+.3f}  |res| > 0.1 in {np.mean(np.abs(res) > 0.1):.2f}")
P("\n== 'error type adds nothing at fixed L': add FR (= FN) or FA (= FP) to the qwen7 fit, score llama8; Delta R^2 [crossed bootstrap 1000]")
for r in U: r["FR"] = r["FN"]; r["FA"] = r["FP"]
for ex in ("FR", "FA"):
    s1 = evaluate(QU, LU, None, "L32h", ex); d = []
    for b in range(500):
        A = cdraw(QU); Bt = cdraw(LU); W = qdraw()
        try: d.append(evaluate(A, Bt, W, "L32h", ex)[0] - evaluate(A, Bt, W)[0])
        except Exception: pass
    t = tost_from_boot(s1[0] - S0[0], d, 0.05)
    P(f"  + {ex}: R2 {s1[0]:.3f} (Delta {s1[0] - S0[0]:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]); TOST |Delta R2| < 0.05 p {t['p']:.4f}, bound {t['bound']:.3f}")
P("\n== runs and seeds")
for st in STS:
    for fam in ("uni", "judge"):
        rr = [r for r in RUNS if r["st"] == st and r["fam"] == fam]
        P(f"  {st} {fam}: runs {len(rr)}, configurations {len({r['cfgkey'] for r in rr})}, seeds per configuration: median {np.median([sum(1 for x in rr if x['cfgkey'] == k) for k in {r['cfgkey'] for r in rr}]):.0f}, range "
          f"{min(sum(1 for x in rr if x['cfgkey'] == k) for k in {r['cfgkey'] for r in rr})}-{max(sum(1 for x in rr if x['cfgkey'] == k) for k in {r['cfgkey'] for r in rr})}")
import csv
with open(f"{HERE}/c4_points.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run", "st", "ds", "fam", "g", "seed", "L", "y_new"])
    for r, L, y in zip(RUNS, Lvals(RUNS), Yvals(RUNS)): w.writerow([r["run"], r["st"], r["ds"], r["fam"], r["g"], r["seed"], f"{L:.4f}", f"{y:.4f}"])
pickle.dump(dict(R2=S0, CI=CI0, p_perm=(np.sum(null >= S0[0]) + 1) / (len(null) + 1)), open(f"{HERE}/c4_summary.pkl", "wb"))
P.save()
