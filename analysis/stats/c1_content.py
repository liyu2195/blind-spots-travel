"""(stats) Claim 1: WHICH wrong answer a judge pays is memorised at home and barely reaches new questions.
Runs: FK20, IDK credit 0.5, judge<J><lv> with J in Q M G O and lv in a35 / a50 / t50 (judge runs); key-trained sim<FP>fn<FN> runs for the DiD.
Unit: (run, question) where the run's final greedy answer x is key-wrong (loose key), well-formed, not IDK, and differs from the untrained greedy.
paid_J(x) = P_J(x) > t_J(lv) (thresholds_<J>.json; t50 = 0.5), P_J looked up by (idx, first line) else (idx, normalised answer).
b_J(q) = share of the untrained student's key-wrong samples on q that J pays at the same level (the question's base rate).
lift_J(run) = mean over units of paid_J(x) - b_J(q);  S(run) = lift_own - mean of lift over the 3 other judges (common support: all 4 verdicts).
PRIMARY estimand: S_new = mean over judge runs of S(run) on NEW questions (duplicate-free), per student and pooled (students weighted equally);
  'transfer ratio' T = S_new / S_home.
Uncertainty: crossed bootstrap (2000): questions resampled within (student, dataset, part) with the same draw for every run, and runs resampled
  within (student, dataset, judge, level) cells. Run-level permutation: judge labels permuted among judge runs within (student, dataset, level)
  (10,000 draws) -> null of S_new and S_home. TOST on S_new with margin 0.05 (about 1/6 of the home lift) and the equivalence bound.
Robustness: changed vs the run's own ck0 re-decode; DiD vs key-trained runs; decode-stable questions only; cell-weighted means.
Output: c1_content_out.txt, c1_rows.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c1_content_out.txt"); P(__doc__)
TH = {J: thresholds(J) for J in J4}
rng = np.random.default_rng(101)
UNITS = []          # dict(st, ds, part, run, kind, own, lv, cell, qpos, D (units x 4), variant)
QN = {}
for st in STS:
    for ds in DSS:
        dsn = DSN[ds]; B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
        JF = pickle.load(open(f"{CACHE}/jf_{dsn}.pkl", "rb")); N = B["N"]; X = B["X"]
        thr = {(J, lv): (0.5 if lv == "t50" else TH[J][f"{st}_{dsn}"][f"t_{lv}"]) for J in J4 for lv in ("a35", "a50", "t50")}
        def pyes(J, i, line, x):
            v = JF[J]["line"].get((i, line)); return v if v is not None else JF[J]["norm"].get((i, x))
        wrong_s = (B["valid"] & ~B["kl"])[:, 1:]          # untrained key-wrong samples
        bJ = {}
        for J in J4:
            for lv in ("a35", "a50", "t50"):
                pj = B["PJ"][J][:, 1:]; paid = (pj > thr[(J, lv)]) & wrong_s & np.isfinite(pj); den = (wrong_s & np.isfinite(pj)).sum(1)
                bJ[(J, lv)] = np.where(den > 0, paid.sum(1) / np.maximum(den, 1), np.nan)
        runs_j = {n: r for n, r in R.items() if r["tag"] == "FK20" and r["c"] == "05" and (re.fullmatch(r"judge[QMGO](a35|a50|t50)", r["g"]) or re.fullmatch(r"sim\d\dfn\d\d", r["g"]))}
        stable = np.array([all(r["z"] is not None and r["z"][i] == X[i][0] for r in runs_j.values()) for i in range(N)])
        parts = {"home": np.where(B["train"])[0], "new": np.where(B["new"])[0]}
        for part, items in parts.items():
            QN[(st, ds, part)] = len(items); pos = {int(i): k for k, i in enumerate(items)}
            for name, r in sorted(runs_j.items()):
                m = re.fullmatch(r"judge([QMGO])(a35|a50|t50)", r["g"]); kind = "judge" if m else "key"
                for variant in ("base", "ck0", "stable"):
                    if variant == "ck0" and r["z"] is None: continue
                    for lv in ([m.group(2)] if m else ["a35", "a50", "t50"]):
                        qp, D = [], []
                        for i in items:
                            x = r["f"][i]
                            if r["o"][i] != 2: continue
                            if variant in ("base", "stable") and x == X[i][0]: continue
                            if variant == "ck0" and x == r["z"][i]: continue
                            if variant == "stable" and not stable[i]: continue
                            pv = [pyes(J, i, r["f_line"][i], x) for J in J4]; bv = [bJ[(J, lv)][i] for J in J4]
                            if any(v is None for v in pv) or any(not np.isfinite(v) for v in bv): continue
                            qp.append(pos[int(i)]); D.append([(pv[k] > thr[(J4[k], lv)]) - bv[k] for k in range(4)])
                        if len(qp) >= 5:
                            UNITS.append(dict(st=st, ds=ds, part=part, run=name, kind=kind, own=m.group(1) if m else None, lv=lv, seed=r["seed"],
                                              cell=(st, ds, m.group(1) if m else "key", lv), qpos=np.array(qp), D=np.array(D, float), variant=variant))
        P(f"{st} {ds}: judge runs {sum(1 for r in runs_j.values() if r['g'].startswith('judge'))}, key runs {sum(1 for r in runs_j.values() if r['g'].startswith('sim'))}, decode-stable share {stable.mean():.3f}")
pickle.dump(dict(units=UNITS, QN=QN), open(f"{HERE}/c1_rows.pkl", "wb"))

def run_values(units, W=None):
    """per unit-row: lift vector (4,) with question weights W[(st, ds, part)]"""
    out = []
    for u in units:
        w = np.ones(len(u["qpos"])) if W is None else W[(u["st"], u["ds"], u["part"])][u["qpos"]]
        if w.sum() == 0: out.append(None); continue
        out.append((w[:, None] * u["D"]).sum(0) / w.sum())
    return out

def spec(l, own):
    k = J4.index(own); return l[k] - np.mean([l[j] for j in range(4) if j != k])

def summarise(units, lifts, runsel=None, did=False, cellw=False):
    """mean S over judge runs per (st, part); runsel: dict cell -> list of unit indices (resampled runs)"""
    keym = defaultdict(list)
    if did:
        for u, l in zip(units, lifts):
            if u["kind"] == "key" and l is not None:
                for J in J4: keym[(u["st"], u["ds"], u["part"], u["lv"], J)].append(spec(l, J))
    vals = defaultdict(list); cells = defaultdict(lambda: defaultdict(list))
    idxs = range(len(units)) if runsel is None else [j for L in runsel.values() for j in L]
    for j in idxs:
        u = units[j]; l = lifts[j]
        if u["kind"] != "judge" or l is None: continue
        s = spec(l, u["own"])
        if did:
            km = keym.get((u["st"], u["ds"], u["part"], u["lv"], u["own"]))
            if not km: continue
            s -= np.mean(km)
        vals[(u["st"], u["part"])].append(s); cells[(u["st"], u["part"])][u["cell"]].append(s)
    out = {}
    for k in vals:
        out[k] = np.mean([np.mean(v) for v in cells[k].values()]) if cellw else np.mean(vals[k])
    for part in ("home", "new"):
        if (STS[0], part) in out and (STS[1], part) in out: out[("pooled", part)] = (out[(STS[0], part)] + out[(STS[1], part)]) / 2
    return out

def boot(units, NB=2000, did=False, cellw=False, seed=7):
    rg = np.random.default_rng(seed); reps = []
    cellidx = defaultdict(list)
    for j, u in enumerate(units): cellidx[(u["cell"], u["part"])].append(j)
    for b in range(NB):
        W = {k: np.bincount(rg.integers(0, n, n), minlength=n).astype(float) for k, n in QN.items()}
        lifts = run_values(units, W)
        sel = {c: [L[t] for t in rg.integers(0, len(L), len(L))] for c, L in cellidx.items()}
        reps.append(summarise(units, lifts, sel, did, cellw))
    return reps

def report(label, units, did=False, cellw=False, NB=2000, tost=True):
    if not did: units = [u for u in units if u["kind"] == "judge"]
    lifts = run_values(units); est = summarise(units, lifts, None, did, cellw); reps = boot(units, NB, did, cellw)
    P(f"\n== {label}")
    res = {}
    for who in ("qwen7", "llama8", "pooled"):
        h = est.get((who, "home"), np.nan); n = est.get((who, "new"), np.nan)
        bh = [r.get((who, "home"), np.nan) for r in reps]; bn = [r.get((who, "new"), np.nan) for r in reps]
        br = [a / b if b else np.nan for a, b in zip(bn, bh)]
        ch, cn, cr = pct(bh), pct(bn), pct(br)
        nr = sum(1 for u, l in zip(units, lifts) if u["kind"] == "judge" and u["part"] == "new" and (who == "pooled" or u["st"] == who) and l is not None)
        line = (f"  {who:6s} home S {h:+.3f} [{ch[0]:+.3f}, {ch[1]:+.3f}]  new S {n:+.3f} [{cn[0]:+.3f}, {cn[1]:+.3f}]  transfer ratio {n / h:.3f} [{cr[0]:.3f}, {cr[1]:.3f}]"
                f"  (judge-run rows on new: {nr})")
        if tost:
            t = tost_from_boot(n, bn, 0.05); line += f"  | TOST(new, margin 0.05) p {t['p']:.4f}, 90% CI [{t['ci90'][0]:+.3f}, {t['ci90'][1]:+.3f}], bound {t['bound']:.3f}"
            res[who] = dict(home=h, new=n, ci_home=ch, ci_new=cn, ratio=n / h, ci_ratio=cr, tost=t, p_ratio_lt_02=float(np.mean(np.array(br) >= 0.2)))
            line += f"  | P*(ratio >= 0.2) {res[who]['p_ratio_lt_02']:.4f}"
        P(line)
    return est, res

MAIN = [u for u in UNITS if u["variant"] == "base"]
est, RES = report("MAIN: changed vs untrained greedy, common support, run-weighted", MAIN)
# per dataset (questions and runs of one dataset)
P("\n== per dataset (main definition; crossed bootstrap 1000)")
for ds in DSS:
    sub = [u for u in MAIN if u["ds"] == ds and u["kind"] == "judge"]; l = run_values(sub); e = summarise(sub, l); rr = boot(sub, 1000)
    P("  " + ds + ": " + "  ".join(f"{st} home {e[(st, 'home')]:+.3f} new {e[(st, 'new')]:+.3f} [{pct([r.get((st, 'new'), np.nan) for r in rr])[0]:+.3f}, {pct([r.get((st, 'new'), np.nan) for r in rr])[1]:+.3f}]" for st in STS))
# run-level permutation of judge labels within (student, dataset, level)
P("\n== run-level permutation: judge labels permuted among judge runs within (student, dataset, level), 10,000 draws")
lifts = run_values(MAIN)
grp = defaultdict(list)
for j, u in enumerate(MAIN):
    if u["kind"] == "judge" and lifts[j] is not None: grp[(u["st"], u["ds"], u["lv"], u["part"])].append(j)
def perm_stat(owners):
    v = defaultdict(list)
    for j, o in owners.items(): v[(MAIN[j]["st"], MAIN[j]["part"])].append(spec(lifts[j], o))
    out = {k: np.mean(x) for k, x in v.items()}
    for part in ("home", "new"): out[("pooled", part)] = (out[("qwen7", part)] + out[("llama8", part)]) / 2
    return out
obs = perm_stat({j: MAIN[j]["own"] for L in grp.values() for j in L})
null = []
for b in range(10000):
    ow = {}
    for k, L in grp.items():
        labs = [MAIN[j]["own"] for j in L]; perm = rng.permutation(len(L))
        for j, t in zip(L, perm): ow[j] = labs[t]
    null.append(perm_stat(ow))
PERM = {}
for who in ("qwen7", "llama8", "pooled"):
    for part in ("home", "new"):
        nv = np.array([x[(who, part)] for x in null]); p = (np.sum(nv >= obs[(who, part)]) + 1) / (len(nv) + 1); PERM[(who, part)] = p
        P(f"  {who:6s} {part}: observed {obs[(who, part)]:+.3f}; null mean {nv.mean():+.4f}, 95th pct {np.percentile(nv, 95):+.4f}, 99.9th {np.percentile(nv, 99.9):+.4f}; one-sided p {p:.5f}")
# robustness
P("\n== robustness (pooled = mean of the two students)")
for lab, sub, did, cw in (("changed vs the run's own ck0 re-decode", [u for u in UNITS if u["variant"] == "ck0"], False, False),
                          ("DiD: minus the same judge's S in key-trained runs (same st, ds, level)", MAIN, True, False),
                          ("decode-stable questions only", [u for u in UNITS if u["variant"] == "stable"], False, False),
                          ("cell-weighted (mean over (st, ds, judge, level) cells)", MAIN, False, True)):
    report(lab, sub, did, cw, NB=1000)
# rank statistic
P("\n== share of judge runs whose own judge has the largest of the four lifts (chance 0.25)")
for st in STS:
    for part in ("home", "new"):
        v = [int(np.argmax(l) == J4.index(u["own"])) for u, l in zip(MAIN, lifts) if u["kind"] == "judge" and u["st"] == st and u["part"] == part and l is not None]
        P(f"  {st} {part}: {sum(v)}/{len(v)} = {np.mean(v):.2f}")
# run / seed inventory
P("\n== runs and seeds (judge runs with >= 5 units on new questions)")
inv = defaultdict(set)
for u in MAIN:
    if u["kind"] == "judge" and u["part"] == "new": inv[(u["st"], u["ds"], u["own"] + u["lv"])].add(u["seed"])
for st in STS:
    P(f"  {st}: " + "; ".join(f"{ds} {g} seeds {sorted(s)}" for (s_, ds, g), s in sorted(inv.items()) if s_ == st))
P(f"  total judge runs (new): {sum(len(s) for s in inv.values())}; cells {len(inv)}")
pickle.dump(dict(RES=RES, PERM=PERM), open(f"{HERE}/c1_summary.pkl", "wb"))
P.save()
