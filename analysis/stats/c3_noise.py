"""(stats) Claim 3: on NEW questions a lenient judge (Qwen judge, P(Yes) > 0.5: judgeQt50) behaves like a random grader with the same error rates
(lenQ: wrong answers paid and right answers rejected at random at the judge's FP / FN rates), FK20, IDK credit 0.5, 6 (student x dataset) cells x
3 seeds per arm. Seeds of the two arms share the batch schedule (same seed = same question order), so runs are paired by seed where both exist.
Outcomes per run on new (duplicate-free) questions: abstention (final greedy is IDK), right (loose key), wrong; and on home questions.
Judge-paid share: share of the run's final key-wrong answers that judge Q would accept at 0.5 (lookup of the final string).
PRIMARY estimand: D = mean over the 6 cells of [mean(judge runs) - mean(lenQ runs)] of new-question abstention; same for right and wrong.
Uncertainty: crossed bootstrap (4000): questions resampled within each cell (same draw for both arms), runs resampled within (cell, arm).
Equivalence: TOST with margin 0.03 (about 2 seed SDs; 1/3 of the leniency curve's out-of-sample MAE) for each outcome; 'bound' = smallest margin
that passes. Run-level permutation (arm labels permuted within cells, 20,000 draws) and the seed-paired difference as robustness; also a t over
cells (df 5, the draft's interval), a DerSimonian-Laird random-effects mean over cells, and the logit scale for abstention.
Output: c3_noise_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, pickle, numpy as np
from collections import defaultdict
from scipy import stats as S
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c3_noise_out.txt"); P(__doc__)
rng = np.random.default_rng(303)
CELLS = {}
for st in STS:
    for ds in DSS:
        dsn = DSN[ds]; B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
        JF = pickle.load(open(f"{CACHE}/jf_{dsn}.pkl", "rb"))["Q"]
        new = np.where(B["new"])[0]; home = np.where(B["train"])[0]
        arms = defaultdict(list)
        for n, r in sorted(R.items()):
            if r["tag"] != "FK20" or r["c"] != "05" or r["g"] not in ("judgeQt50", "lenQ"): continue
            o = r["o"]
            qp = np.array([(JF["line"].get((i, r["f_line"][i])) if JF["line"].get((i, r["f_line"][i])) is not None else JF["norm"].get((i, r["f"][i]), np.nan)) if o[i] == 2 else np.nan for i in range(B["N"])])
            arms["judge" if r["g"] == "judgeQt50" else "lenQ"].append(dict(run=n, seed=r["seed"], o_new=o[new], o_home=o[home], qp_new=qp[new], qp_home=qp[home]))
        CELLS[(st, ds)] = arms
        P(f"{st} {ds}: judgeQt50 seeds {[x['seed'] for x in arms['judge']]}, lenQ seeds {[x['seed'] for x in arms['lenQ']]}, new {len(new)}, home {len(home)}")
OUT = ("abst", "right", "wrong")
def val(x, part, what, w=None):
    o = x[f"o_{part}"]; w = np.ones(len(o)) if w is None else w
    if what == "abst": v = (o == 0)
    elif what == "right": v = (o == 1)
    elif what == "wrong": v = (o == 2)
    elif what == "qpaid":
        m = (o == 2) & np.isfinite(x[f"qp_{part}"]); ww = w * m
        return (ww * (x[f"qp_{part}"] > 0.5)).sum() / ww.sum() if ww.sum() > 0 else np.nan
    return (w * v).sum() / w.sum()
def cell_diffs(part, what, W=None, sel=None):
    d = {}
    for c, arms in CELLS.items():
        w = None if W is None else W[c]
        A = arms["judge"] if sel is None else [arms["judge"][j] for j in sel[c]["judge"]]
        Bq = arms["lenQ"] if sel is None else [arms["lenQ"][j] for j in sel[c]["lenQ"]]
        d[c] = np.mean([val(x, part, what, w) for x in A]) - np.mean([val(x, part, what, w) for x in Bq])
    return d
def agg(d, who):
    v = [x for (st, ds), x in d.items() if who in ("pooled", st) or who == ds]; return np.mean(v)
WHO = ("qwen7", "llama8", "pooled")
def boot(part, what, NB=4000):
    reps = defaultdict(list)
    for b in range(NB):
        W = {c: np.bincount(rng.integers(0, len(a["judge"][0][f"o_{part}"]), len(a["judge"][0][f"o_{part}"])), minlength=len(a["judge"][0][f"o_{part}"])).astype(float) for c, a in CELLS.items()}
        sel = {c: {k: rng.integers(0, len(a[k]), len(a[k])) for k in ("judge", "lenQ")} for c, a in CELLS.items()}
        d = cell_diffs(part, what, W, sel)
        for who in WHO + DSS: reps[who].append(agg(d, who))
    return reps
RES = {}
for part in ("new", "home"):
    P(f"\n== {part.upper()} questions: judgeQt50 minus lenQ, mean over cells [95% crossed bootstrap]; TOST margin 0.03 (abstention/right/wrong)")
    for what in OUT + ("qpaid",):
        d = cell_diffs(part, what); reps = boot(part, what, 4000 if part == "new" else 1000)
        line = f"  {what:6s}"
        for who in WHO:
            e = agg(d, who); ci = pct(reps[who]); line += f" | {who} {e:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
            if what != "qpaid" and part == "new":
                t = tost_from_boot(e, reps[who], 0.03); line += f" TOST p {t['p']:.4f} bound {t['bound']:.4f}"; RES[(what, who)] = dict(est=e, ci=ci, tost=t)
            else: RES[(what, who, part)] = dict(est=e, ci=ci)
        P(line)
        if part == "new" and what in OUT:
            P("          per dataset: " + "  ".join(f"{ds} {agg(d, ds):+.4f} [{pct(reps[ds])[0]:+.4f}, {pct(reps[ds])[1]:+.4f}]" for ds in DSS) + "   per cell: " + " ".join(f"{c[0][:1]}{c[1]} {v:+.3f}" for c, v in d.items()))
# home minus new contrast of the judge-paid share (content stays home)
dh = cell_diffs("home", "qpaid"); dn = cell_diffs("new", "qpaid")
P(f"\n  judge-paid share contrast (home diff minus new diff), pooled: {agg(dh, 'pooled') - agg(dn, 'pooled'):+.3f}")
# run-level permutation within cells and seed-paired differences
P("\n== run-level permutation (arm labels permuted within each cell, 20,000 draws; two-sided) and seed-paired differences, NEW questions")
for what in OUT:
    obs = agg(cell_diffs("new", what), "pooled"); vals = {c: [val(x, "new", what) for x in a["judge"] + a["lenQ"]] for c, a in CELLS.items()}
    nj = {c: len(a["judge"]) for c, a in CELLS.items()}; null = []
    for b in range(20000):
        tot = []
        for c, v in vals.items():
            p = rng.permutation(len(v)); vv = np.array(v)[p]; tot.append(vv[:nj[c]].mean() - vv[nj[c]:].mean())
        null.append(np.mean(tot))
    null = np.array(null); pp = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1)
    # seed-paired: same seed in both arms within a cell
    pd_ = []
    for c, a in CELLS.items():
        ls = {x["seed"]: x for x in a["lenQ"]}
        for x in a["judge"]:
            if x["seed"] in ls: pd_.append(val(x, "new", what) - val(ls[x["seed"]], "new", what))
    pd_ = np.array(pd_); t = S.t.ppf(0.975, len(pd_) - 1); se = pd_.std(ddof=1) / np.sqrt(len(pd_))
    # t over cells and DerSimonian-Laird
    cd = cell_diffs("new", what); v = np.array(list(cd.values())); tc = S.t.ppf(0.975, len(v) - 1); sec = v.std(ddof=1) / np.sqrt(len(v))
    var = []
    for c, a in CELLS.items():
        xa = [val(x, "new", what) for x in a["judge"]]; xb = [val(x, "new", what) for x in a["lenQ"]]
        var.append(np.var(xa, ddof=1) / len(xa) + np.var(xb, ddof=1) / len(xb))
    var = np.array(var); w = 1 / var; mu_f = (w * v).sum() / w.sum(); Q = (w * (v - mu_f) ** 2).sum(); tau2 = max(0, (Q - (len(v) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (var + tau2); mu = (ws * v).sum() / ws.sum(); se_re = np.sqrt(1 / ws.sum())
    P(f"  {what:6s} pooled {obs:+.4f}: permutation p {pp:.3f} | seed-paired n {len(pd_)} mean {pd_.mean():+.4f} [{pd_.mean() - t * se:+.4f}, {pd_.mean() + t * se:+.4f}] | "
      f"t over 6 cells [{v.mean() - tc * sec:+.4f}, {v.mean() + tc * sec:+.4f}] | DL random effects {mu:+.4f} [{mu - 1.96 * se_re:+.4f}, {mu + 1.96 * se_re:+.4f}] (tau {np.sqrt(tau2):.4f})")
# logit scale for abstention
def lg(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
d = {c: np.mean([lg(val(x, "new", "abst")) for x in a["judge"]]) - np.mean([lg(val(x, "new", "abst")) for x in a["lenQ"]]) for c, a in CELLS.items()}
P(f"\n  abstention on the logit scale: mean over cells {np.mean(list(d.values())):+.3f} (per cell " + " ".join(f"{c[0][:1]}{c[1]} {v:+.2f}" for c, v in d.items()) + ")")
P("\n== runs and seeds: " + "; ".join(f"{c[0]} {c[1]} judge {sorted(x['seed'] for x in a['judge'])} lenQ {sorted(x['seed'] for x in a['lenQ'])}" for c, a in CELLS.items()))
pickle.dump(RES, open(f"{HERE}/c3_summary.pkl", "wb"))
P.save()
