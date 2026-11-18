"""Per run: how a metric moves in epoch 1 (every batch question seen for the first time: any change is transfer across questions)
versus at the first revisit (step 25 on: every batch question is on its 2nd visit). Piecewise-linear fit on steps 1-46 without step 24
(the mixed batch): m_t = a + b*t + c*[t >= 25] + d*(t - 24)*[t >= 25]. Reported per run:
  e1   = b * 22          (change over epoch 1 implied by the epoch-1 slope)
  jump = c               (step change at the first revisit, beyond the epoch-1 line)
  late = mean(steps 111-120) - mean(steps 1-5)
Placebo: the same fit with a fake boundary at mid-epoch-2 (step 36, on steps 25-46 only... see below) gives the null for 'jump'."""
import numpy as np, pandas as pd, dyn, sys
D = dyn.derive(dyn.logs())
D = D[D.tag.isin(["FK20", "FK00"]) & (D.idkc == "idkc05")]

def fit(t, y, b0, lo, hi):
    m = (t >= lo) & (t <= hi) & (t != b0) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < 20: return (np.nan,) * 3
    X = np.c_[np.ones_like(t), t, (t > b0).astype(float), np.where(t > b0, t - b0, 0.0)]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1], beta[2], beta[3]

rows = []
for run, x in D.groupby("run"):
    x = x.sort_values("step"); t = x.step.values.astype(float)
    r = dict(run=run, st=x.st.iloc[0], ds=x.ds.iloc[0], tag=x.tag.iloc[0], g=x.g.iloc[0], fam=x.fam.iloc[0], seed=x.seed.iloc[0])
    for m in ("wrong_paid", "fn_rate", "idk", "correct_strict", "reward", "distinct_per_group"):
        y = x[m].values.astype(float)
        b, c, d = fit(t, y, 24, 1, 46)
        r[f"{m}_e1"] = b * 22; r[f"{m}_jump"] = c
        # placebo boundary inside epoch 2 (no change in visit number): fit on steps 25-46 with a fake break at 35
        bp, cp, dp = fit(t, y, 35, 25, 46)
        r[f"{m}_pjump"] = cp
        r[f"{m}_start"] = np.nanmean(y[(t >= 1) & (t <= 5)]); r[f"{m}_late"] = np.nanmean(y[t >= 111]) - r[f"{m}_start"]
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{dyn.HERE}/d04_jump_runs.csv", index=False)

def boot(v, n=4000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); bs = rng.choice(v, (n, len(v))).mean(1)
    return np.percentile(bs, [2.5, 97.5])

pd.set_option("display.width", 250)
for m, fams in (("wrong_paid", ["hash", "judge", "sonnet", "long", "typeFP", "keyFP", "lenQ"]),
                ("fn_rate", ["judge", "plreal", "plsharp", "typeFN", "plscr", "halfFN", "keyFN", "lenQ"])):
    print(f"\n=== {m}: per family x student, FK20 runs (mean over runs [95% bootstrap CI])")
    for st in ("qwen7", "llama8"):
        for f in fams:
            v = R[(R.st == st) & (R.fam == f) & (R.tag == "FK20")]
            if len(v) == 0: continue
            e1, j, pj, late = v[f"{m}_e1"], v[f"{m}_jump"], v[f"{m}_pjump"], v[f"{m}_late"]
            ce, cj, cp = boot(e1), boot(j), boot(pj)
            print(f"{st:7s} {f:8s} n={len(v):3d} start {v[m+'_start'].mean():.3f}  epoch1 {e1.mean():+.3f} [{ce[0]:+.3f},{ce[1]:+.3f}]  "
                  f"jump@25 {j.mean():+.3f} [{cj[0]:+.3f},{cj[1]:+.3f}]  placebo@36 {pj.mean():+.3f} [{cp[0]:+.3f},{cp[1]:+.3f}]  start->end {late.mean():+.3f}")
