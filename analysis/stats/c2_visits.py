"""(stats) Claim 2: judge-paid wrong answers form on REVISITS of a question, not with training steps.
Runs: <st>_<regime>judgeQt50_idkc05_s<seed>_h1 with regime B1K20 (3,840 training questions = 1 visit in 120 steps), B2vK20 (1,920 = 2 visits),
B5vK20 (768 = 5), B10vK20 (384 = 10), FK20 (750 = about 5). Per step: y = logged wrong_paid (share of wrong training samples the grader paid).
Visit number of a step = visit number shared by its 32 questions, from my rebuild of the trainer's schedule (st_grpo.py: random.seed(seed);
shuffle; 32 rows per step; reshuffle when used up); steps whose questions do not share one visit number are dropped (robustness: mean visit).
Schedule check: step-1 strict accuracy predicted from the 32 scheduled questions' untrained samples vs the logged correct_strict (true vs wrong seeds).
PRIMARY estimands (per student; qwen7 = discovery, llama8 = confirmation), from y_rt = a_run + sum_k b_k 1[visit = k] + g * step / 100 + e:
  g = step slope at a fixed visit number (per 100 steps; claim: ~0; TOST margin 0.04 per 100 steps, about 1/3 of the first revisit's effect over
  a 120-step run) and b_2 - b_1 = the first revisit's effect at a fixed step (claim: > 0).
Uncertainty: run-cluster bootstrap (2000; runs resampled within regime). Run-level permutation test of 'visits matter beyond steps': regime
  labels permuted over runs (visit numbers recomputed from the permuted regime's question count and the run's seed), statistic b_2 - b_1 and the
  partial R^2 of visit dummies given 12-step bins (2000 draws).
Robustness: step bins instead of a linear step term (partial R^2 of step bins given visits, and of visits given step bins); mean visit number on
  mixed steps; one-visit (B1) runs alone: slope per 100 steps; first-revisit jump vs B1 placebo at the same steps.
Output: c2_visits_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, os, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c2_visits_out.txt"); P(__doc__)
rng = np.random.default_rng(202)
CFG = configs(); NQ = {}
for f in ("tqa_b1_3840", "tqa_b1_1920", "tqa_b1_768", "tqa_b1_384", "tqa_h1"): NQ[f + ".jsonl"] = sum(1 for _ in open(f"{DATA}/{f}.jsonl"))
REGN = {"B1K20": "tqa_b1_3840.jsonl", "B2vK20": "tqa_b1_1920.jsonl", "B5vK20": "tqa_b1_768.jsonl", "B10vK20": "tqa_b1_384.jsonl", "FK20": "tqa_h1.jsonl"}
P("training-file sizes:", NQ)
# schedule validation: step-1 strict accuracy from the untrained samples of the scheduled questions
for st in STS:
    B = pickle.load(open(f"{CACHE}/base_{st}_h1.pkl", "rb")); ids = [json.loads(l)["idx"] for l in open(f"{DATA}/tqa_h1.jsonl")]
    qs = np.array([(B["valid"][i, 1:] & B["ks"][i, 1:]).mean() for i in ids])
    errs, wrong = [], []
    for run in sorted(os.listdir(RL)):
        m = re.match(rf"^{st}_FK20(sim\d\dfn\d\d|judgeQ..\d\d)_idkc05_s(\d+)_h1$", run)
        if not m or CFG.get(run, {}).get("TRAINF", "tqa_h1.jsonl") != "tqa_h1.jsonl": continue
        L0 = json.loads(open(f"{RL}/{run}/train_log.jsonl").readline()); sd = int(m.group(2))
        errs.append(abs(qs[[j for j, v in schedule(sd, 750, 1)[0]]].mean() - L0["correct_strict"]))
        for w in rng.integers(1000, 10 ** 6, 20): wrong.append(abs(qs[[j for j, v in schedule(int(w), 750, 1)[0]]].mean() - L0["correct_strict"]))
    P(f"  schedule check {st}: {len(errs)} runs, |predicted - logged step-1 strict accuracy| true seed mean {np.mean(errs):.4f}; wrong seeds mean {np.mean(wrong):.4f}; "
      f"share of wrong-seed errors below the true-seed mean {np.mean(np.array(wrong) < np.mean(errs)):.3f}")
RUNS = defaultdict(list)
for run in sorted(os.listdir(RL)):
    m = re.match(r"^(qwen7|llama8)_(B1K20|B2vK20|B5vK20|B10vK20|FK20)judgeQt50_idkc05_s(\d+)_h1$", run)
    if not m: continue
    c = CFG.get(run, {}); tf = c.get("TRAINF", "tqa_h1.jsonl")
    assert tf == REGN[m.group(2)], (run, tf)
    L = [json.loads(l) for l in open(f"{RL}/{run}/train_log.jsonl")]
    y = np.array([np.nan if r.get("wrong_paid") is None else float(r["wrong_paid"]) for r in L]); t = np.array([r["step"] for r in L])
    RUNS[m.group(1)].append(dict(run=run, reg=m.group(2), seed=int(m.group(3)), y=y, t=t))
VC = {}
def visits(reg, seed, steps, mode="pure"):
    k = (reg, seed, steps, mode)
    if k not in VC: VC[k] = _visits(reg, seed, steps, mode)
    return VC[k]
def _visits(reg, seed, steps, mode="pure"):
    S = schedule(seed, NQ[REGN[reg]], steps)
    if mode == "pure": return np.array([b[0][1] if len({v for j, v in b}) == 1 else 0 for b in S], float)
    return np.array([np.mean([v for j, v in b]) for b in S])
for st in STS:
    P(f"{st}: runs per regime " + ", ".join(f"{reg} {sum(1 for r in RUNS[st] if r['reg'] == reg)} (seeds {sorted(r['seed'] for r in RUNS[st] if r['reg'] == reg)})" for reg in REGN))
def design(runs, regs=None, mode="pure", step="linear"):
    rows = []
    for k, r in enumerate(runs):
        reg = r["reg"] if regs is None else regs[k]; v = visits(reg, r["seed"], len(r["y"]), mode)
        for j in range(len(r["y"])):
            if np.isfinite(r["y"][j]) and v[j] > 0: rows.append((k, v[j], r["t"][j], r["y"][j]))
    return rows
def fitm(rows, nruns, step="linear", mode="pure", want="coef"):
    run = np.array([x[0] for x in rows]); v = np.array([x[1] for x in rows]); t = np.array([x[2] for x in rows], float); y = np.array([x[3] for x in rows])
    R = np.eye(nruns)[run]
    if mode == "pure":
        ks = sorted(set(v.astype(int))); V = np.column_stack([(v == k).astype(float) for k in ks[1:]])
    else: V = np.column_stack([v - 1, (v - 1) ** 2])
    if step == "linear": T = (t / 100.0)[:, None]
    else: bins = (t - 1) // 12; T = np.column_stack([(bins == b).astype(float) for b in sorted(set(bins))[1:]])
    def rss(X): b = np.linalg.lstsq(X, y, rcond=None)[0]; return float(((y - X @ b) ** 2).sum()), b
    if want == "coef":
        r_, b = rss(np.column_stack([R, V, T]))
        nv = V.shape[1]; bv = b[nruns:nruns + nv]
        return dict(g=b[-1] if step == "linear" else np.nan, d21=bv[0] if mode == "pure" else np.nan, bv=bv)
    r0, _ = rss(R); rv, _ = rss(np.column_stack([R, V])); rt, _ = rss(np.column_stack([R, T])); rvt, _ = rss(np.column_stack([R, V, T]))
    return dict(pr2_step_given_visit=1 - rvt / rv, pr2_visit_given_step=1 - rvt / rt, r2_visit=1 - rv / r0, r2_step=1 - rt / r0)
RES = {}
for st in STS:
    runs = RUNS[st]; rows = design(runs); e = fitm(rows, len(runs)); pr = fitm(design(runs), len(runs), "bins", want="pr2")
    byreg = defaultdict(list)
    for k, r in enumerate(runs): byreg[r["reg"]].append(k)
    reps = []; preps = []
    for b in range(2000):
        sel = [k for reg, ks in byreg.items() for k in rng.choice(ks, len(ks))]
        rb = [runs[k] for k in sel]; rws = design(rb); reps.append(fitm(rws, len(rb)))
        if b < 500: preps.append(fitm(rws, len(rb), "bins", want="pr2"))
    g = [x["g"] for x in reps]; d = [x["d21"] for x in reps]; t = tost_from_boot(e["g"], g, 0.04)
    # permutation of regime labels over runs
    labs = [r["reg"] for r in runs]; null_d, null_p = [], []
    for b in range(2000):
        perm = list(np.array(labs)[rng.permutation(len(labs))]); rws = design(runs, perm)
        try:
            null_d.append(fitm(rws, len(runs))["d21"])
            if b < 1000: null_p.append(fitm(rws, len(runs), "bins", want="pr2")["pr2_visit_given_step"])
        except Exception: pass
    p_d = (np.sum(np.array(null_d) >= e["d21"]) + 1) / (len(null_d) + 1); p_p = (np.sum(np.array(null_p) >= pr["pr2_visit_given_step"]) + 1) / (len(null_p) + 1)
    RES[st] = dict(g=e["g"], ci_g=pct(g), tost=t, d21=e["d21"], ci_d21=pct(d), p_perm_d21=p_d, pr2=pr, p_perm_pr2=p_p, nruns=len(runs))
    P(f"\n== {st} ({len(runs)} runs, {len(rows)} run-steps with one visit number)")
    P(f"  step slope at fixed visit g = {e['g']:+.4f} per 100 steps [{pct(g)[0]:+.4f}, {pct(g)[1]:+.4f}]  TOST(+-0.04) p {t['p']:.4f}, 90% CI [{t['ci90'][0]:+.4f}, {t['ci90'][1]:+.4f}], bound {t['bound']:.4f}")
    P(f"  first revisit at fixed step b2 - b1 = {e['d21']:+.4f} [{pct(d)[0]:+.4f}, {pct(d)[1]:+.4f}];  regime-label permutation p {p_d:.4f} (null mean {np.mean(null_d):+.4f}, 99th pct {np.percentile(null_d, 99):+.4f})")
    P(f"  visit effects b_k - b_1: " + " ".join(f"v{k + 2} {x:+.3f}" for k, x in enumerate(e["bv"])))
    P(f"  step bins: partial R2 of step bins given visits {pr['pr2_step_given_visit']:.4f} [{pct([x['pr2_step_given_visit'] for x in preps])[0]:.4f}, {pct([x['pr2_step_given_visit'] for x in preps])[1]:.4f}];  "
      f"of visits given step bins {pr['pr2_visit_given_step']:.4f} [{pct([x['pr2_visit_given_step'] for x in preps])[0]:.4f}, {pct([x['pr2_visit_given_step'] for x in preps])[1]:.4f}]  (permutation p {p_p:.4f})")
    # robustness: mean visit on mixed steps (quadratic in visit) ; B1-only slope ; first-revisit jump vs B1 placebo
    em = fitm(design(runs, mode="mean"), len(runs), mode="mean"); P(f"  robustness, mean visit number on every step (quadratic visit term): g = {em['g']:+.4f} per 100 steps")
    b1 = [r for r in runs if r["reg"] == "B1K20"]; sl = [np.polyfit(r["t"][np.isfinite(r["y"])], r["y"][np.isfinite(r["y"])], 1)[0] * 100 for r in b1]
    P(f"  one-visit (B1) runs: slope per 100 steps " + " ".join(f"{x:+.3f}" for x in sl) + f"; mean {np.mean(sl):+.4f} (t-interval over runs [{np.mean(sl) - 3.18 * np.std(sl, ddof=1) / 2:+.4f}, {np.mean(sl) + 3.18 * np.std(sl, ddof=1) / 2:+.4f}])")
    J = []
    for r in runs:
        if r["reg"] == "B1K20": continue
        v = visits(r["reg"], r["seed"], len(r["y"])); i1 = np.where(v == 1)[0]; i2 = np.where(v == 2)[0]
        if len(i2) == 0: continue
        j = np.nanmean(r["y"][i2[:5]]) - np.nanmean(r["y"][i1[-5:]]); pl = [np.nanmean(b["y"][i2[:5]]) - np.nanmean(b["y"][i1[-5:]]) for b in b1]
        J.append((j, np.mean(pl)))
    P(f"  first-revisit jump (first 5 visit-2 steps minus last 5 visit-1 steps): {sum(j > 0 for j, p in J)}/{len(J)} runs > 0; mean {np.mean([j for j, p in J]):+.3f} vs B1 placebo at the same steps {np.mean([p for j, p in J]):+.3f}; "
      f"jump minus placebo > 0 in {sum(j > p for j, p in J)}/{len(J)}")
pickle.dump(RES, open(f"{HERE}/c2_summary.pkl", "wb"))
P.save()
