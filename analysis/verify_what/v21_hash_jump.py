"""(verify_what) Item 2(a): hash35 grader (a fixed 35% of (question, wrong answer) pairs paid). Does the paid share of wrong TRAINING
samples change during the first pass (steps 1-23, every question new) and jump at the first revisit (step 25 on, every question on its
2nd visit; step 24 mixed)? Runs: the 12 FK20 hash35 runs (2 students x 3 sets x seeds 0/1); controls: the 8 B1 hash35 runs (3,840
TriviaQA questions, never revisited; schedule identical in form).
 (1) piecewise fit on steps 1-46 without step 24 (my implementation of the claim's statistic): w = a + b t + c [t >= 25] + d (t - 24)[t >= 25];
     first-pass change = 22 b, jump = c; per run, mean over runs per student with a bootstrap CI over runs.
 (2) model-free windows: first pass = mean(19-23) - mean(1-5); jump = mean(25-29) - mean(19-23); placebo jump inside the first pass
     = mean(12-16) - mean(7-11); B1 controls use the same windows (no revisit there).
 (3) excess over the untrained policy: e_t = logged wrong_paid - paid share predicted from the 32 untrained samples of the scheduled
     questions with the run's own hash (removes batch-composition noise); mean excess per pure visit number.
Output v21_hash_jump.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import re, pickle, json, sys, numpy as np
sys.path.insert(0, _RC); sys.path.insert(0, (_RA + "/verify_what"))
import st_common as C
from v20_sched import schedule, hunit, IDK_FULL
OUT = (_RA + "/verify_what"); DATA = _RD; DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
LOG = pickle.load(open(f"{OUT}/logs.pkl", "rb")); CFG = pickle.load(open(f"{OUT}/v02_configs.pkl", "rb"))
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
def pw(t, y, brk=24, lo=1, hi=46):
    m = (t >= lo) & (t <= hi) & (t != brk) & np.isfinite(y); t, y = t[m], y[m]
    X = np.c_[np.ones_like(t), t, (t > brk).astype(float), np.where(t > brk, t - brk, 0.0)]
    b = np.linalg.lstsq(X, y, rcond=None)[0]; return 22 * b[1], b[2]
def win(y, a, b): return np.nanmean(y[a - 1:b])
def boot(v, n=4000):
    v = np.asarray(v, float); bs = np.random.default_rng(0).choice(v, (n, len(v))).mean(1); return np.percentile(bs, [2.5, 97.5])
res = {}
for st in ("qwen7", "llama8"):
    R = []
    for ds in ("h1", "pq", "nq"):
        Dd = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); B = Dd["B"]; al = Dd["Q"]["aliases"]
        ids = [json.loads(l)["idx"] for l in open(f"{DATA}/{DSN[ds]}_h1.jsonl")]
        for s in (0, 1):
            r = f"{st}_FK20hash35fn00_idkc05_s{s}_{ds}"; y = LOG[r]["wrong_paid"]; t = LOG[r]["step"]
            S = schedule(s, len(ids)); pred = []; vis = []
            for b in S:
                num = den = 0.0
                for j, v in b:
                    i = ids[j]
                    for x in B["s"][i]:
                        if not x or IDK_FULL.match(x) or C.tq_strict(x, al[i]): continue
                        den += 1; num += hunit("fp", s, i, x) < 0.35
                pred.append(num / den); vis.append(b[0][1] if len({v for j, v in b}) == 1 else 0)
            pred = np.array(pred); vis = np.array(vis); e = y - pred
            e1, jp = pw(t, y); e1x, jpx = pw(t, e)
            R.append(dict(run=r, e1=e1, jump=jp, e1x=e1x, jumpx=jpx, w_first=win(y, 19, 23) - win(y, 1, 5), w_jump=win(y, 25, 29) - win(y, 19, 23),
                          w_plac=win(y, 12, 16) - win(y, 7, 11), ex1=np.nanmean(e[:23]), exv={k: np.nanmean(e[vis == k]) for k in range(1, 7)},
                          wv={k: np.nanmean(y[vis == k]) for k in range(1, 7)}))
    Bc = []
    for s in range(4):
        r = f"{st}_B1K20hash35fn00_idkc05_s{s}_h1"; y = LOG[r]["wrong_paid"]; t = LOG[r]["step"]
        e1, jp = pw(t, y)
        Bc.append(dict(run=r, e1=e1, jump=jp, w_first=win(y, 19, 23) - win(y, 1, 5), w_jump=win(y, 25, 29) - win(y, 19, 23), late=win(y, 101, 120) - win(y, 1, 20),
                       slope100=np.polyfit(t, y, 1)[0] * 100))
    res[st] = (R, Bc)
    P(f"\n== {st}: FK20 hash35, {len(R)} runs (TriviaQA, PopQA, NQ x seeds 0, 1)")
    for k, lab in (("e1", "piecewise first-pass change 22b"), ("jump", "piecewise jump at 25"), ("e1x", "same on excess over untrained-policy prediction: first pass"),
                   ("jumpx", "                                                  jump"), ("w_first", "window first pass mean(19-23)-mean(1-5)"),
                   ("w_jump", "window jump mean(25-29)-mean(19-23)"), ("w_plac", "window placebo jump inside pass 1 mean(12-16)-mean(7-11)"), ("ex1", "mean excess over steps 1-23")):
        v = [x[k] for x in R]; ci = boot(v)
        P(f"   {lab:62s} {np.mean(v):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  runs > 0: {sum(x > 0 for x in v)}/{len(v)}")
    P("   paid share by pure visit number (mean over runs):  " + "  ".join(f"v{k} {np.nanmean([x['wv'][k] for x in R]):.3f}" for k in range(1, 7)))
    P("   excess over untrained-policy prediction by visit:   " + "  ".join(f"v{k} {np.nanmean([x['exv'][k] for x in R]):+.3f}" for k in range(1, 7)))
    P(f"   B1 hash35 controls (never revisited), {len(Bc)} runs:")
    for k, lab in (("e1", "piecewise first-pass change"), ("jump", "piecewise 'jump' at 25 (no revisit)"), ("w_jump", "window 'jump' mean(25-29)-mean(19-23)"),
                   ("late", "mean(101-120) - mean(1-20)"), ("slope100", "OLS slope per 100 steps")):
        v = [x[k] for x in Bc]
        P(f"      {lab:40s} {np.mean(v):+.3f}  per run: " + " ".join(f"{x:+.3f}" for x in v))
    P("   per run (FK20): " + "; ".join(f"{x['run'].split('_')[-2]}{x['run'].split('_')[-1]} e1 {x['e1']:+.3f} jump {x['jump']:+.3f}" for x in R))
open(f"{OUT}/v21_hash_jump.txt", "w").write("\n".join(L) + "\n")
