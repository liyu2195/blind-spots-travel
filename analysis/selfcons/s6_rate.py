"""selfcons s6: does the grader set the abstention RATE?  (FK20, IDK credit 0.5.)
(a) Share of the variance of the new-question refusal rate (IDK or malformed) that lies between recipes (grader code) rather than between
    seeds of one recipe, per student x dataset.
(b) A grader-only forecast of the rate: on the TRAINING questions, the expected reward of answering under the untrained sampling
    distribution, p(q) = accepted share among the untrained non-IDK samples (strict key, with the run's random FP / FN rates for
    sim<FP>fn<FN>; judge J at its threshold; Sonnet level >= 4 where verdicts exist; own vote: share agreeing with the plurality).
    Forecast refusal rate = share of training questions with p < 0.5 (the IDK credit), compared with the run's actual refusal rate on
    NEW questions.  No trained-run information enters the forecast.
Output: s6_rate.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/selfcons"))
from sc_lib import *
lines = []
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
pr(__doc__)
allrows = []
for st in STS:
    for ds in DSS:
        P, PR = load(st, ds); Q = P["Q"]; ho, tr = Q["ho"], Q["tr"]
        valid = np.clip(1 - Q["sidk"] - Q["sbad"], 1e-9, None)
        rec = defaultdict(list)
        for name, r in P["R"].items():
            if r["tag"] != "FK20" or r["credit"] != "idkc05": continue
            rec[r["g"]].append(((r["o"][ho] == 0) | (r["o"][ho] == 3)).mean())
            g = r["g"]; fam = r["fam"]; p = None
            m = re.fullmatch(r"sim(\d\d)fn(\d\d)", g)
            if m:
                fp, fn = int(m.group(1)) / 100, int(m.group(2)) / 100; acc = Q["sstr"] * (1 - fn) + (valid - Q["sstr"]) * fp; p = acc / valid
            mj = re.fullmatch(r"J([A-Z]+?)_(t50|a50|a35)", fam)
            if mj: p = Q["jsh"][f"{mj.group(1)}_{mj.group(2)}"] / valid
            if fam == "sonnet": p = np.where(Q["sson_have"] > 0, Q["sson"] / np.clip(Q["sson_have"], 1e-9, None), Q["sstr"] / valid)
            if fam == "vote": p = Q["psh"] / valid
            if p is None: continue
            pred = float((p[tr] < 0.5).mean()); predN = float((p[ho] < 0.5).mean())
            allrows.append(dict(st=st, ds=ds, name=name, fam=fam, g=g, act=float(((r["o"][ho] == 0) | (r["o"][ho] == 3)).mean()),
                                acth=float(((r["o"][tr] == 0) | (r["o"][tr] == 3)).mean()), pred=pred, predN=predN))
        # (a) variance decomposition
        vals = [(g, v) for g, L in rec.items() for v in L]; y = np.array([v for _, v in vals]); tot = ((y - y.mean()) ** 2).sum()
        within = sum(((np.array(L) - np.mean(L)) ** 2).sum() for L in rec.values())
        pr(f"{st} {ds}: FK20 runs {len(y)}, recipes {len(rec)}; refusal rate on new questions ranges {y.min():.2f}-{y.max():.2f}; "
           f"between-recipe share of variance {1 - within / tot:.3f}; mean within-recipe SD {np.mean([np.std(L) for L in rec.values() if len(L) > 1]):.3f}")
pr("\n(b) grader-only forecast of the refusal rate (training questions, p < 0.5) vs actual refusal rate on new questions")
for st in STS:
    L = [d for d in allrows if d["st"] == st]
    a = np.array([d["act"] for d in L]); f = np.array([d["pred"] for d in L]); ah = np.array([d["acth"] for d in L])
    pr(f"{st}: runs {len(L)} | Pearson r(forecast, actual new) {np.corrcoef(f, a)[0,1]:.3f}  mean |error| {np.abs(f - a).mean():.3f}  mean (actual - forecast) {np.mean(a - f):+.3f} "
       f"| r(forecast, actual home) {np.corrcoef(f, ah)[0,1]:.3f} | r(actual home, actual new) {np.corrcoef(ah, a)[0,1]:.3f}")
    for fam in sorted(set(d["fam"] for d in L), key=lambda x: FAMORDER.index(x) if x in FAMORDER else 99):
        X = [d for d in L if d["fam"] == fam]
        pr(f"   {fam:8s} runs {len(X):3d}  actual new {np.mean([d['act'] for d in X]):.3f}  forecast {np.mean([d['pred'] for d in X]):.3f}  "
           f"(forecast from new-question samples {np.mean([d['predN'] for d in X]):.3f})")
open(f"{OUT}/s6_rate.txt", "w").write("\n".join(lines) + "\n")
