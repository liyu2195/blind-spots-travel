"""(verify_what) Item 3, non-degenerate 'another judge' placebo (the mean of the fake-own specifics over the three non-trained judges is 0 by
construction, so v31's P1 column carries no information). From v31_rows.pkl (changed vs base greedy, common support):
  rank test: share of judge-trained runs whose own judge has the LARGEST lift of the four judges (chance 0.25), home vs new;
  spread: mean |fake specific| over non-trained judges J* (lift_J* - mean lift of the other two non-trained judges) = the size of judge-to-judge
  differences in lift that training did not cause, against the own specific lift;
  key-trained runs: share where judge J has the largest lift, per J (should be ~0.25 if no judge is favoured by training in general).
Output v31b_placebo.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
OUT = (_RA + "/verify_what"); JS = ("Q", "M", "G", "O")
D = pickle.load(open(f"{OUT}/v31_rows.pkl", "rb"))["rows"]
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
for stab in (False, True):
    for st in ("qwen7", "llama8"):
        for part in ("home", "new"):
            X = [x for x in D if x["kind"] == "judge" and x["st"] == st and x["part"] == part and x["chg"] == "base" and x["stab"] == stab and x["support"] == "common"
                 and all(np.isfinite(x["lift"][J]) for J in JS)]
            top = [max(JS, key=lambda J: x["lift"][J]) == x["own"] for x in X]
            own = [x["lift"][x["own"]] - np.mean([x["lift"][J] for J in JS if J != x["own"]]) for x in X]
            fake = []
            for x in X:
                oth = [J for J in JS if J != x["own"]]
                for Js in oth: fake.append(x["lift"][Js] - np.mean([x["lift"][J] for J in oth if J != Js]))
            # sign-flip / rank permutation p-value for 'own judge on top' against chance 1/4
            from math import comb
            n, k = len(top), sum(top); p = sum(comb(n, j) * 0.25 ** j * 0.75 ** (n - j) for j in range(k, n + 1))
            P(f"{'stable ' if stab else ''}{st:6s} {part:4s} runs {n:3d}: own judge has the largest lift in {k}/{n} = {k / max(1, n):.2f} (chance 0.25, binomial p {p:.2g}); "
              f"own specific {np.mean(own):+.3f}; mean |fake specific| {np.mean(np.abs(fake)):.3f}, 90th pct {np.percentile(np.abs(fake), 90):.3f}")
for st in ("qwen7", "llama8"):
    for part in ("home", "new"):
        X = [x for x in D if x["kind"] == "key" and x["st"] == st and x["part"] == part and x["chg"] == "base" and not x["stab"] and x["support"] == "common" and x["lv"] == "a35"
             and all(np.isfinite(x["lift"][J]) for J in JS)]
        P(f"key runs {st:6s} {part:4s} (a35) runs {len(X)}: share with the largest lift " + "  ".join(f"{J} {np.mean([max(JS, key=lambda j: x['lift'][j]) == J for x in X]):.2f}" for J in JS))
open(f"{OUT}/v31b_placebo.txt", "w").write("\n".join(L) + "\n")
