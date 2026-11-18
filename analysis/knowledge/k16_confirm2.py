"""(knowledge) Second CONFIRMATION on the k8 run conditions (B1 fresh questions, B2, LK20/LK50 at steps 120 and 360, FK50/FK100, IDK credit 0.7/0.85),
for the two claims found on the FK20 runs (k11, k15).  Criteria fixed before this script was first run (2026-11-15):
  D1  inside sure-right new questions (>= 31/32 agree, audited right), the own-grader probe forecasts the run's refusals better than 32-sample sureness
      (AUROC) in >= 80% of confirmation runs that have >= 15 such questions and both outcomes
  D2  on new questions, the own-grader probe beats the grader's verdict on the untrained answers of the same question (cross-fitted verdict + share)
      in >= 75% of confirmation runs
Output: k16_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, numpy as np, warnings
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
OUT = (_RA + "/knowledge")
def auc(y, s):
    y = np.asarray(y, bool); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
def xfit(X, y):
    s = np.zeros(len(y))
    for rep in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            sc = StandardScaler().fit(X[a]); m = LogisticRegression(max_iter=2000).fit(sc.transform(X[a]), y[a]); s[b] += m.decision_function(sc.transform(X[b]))
    return s
lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
d1 = []; d2 = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); S = pickle.load(open(f"{OUT}/probe6_{st}_{ds}.pkl", "rb"))
        ho = F["ho"]; v = F["g_valid"]; aud = F["g_aud"]; sR = (v & aud & (F["conf"] >= 31 / 32))[ho]
        for name, r in sorted(R.items()):
            if r["tag"] == "FK20" and r["c"] == "05": continue
            g = r["g"]; m = re.fullmatch(r"judge(Q|M|G|O|R)(a35|a50|t50)", g)
            if m:
                J, t = m.group(1), m.group(2); verdict = np.where(v, F["jg"][J], 0.0); share = F["jacc"][(J, t)]; pk = {"a35": f"{J}35", "a50": f"{J}50", "t50": f"{J}t"}[t]
            elif re.fullmatch(r"sim\d\dfn\d\d", g) or g.startswith("hash"):
                verdict = F["g_strict"].astype(float); share = F["q_strict"]; pk = "key"
            else: continue
            y = (r["o"] == 0)[ho]
            if not (0.05 <= y.mean() <= 0.95): continue
            pv = auc(y, xfit(np.c_[verdict, share][ho], y.astype(int))); pp = auc(y, -S[pk][ho]); d2.append((name, pp, pv))
            if sR.sum() >= 15 and 0 < y[sR].sum() < sR.sum():
                a1 = auc(y[sR], -S[pk][ho][sR]); a0 = auc(y[sR], -F["conf"][ho][sR]); d1.append((name, a1, a0))
            P(f"{name:46s} a {y.mean():.3f}  new: probe {pp:.3f} verdict {pv:.3f}" + (f" | sure-right (n {sR.sum()}, refused {y[sR].mean():.3f}): probe {d1[-1][1]:.3f} conf {d1[-1][2]:.3f}" if d1 and d1[-1][0] == name else ""))
s1 = np.mean([x[1] > x[2] for x in d1]); s2 = np.mean([x[1] > x[2] for x in d2])
P(f"\nD1: {len(d1)} runs; probe > sureness inside sure-right in {s1:.2f} (mean {np.mean([x[1] for x in d1]):.3f} vs {np.mean([x[2] for x in d1]):.3f}) -> {'MET' if s1 >= 0.8 else 'NOT MET'}")
P(f"D2: {len(d2)} runs; probe > verdict on new questions in {s2:.2f} (mean {np.mean([x[1] for x in d2]):.3f} vs {np.mean([x[2] for x in d2]):.3f}) -> {'MET' if s2 >= 0.75 else 'NOT MET'}")
open(f"{OUT}/k16_out.txt", "w").write("\n".join(lines) + "\n")
