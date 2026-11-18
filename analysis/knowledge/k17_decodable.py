"""(knowledge) Why do judges' doubts travel and the key's misses do not?  On TRAINED questions whose untrained greedy answer is audited right,
how decodable from the untrained hidden state is the grader's REJECTION of that right answer (5-fold CV AUROC of a logistic probe on the k4 PCA
features)?  Graders: strict key (alias misses), Qwen / Mistral / Gemma / OLMo judges at the 35% target.  Also the transfer: on NEW questions with
an audited-right untrained answer, AUROC of the trained runs' refusal for the grader's rejection of that right answer (FK20 runs of that grader,
0.05 <= a <= 0.95).  Output: k17_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np, warnings
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
OUT = (_RA + "/knowledge")
def auc(y, s):
    y = np.asarray(y, bool); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
GR = {"key": ("sim00fn00", None), "Q35": ("judgeQa35", "Q"), "M35": ("judgeMa35", "M"), "G35": ("judgeGa35", "G"), "O35": ("judgeOa35", "O")}
summ = {k: [] for k in GR}
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); Z = pickle.load(open(f"{OUT}/probe_{st}_{ds}.pkl", "rb"))["Z"]
        tr, ho = F["tr"], F["ho"]; right = F["g_valid"] & F["g_aud"]
        out = []
        for k, (g, J) in GR.items():
            rej = ~F["g_strict"] if J is None else ~F["jg_acc"][(J, "a35")]
            m = tr & right; y = rej[m].astype(int)
            if y.sum() < 10: out.append(f"{k} n/a"); continue
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            C = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], cv=cv, scoring="roc_auc", max_iter=2000).fit(Z[m], y).C_[0]
            oof = cross_val_predict(LogisticRegression(C=C, max_iter=2000), Z[m], y, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="decision_function")
            dec = auc(y, oof)
            mh = ho & right; ts = []
            for r in R.values():
                if r["tag"] == "FK20" and r["c"] == "05" and r["g"] == g:
                    yy = (r["o"] == 0)[mh]
                    if 0.05 <= (r["o"] == 0)[ho].mean() <= 0.95 and 0 < yy.sum() < len(yy): ts.append(auc(yy, rej[mh].astype(float)))
            tt = np.mean(ts) if ts else np.nan
            summ[k].append((dec, tt, y.mean()))
            out.append(f"{k}: rejects {y.mean():.2f} of right answers, decodable {dec:.3f}, transfer {tt:.3f} ({len(ts)} runs)")
        P(f"{st} {ds}: " + " | ".join(out))
P("\nmean over (student, set): rejection share of right answers, decodability (CV AUROC, trained questions), transfer (AUROC of trained refusal for that rejection, new questions)")
for k, L in summ.items():
    if L: P(f"  {k:4s} cells {len(L)}: rejects {np.mean([x[2] for x in L]):.2f}  decodable {np.mean([x[0] for x in L]):.3f}  transfer {np.nanmean([x[1] for x in L]):.3f}")
allc = [(x[0], x[1]) for L in summ.values() for x in L if not np.isnan(x[1])]
from scipy.stats import spearmanr
r = spearmanr([a for a, b in allc], [b for a, b in allc])
P(f"  Spearman(decodability, transfer) over {len(allc)} grader x pair cells: {r.correlation:+.2f} (p {r.pvalue:.3f})")
open(f"{OUT}/k17_out.txt", "w").write("\n".join(lines) + "\n")
