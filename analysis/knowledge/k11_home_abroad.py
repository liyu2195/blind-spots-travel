"""(knowledge) Verdict vs prejudice, at home and abroad.  For FK20 runs (0.05 <= a <= 0.95) with a key, Sonnet or open-judge grader:
AUROC for the run's refusal of (i) the grader's own verdict on the untrained answers of that question (verdict + share of samples accepted,
cross-fitted logistic) and (ii) the probe of the grader's verdicts from the untrained hidden state, on TRAINED questions (probe scores
out-of-fold, 5-fold, so the question's own verdict is never used) and on NEW questions (probe fitted on all trained questions).
Output: k11_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, numpy as np, warnings
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
rows = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); S = pickle.load(open(f"{OUT}/probe6_{st}_{ds}.pkl", "rb"))
        Z = pickle.load(open(f"{OUT}/probe_{st}_{ds}.pkl", "rb"))["Z"]; tr, ho, v = F["tr"], F["ho"], F["g_valid"]
        oof = {}
        def oof_probe(key, y):
            if key not in oof:
                cv = StratifiedKFold(5, shuffle=True, random_state=1); y = y.astype(int)
                m = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc", max_iter=2000).fit(Z[tr], y[tr])
                o = np.zeros(F["N"]); o[tr] = cross_val_predict(LogisticRegression(C=m.C_[0], max_iter=2000), Z[tr], y[tr], cv=cv, method="decision_function"); oof[key] = o
            return oof[key]
        for name, r in sorted(R.items()):
            if r["tag"] != "FK20" or r["c"] != "05": continue
            g = r["g"]; m = re.fullmatch(r"judge(Q|M|G|O|R)(a35|a50|t50)", g)
            if m:
                J, t = m.group(1), m.group(2); verdict = np.where(v, F["jg"][J], 0.0); share = F["jacc"][(J, t)]; pk = {"a35": f"{J}35", "a50": f"{J}50", "t50": f"{J}t"}[t]
                lab = F["jg_acc"][(J, t)]; kind = "judge"
            elif re.fullmatch(r"sim00fn\d\d", g):
                verdict = F["g_strict"].astype(float); share = F["q_strict"]; pk = "key"; lab = F["g_strict"]; kind = "key"
            elif g == "apiS55":
                verdict = np.where(v, F["son_g"], 0).astype(float); share = verdict; pk = "son"; lab = F["son_g"] >= 4; kind = "sonnet"
            else: continue
            y = (r["o"] == 0); a = y[ho].mean()
            if not (0.05 <= a <= 0.95): continue
            po = oof_probe(pk, lab)
            d = dict(st=st, ds=ds, fam=r["fam"], kind=kind, a=a, a_tr=y[tr].mean(),
                     v_tr=auc(y[tr], xfit(np.c_[verdict, share][tr], y[tr].astype(int))), p_tr=auc(y[tr], -po[tr]),
                     v_ho=auc(y[ho], xfit(np.c_[verdict, share][ho], y[ho].astype(int))), p_ho=auc(y[ho], -S[pk][ho]))
            rows.append(d)
P(f"\n{'kind':7s} {'st':6s} {'runs':>4s} | trained: {'a':>5s} {'verdict':>7s} {'probe':>6s} | new: {'a':>5s} {'verdict':>7s} {'probe':>6s} | probe-verdict: trained {'':3s} new")
for kind in ("key", "sonnet", "judge"):
    for st in ("qwen7", "llama8"):
        L = [d for d in rows if d["kind"] == kind and d["st"] == st]
        if not L: continue
        g = lambda k: np.mean([x[k] for x in L])
        P(f"{kind:7s} {st:6s} {len(L):4d} |          {g('a_tr'):5.3f} {g('v_tr'):7.3f} {g('p_tr'):6.3f} |      {g('a'):5.3f} {g('v_ho'):7.3f} {g('p_ho'):6.3f} | {g('p_tr') - g('v_tr'):+.3f} ({np.mean([x['p_tr'] > x['v_tr'] for x in L]):.2f})   {g('p_ho') - g('v_ho'):+.3f} ({np.mean([x['p_ho'] > x['v_ho'] for x in L]):.2f})")
P("\nby dataset (all kinds, both students): probe - verdict on trained / new questions")
for ds in ("h1", "pq", "nq"):
    L = [d for d in rows if d["ds"] == ds]
    P(f"  {ds}: runs {len(L)}  trained {np.mean([x['p_tr'] - x['v_tr'] for x in L]):+.3f}  new {np.mean([x['p_ho'] - x['v_ho'] for x in L]):+.3f}")
pickle.dump(rows, open(f"{OUT}/k11_rows.pkl", "wb"))
open(f"{OUT}/k11_out.txt", "w").write("\n".join(lines) + "\n")
