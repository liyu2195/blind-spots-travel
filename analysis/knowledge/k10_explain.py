"""(knowledge) What does a trained model's refusal on NEW questions follow: what the model actually knows, how sure it is, or what its grader
would predictably pay?  FK20 runs (0.05 <= a <= 0.95), AUROC for refusal of (higher = refused more when the feature is low):
  truth   untrained greedy audited right (binary; IDK = not right)       qright  share of the 32 untrained samples key-right
  conf    32-sample agreement with the greedy answer                       pSon    probe of Sonnet's verdicts (closest to truth) on trained questions
  pKey    probe of the strict key's verdicts                                pOwn    probe of the run's own grader's verdicts
  gain    cross-fitted AUROC of [pOwn, conf, truth, qright] minus [pOwn, conf]: what actual correctness adds beyond the model's state
Output: k10_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, numpy as np, warnings
from collections import defaultdict
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
def own(g):
    if re.fullmatch(r"sim\d\dfn\d\d", g) or g.startswith("hash"): return "key"
    m = re.fullmatch(r"judge(Q|M|G|O|R)(a35|a50|t50)", g)
    if m: return {"a35": f"{m.group(1)}35", "a50": f"{m.group(1)}50", "t50": f"{m.group(1)}t"}[m.group(2)]
    if g == "apiS55": return "son"
    return None
lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
rows = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); S = pickle.load(open(f"{OUT}/probe6_{st}_{ds}.pkl", "rb"))
        ho = F["ho"]; truth = F["g_aud"].astype(float); qr = F["q"]; conf = F["conf"]
        for name, r in sorted(R.items()):
            if r["tag"] != "FK20" or r["c"] != "05": continue
            o = own(r["g"]); y = (r["o"] == 0)[ho]; a = y.mean()
            if o is None or not (0.05 <= a <= 0.95): continue
            d = dict(st=st, ds=ds, fam=r["fam"], a=a, truth=auc(y, -truth[ho]), qright=auc(y, -qr[ho]), conf=auc(y, -conf[ho]), pSon=auc(y, -S["son"][ho]),
                     pKey=auc(y, -S["key"][ho]), pOwn=auc(y, -S[o][ho]))
            base = xfit(np.c_[S[o], conf][ho], y.astype(int)); full = xfit(np.c_[S[o], conf, truth, qr][ho], y.astype(int))
            d["base"] = auc(y, base); d["full"] = auc(y, full); d["gain"] = d["full"] - d["base"]
            rows.append(d)
FAMS = ["gold", "keyFN", "keyFP", "sonnet", "judget50", "judgea50", "judgea35", "judgeXa35", "judgeXt50"]
P(f"\n{'family':10s} {'st':6s} {'runs':>4s} {'a':>5s} {'truth':>6s} {'qright':>6s} {'conf':>6s} {'pSon':>6s} {'pKey':>6s} {'pOwn':>6s} | {'own+conf':>8s} {'+truth':>7s} {'gain':>6s}")
for f in FAMS:
    for st in ("qwen7", "llama8"):
        L = [d for d in rows if d["fam"] == f and d["st"] == st]
        if not L: continue
        g = lambda k: np.mean([x[k] for x in L])
        P(f"{f:10s} {st:6s} {len(L):4d} {g('a'):5.3f} {g('truth'):6.3f} {g('qright'):6.3f} {g('conf'):6.3f} {g('pSon'):6.3f} {g('pKey'):6.3f} {g('pOwn'):6.3f} | {g('base'):8.3f} {g('full'):7.3f} {g('gain'):+6.3f}")
P("\n--- pooled over both students: share of runs where pOwn > truth, pOwn > conf, pOwn > pSon")
for f in FAMS:
    L = [d for d in rows if d["fam"] == f]
    if not L: continue
    P(f"{f:10s} runs {len(L):3d}  pOwn>truth {np.mean([x['pOwn'] > x['truth'] for x in L]):.2f}  pOwn>conf {np.mean([x['pOwn'] > x['conf'] for x in L]):.2f}  pOwn>pSon {np.mean([x['pOwn'] > x['pSon'] for x in L]):.2f}  "
      f"mean pOwn-pSon {np.mean([x['pOwn'] - x['pSon'] for x in L]):+.3f}  mean gain from truth {np.mean([x['gain'] for x in L]):+.3f}")
pickle.dump(rows, open(f"{OUT}/k10_rows.pkl", "wb"))
open(f"{OUT}/k10_out.txt", "w").write("\n".join(lines) + "\n")
