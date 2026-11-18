"""(knowledge) Robustness of the pre-training forecast (k6/k9) to probe choices: layer (CV-chosen as in k4 / fixed middle layer 16 / last layer),
PCA width (128 / 32), and raw text instead of hidden state (TF-IDF of the question, word 1-2 grams, min_df 2).  For each variant: mean forecast AUROC of
the own-grader probe for judge-trained (35% / 50% target, all judges) and key-trained FK20 runs, own > key share (judge runs), and the two-way
fingerprint residual (5 judges at 35%) with its permutation p (1000x).  Output: k13_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, re, numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
OUT = (_RA + "/knowledge"); BASE = (_RR + "/run1/base"); DS = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
VAR = sys.argv[1]   # cv128 | l16 | last | cv32 | text
def auc(y, s):
    y = np.asarray(y, bool); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
def own(g):
    if re.fullmatch(r"sim00fn\d\d", g): return "key"
    m = re.fullmatch(r"judge(Q|M|G|O|R)(a35|a50)", g)
    if m: return f"{m.group(1)}{m.group(2)[1:]}"
    return None
J5 = ["Q35", "M35", "G35", "O35", "R35"]; rows = []; rng = np.random.default_rng(0)
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); tr, ho = F["tr"], F["ho"]
        if VAR == "text":
            Z = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(np.array(F["question"])[tr]).transform(F["question"])
        else:
            d = np.load(f"{BASE}/{st}_{DS[ds]}_1500_hidden.npz"); layers = list(d["layers"])
            l = {"cv128": pickle.load(open(f"{OUT}/probe_{st}_{ds}.pkl", "rb"))["layer"], "cv32": pickle.load(open(f"{OUT}/probe_{st}_{ds}.pkl", "rb"))["layer"], "l16": 16, "last": layers[-1]}[VAR]
            X = d["h"][:, layers.index(l), :].astype(np.float32); del d
            k = 32 if VAR == "cv32" else 128
            Zs = StandardScaler().fit(X[tr]).transform(X); Z = PCA(k, random_state=0).fit(Zs[tr]).transform(Zs); del X, Zs
        T = {"key": F["g_strict"]}
        for J in ("Q", "M", "G", "O", "R"):
            T[f"{J}35"] = F["jg_acc"][(J, "a35")]; T[f"{J}50"] = F["jg_acc"][(J, "a50")]
        S = {}
        for kk, y in T.items():
            m = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10], cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc", max_iter=3000).fit(Z[tr], y[tr].astype(int))
            S[kk] = m.decision_function(Z)
        for name, r in sorted(R.items()):
            if r["tag"] != "FK20" or r["c"] != "05": continue
            o = own(r["g"]); y = (r["o"] == 0)[ho]; a = y.mean()
            if o is None or not (0.05 <= a <= 0.95): continue
            rows.append(dict(st=st, ds=ds, own=o, kind="key" if o == "key" else "judge", p_own=auc(y, -S[o][ho]), p_key=auc(y, -S["key"][ho]), **{f"p_{j}": auc(y, -S[j][ho]) for j in J5}))
out = [f"variant {VAR}"]
for kind in ("judge", "key"):
    L = [d for d in rows if d["kind"] == kind]
    out.append(f"  {kind}-trained runs {len(L)}: own-probe forecast AUROC mean {np.mean([d['p_own'] for d in L]):.3f} (qwen7 {np.mean([d['p_own'] for d in L if d['st'] == 'qwen7']):.3f}, llama8 {np.mean([d['p_own'] for d in L if d['st'] == 'llama8']):.3f})"
               + (f"; own > key in {np.mean([d['p_own'] > d['p_key'] for d in L]):.2f}" if kind == "judge" else ""))
eff = []; tot_null = np.zeros(1000)
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        L = [d for d in rows if d["st"] == st and d["ds"] == ds and d["own"] in J5]
        if len({d["own"] for d in L}) < 2: continue
        M = np.array([[d[f"p_{j}"] for j in J5] for d in L]); ow = np.array([J5.index(d["own"]) for d in L])
        Rm = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean(); eff.extend(Rm[np.arange(len(L)), ow])
        for b in range(1000): tot_null[b] += Rm[np.arange(len(L)), rng.permutation(ow)].sum()
out.append(f"  fingerprint: own-cell residual {np.mean(eff):+.4f} over {len(eff)} runs, permutation p = {(np.sum(tot_null >= np.sum(eff)) + 1) / 1001:.4f}")
print("\n".join(out), flush=True)
open(f"{OUT}/k13_out.txt", "a").write("\n".join(out) + "\n")
