"""(verify_map) Other predictors of each judge's a35 verdict on the untrained greedy answer, fitted ONLY on training questions (as v02):
  p64    L2 logistic regression on the first 64 PCA components of layer 16 (same input as mlp16; separates nonlinearity from input size)
  mlp16  one hidden layer of 64 ReLU units (lbfgs) on the same 64 components; alpha by 5-fold CV log-loss
  knn16  cosine k-NN on layer-16 PCA-256 (k by CV)
  tfidf  L2 logistic regression on word 1-2-gram + char 3-5-gram TF-IDF of the question text (vocabulary fitted on training questions)
  L12 / L20: the v02 linear probe on layers 12 / 20 are produced by v02_probe.py itself.
Saved: P on all questions (final fit), OOF on training questions. Usage: nice -n 10 timeout 280 python3 v02b_predictors.py qwen7 nq
-> cache/q_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, time, warnings
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
warnings.filterwarnings("ignore")
st, ds = sys.argv[1], sys.argv[2]; t0 = time.time()
D = load(st, ds); N = D["N"]; tr = np.where(D["train"])[0]
z = np.load(f"{BASE}/{st}_{DSN[ds]}_1500_hidden.npz"); li = list(z["layers"]).index(16); Hh = z["h"][:, li, :].astype(np.float32); del z
mu = Hh[tr].mean(0); sd = Hh[tr].std(0) + 1e-3; Hh = (Hh - mu) / sd
Z = PCA(256, random_state=0).fit(Hh[tr]).transform(Hh).astype(np.float64); Z /= Z[tr].std(0) + 1e-6; del Hh
Z64 = np.ascontiguousarray(Z[:, :64]); Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True); S = Zn @ Zn.T
qs = D["question"]; vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([qs[i] for i in tr])
vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit([qs[i] for i in tr])
TF = hstack([vw.transform(qs), vc.transform(qs)]).tocsr()
v = np.isin(D["cd"][:, 0], (1, 2))
KINDS = {
    "p64": (lambda hp: (lambda a, b: LogisticRegression(C=hp, max_iter=4000).fit(Z64[a], Yall[a]).predict_proba(Z64[b])[:, 1]), (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)),
    "mlp16": (lambda hp: (lambda a, b: MLPClassifier((64,), alpha=hp, solver="lbfgs", max_iter=400, random_state=0).fit(Z64[a], Yall[a]).predict_proba(Z64[b])[:, 1]), (1.0, 10.0, 100.0)),
    "tfidf": (lambda hp: (lambda a, b: LogisticRegression(C=hp, max_iter=4000).fit(TF[a], Yall[a]).predict_proba(TF[b])[:, 1]), (0.3, 1.0, 3.0, 10.0, 30.0)),
}
def knn(hp):
    def f(a, b):
        Sq = S[np.ix_(b, a)].copy(); Sq[b[:, None] == a[None, :]] = -np.inf
        nn = np.argpartition(-Sq, kth=hp - 1, axis=1)[:, :hp]; p0 = Yall[a].mean(); return (Yall[a][nn].sum(1) + 2 * p0) / (hp + 2)
    return f
KINDS["knn16"] = (knn, (5, 10, 20, 40, 80))
out = {k: {} for k in KINDS}
for J in J5:
    Yall = (D["PJ"][J][:, 0] <= thr(J, st, ds, "a35")).astype(int)
    rows = tr[v[tr]]; yy = Yall[rows]
    for kname, (mk, grid) in KINDS.items():
        best = None
        for hp in grid:
            oof = np.zeros(len(rows)); f = mk(hp)
            for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(rows, yy): oof[b] = f(rows[a], rows[b])
            s = log_loss(yy, np.clip(oof, 1e-4, 1 - 1e-4), labels=[0, 1])
            if best is None or s < best[0]: best = (s, hp, oof)
        Pall = mk(best[1])(rows, np.arange(N)); O = np.full(N, np.nan); O[rows] = best[2]
        nv = np.where(D["new"] & v)[0]
        out[kname][f"rej_{J}_a35"] = dict(p=Pall, oof=O, hp=best[1], cv_auc=roc_auc_score(yy, best[2]), new_auc=roc_auc_score(Yall[nv], Pall[nv]))
pickle.dump(out, open(f"{CACHE}/q_{st}_{ds}.pkl", "wb"))
print(f"{st} {ds} ({time.time() - t0:.0f}s): " + " | ".join(f"{k} new AUC " + "/".join(f"{out[k][f'rej_{J}_a35']['new_auc']:.3f}" for J in J5) for k in out), flush=True)
