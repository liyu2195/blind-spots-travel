"""(stats) Linear probes of the UNTRAINED student's last-prompt-token hidden state, fitted ONLY on trained questions (own code).
Representation: layer L (16 default; 24 robustness), standardised with trained-question means / sds, PCA 256 fitted on trained questions.
Targets (untrained greedy answer, trained questions with a valid greedy answer): rej_J = judge J (Q M G O) at its a35 cut rejects the answer;
shuf_J = the same labels permuted over trained questions (placebo probe). Classifier: L2 logistic regression, C from {1e-3, 3e-3, 1e-2, 3e-2, 0.1}
by 5-fold CV AUROC on trained questions. Saved: decision scores for all 1,500 questions, 5-fold out-of-fold scores on trained questions, CV AUROC,
and the PCA features (for nearest-neighbour proxies). Usage: nice -n 10 python3 -I s03_probes.py qwen7 pq [layer] -> cache/probe_<st>_<ds>_L<l>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
st, ds = sys.argv[1], sys.argv[2]; layer = int(sys.argv[3]) if len(sys.argv) > 3 else 16; dsn = DSN[ds]
B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); N = B["N"]; TH = {J: thresholds(J)[f"{st}_{dsn}"]["t_a35"] for J in J4}
z = np.load(f"{BASE}/{st}_{dsn}_1500_hidden.npz"); assert (z["idx"] == np.arange(N)).all()
li = list(z["layers"]).index(layer); Hh = z["h"][:, li, :].astype(np.float32); del z
tr = np.where(B["train"])[0]; mu = Hh[tr].mean(0); sd = Hh[tr].std(0) + 1e-3; Hh = (Hh - mu) / sd
Z = PCA(256, random_state=0).fit(Hh[tr]).transform(Hh).astype(np.float32); del Hh
v = B["valid"][:, 0]; trv = tr[v[tr]]
def fit(y):
    yt = y[trv]; cv = StratifiedKFold(5, shuffle=True, random_state=0); best = None
    for c in (1e-3, 3e-3, 1e-2, 3e-2, 0.1):
        o = cross_val_predict(LogisticRegression(C=c, max_iter=2000), Z[trv], yt, cv=cv, method="decision_function"); a = roc_auc_score(yt, o)
        if best is None or a > best[0]: best = (a, c, o)
    m = LogisticRegression(C=best[1], max_iter=2000).fit(Z[trv], yt); oof = np.full(N, np.nan); oof[trv] = best[2]
    return dict(score=m.decision_function(Z), oof=oof, C=best[1], cv_auc=best[0])
out = dict(st=st, ds=ds, layer=layer, Z=Z, probes={})
rng = np.random.default_rng(31)
msg = []
for J in J4:
    y = np.zeros(N, int); y[v] = (B["PJ"][J][v, 0] <= TH[J]).astype(int)
    out["probes"][f"rej_{J}"] = fit(y)
    ys = y.copy(); ys[trv] = rng.permutation(y[trv]); out["probes"][f"shuf_{J}"] = fit(ys)
    hv = B["new"] & v
    msg.append(f"{J}: cvAUC {out['probes'][f'rej_{J}']['cv_auc']:.3f} newAUC {roc_auc_score(y[hv], out['probes'][f'rej_{J}']['score'][hv]):.3f} "
               f"(shuffled-label probe newAUC {roc_auc_score(y[hv], out['probes'][f'shuf_{J}']['score'][hv]):.3f})")
pickle.dump(out, open(f"{CACHE}/probe_{st}_{ds}_L{layer}.pkl", "wb"))
print(f"{st} {ds} L{layer}: " + " | ".join(msg), flush=True)
