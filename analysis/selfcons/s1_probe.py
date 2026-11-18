"""selfcons s1: linear probes on the UNTRAINED student's hidden state (last prompt token), fitted on the 750 TRAINING questions only,
scored on all 1,500 (so new-question scores are out of sample).  Writes probe_<st>_<ds>.pkl and appends to s1_probe.txt.
Method: one layer (chosen by 5-fold CV AUROC of target 'pik' on training questions, layers 8-28), standardise + PCA(128) fitted on
training questions, logistic regression with C chosen by 5-fold CV.  Targets (all known before any RL run):
  pik     P(IK)-style: the untrained greedy answer is right by the loose key (IDK / malformed = not right)
  pikA    same with audited labels (loose key OR Sonnet >= 4)
  sure    grader-free: >= 31 of 32 untrained samples agree with the greedy answer (no labels at all)
  pay_<g> the grader g would pay the untrained greedy answer (strict key; judge J at op; Sonnet >= 4; own-vote = greedy is the plurality)
Also a 'combo' score: logistic model on [conf, plurality share, sample IDK share, sample key share is NOT used (needs labels on new qs)]
  -> combo = logistic on [conf, psh, sidk, ncl, probe pik, probe sure] fitted on training questions to the loose-key label.
Usage: python3 s1_probe.py qwen7:pq ..."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, time, numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
OUT = (_RA + "/selfcons"); BASE = (_RR + "/run1/base")
DS = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}

def auc(y, s):
    y = np.asarray(y, bool); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

def zfeat(X, tr, k=128):
    sc = StandardScaler().fit(X[tr]); Z = sc.transform(X); p = PCA(k, random_state=0).fit(Z[tr]); return p.transform(Z)

def fit(Z, y, tr, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    m = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], cv=cv, scoring="roc_auc", max_iter=3000).fit(Z[tr], y[tr])
    oof = cross_val_predict(LogisticRegression(C=m.C_[0], max_iter=3000), Z[tr], y[tr], cv=cv, method="decision_function")
    return m.decision_function(Z), oof

def run(st, ds):
    t0 = time.time(); P = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); Q = P["Q"]; del P
    tr, ho = Q["tr"], Q["ho"]; N = Q["N"]
    d = np.load(f"{BASE}/{st}_{DS[ds]}_1500_hidden.npz"); layers = [int(x) for x in d["layers"]]; assert (d["idx"] == np.arange(N)).all()
    H = d["h"]
    pik = Q["gkey"].astype(int); pikA = Q["gaud"].astype(int); sure = (Q["conf"] >= 31 / 32).astype(int)
    best = None; log = []
    for l in [l for l in layers if 8 <= l <= 28]:
        Z = zfeat(H[:, layers.index(l), :].astype(np.float32), tr)
        _, oof = fit(Z, pik, tr); a = auc(pik[tr], oof); log.append((l, round(float(a), 3)))
        if best is None or a > best[1]: best = (l, a, Z)
    l, a_cv, Z = best; del H, d
    tg = dict(pik=pik, pikA=pikA, sure=sure, pay_key=Q["gstr"].astype(int), pay_son=pikA, pay_vote=Q["vote"].astype(int))
    for k, v in Q["accg"].items(): tg[f"pay_{k}"] = v.astype(int)
    S = {}; OOF = {}
    for k, y in tg.items():
        if y[tr].sum() < 10 or (1 - y[tr]).sum() < 10: continue
        S[k], OOF[k] = fit(Z, y, tr)
    # combo: sampling features + the two probe scores (out-of-fold on training questions to avoid leakage into the fit)
    Xc = np.c_[Q["conf"], Q["psh"], Q["sidk"], Q["ncl"], S["pik"], S["sure"]]
    Xc_tr = Xc.copy(); Xc_tr[np.where(tr)[0], 4] = OOF["pik"]; Xc_tr[np.where(tr)[0], 5] = OOF["sure"]
    sc = StandardScaler().fit(Xc_tr[tr]); lm = LogisticRegression(max_iter=3000).fit(sc.transform(Xc_tr[tr]), pik[tr])
    S["combo"] = lm.decision_function(sc.transform(Xc))
    out = dict(layer=l, cv=log, S=S)
    pickle.dump(out, open(f"{OUT}/probe_{st}_{ds}.pkl", "wb"))
    v = Q["gval"] & ho; aud = Q["gaud"]
    msg = (f"{st} {ds}: layer {l} cv {log} | new answered {v.sum()} | AUROC audited-right on new answered: conf {auc(aud[v], Q['conf'][v]):.3f} "
           f"pik {auc(aud[v], S['pik'][v]):.3f} pikA {auc(aud[v], S['pikA'][v]):.3f} sure {auc(aud[v], S['sure'][v]):.3f} combo {auc(aud[v], S['combo'][v]):.3f} "
           f"| AUROC of probe for its own target on new qs: " + " ".join(f"{k} {auc(tg[k][ho], S[k][ho]):.2f}" for k in S if k in tg) + f" | {time.time()-t0:.0f}s")
    print(msg, flush=True); open(f"{OUT}/s1_probe.txt", "a").write(msg + "\n")

if __name__ == "__main__":
    for a in sys.argv[1:]:
        st, ds = a.split(":"); run(st, ds)
