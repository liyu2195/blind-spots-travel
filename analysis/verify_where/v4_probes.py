"""(verify_where) Linear probes of the UNTRAINED student's hidden state (last prompt token), fitted ONLY on trained questions; own code.
Representation: one layer (default 16), standardised with trained-question statistics, PCA 256 fitted on trained questions.
Classifier: L2 logistic regression, C chosen by 5-fold CV AUROC on trained questions only. Scores saved for all 1,500 questions, plus
5-fold out-of-fold scores on trained questions (for home comparisons) and a shuffled-label placebo probe per judge target.
Targets on the untrained greedy answer:
  rejV_<J>   judge J (a35) rejects the greedy answer, fitted on trained questions with a VALID greedy answer (item 3)
  pay_<g>    grader g would pay the greedy answer, IDK / malformed = not paid, fitted on all trained questions (item 4):
             key (strict key), son (Sonnet >= 4), Q35 M35 G35 O35 R35, Q50, Qt (P(Yes) > 0.5), aud (loose key OR Sonnet: grader-free truth),
             sure (>= 31/32 samples agree; label-free), nj35 (number of the 5 judges paying at a35 >= 3: a 'consensus judge')
Also saves the new-question verdict inputs. One hidden-state file in memory at a time.
Usage: nice -n 10 python3 v4_probes.py qwen7 pqa [layer]   -> cache/probe_<st>_<dsn>_L<layer>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
C_ = (_RA + "/verify_where/cache"); BASE = (_RR + "/run1/base")
J5 = ["Q", "M", "G", "O", "R"]; CS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

def fit(Z, y, tr_idx, seed=0):
    y = np.asarray(y, int); yt = y[tr_idx]; cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    best = None
    for c in CS:
        o = cross_val_predict(LogisticRegression(C=c, max_iter=3000), Z[tr_idx], yt, cv=cv, method="decision_function")
        a = roc_auc_score(yt, o)
        if best is None or a > best[0]: best = (a, c, o)
    m = LogisticRegression(C=best[1], max_iter=3000).fit(Z[tr_idx], yt)
    return dict(score=m.decision_function(Z), oof=best[2], C=best[1], cv_auc=best[0])

def main(st, dsn, layer=16):
    B = pickle.load(open(f"{C_}/base_{st}_{dsn}.pkl", "rb")); N = 1500
    z = np.load(f"{BASE}/{st}_{dsn}_1500_hidden.npz"); assert (z["idx"] == np.arange(N)).all()
    li = list(z["layers"]).index(layer); H = z["h"][:, li, :].astype(np.float32); del z
    tr = np.where(B["train"])[0]
    mu = H[tr].mean(0); sd = H[tr].std(0) + 1e-3; Hs = (H - mu) / sd; del H
    Z = PCA(256, random_state=0).fit(Hs[tr]).transform(Hs); del Hs
    v = B["valid"][:, 0]; TH = B["TH"]; PJ = {J: B["PJ"][J][:, 0] for J in B["PJ"]}
    out = dict(st=st, dsn=dsn, layer=layer, probes={})
    trv = tr[v[tr]]
    for J in J5:
        y = np.zeros(N, int); y[v] = (PJ[J][v] <= TH[J]["a35"]).astype(int)
        f = fit(Z, y, trv); out["probes"][f"rejV_{J}"] = f
        rng = np.random.default_rng(11); ys = y.copy(); ys[trv] = rng.permutation(y[trv])
        out["probes"][f"shufV_{J}"] = fit(Z, ys, trv)
    pay = {"key": v & B["ks"][:, 0], "son": v & (B["son"][:, 0] >= 4), "aud": v & (B["kl"][:, 0] | (B["son"][:, 0] >= 4)), "sure": B["conf"] >= 31 / 32,
           "Q50": v & (PJ["Q"] > TH["Q"]["a50"]), "Qt": v & (PJ["Q"] > 0.5)}
    for J in J5: pay[f"{J}35"] = v & (PJ[J] > TH[J]["a35"])
    pay["nj35"] = np.sum([pay[f"{J}35"] for J in J5], 0) >= 3
    for k, y in pay.items():
        out["probes"][f"pay_{k}"] = fit(Z, y.astype(int), tr)
    pickle.dump(out, open(f"{C_}/probe_{st}_{dsn}_L{layer}.pkl", "wb"))
    ho = B["held"]; hv = ho & v
    msg = [f"{st} {dsn} L{layer}:"]
    for J in J5:
        y = (PJ[J] <= TH[J]["a35"]); p = out["probes"][f"rejV_{J}"]; s = out["probes"][f"shufV_{J}"]
        msg.append(f"rejV_{J} cvAUC {p['cv_auc']:.3f} newAUC {roc_auc_score(y[hv], p['score'][hv]):.3f} (shuffled {roc_auc_score(y[hv], s['score'][hv]):.3f})")
    for k, y in pay.items():
        msg.append(f"pay_{k} newAUC {roc_auc_score(y[ho], out['probes'][f'pay_{k}']['score'][ho]):.3f}")
    print(" | ".join(msg), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
