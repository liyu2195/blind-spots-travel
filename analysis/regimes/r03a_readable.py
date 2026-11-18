"""r03a: are a grader's false rejections READABLE by the student before training?
For each (student, dataset, grader setting): questions whose untrained greedy answer is audited right (and, for judges and Sonnet,
accepted by the training key; for the key, accepted by Sonnet). Label = the grader rejects that answer.
Probe = L2 logistic regression on the untrained student's hidden state (layer 16, last prompt token) + untrained agreement (conf),
fitted on TRAINING questions only, scored by AUROC on NEW questions. Baselines: conf alone; the same probe with labels permuted
within the training set (20 draws; placebo). Readable part = probe AUROC minus conf-only AUROC.
Output: r03a_readable.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, numpy as np
sys.path.insert(0, (_RA + "/regimes"))
from rg import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SET = [("Q_t50", "Q", "t50"), ("QS_t50", "QS", "t50"), ("M_t50", "M", "t50"), ("G_t50", "G", "t50"), ("O_t50", "O", "t50"), ("R_t50", "R", "t50"),
       ("Q_a35", "Q", "a35"), ("QS_a35", "QS", "a35"), ("sonnet", None, "son"), ("key_alias", None, "alias")]
out = open(f"{HERE}/r03a_readable.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")

def labels(Q, J, op, st, dsn):
    base = Q["gval"] & Q["aud"]
    if J is not None:
        U = base & Q["kstrict"] & np.isfinite(Q["pj"][J]); y = Q["pj"][J] <= thr(J, st, dsn, op)
    elif op == "son":
        U = base & Q["kstrict"] & (Q["son"] >= 0); y = Q["son"] < 4
    else:  # key's missing aliases: Sonnet-right answers the loose key rejects, vs answers the strict key accepts
        U = Q["gval"] & (Q["son"] >= 4) & (Q["kstrict"] | ~Q["key"]); y = ~Q["key"]
    return U, y

pr("st ds setting | n_train n_new rej_rate_new | AUROC new: probe(h16+conf)  conf-only  readable(probe-conf)  permuted-label probe mean [max of 20]")
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        P = build(st, ds); Q = P["Q"]; dsn = Q["dsn"]
        H = np.load(f"{BASE}/{st}_{dsn}_1500_hidden.npz"); li = list(H["layers"]).index(16)
        pos = {int(i): k for k, i in enumerate(H["idx"])}; X = H["h"][[pos[int(i)] for i in Q["idx"]], li, :].astype(np.float32); del H
        X = np.concatenate([X, Q["conf"][:, None].astype(np.float32) * 10], axis=1)
        for name, J, op in SET:
            U, y = labels(Q, J, op, st, dsn); tr = U & Q["train"]; te = U & Q["held"]
            if y[tr].sum() < 10 or y[te].sum() < 10 or (~y[tr]).sum() < 10: pr(f"{st} {ds} {name:9s} | too few"); continue
            sc = StandardScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
            m = LogisticRegression(C=0.002, max_iter=2000).fit(Xtr, y[tr]); a = roc_auc_score(y[te], m.decision_function(Xte))
            ac = roc_auc_score(y[te], -Q["conf"][te])
            rng = np.random.default_rng(0); pa = []
            for _ in range(20):
                mp = LogisticRegression(C=0.002, max_iter=2000).fit(Xtr, rng.permutation(y[tr])); pa.append(roc_auc_score(y[te], mp.decision_function(Xte)))
            pr(f"{st} {ds} {name:9s} | {tr.sum():4d} {te.sum():4d} {y[te].mean():.3f} | probe {a:.3f}  conf {ac:.3f}  readable {a - ac:+.3f}  permuted {np.mean(pa):.3f} [{np.max(pa):.3f}]")
        del X
out.close()
