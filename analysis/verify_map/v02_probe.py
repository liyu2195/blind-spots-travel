"""(verify_map) Linear probes of each grader's verdict on the untrained greedy answer, fitted ONLY on training questions.
Input: layer-L hidden state of the untrained student at the last prompt token (base/<st>_<dsn>_1500_hidden.npz), standardised with training-question
moments, PCA to 256 dims fitted on training questions, L2 logistic regression; C chosen by 5-fold CV log-loss on the training questions.
Saved per target: P (fit on all eligible training rows, applied to all 1,500 questions) and OOF (5-fold out-of-fold on training rows).
Targets: rej_<J>_<op> (J rejects the valid greedy answer; eligible = training questions with a valid greedy answer), J in Q M G O R,
op in a35 a50 t50; rej_key (strict key rejects); aud (greedy audited right: loose key or Sonnet >= 4; all training questions);
sure (valid and >= 31/32 samples agree; all training questions); acc_wrong_<J>_a35 (J accepts, fitted on audited-wrong valid training answers only).
Usage: nice -n 10 timeout 280 python3 v02_probe.py qwen7 nq 16   -> cache/p_<st>_<ds>_L<layer>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, time, warnings
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
warnings.filterwarnings("ignore")
st, ds, L = sys.argv[1], sys.argv[2], int(sys.argv[3]); t0 = time.time()
D = load(st, ds); N = D["N"]; tr = np.where(D["train"])[0]
z = np.load(f"{BASE}/{st}_{DSN[ds]}_1500_hidden.npz"); assert (z["idx"] == np.arange(N)).all()
li = list(z["layers"]).index(L); Hh = z["h"][:, li, :].astype(np.float32); del z
mu = Hh[tr].mean(0); sd = Hh[tr].std(0) + 1e-3; Hh = (Hh - mu) / sd
Z = PCA(256, random_state=0).fit(Hh[tr]).transform(Hh).astype(np.float64); Z /= Z[tr].std(0) + 1e-6; del Hh
cd0 = D["cd"][:, 0]; v = np.isin(cd0, (1, 2)); aud = v & ((cd0 == 1) | (D["son0"] >= 4)); wrong = v & (cd0 == 2) & (D["son0"] >= 1) & (D["son0"] <= 3)
T = {}
for J in J5:
    for op in ("a35", "a50", "t50"):
        t = thr(J, st, ds, op); T[f"rej_{J}_{op}"] = ((D["PJ"][J][:, 0] <= t).astype(int), v)
    t = thr(J, st, ds, "a35"); T[f"acc_wrong_{J}_a35"] = ((D["PJ"][J][:, 0] > t).astype(int), wrong)
T["rej_key"] = ((~D["ks"][:, 0]).astype(int), v)
T["aud"] = (aud.astype(int), np.ones(N, bool)); T["sure"] = ((v & (D["conf"] >= 31 / 32)).astype(int), np.ones(N, bool))
GRID = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)
out = dict(st=st, ds=ds, layer=L, T={})
for name, (y, el) in T.items():
    rows = tr[el[tr]]; yy = y[rows]
    if yy.sum() < 8 or (1 - yy).sum() < 8: continue
    best = None
    for c in GRID:
        oof = np.zeros(len(rows))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(rows, yy):
            oof[b] = LogisticRegression(C=c, max_iter=4000).fit(Z[rows[a]], yy[a]).predict_proba(Z[rows[b]])[:, 1]
        s = log_loss(yy, np.clip(oof, 1e-4, 1 - 1e-4), labels=[0, 1])
        if best is None or s < best[0]: best = (s, c, oof)
    P = LogisticRegression(C=best[1], max_iter=4000).fit(Z[rows], yy).predict_proba(Z)[:, 1]
    O = np.full(N, np.nan); O[rows] = best[2]
    nv = np.where(D["new"] & el)[0]
    auc_new = roc_auc_score(y[nv], P[nv]) if 0 < y[nv].sum() < len(nv) else np.nan
    out["T"][name] = dict(p=P, oof=O, C=best[1], cv_auc=roc_auc_score(yy, best[2]), new_auc=auc_new, n=len(rows), rate=float(yy.mean()))
pickle.dump(out, open(f"{CACHE}/p_{st}_{ds}_L{L}.pkl", "wb"))
print(f"{st} {ds} L{L} ({time.time() - t0:.0f}s): " + " | ".join(f"{k} cv {r['cv_auc']:.3f} new {r['new_auc']:.3f}" for k, r in out["T"].items()
                                                         if k in ("rej_Q_a35", "rej_M_a35", "rej_G_a35", "rej_O_a35", "acc_wrong_Q_a35", "aud", "sure")), flush=True)
