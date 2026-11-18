"""(meta) Second visibility measure for false-reject sets: a linear probe on the UNTRAINED student's own hidden state at the last prompt token
(base/<st>_<set>_1500_hidden.npz, middle stored layer, PCA 64 fitted on trained questions, L2 logistic C 0.05), fitted on trained untrained-right
questions (label rej_q >= .5), scored on new questions (AUC). Joined with the FR transfer points of m13. One hidden-state file in memory at a time.
Output stdout (out_m17_hidden_vis.txt), cache/transfer_points_hidden.csv."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, json, numpy as np, pandas as pd, gc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
OUT = (_RA + "/meta/cache"); BASE = (_RR + "/run1/base")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
T = pd.read_pickle(f"{OUT}/law2.pkl")
rows = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        P = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); Qf = P["Qf"]; HOME, NEW = P["HOME"], P["NEW"]
        z = np.load(f"{BASE}/{st}_{DSN[ds]}_1500_hidden.npz"); hh = z["h"]; ids = z["idx"]; L = hh.shape[1] // 2
        hl = hh[:, L, :].astype(np.float32); del hh, z; gc.collect()
        H = {int(i): hl[k] for k, i in enumerate(ids)}
        Xh_all = np.stack([H[i] for i in HOME]); mu = Xh_all.mean(0); sd = Xh_all.std(0) + 1e-3
        pca = PCA(64, random_state=0).fit((Xh_all - mu) / sd)
        emb = lambda items: pca.transform((np.stack([H[i] for i in items]) - mu) / sd)
        Eh, En = emb(HOME), emb(NEW)
        gh = np.array([Qf[i]["gcode"] for i in HOME]); gn = np.array([Qf[i]["gcode"] for i in NEW])
        D = T[(T.st == st) & (T.ds == ds) & T["std"]]
        seen = {}
        jkeys = {P["R"][r]["spec"]["J"]: P["R"][r]["speckey"] for r in D.run if P["R"][r]["spec"]["kind"] == "judge" and "a35" in r and not P["R"][r]["spec"].get("quorum")}
        for run, sub in zip(D.run, D["sub"]):
            key = P["R"][run]["speckey"]
            if sub in seen: continue
            Sq = P["SPECQ"][key]; rh = Sq["home"]["rej"]; rn = Sq["new"]["rej"]
            sp = P["R"][run]["spec"]
            if sp["kind"] == "key" and sp.get("knows") and "real" in sp["knows"]:
                J = sp["knows"].split("real")[1][0]
                if J in jkeys: rn = P["SPECQ"][jkeys[J]]["new"]["rej"]
            mh = (gh == 1) & np.isfinite(rh); mn = (gn == 1) & np.isfinite(rn)
            yh = (rh[mh] >= 0.5).astype(int); yn = (rn[mn] >= 0.5).astype(int)
            if yh.sum() < 15 or (1 - yh).sum() < 15 or yn.sum() < 5 or (1 - yn).sum() < 5: continue
            m = LogisticRegression(C=0.05, max_iter=3000).fit(Eh[mh], yh)
            seen[sub] = roc_auc_score(yn, m.predict_proba(En[mn])[:, 1])
            rows.append(dict(st=st, ds=ds, sub=sub, auc_hidden=seen[sub]))
        print(st, ds, {k: round(v, 3) for k, v in seen.items()}, flush=True)
        del H, Xh_all; gc.collect()
Hv = pd.DataFrame(rows)
P = pd.read_csv(f"{OUT}/transfer_points.csv"); P = P[P.effect == "FR refusal"].merge(Hv, on=["st", "ds", "sub"], how="left")
P["gini_hidden"] = 2 * P.auc_hidden - 1; P.to_csv(f"{OUT}/transfer_points_hidden.csv", index=False)
d = P[np.isfinite(P.gini_hidden)]
print(f"\nFR points with a hidden-state visibility: {len(d)}; Spearman(transfer, Gini_hidden) {spearmanr(d.gini_hidden, d.transfer)[0]:+.3f}; "
      f"Spearman(transfer, Gini_text) {spearmanr(d.gini, d.transfer)[0]:+.3f}; Spearman(Gini_hidden, Gini_text) {spearmanr(d.gini, d.gini_hidden)[0]:+.3f}")
a = d[d.st == "qwen7"]; b = d[d.st == "llama8"]
for col in ("gini", "gini_hidden"):
    k = np.sum(a[col] * a.transfer) / np.sum(a[col]**2); r = b.transfer - k * b[col]
    print(f"  T = k * {col}: k (qwen7) {k:.3f} -> llama8 MAE {np.mean(np.abs(r)):.3f} R2 {1 - np.sum(r**2) / np.sum((b.transfer - b.transfer.mean())**2):+.3f}")
pd.set_option("display.width", 250)
print(d[["st", "ds", "sub", "transfer", "gini", "gini_hidden"]].sort_values("gini_hidden").round(3).to_string(index=False))
