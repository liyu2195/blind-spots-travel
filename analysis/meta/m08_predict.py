"""(meta) Visibility of a grader's mistakes, measured on the UNTRAINED student's answers only (no training run is used).
FR (false rejects, question level): questions whose untrained greedy answer is right; label = the grader rejects >= half of the strictly-right samples.
FA (false accepts, answer level): distinct (question, wrong answer) pairs of the 32 samples; label = accepted (probability >= 0.5).
Features: TF-IDF of the question text (FR, FA) and of the answer text plus its length and sample frequency (FA).
AUC_new = logistic model fitted on the trained questions, evaluated on the new questions; AUC_cv = 5-fold within trained questions (grouped by question).
Gini = 2 AUC - 1. Writes cache/visibility.csv (one row per (student, dataset, grader spec)). Usage: python3 m08_predict.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, glob, numpy as np, pandas as pd, re
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, BASE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
OUT = (_RA + "/meta/cache")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
rows = []
def auc_fit(Xh, yh, Xn, yn, groups):
    out = {}
    if len(set(yh)) < 2 or min(np.bincount(yh)) < 10: return dict(auc_cv=np.nan, auc_new=np.nan)
    p = np.zeros(len(yh)); gkf = GroupKFold(5)
    for tr, te in gkf.split(Xh, yh, groups):
        if len(set(yh[tr])) < 2: p[te] = 0.5; continue
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xh[tr], yh[tr]); p[te] = m.predict_proba(Xh[te])[:, 1]
    out["auc_cv"] = roc_auc_score(yh, p)
    if Xn is not None and len(set(yn)) == 2:
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xh, yh); out["auc_new"] = roc_auc_score(yn, m.predict_proba(Xn)[:, 1])
    else: out["auc_new"] = np.nan
    return out
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        P = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); dsn = DSN[ds]
        probs, H1, HELD = setup(dsn); Qf = P["Qf"]; HOME, NEW = P["HOME"], P["NEW"]
        U = {}
        for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")): U[(r["idx"], r["s"])] = sa(r)
        qtext = {i: probs[i]["question"] for i in probs}
        vq = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([qtext[i] for i in HOME])
        # per-question extra features: the student's own confidence
        def qfeat(items):
            return np.array([[Qf[i]["conf"], Qf[i]["q_loose"]] for i in items])
        # which spec keys: one per sub-family (first run), standard runs only
        T = pd.read_pickle(f"{OUT}/law2.pkl"); T = T[(T.st == st) & (T.ds == ds) & T["std"]]
        keys = {}
        for run in T.run:
            r = P["R"][run]; sub = T.loc[T.run == run, "sub"].iloc[0]; fam = T.loc[T.run == run, "fam"].iloc[0]
            if fam in ("gold", "keyFP", "keyFN", "mixU"): continue
            if sum(1 for v in keys.values() if v == (fam, sub)) >= 2: continue
            keys.setdefault(r["speckey"], (fam, sub))
        for key, (fam, sub) in keys.items():
            Sq = P["SPECQ"][key]; d = dict(st=st, ds=ds, fam=fam, sub=sub, speckey=key)
            # ---- FR, question level
            lab = {}
            for part, items in (("home", HOME), ("new", NEW)):
                rej = Sq[part]["rej"]; sel = [k for k, i in enumerate(items) if Qf[i]["gcode"] == 1 and np.isfinite(rej[k])]
                lab[part] = ([items[k] for k in sel], (rej[sel] >= 0.5).astype(int))
            (ih, yh), (inn, yn) = lab["home"], lab["new"]
            d["FR_pos_home"] = float(yh.mean()) if len(yh) else np.nan; d["FR_pos_new"] = float(yn.mean()) if len(yn) else np.nan
            if len(yh) >= 50:
                Xh = vq.transform([qtext[i] for i in ih]); Xn = vq.transform([qtext[i] for i in inn]) if len(inn) else None
                o = auc_fit(Xh, yh, Xn, yn, np.array(ih)); d["FRq_auc_cv"], d["FRq_auc_new"] = o["auc_cv"], o["auc_new"]
                Xh2 = hstack([Xh, csr_matrix(qfeat(ih))]).tocsr(); Xn2 = hstack([Xn, csr_matrix(qfeat(inn))]).tocsr() if len(inn) else None
                o = auc_fit(Xh2, yh, Xn2, yn, np.array(ih)); d["FRqc_auc_cv"], d["FRqc_auc_new"] = o["auc_cv"], o["auc_new"]
            # ---- FA, answer level (distinct pairs)
            pairs = {}
            for part in ("home", "new"):
                W = Sq[part]; agg = {}
                for i, s_, a in zip(W["w_idx"], W["w_s"], W["w_acc"]):
                    x = U.get((int(i), int(s_)), "")
                    if not x: continue
                    e = agg.setdefault((int(i), x), [0.0, 0]); e[0] += float(a); e[1] += 1
                pairs[part] = [(i, x, v[0] / v[1], v[1]) for (i, x), v in agg.items()]
            if len(pairs["home"]) >= 100:
                va = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True).fit([x for _, x, _, _ in pairs["home"]])
                def X(pl):
                    A = va.transform([x for _, x, _, _ in pl]); Qm = vq.transform([qtext[i] for i, _, _, _ in pl])
                    ex = np.array([[len(x.split()), np.log1p(n), Qf[i]["conf"]] for i, x, _, n in pl])
                    return hstack([A, Qm, csr_matrix(ex)]).tocsr()
                yh = np.array([a > 1e-9 for _, _, a, _ in pairs["home"]], int); yn = np.array([a > 1e-9 for _, _, a, _ in pairs["new"]], int)
                d["FA_pos_home"] = float(yh.mean()); d["FA_pos_new"] = float(yn.mean()) if len(yn) else np.nan
                o = auc_fit(X(pairs["home"]), yh, X(pairs["new"]) if len(pairs["new"]) else None, yn, np.array([i for i, _, _, _ in pairs["home"]]))
                d["FAa_auc_cv"], d["FAa_auc_new"] = o["auc_cv"], o["auc_new"]
                # question-only features for FA (is the grader's leniency a property of the question?)
                Xq = lambda pl: hstack([vq.transform([qtext[i] for i, _, _, _ in pl]), csr_matrix(np.array([[Qf[i]["conf"]] for i, _, _, _ in pl]))]).tocsr()
                o = auc_fit(Xq(pairs["home"]), yh, Xq(pairs["new"]) if len(pairs["new"]) else None, yn, np.array([i for i, _, _, _ in pairs["home"]]))
                d["FAq_auc_cv"], d["FAq_auc_new"] = o["auc_cv"], o["auc_new"]
            rows.append(d)
            print(st, ds, f"{fam:7s} {sub:14s}", " ".join(f"{k} {d[k]:.3f}" for k in ("FRq_auc_cv", "FRq_auc_new", "FRqc_auc_new", "FAa_auc_cv", "FAa_auc_new", "FAq_auc_new") if k in d and np.isfinite(d[k])), flush=True)
V = pd.DataFrame(rows); V.to_csv(f"{OUT}/visibility.csv", index=False)
