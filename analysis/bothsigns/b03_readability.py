"""(bothsigns) b03: readability of every arm's error, one definition for all arms.
Readability = AUC on NEW questions of a probe fitted on TRAINING questions only, predicting whether the grader errs on the untrained student's
greedy answer (the affected-item label of b02, on the same population: FR arms: strict-right greedy answers, label = rejected; FA / form arms:
audited-wrong greedy answers, label = paid / paid form; key misses: audited-right greedy answers, label = key misses; placebo: pseudo-random half).
Primary probe (hidQA): L2 logistic regression on what the student sees when it produces the item: the untrained student's hidden state at the
last prompt token (layer 16; standardised, top 128 principal components of the training questions, scaled to unit mean row norm) plus its
answer's text (TF-IDF of answer tokens, word count, long-answer and digit flags); C chosen by 5-fold CV AUC on training items. Variants: hidden
state only (layers 16, 24); TF-IDF of the question; TF-IDF of question + answer. Also: hidQA AUC within quartiles of the student's sample
agreement (readability beyond confidence) and the AUC of confidence alone. CI: 300 bootstrap resamples of the new items (probe fixed).
Usage: nice -n 10 python3 b03_readability.py <st> <ds>  -> cache/b03_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, pickle, time
import numpy as np
sys.path.insert(0, (_RA + "/bothsigns"))
from b00_common import *
import b02_effects as E
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from scipy.stats import rankdata

CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)          # average ranks for ties
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_probe(Xtr, ytr, Xte):
    best, bc = -1, CS[0]
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for c in CS:
        sc = []
        for tr, va in skf.split(Xtr, ytr):
            m = LogisticRegression(C=c, max_iter=2000).fit(Xtr[tr], ytr[tr]); sc.append(auc(ytr[va], m.decision_function(Xtr[va])))
        if np.nanmean(sc) > best:
            best, bc = np.nanmean(sc), c
    m = LogisticRegression(C=bc, max_iter=2000).fit(Xtr, ytr)
    return m.decision_function(Xte), bc, best


def main(st, ds):
    t0 = time.time(); dsn = DSN[ds]
    U, M = E.build_units(st, ds)
    P = pickle.load(open(f"{OUT}/cache/cache_{st}_{ds}.pkl", "rb")); Q = P["Q"]; idx = Q["idx"]; part = Q["part"]
    feats = {}
    z = np.load(f"{BASE}/{st}_{dsn}_1500_hidden.npz")
    pos = {int(i): k for k, i in enumerate(z["idx"])}; order = np.array([pos[int(i)] for i in idx]); layers = list(z["layers"])
    Hall = z["h"]
    for L in (16, 24):
        H = Hall[:, layers.index(L), :].astype(np.float32)[order]
        mu = H[part == 0].mean(0); sd = H[part == 0].std(0) + 1e-6; Hs = (H - mu) / sd
        _, _, Vt = np.linalg.svd(Hs[part == 0], full_matrices=False)
        feats[f"hid{L}"] = Hs @ Vt[:128].T
        del H, Hs
    del Hall, z
    qv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([Q["question"][k] for k in range(len(idx)) if part[k] == 0])
    feats["txtQ"] = qv.transform(Q["question"]).tocsr()
    ans = ["ANS_" + " ANS_".join(Q["g"][k].split()) if Q["g"][k] else "ANS_none" for k in range(len(idx))]
    av = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, token_pattern=r"\S+").fit([ans[k] for k in range(len(idx)) if part[k] == 0])
    lenf = csr_matrix(np.stack([np.minimum(Q["gwords"], 8) / 8.0, (Q["gwords"] >= LONG_WORDS).astype(float),
                                np.array([any(ch.isdigit() for ch in g) for g in Q["g"]], float)], 1))
    feats["txtQA"] = hstack([feats["txtQ"], av.transform(ans), lenf]).tocsr()
    hq = feats["hid16"] / np.sqrt((feats["hid16"][part == 0] ** 2).sum(1).mean())          # unit mean row norm, like the TF-IDF block
    feats["hidQA"] = hstack([csr_matrix(hq), av.transform(ans), lenf]).tocsr()              # PRIMARY: what the student sees = question state + its answer
    conf = Q["conf"]
    rng = np.random.default_rng(3); res = []
    for u in U:
        lab = np.asarray(u["lab"], bool); pop = np.asarray(u["labpop"], bool)
        tr = pop & (part == 0); te = pop & (part == 1)
        ytr = lab[tr].astype(int); yte = lab[te]
        r = dict(arm=u["arm"], sign=u["sign"], st=st, ds=ds, n_tr=int(tr.sum()), pos_tr=int(ytr.sum()), n_te=int(te.sum()), pos_te=int(yte.sum()),
                 runs=u["runs"])
        for fk, X in feats.items():
            if ytr.sum() < 5 or (len(ytr) - ytr.sum()) < 5 or yte.sum() < 3 or (len(yte) - yte.sum()) < 3:
                r[fk] = np.nan; r[f"{fk}_bs"] = np.full(300, np.nan); continue
            s, c, cv = fit_probe(X[tr], ytr, X[te])
            a = auc(yte, s); bs = np.empty(300)
            n = len(yte)
            for b in range(300):
                ii = rng.integers(0, n, n); bs[b] = auc(yte[ii], s[ii])
            r[fk] = a; r[f"{fk}_bs"] = bs; r[f"{fk}_C"] = c; r[f"{fk}_cv"] = cv
            if fk == "hidQA":       # readability beyond the student's own confidence: AUC within quartiles of sample agreement (pair-weighted)
                cf = conf[te]; qs = np.quantile(cf, [0.25, 0.5, 0.75]); stt = np.digitize(cf, qs)
                num = den = 0.0
                for g in np.unique(stt):
                    m = stt == g; n1 = yte[m].sum(); n0 = m.sum() - n1
                    if n1 > 0 and n0 > 0:
                        num += auc(yte[m], s[m]) * n1 * n0; den += n1 * n0
                r["hidQA_strat"] = num / den if den > 0 else np.nan
                r["conf_auc"] = auc(yte, cf) if True else np.nan     # how well confidence alone ranks the label (low conf -> higher score)
                r["conf_auc"] = max(r["conf_auc"], 1 - r["conf_auc"]) if np.isfinite(r["conf_auc"]) else np.nan
        res.append(r)
        print(f"{st} {ds} {u['arm']:18s} {u['sign']:9s} n {r['n_tr']}/{r['n_te']} pos {r['pos_tr']}/{r['pos_te']} | " +
              " ".join(f"{fk} {r[fk]:.3f}" for fk in feats) + f" ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(res, open(f"{OUT}/cache/b03_{st}_{ds}.pkl", "wb"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
