"""(dose) Predictors of each grader's verdict, fitted ONLY on TRAINING questions (own code). One (student, dataset) and a list of predictor kinds per call.
Targets (labels on training questions; the answer is the untrained greedy answer):
  rej targets, fitted on training questions with a valid (non-IDK, well-formed) greedy answer, label 1 = the grader rejects it:
    J_a35, J_t50 for J in Q M G O R; Q_a50 (judge J at the cut accepting 35% / 50% of untrained answers, or P(Yes) > 0.5);
    key (the strict key = the training reward); son (Sonnet level < 4; Sonnet-graded answers only)
  fr targets ('fr_' prefix; lin16 and tfidf only): the same, fitted only on training answers that are audited right (false rejections)
  grader-free targets, all training questions: aud (greedy answer audited right: loose key or Sonnet >= 4; IDK / malformed = 0),
    sure (valid and >= 31/32 samples agree)
  planted sets (PopQA): membership in the knows file of the arm (type, hid, realM, realO, scrM, sharpM, sharpO), fitted on training questions with
    an audited-right greedy answer (the planted graders reject right answers; the pl* sets were drawn only among answerable questions)
Kinds: linL (L2 logistic regression on PCA-256 of the standardised layer-L hidden state of the untrained student, last prompt token; L = 4 8 12 16 20
  24 and last), mlp16 (one hidden layer of 64 ReLU units, lbfgs, on the first 64 PCA components of layer 16), lin16p64 (logistic regression on the
  same 64 components: separates nonlinearity from input size), knn16 (cosine k-NN on layer-16 PCA-256),
  tfidf (L2 logistic regression on word 1-2-gram + char 3-5-gram TF-IDF of the question text, vocabulary fitted on training questions),
  knnE (cosine k-NN on SmolLM2-135M question embeddings, d02_embed.py). Standardisation, PCA and vocabularies use training questions only.
Hyper-parameters (C, alpha or k) by 5-fold CV log-loss on the training questions; saved: P(reject) for all 1,500 questions (final fit on all
eligible training questions), 5-fold out-of-fold P on training questions (for HOME analyses), CV AUROC. Text / embedding kinds on PopQA also
score the capitals probe questions (pqq, pqp). Usage: nice -n 10 timeout 280 python3 d03_predictors.py qwen7 pq lin16 tfidf ...
-> cache/pred_<st>_<ds>_<kind>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, time, pickle, json, numpy as np, warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
warnings.filterwarnings("ignore")
sys.path.insert(0, (_RA + "/dose"))
from d00_common import *
st, ds = sys.argv[1], sys.argv[2]; KINDS = sys.argv[3:]; dsn = DSN[ds]; t00 = time.time()
B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); N = B["N"]; TH = thresholds()
tr = np.where(B["train"])[0]; v = B["valid"][:, 0]; g0 = B["PJ"]
aud = v & (B["kl"][:, 0] | (B["son"][:, 0] >= 4))
T = {}   # name -> (labels over N, eligible mask over N)
for J in J5:
    for op in ("a35", "t50") + (("a50",) if J == "Q" else ()):
        t = 0.5 if op == "t50" else TH[J][f"{st}_{dsn}"][f"t_{op}"]
        y = (g0[J][:, 0] <= t).astype(int); T[f"{J}_{op}"] = (y, v.copy()); T[f"fr_{J}_{op}"] = (y, v & aud)
T["key"] = ((~B["ks"][:, 0]).astype(int), v.copy()); T["fr_key"] = ((~B["ks"][:, 0]).astype(int), v & aud)
T["son"] = ((B["son"][:, 0] < 4).astype(int), v & (B["son"][:, 0] > 0))
T["aud"] = (aud.astype(int), np.ones(N, bool)); T["sure"] = ((v & (B["conf"] >= 31 / 32)).astype(int), np.ones(N, bool))
if ds == "pq":
    for nm, f in (("type", "pqa_knows_type.jsonl"), ("hid", "pqa_knows_hid.jsonl"), ("realM", f"pqa_knows_realM_{st}.jsonl"), ("realO", f"pqa_knows_realO_{st}.jsonl"),
                  ("scrM", f"pqa_knows_scrM_{st}.jsonl"), ("sharpM", f"pqa_knows_sharpM_{st}.jsonl"), ("sharpO", f"pqa_knows_sharpO_{st}.jsonl")):
        kk = {r["idx"]: bool(r["knows"]) for r in jl(f"{DATA}/{f}")}; y = np.zeros(N, int); m = np.zeros(N, bool)
        for i, b in kk.items(): y[i] = int(b); m[i] = True
        T[f"pl_{nm}"] = (y, m & B["train"] & aud)   # planted sets reject RIGHT answers (pl* sets were drawn among answerable questions only)
FR_KINDS = ("lin16", "tfidf")
CAPQ = {s: [p["question"] for p in jl(f"{DATA}/{s}_1500.jsonl")] for s in ("pqq", "pqp")} if ds == "pq" else {}
cv = lambda y: StratifiedKFold(5, shuffle=True, random_state=0)
def ll(y, p): return log_loss(y, np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1])
def select(y_all, elig, folds_fn, fit_pred, grid):
    """fit_pred(train_rows, pred_rows, hp) -> P for pred_rows. Returns P over N (fit on all eligible train rows), oof over N, hp, cv ll, cv auc."""
    rows = tr[elig[tr]]; y = y_all[rows]
    if y.sum() < 8 or (1 - y).sum() < 8: return None
    best = None
    for hp in grid:
        oof = np.zeros(len(rows))
        for a, b in cv(y).split(rows, y): oof[b] = fit_pred(rows[a], rows[b], hp)
        s = ll(y, oof)
        if best is None or s < best[0]: best = (s, hp, oof)
    P = fit_pred(rows, np.arange(N), best[1]); oofN = np.full(N, np.nan); oofN[rows] = best[2]
    return dict(p=P, oof=oofN, hp=best[1], cv_ll=best[0], cv_auc=roc_auc_score(y, best[2]), n_fit=len(rows), rate=float(y.mean()))
def lr_fp(Z, y_all, Zext=None):
    def f(a, b, c, ext=False):
        m = LogisticRegression(C=c, max_iter=3000).fit(Z[a], y_all[a]); return m.predict_proba(Zext if ext else Z[b])[:, 1]
    return f
def knn_fp(S, y_all, Sext=None):
    """S: N x N similarity (rows = query, cols = all questions); only columns in the fitting rows are used as neighbours."""
    def f(a, b, k, ext=False):
        Sq = (Sext if ext else S[b])[:, a].copy()
        if not ext:
            Sq[np.isin(b, a)[:, None] & (b[:, None] == a[None, :])] = -np.inf   # never use itself
        nn = np.argpartition(-Sq, kth=min(k, Sq.shape[1] - 1) - 1, axis=1)[:, :k]; p0 = y_all[a].mean()
        return (y_all[a][nn].sum(1) + 2 * p0) / (k + 2)
    return f
def mlp_fp(Z, y_all, Zext=None):
    def f(a, b, al, ext=False):
        m = MLPClassifier(hidden_layer_sizes=(64,), alpha=al, solver="lbfgs", max_iter=400, random_state=0).fit(Z[a], y_all[a])
        return m.predict_proba(Zext if ext else Z[b])[:, 1]
    return f
def hidden(layer):
    z = np.load(f"{BASE}/{st}_{dsn}_1500_hidden.npz"); assert (z["idx"] == np.arange(N)).all(); L = list(z["layers"])
    li = L.index(layer if layer > 0 else L[-1]); H = z["h"][:, li, :].astype(np.float32); del z
    mu = H[tr].mean(0); sd = H[tr].std(0) + 1e-3; H = (H - mu) / sd
    Z = PCA(256, random_state=0).fit(H[tr]).transform(H).astype(np.float32); Z /= Z[tr].std(0) + 1e-6
    return Z, (L[-1] if layer <= 0 else layer)
for kind in KINDS:
    t0 = time.time(); out = dict(st=st, ds=ds, kind=kind, targets={}); ext = {}
    if kind.startswith("lin") and kind != "lin16p64":
        L_ = kind[3:]; Z, lay = hidden(-1 if L_ == "last" else int(L_)); out["layer"] = lay
        mk = lambda y: lr_fp(Z, y); grid = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)
    elif kind == "mlp16":
        Z, _ = hidden(16); Z = Z[:, :64]; mk = lambda y: mlp_fp(Z, y); grid = (1.0, 10.0, 100.0)
    elif kind == "lin16p64":   # control for mlp16: the same 64-component input, linear
        Z, _ = hidden(16); Z = np.ascontiguousarray(Z[:, :64]); mk = lambda y: lr_fp(Z, y); grid = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
    elif kind == "knn16":
        Z, _ = hidden(16); Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True); S = Zn @ Zn.T; mk = lambda y: knn_fp(S, y); grid = (5, 10, 20, 40, 80)
    elif kind == "tfidf":
        qs = B["question"]; vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([qs[i] for i in tr])
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit([qs[i] for i in tr])
        tf = lambda L: hstack([vw.transform(L), vc.transform(L)]).tocsr(); Z = tf(qs)
        for s, L in CAPQ.items(): ext[s] = tf(L)
        mk = lambda y: lr_fp(Z, y); grid = (0.3, 1.0, 3.0, 10.0, 30.0)
    elif kind == "knnE":
        e = np.load(f"{CACHE}/emb_{dsn}.npz"); assert (e["idx"] == np.arange(N)).all(); E = e["e"]; S = E @ E.T
        for s in CAPQ: ext[s] = np.load(f"{CACHE}/emb_{s}.npz")["e"] @ E.T
        mk = lambda y: knn_fp(S, y); grid = (5, 10, 20, 40, 80)
    else: raise ValueError(kind)
    for name, (y, elig) in T.items():
        if name.startswith("fr_") and kind not in FR_KINDS: continue
        fp = mk(y); r = select(y, elig, cv, fp, grid)
        if r is None: continue
        if ext:
            rows = tr[elig[tr]]; r["cap"] = {}
            for s, M in ext.items():
                if kind == "tfidf": r["cap"][s] = LogisticRegression(C=r["hp"], max_iter=3000).fit(Z[rows], y[rows]).predict_proba(M)[:, 1]
                else: r["cap"][s] = knn_fp(S, y, M)(rows, None, r["hp"], ext=True)
        out["targets"][name] = r
    pickle.dump(out, open(f"{CACHE}/pred_{st}_{ds}_{kind}.pkl", "wb"))
    nv = B["new"] & v
    msg = []
    for name in ("Q_a35", "M_a35", "Q_t50", "key", "aud"):
        r = out["targets"].get(name)
        if r is None: continue
        y = T[name][0]; m = nv if name != "aud" else B["new"]
        msg.append(f"{name} cv {r['cv_auc']:.3f} new {roc_auc_score(y[m], r['p'][m]):.3f} hp {r['hp']}")
    print(f"{st} {ds} {kind} ({time.time() - t0:.0f}s, {len(out['targets'])} targets): " + " | ".join(msg), flush=True)
