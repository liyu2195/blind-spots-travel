"""(verify_where) Item 5: 'visibility law' for grader-specific selective refusal (false-reject maps). Own code from raw files.
Questions: untrained greedy answer valid and loose-key right; home = trained, new = held-out (duplicate-free).
Rejected set per grader: judges J at t (a35 / a50 / t50): >= half of the untrained strictly-right samples have P(Yes) <= t.
  Planted (PopQA): fnB80 / fnB50 (FNK on 'knows' types), fnA80 (reverse), fnH80 / fnH60 (random half: final/hid_halves.json on new),
  plrealJ (copied judge rejections; new = judge J's own a35 set), plsharpJ (type-wide; new = prop in scramble_sets types). Home sets from the
  trainer's knows files. plscrM has no new-question set (skipped).
Effect(run, part) = [IDK on rejected - IDK on kept] - reference, reference = the same contrast in uniform random-error key runs
  (sim<FP>fn<FN>, lenQ; non-collapsed) as a function of their IDK rate on that part: linear (claimed), quadratic, or mean of the 3 nearest.
Point = (student, set, grader code), >= 2 runs; transfer = mean new effect / mean home effect (home >= 0.05), plus alternatives.
Visibility = Gini (2 AUC - 1) of a model fitted on home questions (label = rejected) and scored on new: (text) TF-IDF of the question +
  untrained confidence + share of samples key-right, logistic C = 1; (hidden) layer-16 hidden state PCA 64, logistic C by CV.
Spearman(transfer, Gini) with bootstrap-over-points 95% CI; within judges only and within planted graders only.
Output: v7_out.txt, v7_points.csv"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, json, numpy as np, pandas as pd, warnings
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
warnings.filterwarnings("ignore")
C = (_RA + "/verify_where/cache"); OUT = (_RA + "/verify_where"); DATA = _RD; FIN = (_RX + "/final")
BASE = (_RR + "/run1/base"); CELLS = [(st, d) for st in ("qwen7", "llama8") for d in ("pqa", "nqo", "tqa")]
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
P(__doc__)
def jl(p): return [json.loads(l) for l in open(p)]
HID = json.load(open(f"{FIN}/hid_halves.json")); SCR = json.load(open(f"{FIN}/scramble_sets.json"))
pts = []
for st, dsn in CELLS:
    B = pickle.load(open(f"{C}/base_{st}_{dsn}.pkl", "rb")); R = pickle.load(open(f"{C}/runs_{st}_{dsn}.pkl", "rb"))
    v0 = B["valid"][:, 0]; right = v0 & B["kl"][:, 0]
    part = {"home": np.where(B["train"])[0], "new": np.where(B["held"])[0]}
    sr = B["ks"][:, 1:] & B["valid"][:, 1:]; nsr = sr.sum(1)
    def judge_rej(J, t):
        r = np.full(1500, np.nan); m = nsr > 0
        r[m] = ((B["PJ"][J][:, 1:] <= t) & sr).sum(1)[m] / nsr[m]; return r >= 0.5, m
    REJ = {}
    codes = sorted({r["g"] for r in R.values() if r["c"] == 5})
    for g in codes:
        m = re.fullmatch(r"judge([QMGOR])(a35|a50|t50)", g)
        if m:
            J, t = m.groups(); rj, ok = judge_rej(J, B["TH"][J][t]); REJ[g] = (rj, ok); continue
        if dsn != "pqa": continue
        prop = B["prop"]
        if g in ("fnB80", "fnB50", "fnA80"):
            kt = {x["idx"]: x["knows"] for x in jl(f"{DATA}/pqa_knows_type.jsonl")}; types = {prop[i] for i, k in kt.items() if k}
            kn = np.array([kt[i] if i in kt else (prop[i] in types) for i in range(1500)]); REJ[g] = (kn if g != "fnA80" else ~kn, np.ones(1500, bool))
        elif g in ("fnH80", "fnH60"):
            REJ[g] = (np.array([bool(HID[str(i)]) for i in range(1500)]), np.ones(1500, bool))
        elif g.startswith("plreal") or g.startswith("plsharp"):
            J = g[-1]; kf = {x["idx"]: x["knows"] for x in jl(f"{DATA}/pqa_knows_{g[2:]}_{st}.jsonl")}
            if g.startswith("plreal"): rjJ, okJ = judge_rej(J, B["TH"][J]["a35"]); newv = rjJ; ok = okJ.copy()
            else: ty = set(SCR[f"{st}|sharp{J}"]["types"]); newv = np.array([prop[i] in ty for i in range(1500)]); ok = np.ones(1500, bool)
            kn = np.array([kf[i] if i in kf else newv[i] for i in range(1500)]); ok[B["train"]] = True; REJ[g] = (kn, ok)
    # uniform reference runs
    UNI = [r for r in R.values() if r["c"] == 5 and (re.fullmatch(r"sim\d\dfn\d\d", r["g"]) or r["g"] == "lenQ")]
    for g, (rj, ok) in REJ.items():
        runs = [r for r in R.values() if r["c"] == 5 and r["g"] == g]
        if len(runs) < 2: continue
        rec = dict(st=st, ds=dsn, g=g, fam="judge" if g.startswith("judge") else "planted", nruns=len(runs))
        for pn, idx in part.items():
            q = idx[right[idx] & ok[idx]]; A = q[rj[q]]; K = q[~rj[q]]
            rec[f"nA_{pn}"] = len(A); rec[f"nK_{pn}"] = len(K)
            if len(A) < 15 or len(K) < 15: continue
            ua = np.array([(u["o"][idx] == 0).mean() for u in UNI]); uc = np.array([(u["o"][A] == 0).mean() - (u["o"][K] == 0).mean() for u in UNI])
            keep = ua < 0.95; ua, uc = ua[keep], uc[keep]
            b1 = np.polyfit(ua, uc, 1); b2 = np.polyfit(ua, uc, 2)
            effs = defaultdict(list)
            for r in runs:
                a = (r["o"][idx] == 0).mean()
                if a >= 0.95: continue
                con = (r["o"][A] == 0).mean() - (r["o"][K] == 0).mean()
                effs["lin"].append(con - np.polyval(b1, a)); effs["quad"].append(con - np.polyval(b2, a))
                effs["nn3"].append(con - uc[np.argsort(np.abs(ua - a))[:3]].mean()); effs["raw"].append(con)
            for k_, vv in effs.items(): rec[f"{k_}_{pn}"] = vv
            rec[f"refsd_{pn}"] = float(np.std(uc - np.polyval(b1, ua)))
        # visibility (text + confidence) on right-greedy questions with a defined label
        qh = part["home"][right[part["home"]] & ok[part["home"]]]; qn = part["new"][right[part["new"]] & ok[part["new"]]]
        yh = rj[qh].astype(int); yn = rj[qn].astype(int)
        if yh.sum() >= 10 and (1 - yh).sum() >= 10 and 0 < yn.sum() < len(yn):
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([B["question"][i] for i in qh])
            q32 = (B["kl"][:, 1:] & B["valid"][:, 1:]).mean(1)
            X = lambda q: hstack([vec.transform([B["question"][i] for i in q]), csr_matrix(np.c_[B["conf"][q], q32[q]])]).tocsr()
            m_ = LogisticRegression(C=1.0, max_iter=3000).fit(X(qh), yh); rec["auc_text"] = roc_auc_score(yn, m_.predict_proba(X(qn))[:, 1])
            rec["qh"] = qh; rec["qn"] = qn; rec["yh"] = yh; rec["yn"] = yn
        pts.append(rec)
    # hidden-state visibility (one file at a time)
    z = np.load(f"{BASE}/{st}_{dsn}_1500_hidden.npz"); H = z["h"][:, list(z["layers"]).index(16), :].astype(np.float32); del z
    tri = part["home"]; mu = H[tri].mean(0); sd = H[tri].std(0) + 1e-3; Zh = PCA(64, random_state=0).fit((H[tri] - mu) / sd).transform((H - mu) / sd); del H
    for rec in pts:
        if rec["st"] != st or rec["ds"] != dsn or "qh" not in rec: continue
        best = None
        for c in (0.003, 0.01, 0.03, 0.1, 0.3):
            o = cross_val_predict(LogisticRegression(C=c, max_iter=3000), Zh[rec["qh"]], rec["yh"], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
            a = roc_auc_score(rec["yh"], o); best = (a, c) if best is None or a > best[0] else best
        m_ = LogisticRegression(C=best[1], max_iter=3000).fit(Zh[rec["qh"]], rec["yh"]); rec["auc_hid"] = roc_auc_score(rec["yn"], m_.decision_function(Zh[rec["qn"]]))
    P(f"{st} {dsn}: graders with >= 2 runs and a rejection set: {sum(1 for p in pts if p['st'] == st and p['ds'] == dsn)}; uniform reference runs {len(UNI)}")
rows = []
for p in pts:
    if "lin_home" not in p or "lin_new" not in p or "auc_text" not in p: continue
    d = {k: p[k] for k in ("st", "ds", "g", "fam", "nruns", "nA_home", "nA_new", "auc_text", "auc_hid")}
    for ref in ("lin", "quad", "nn3", "raw"):
        h = np.mean(p[f"{ref}_home"]); n = np.mean(p[f"{ref}_new"]); d[f"home_{ref}"] = h; d[f"new_{ref}"] = n; d[f"T_{ref}"] = n / h if h >= 0.05 else np.nan
    hh = np.array(p["lin_home"]); nn = np.array(p["lin_new"]); kk = min(len(hh), len(nn))
    d["T_runratio"] = np.mean(nn[:kk] / hh[:kk]) if (hh[:kk] >= 0.05).all() else np.nan
    d["refsd_new"] = p.get("refsd_new"); d["gini_text"] = 2 * d["auc_text"] - 1; d["gini_hid"] = 2 * d["auc_hid"] - 1
    rows.append(d)
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/v7_points.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
P(D[["st", "ds", "g", "nruns", "nA_home", "nA_new", "home_lin", "new_lin", "T_lin", "T_quad", "T_nn3", "gini_text", "gini_hid"]].sort_values("gini_text").round(3).to_string(index=False))
rng = np.random.default_rng(3)
def sp(d, x, y, B=2000):
    d = d[np.isfinite(d[x]) & np.isfinite(d[y])]
    if len(d) < 5: return f"n {len(d)}"
    r = spearmanr(d[x], d[y])[0]; bs = []
    for _ in range(B):
        i = rng.integers(0, len(d), len(d)); s = spearmanr(d[x].values[i], d[y].values[i])[0]
        if np.isfinite(s): bs.append(s)
    perm = np.mean([abs(spearmanr(d[x], rng.permutation(d[y].values))[0]) >= abs(r) for _ in range(2000)])
    return f"n {len(d):3d} Spearman {r:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] perm p {perm:.4f}"
P("\n==== Spearman(transfer, visibility)")
for lab, sub in (("all points", D), ("judges only", D[D.fam == "judge"]), ("planted only", D[D.fam == "planted"]), ("qwen7", D[D.st == "qwen7"]), ("llama8", D[D.st == "llama8"]),
                 ("judges, a35 only", D[D.g.str.endswith("a35") & (D.fam == "judge")]), ("judges, PopQA excluded", D[(D.fam == "judge") & (D.ds != "pqa")])):
    P(f"  {lab:24s} text-Gini: {sp(sub, 'gini_text', 'T_lin')} | hidden-Gini: {sp(sub, 'gini_hid', 'T_lin')}")
P("\n  transfer definitions and reference curves (all points, text-Gini):")
for tcol in ("T_lin", "T_quad", "T_nn3", "T_raw", "T_runratio", "new_lin"):
    P(f"    {tcol:11s} all: {sp(D, 'gini_text', tcol)} | judges only: {sp(D[D.fam == 'judge'], 'gini_text', tcol)}")
P(f"\n  judges only: Spearman(new effect, home effect) {spearmanr(D[D.fam == 'judge'].home_lin, D[D.fam == 'judge'].new_lin)[0]:+.2f}; "
  f"Spearman(gini_text, home effect) {spearmanr(D[D.fam == 'judge'].gini_text, D[D.fam == 'judge'].home_lin)[0]:+.2f}; Spearman(gini_text, dataset is PopQA) {spearmanr(D[D.fam == 'judge'].gini_text, D[D.fam == 'judge'].ds == 'pqa')[0]:+.2f}")
dj = D[D.fam == "judge"]
for ds in ("pqa", "nqo", "tqa"):
    s = dj[dj.ds == ds]; P(f"    judges within {ds}: {sp(s, 'gini_text', 'T_lin', 500)}")
P(f"\n  noise floor: typical sd of the uniform-run contrast around the reference line on new questions: {np.nanmedian(D.refsd_new):.3f} (single run)")
open(f"{OUT}/v7_out.txt", "w").write("\n".join(lines) + "\n")
