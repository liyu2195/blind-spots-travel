"""(verify_where) Item 1: are judges' false rejects question-level and false accepts answer-level? Own re-derivation from raw files.
Data: untrained students' 32 sampled valid answers on the TRAINED questions (Sonnet verdict for every unique string there).
Truth: aud = loose key OR Sonnet >= 4 (as claimed); also key-only, Sonnet-only, consensus (key and Sonnet agree; disagreements dropped).
Graders: judges Q M G O R QS at P(Yes) > 0.5 (t50), a50, a35 thresholds (final/thresholds_<J>.json).
(A) eta2 = between-question share of the sample-level verdict variance, vs rate-matched answer-level nulls
    (per-sample random; per (question, string) hash; 5 hash seeds averaged).
(B) one-way ICC(1) of the error indicator, unit = unique (question, normalised string), grouped by question (groups >= 2):
    FR among right strings, FA among wrong strings (the claimed statistic), per truth definition.
(C) rate-matched: per judge and cell, threshold t_r with FNR = r for FR and t'_r with FPR = r for FA (r = 0.05, 0.10, 0.20, 0.30); ICC at equal rates.
(D) threshold-free: ICC of the rank-normalised logit P(Yes) among right strings vs among wrong strings (the latent variable of both error indicators).
(E) group composition: FR and FA ICC computed on the SAME questions (>= 2 right and >= 2 wrong strings); and paraphrase level: ICC grouped by
    (question, answer cluster) vs between clusters of the same question (cluster means grouped by question).
(F) error sharing between judges (phi) for FR vs FA at a35, t50 and at matched rates (each judge thresholded to FNR = FPR = r on that cell).
Output: v2_out.txt, v2_rows.pkl.  Usage: nice -n 10 python3 v2_judge_errors.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, hashlib, itertools, numpy as np
from scipy.stats import norm, rankdata
C = (_RA + "/verify_where/cache"); OUT = (_RA + "/verify_where")
JUD = ["Q", "M", "G", "O", "R", "QS"]; J5 = ["Q", "M", "G", "O", "R"]
CELLS = [(st, d) for d in ("pqa", "nqo", "tqa") for st in ("qwen7", "llama8")]

def icc1(y, g):
    """one-way random-effects ICC(1) (ANOVA estimator) of y grouped by g; groups with < 2 members dropped."""
    y = np.asarray(y, float); g = np.asarray(g)
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True); keep = cnt[inv] >= 2
    y, g = y[keep], g[keep]
    if len(y) < 20: return np.nan
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True); k = len(cnt); n = len(y)
    if k < 3: return np.nan
    gm = np.bincount(inv, weights=y) / cnt; m = y.mean()
    msb = (cnt * (gm - m) ** 2).sum() / (k - 1); msw = ((y - gm[inv]) ** 2).sum() / (n - k)
    n0 = (n - (cnt ** 2).sum() / n) / (k - 1); den = msb + (n0 - 1) * msw
    return (msb - msw) / den if den > 0 else np.nan

def eta2(y, g):
    y = np.asarray(y, float); tot = ((y - y.mean()) ** 2).sum()
    if tot == 0: return np.nan
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True); gm = np.bincount(inv, weights=y) / cnt
    return (cnt * (gm - y.mean()) ** 2).sum() / tot

def hu(*p): return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:12], 16) / 16 ** 12
def lg(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def rn(x):  # rank-normal scores
    return norm.ppf((rankdata(x) - 0.5) / len(x))
def phi(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

rows = []; lines = []
def P(s=""): print(s, flush=True); lines.append(s)
P(__doc__)
for st, dsn in CELLS:
    B = pickle.load(open(f"{C}/base_{st}_{dsn}.pkl", "rb"))
    tr = np.where(B["train"])[0]
    # sample-level arrays (trained questions, sampled valid answers)
    qi, ki = np.nonzero(B["valid"][tr][:, 1:]); qi = tr[qi]; ki = ki + 1
    xs = [B["X"][i][k] for i, k in zip(qi, ki)]
    T = {"aud": B["kl"][qi, ki] | (B["son"][qi, ki] >= 4), "key": B["kl"][qi, ki], "son": B["son"][qi, ki] >= 4}
    cons_ok = B["kl"][qi, ki] == (B["son"][qi, ki] >= 4)
    PJ = {J: B["PJ"][J][qi, ki] for J in JUD}
    # unique (question, string) units: mean P(Yes) over occurrences; cluster id of the string
    ukey = {}; uq = []; ux = []; ucl = []
    uid = np.zeros(len(qi), int)
    for n_, (i, x, k) in enumerate(zip(qi, xs, ki)):
        kk = (int(i), x)
        if kk not in ukey: ukey[kk] = len(uq); uq.append(int(i)); ux.append(x); ucl.append(int(B["clu"][i, k]))
        uid[n_] = ukey[kk]
    uq = np.array(uq); ucl = np.array(ucl); nu = len(uq); cnt = np.bincount(uid, minlength=nu)
    def umean(v): return np.bincount(uid, weights=v.astype(float), minlength=nu) / cnt
    UT = {t: umean(T[t]) > 0.5 for t in T}; ucons = umean(cons_ok) > 0.5
    UP = {J: umean(PJ[J]) for J in JUD}
    TH = B["TH"]
    P(f"\n== {st} {dsn}: {len(qi)} valid samples on {len(np.unique(qi))} trained questions, {nu} unique strings; right share (samples) aud {T['aud'].mean():.3f} key {T['key'].mean():.3f} son {T['son'].mean():.3f}; "
      f"key/Sonnet disagree on {(~cons_ok).mean():.3f} of samples")
    P(f"   truth eta2 (aud) {eta2(T['aud'], qi):.3f}")
    for J in JUD:
        for rule in ("t50", "a50", "a35"):
            t = TH[J][rule]; V = PJ[J] > t; Ta = T["aud"]
            fnr = (~V & Ta).sum() / Ta.sum(); fpr = (V & ~Ta).sum() / (~Ta).sum()
            rng = np.random.default_rng(7); Vr = np.where(Ta, rng.random(len(Ta)) >= fnr, rng.random(len(Ta)) < fpr)
            eh = []
            for seed in range(5):
                h = np.array([hu(seed, i, x) for i, x in zip(uq, ux)])[uid]; eh.append(eta2(np.where(Ta, h >= fnr, h < fpr), qi))
            d = dict(st=st, ds=dsn, J=J, rule=rule, A=V.mean(), FNR=fnr, FPR=fpr, eta=eta2(V, qi), eta_rand=eta2(Vr, qi), eta_hash=float(np.mean(eh)), eta_truth=eta2(Ta, qi))
            # (B) unique-string ICC per truth definition
            Vu = UP[J] > t
            for tn in ("aud", "key", "son", "cons"):
                Tu = UT["aud"] if tn == "cons" else UT[tn]; m = ucons if tn == "cons" else np.ones(nu, bool)
                R_ = m & Tu; W_ = m & ~Tu
                d[f"iccFR_{tn}"] = icc1(~Vu[R_], uq[R_]); d[f"iccFA_{tn}"] = icc1(Vu[W_], uq[W_])
                d[f"uFNR_{tn}"] = (~Vu[R_]).mean(); d[f"uFPR_{tn}"] = Vu[W_].mean()
            rows.append(d)
        # (C) rate-matched thresholds on unique strings (aud truth)
        R_ = UT["aud"]; W_ = ~UT["aud"]; pr = UP[J][R_]; pw = UP[J][W_]
        for r in (0.05, 0.10, 0.20, 0.30):
            tfr = np.quantile(pr, r); tfa = np.quantile(pw, 1 - r)
            rows.append(dict(st=st, ds=dsn, J=J, rule=f"rate{r:.2f}", iccFR_aud=icc1(pr < tfr, uq[R_]), iccFA_aud=icc1(pw > tfa, uq[W_]),
                             uFNR_aud=(pr < tfr).mean(), uFPR_aud=(pw > tfa).mean()))
        # (D) threshold-free latent ICC (rank-normal logit P(Yes)) among right vs among wrong strings
        d = dict(st=st, ds=dsn, J=J, rule="latent", iccFR_aud=icc1(rn(lg(pr)), uq[R_]), iccFA_aud=icc1(rn(lg(pw)), uq[W_]))
        # (E) same questions only; and paraphrase vs between-cluster structure (latent score)
        qs_r = set(np.unique(uq[R_], return_counts=True)[0][np.unique(uq[R_], return_counts=True)[1] >= 2])
        qs_w = set(np.unique(uq[W_], return_counts=True)[0][np.unique(uq[W_], return_counts=True)[1] >= 2])
        both = np.array([q in qs_r and q in qs_w for q in uq])
        d["n_both_q"] = len(qs_r & qs_w)
        d["iccFR_same"] = icc1(rn(lg(UP[J][R_ & both])), uq[R_ & both]); d["iccFA_same"] = icc1(rn(lg(UP[J][W_ & both])), uq[W_ & both])
        t35 = TH[J]["a35"]
        d["iccFR_same35"] = icc1(UP[J][R_ & both] <= t35, uq[R_ & both]); d["iccFA_same35"] = icc1(UP[J][W_ & both] > t35, uq[W_ & both])
        zr = rn(lg(pr)); zw = rn(lg(pw))
        d["iccFR_withinclu"] = icc1(zr, uq[R_] * 1000 + ucl[R_]); d["iccFA_withinclu"] = icc1(zw, uq[W_] * 1000 + ucl[W_])
        # between clusters of the same question: cluster means of the latent score, grouped by question
        for lab, msk, z in (("FR", R_, zr), ("FA", W_, zw)):
            ck = uq[msk] * 1000 + ucl[msk]; u_, inv = np.unique(ck, return_inverse=True); cm = np.bincount(inv, weights=z) / np.bincount(inv)
            d[f"icc{lab}_betweenclu"] = icc1(cm, u_ // 1000); d[f"n{lab}_multiclu_q"] = int((np.unique(u_ // 1000, return_counts=True)[1] >= 2).sum())
        rows.append(d)
    # (F) error sharing between judges: phi of FR (right strings) and FA (wrong strings), a35 / t50 / matched rates
    R_ = UT["aud"]; W_ = ~UT["aud"]
    for rule in ("a35", "t50", "rate0.10", "rate0.20"):
        fr, fa = [], []
        for a, b in itertools.combinations(J5, 2):
            if rule.startswith("rate"):
                r = float(rule[4:])
                ea = UP[a][R_] < np.quantile(UP[a][R_], r); eb = UP[b][R_] < np.quantile(UP[b][R_], r)
                fa_a = UP[a][W_] > np.quantile(UP[a][W_], 1 - r); fa_b = UP[b][W_] > np.quantile(UP[b][W_], 1 - r)
            else:
                ea = UP[a][R_] <= TH[a][rule]; eb = UP[b][R_] <= TH[b][rule]; fa_a = UP[a][W_] > TH[a][rule]; fa_b = UP[b][W_] > TH[b][rule]
            fr.append(phi(ea, eb)); fa.append(phi(fa_a, fa_b))
        rows.append(dict(st=st, ds=dsn, J="pairs", rule=f"phi_{rule}", phiFR=np.nanmean(fr), phiFA=np.nanmean(fa)))
pickle.dump(rows, open(f"{OUT}/v2_rows.pkl", "wb"))
# ---------------- summaries
def sel(rule, J=None): return [r for r in rows if r["rule"] == rule and (J is None or r["J"] == J) and r["J"] != "pairs"]
P("\n==== (A) between-question share of verdict variance (36 judge x cell values per rule)")
for rule in ("t50", "a50", "a35"):
    s = sel(rule); e = np.array([r["eta"] for r in s]); h = np.array([r["eta_hash"] for r in s]); rr = np.array([r["eta_rand"] for r in s])
    P(f"  {rule}: eta2 mean {e.mean():.3f} (min {e.min():.3f} max {e.max():.3f}); hash null {h.mean():.3f}; random null {rr.mean():.3f}; judge - hash null mean {np.mean(e - h):+.3f} "
      f"(min {np.min(e - h):+.3f} max {np.max(e - h):+.3f}, > 0 in {(e > h).sum()}/{len(e)}); FNR {np.mean([r['FNR'] for r in s]):.3f} FPR {np.mean([r['FPR'] for r in s]):.3f}")
a35 = {(r["st"], r["ds"], r["J"]): r["eta"] for r in sel("a35")}; t50 = {(r["st"], r["ds"], r["J"]): r["eta"] for r in sel("t50")}
P(f"  a35 > t50 in {sum(a35[k] > t50[k] for k in a35)}/{len(a35)} cells; truth eta2 range {min(r['eta_truth'] for r in sel('a35')):.3f}-{max(r['eta_truth'] for r in sel('a35')):.3f}")
P("\n==== (B) unique-string ICC of false rejects vs false accepts, by truth definition (mean over 36 judge x cell; FR > FA count)")
for rule in ("t50", "a50", "a35"):
    for tn in ("aud", "key", "son", "cons"):
        s = sel(rule); a = np.array([r[f"iccFR_{tn}"] for r in s]); b = np.array([r[f"iccFA_{tn}"] for r in s]); ok = np.isfinite(a) & np.isfinite(b)
        P(f"  {rule} truth={tn:4s}: ICC FR {np.mean(a[ok]):.3f}  FA {np.mean(b[ok]):.3f}  FR > FA in {(a[ok] > b[ok]).sum()}/{ok.sum()}; unique-string rates FNR {np.mean([r[f'uFNR_{tn}'] for r in s]):.3f} FPR {np.mean([r[f'uFPR_{tn}'] for r in s]):.4f}")
P("\n==== (C) rate-matched thresholds (unique strings, aud truth): FR at FNR = r vs FA at FPR = r")
for r_ in ("rate0.05", "rate0.10", "rate0.20", "rate0.30"):
    s = sel(r_); a = np.array([r["iccFR_aud"] for r in s]); b = np.array([r["iccFA_aud"] for r in s]); ok = np.isfinite(a) & np.isfinite(b)
    P(f"  {r_}: ICC FR {np.mean(a[ok]):.3f}  FA {np.mean(b[ok]):.3f}  FR > FA in {(a[ok] > b[ok]).sum()}/{ok.sum()}  (realised FNR {np.mean([r['uFNR_aud'] for r in s]):.3f}, FPR {np.mean([r['uFPR_aud'] for r in s]):.3f})")
P("\n==== (D, E) threshold-free latent score ICC and group composition")
s = sel("latent")
for k_, lab in (("iccFR_aud", "latent ICC among right strings (FR side)"), ("iccFA_aud", "latent ICC among wrong strings (FA side)"),
                ("iccFR_same", "same questions only, latent, right"), ("iccFA_same", "same questions only, latent, wrong"),
                ("iccFR_same35", "same questions only, a35 FR indicator"), ("iccFA_same35", "same questions only, a35 FA indicator"),
                ("iccFR_withinclu", "within answer cluster (paraphrases), right"), ("iccFA_withinclu", "within answer cluster (paraphrases), wrong"),
                ("iccFR_betweenclu", "between clusters of a question, right"), ("iccFA_betweenclu", "between clusters of a question, wrong")):
    v = np.array([r[k_] for r in s], float); ok = np.isfinite(v)
    P(f"  {lab:48s} mean {np.mean(v[ok]):.3f} (n {ok.sum()}, range {np.min(v[ok]):.2f}-{np.max(v[ok]):.2f})")
a = np.array([r["iccFR_aud"] for r in s]); b = np.array([r["iccFA_aud"] for r in s]); P(f"  latent FR > FA in {(a > b).sum()}/{len(a)}")
a = np.array([r["iccFR_same"] for r in s]); b = np.array([r["iccFA_same"] for r in s]); ok = np.isfinite(a) & np.isfinite(b); P(f"  same-question latent FR > FA in {(a[ok] > b[ok]).sum()}/{ok.sum()}; questions with >= 2 right and >= 2 wrong strings per cell: {sorted(set(r['n_both_q'] for r in s))}")
a = np.array([r["iccFR_same35"] for r in s]); b = np.array([r["iccFA_same35"] for r in s]); ok = np.isfinite(a) & np.isfinite(b); P(f"  same-question a35 indicator FR > FA in {(a[ok] > b[ok]).sum()}/{ok.sum()}")
P(f"  questions with >= 2 right clusters per cell: {sorted(set(r['nFR_multiclu_q'] for r in s))}; with >= 2 wrong clusters: {sorted(set(r['nFA_multiclu_q'] for r in s))}")
P("\n==== (F) error sharing between the 5 judges (mean phi over 10 pairs; 6 cells)")
for rule in ("a35", "t50", "rate0.10", "rate0.20"):
    s = [r for r in rows if r["rule"] == f"phi_{rule}"]
    P(f"  {rule:8s}: FR phi " + " ".join(f"{r['phiFR']:.2f}" for r in s) + " | FA phi " + " ".join(f"{r['phiFA']:.2f}" for r in s) + f" | FR > FA in {sum(r['phiFR'] > r['phiFA'] for r in s)}/6")
open(f"{OUT}/v2_out.txt", "w").write("\n".join(lines) + "\n")
