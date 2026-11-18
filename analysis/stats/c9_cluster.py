"""(stats) Claim 9: judges' false rejections are tied to the question, false accepts to the answer.
Data: the untrained students' 32 sampled valid (non-IDK) answers on the TRAINED questions; unit = unique (question, normalised string), P(Yes) =
mean over its occurrences; truth 'aud' = loose key OR Sonnet >= 4 (frontierH / frontierA); a unit is right if the majority of its occurrences are.
For judge J (Q M G O R QS) and cell (student, dataset): FR indicator among right units = P(Yes) <= t_J(a35); FA indicator among wrong units =
P(Yes) > t_J(a35). ICC(1) (one-way ANOVA estimator, groups = questions with >= 2 units of that kind).
PRIMARY estimand: D = mean over the 6 judges x 3 datasets of [ICC_FR - ICC_FA] at the a35 cut, per student and pooled.
Uncertainty: question-cluster bootstrap (2000) within (student, dataset), the same draw for every judge (judges share questions and strings);
  a resampled question enters as a new group each time it is drawn.
Robustness: rate-matched cuts (FNR = FPR = 0.10 / 0.20 per judge and cell); threshold-free latent score (rank-normal logit P(Yes)) among right vs
  wrong units; key-only truth; questions with >= 2 right AND >= 2 wrong units only; the default cut P(Yes) > 0.5.
Output: c9_cluster_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
from collections import defaultdict
from scipy.stats import norm, rankdata
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/c9_cluster_out.txt"); P(__doc__)
rng = np.random.default_rng(909)
TH = {J: thresholds(J) for J in JALL}
def lg(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def rn(x): return norm.ppf((rankdata(x) - 0.5) / len(x))
def gstats(y, g):
    """per-group (n, sum, sumsq) for groups with >= 2 members; y float, g int group labels (question ids)."""
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    s1 = np.bincount(inv, weights=y); s2 = np.bincount(inv, weights=y * y); keep = cnt >= 2
    return u[keep], cnt[keep].astype(float), s1[keep], s2[keep]
def icc_w(n, s1, s2, c):
    """ICC(1) with group replication counts c."""
    k = c.sum(); N = (c * n).sum()
    if k < 3 or N < 20: return np.nan
    gm = s1 / n; m = (c * s1).sum() / N
    msb = (c * n * (gm - m) ** 2).sum() / (k - 1); msw = (c * (s2 - n * gm ** 2)).sum() / (N - k)
    n0 = (N - (c * n ** 2).sum() / N) / (k - 1); den = msb + (n0 - 1) * msw
    return (msb - msw) / den if den > 0 else np.nan
CELL = {}
for st in STS:
    for ds in DSS:
        dsn = DSN[ds]; B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb"))
        tr = np.where(B["train"])[0]
        qi, ki = np.nonzero(B["valid"][tr][:, 1:]); qi = tr[qi]; ki = ki + 1
        key = {}; uq = []; uid = np.zeros(len(qi), int)
        for n_, (i, k) in enumerate(zip(qi, ki)):
            kk = (int(i), B["X"][i][k])
            if kk not in key: key[kk] = len(uq); uq.append(int(i))
            uid[n_] = key[kk]
        uq = np.array(uq); nu = len(uq); cnt = np.bincount(uid, minlength=nu)
        um = lambda v: np.bincount(uid, weights=v.astype(float), minlength=nu) / cnt
        T = {"aud": um(B["kl"][qi, ki] | (B["son"][qi, ki] >= 4)) > 0.5, "key": um(B["kl"][qi, ki]) > 0.5}
        UP = {J: um(np.nan_to_num(B["PJ"][J][qi, ki], nan=0.5)) for J in JALL}
        CELL[(st, ds)] = dict(uq=uq, T=T, UP=UP, thr={J: TH[J][f"{st}_{dsn}"]["t_a35"] for J in JALL}, nq=len(tr), tr=tr)
        P(f"{st} {ds}: {len(qi)} valid samples, {nu} unique strings on {len(np.unique(uq))} questions; right share (units, aud) {T['aud'].mean():.3f}")
def indicators(c, J, variant):
    """returns (y_FR, g_FR, y_FA, g_FA) for one judge in one cell under a variant"""
    tn = "key" if variant == "key" else "aud"; R_ = c["T"][tn]; W_ = ~R_; p = c["UP"][J]; uq = c["uq"]
    if variant in ("a35", "key", "same"): t = c["thr"][J]; yr = (p[R_] <= t).astype(float); yw = (p[W_] > t).astype(float)
    elif variant == "t50": yr = (p[R_] <= 0.5).astype(float); yw = (p[W_] > 0.5).astype(float)
    elif variant.startswith("rate"):
        r = float(variant[4:]); yr = (p[R_] < np.quantile(p[R_], r)).astype(float); yw = (p[W_] > np.quantile(p[W_], 1 - r)).astype(float)
    elif variant == "latent": yr = rn(lg(p[R_])); yw = rn(lg(p[W_]))
    gr, gw = uq[R_], uq[W_]
    if variant == "same":
        qr, cr = np.unique(gr, return_counts=True); qw, cw = np.unique(gw, return_counts=True)
        both = set(qr[cr >= 2]) & set(qw[cw >= 2]); mr = np.isin(gr, list(both)); mw = np.isin(gw, list(both))
        yr, gr, yw, gw = yr[mr], gr[mr], yw[mw], gw[mw]
    return yr, gr, yw, gw
def prepare(variant):
    out = {}
    for (st, ds), c in CELL.items():
        for J in JALL:
            yr, gr, yw, gw = indicators(c, J, variant)
            out[(st, ds, J)] = (gstats(yr, gr), gstats(yw, gw))
    return out
def stat(PR, Wq=None):
    res = {}
    for (st, ds, J), ((ur, nr, s1r, s2r), (uw, nw, s1w, s2w)) in PR.items():
        if Wq is None: cr = np.ones(len(ur)); cw = np.ones(len(uw))
        else: w = Wq[(st, ds)]; cr = w[ur]; cw = w[uw]
        res[(st, ds, J)] = (icc_w(nr, s1r, s2r, cr), icc_w(nw, s1w, s2w, cw))
    return res
def agg(res, who):
    v = [(a, b) for (st, ds, J), (a, b) in res.items() if who in ("pooled", st, ds, J) and np.isfinite(a) and np.isfinite(b)]
    return np.mean([a for a, b in v]), np.mean([b for a, b in v]), np.mean([a - b for a, b in v]), sum(a > b for a, b in v), len(v)
def report(variant, NB=2000):
    PR = prepare(variant); res = stat(PR); reps = defaultdict(list)
    for b in range(NB):
        Wq = {}
        for (st, ds), c in CELL.items():
            w = np.zeros(1500); w[c["tr"]] = np.bincount(rng.integers(0, c["nq"], c["nq"]), minlength=c["nq"]); Wq[(st, ds)] = w
        rb = stat(PR, Wq)
        for who in ("qwen7", "llama8", "pooled") + DSS + JALL: reps[who].append(agg(rb, who)[2])
    P(f"\n== {variant}: ICC_FR vs ICC_FA (mean over judge x dataset cells) [95% question-cluster bootstrap of the difference]")
    out = {}
    for who in ("qwen7", "llama8", "pooled"):
        fr, fa, d, k, n = agg(res, who); ci = pct(reps[who]); out[who] = (d, ci, fr, fa, k, n)
        P(f"  {who:6s} FR {fr:.3f}  FA {fa:.3f}  diff {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  FR > FA in {k}/{n};  bootstrap P(diff <= 0) {np.mean(np.array(reps[who]) <= 0):.4f}")
    P("  per dataset: " + "  ".join(f"{ds} {agg(res, ds)[2]:+.3f} [{pct(reps[ds])[0]:+.3f}, {pct(reps[ds])[1]:+.3f}]" for ds in DSS))
    P("  per judge:   " + "  ".join(f"{J} {agg(res, J)[2]:+.3f} [{pct(reps[J])[0]:+.3f}, {pct(reps[J])[1]:+.3f}]" for J in JALL))
    return out
MAIN = report("a35")
ROB = {}
for v in ("rate0.10", "rate0.20", "latent", "key", "same", "t50"):
    ROB[v] = report(v, 1000)
pickle.dump(dict(MAIN=MAIN, ROB=ROB), open(f"{HERE}/c9_summary.pkl", "wb"))
P.save()
