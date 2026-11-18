"""(judges) Q1: is a grader's verdict / mistake about the QUESTION or about the particular ANSWER?
Data: untrained students' 32 sampled valid answers on the TRAINED questions (where Sonnet verdicts exist for every unique answer).
Truth T = loose key OR Sonnet level >= 4 ("audited"). Graders: judges Q M G O R QS at threshold 0.5 (t50) and at their label-free
50% / 35% acceptance thresholds (a50, a35; thresholds_<J>.json), the strict key (trainer's gold reward), and two nulls with the judge's
own error rates: per-sample random (sim-style) and per-(question, answer string) hash (hash-style).
Measures (all on trained questions, valid sampled answers):
  A, FNR = P(reject | T), FPR = P(accept | not T);
  ICC_FN = one-way ICC of the false-reject indicator among right answers, by question (sample level, and unique answer strings);
  ICC_FP = the same for false accepts among wrong answers;
  allnone_FN = share of false rejects that sit on questions where the grader rejects EVERY right sample;
  eta2_V = between-question share of the verdict variance; cl_V = between-cluster-within-question share; res_V = within-cluster share.
Usage: nice -n 10 python3 j1_decomp.py > j1_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, numpy as np, hashlib
HERE = (_RA + "/judges")
JUD = ["Q", "M", "G", "O", "R", "QS"]
TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JUD}

def icc(y, g):
    """one-way random-effects ICC(1) of y grouped by g (groups with >= 2 members)."""
    y = np.asarray(y, float); g = np.asarray(g)
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2; y = y[keep]; g = g[keep]
    if len(y) < 10: return np.nan
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    k = len(u); N = len(y); m = y.mean()
    gm = np.bincount(inv, weights=y) / cnt
    ssb = (cnt * (gm - m) ** 2).sum(); ssw = ((y - gm[inv]) ** 2).sum()
    if k < 2 or N - k < 1: return np.nan
    msb = ssb / (k - 1); msw = ssw / (N - k); n0 = (N - (cnt ** 2).sum() / N) / (k - 1)
    d = msb + (n0 - 1) * msw
    return (msb - msw) / d if d > 0 else np.nan

def shares(y, q, c):
    """three-level sums of squares: question / cluster within question / residual (shares of the total)."""
    y = np.asarray(y, float); m = y.mean(); tot = ((y - m) ** 2).sum()
    if tot == 0: return (np.nan,) * 3
    _, iq = np.unique(q, return_inverse=True); qm = np.bincount(iq, weights=y) / np.bincount(iq)
    key = q.astype(np.int64) * 1000 + c.astype(np.int64); _, ic = np.unique(key, return_inverse=True)
    cm = np.bincount(ic, weights=y) / np.bincount(ic)
    ssq = ((qm[iq] - m) ** 2).sum(); ssc = ((cm[ic] - qm[iq]) ** 2).sum(); ssr = ((y - cm[ic]) ** 2).sum()
    return ssq / tot, ssc / tot, ssr / tot

def hu(*p): return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:12], 16) / 16 ** 12

def allnone(rej, q):
    """share of rejections (rej True) that are on questions where every member is rejected (questions with >= 2 members)."""
    u, inv, cnt = np.unique(q, return_inverse=True, return_counts=True)
    frac = np.bincount(inv, weights=rej.astype(float)) / cnt
    ok = (cnt[inv] >= 2)
    return (rej & ok & (frac[inv] == 1)).sum() / max(1, (rej & ok).sum())

def uniq_level(err, q, X, mask):
    """ICC on unique (question, answer string) pairs: one row per distinct string (verdicts are per string anyway)."""
    seen = {}; ys = []; gs = []
    for k in np.where(mask)[0]:
        key = (int(q[k]), X[k])
        if key in seen: continue
        seen[key] = 1; ys.append(err[k]); gs.append(int(q[k]))
    return icc(ys, gs), len(ys)

def run(st, dsn, out):
    D = pickle.load(open(f"{HERE}/cache_{st}_{dsn}.pkl", "rb"))
    tr = D["Q"]["train"][D["idx"]]; m = tr & D["valid"] & (D["s"] >= 0)
    q = D["idx"][m]; c = D["clu"][m]; X = [D["X"][k] for k in np.where(m)[0]]
    T = (D["kl"] | (D["son"] >= 4))[m]; ks = D["ks"][m]
    rows = []
    def measure(name, V):
        V = np.asarray(V, bool); fn = ~V & T; fp = V & ~T
        r = dict(g=name, A=V.mean(), FNR=fn[T].mean(), FPR=fp[~T].mean(),
                 iccFN=icc(fn[T], q[T]), iccFP=icc(fp[~T], q[~T]),
                 iccFNu=uniq_level(fn, q, X, T)[0], iccFPu=uniq_level(fp, q, X, ~T)[0],
                 allnoneFN=allnone(fn[T], q[T]), allnoneFP=allnone(fp[~T], q[~T]))
        r["eta2_V"], r["cl_V"], r["res_V"] = shares(V, q, c)
        rows.append(r); return r
    measure("truth(audited)", T)
    measure("key_strict", ks)
    for J in JUD:
        P = D["PJ"][J][m]; th = TH[J][f"{st}_{dsn}"]
        for tag, t in (("t50", 0.5), ("a50", th["t_a50"]), ("a35", th["t_a35"])):
            V = P > t; r = measure(f"{J}{tag}", V)
            if J in ("Q", "M") and tag in ("t50", "a35"):
                # nulls with the same error rates: per-sample random, and per-(question, string) hash
                rng = np.random.default_rng(7)
                Vr = np.where(T, rng.random(len(T)) >= r["FNR"], rng.random(len(T)) < r["FPR"]); measure(f"  null_rand({J}{tag})", Vr)
                h = np.array([hu("x", int(q[k]), X[k]) for k in range(len(q))])
                Vh = np.where(T, h >= r["FNR"], h < r["FPR"]); measure(f"  null_hash({J}{tag})", Vh)
    print(f"\n== {st} {dsn}: trained questions, {m.sum()} valid sampled answers, {len(np.unique(q))} questions, audited-right share {T.mean():.3f}", file=out)
    print(f"{'grader':24s} {'A':>5s} {'FNR':>5s} {'FPR':>5s} | {'iccFN':>6s} {'iccFNu':>6s} {'allN_FN':>7s} | {'iccFP':>6s} {'iccFPu':>6s} {'allN_FP':>7s} | {'q_V':>5s} {'cl_V':>5s} {'res_V':>5s}", file=out)
    for r in rows:
        print(f"{r['g']:24s} {r['A']:5.3f} {r['FNR']:5.3f} {r['FPR']:5.3f} | {r['iccFN']:6.3f} {r['iccFNu']:6.3f} {r['allnoneFN']:7.3f} | {r['iccFP']:6.3f} {r['iccFPu']:6.3f} {r['allnoneFP']:7.3f} | {r['eta2_V']:5.3f} {r['cl_V']:5.3f} {r['res_V']:5.3f}", file=out)
    return rows

if __name__ == "__main__":
    out = sys.stdout; R = {}
    for dsn in ("pqa", "nqo", "tqa"):
        for st in ("qwen7", "llama8"):
            R[(st, dsn)] = run(st, dsn, out)
    pickle.dump(R, open(f"{HERE}/j1_rows.pkl", "wb"))
