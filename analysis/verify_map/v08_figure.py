"""(verify_map) Proposed central figure: 'the student inherits the judge's READABLE accept/reject map, in both directions'.
Design (no reference runs, no extrapolation): judge-vs-judge contrast on NEW questions.
  For each (student, dataset) and each unordered pair of judges (K, J) with strict (a35) runs (FK20, credit 0.5, new abstention < 0.95):
  x = z-scored [logit P_K(reject) - logit P_J(reject)] from the layer-16 linear probes fitted ONLY on training-question verdicts (v02)
      ('K's readable map says reject, relative to J's');
  y = R_K - R_J, the difference in final refusal rate between K-trained and J-trained runs on the question.
  FWL within pair: x and y residualised on [1, conf, logit P_aud, logit P_sure] (grader-free; removes the pair's abstention gap and linear generic
  difficulty). Panels: RIGHT = untrained greedy answer audited right (judge rejections are false rejections; prediction: y rises with x);
  WRONG = audited wrong (judge acceptances are false accepts; prediction: y rises with x = the K-run answers where K's map says pay).
  Curve: mean y per quintile of x within pair; slope = OLS slope of y on x (refusal points per SD). Pooling: pairs within cell, cells within student,
  students averaged. Orientation of a pair does not matter (x and y both flip sign).
  Also: the same on TRAINING questions (out-of-fold probes) and with the actual verdict contrast (only-K-rejects minus only-J-rejects, i.e. the
  crossover X) for comparison; placebo = judge labels permuted over the cell's a35 runs (500 draws).
  Variants: no FWL; known-sure right answers only; near-duplicates removed; best probe (p64).
Bootstrap: 1000 draws (questions within cell, runs within judge). Usage: python3 v08_figure.py main|var|var2 NB NPERM -> v08_figure_<which>_out.txt, cache/v08_fig_<which>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, itertools, time
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
P = Log(f"{HERE}/v08_figure_out.txt"); P(__doc__); t00 = time.time()
rng = np.random.default_rng(88)

def prep(st, ds, part="new", probe="lin16"):
    D = load(st, ds); P16 = pickle.load(open(f"{CACHE}/p_{st}_{ds}_L16.pkl", "rb"))["T"]; Q2 = pickle.load(open(f"{CACHE}/q_{st}_{ds}.pkl", "rb"))
    key = "p" if part == "new" else "oof"; base = D["new"] if part == "new" else D["train"]
    cd0 = D["cd"][:, 0]; v = np.isin(cd0, (1, 2)); s0 = D["son0"]
    masks = {"right": base & v & ((cd0 == 1) | (s0 >= 4)), "wrong": base & v & (cd0 == 2) & (s0 >= 1) & (s0 <= 3),
             "ks": base & v & ((cd0 == 1) | (s0 >= 4)) & (D["conf"] >= 31 / 32),
             "right_nodup": base & v & ((cd0 == 1) | (s0 >= 4)) & ~D["neardup"], "wrong_nodup": base & v & (cd0 == 2) & (s0 >= 1) & (s0 <= 3) & ~D["neardup"]}
    Y = {}
    for n, r in sorted(D["runs"].items()):
        m = re.fullmatch(r"judge([QMGOR])a35", r["g"])
        if not m or (r["o"][D["new"]] == 0).mean() >= 0.95: continue
        Y.setdefault(m.group(1), []).append((r["o"] == 0).astype(float))
    Y = {J: np.array(x) for J, x in Y.items()}
    src = P16 if probe == "lin16" else Q2[probe]
    pr = {J: logit(src[f"rej_{J}_a35"][key]) for J in Y}
    rej = {J: D["PJ"][J][:, 0] <= thr(J, st, ds, "a35") for J in Y}
    G = np.column_stack([D["conf"], logit(P16["aud"][key]), logit(P16["sure"][key])])
    return dict(masks={k: np.where(m)[0] for k, m in masks.items()}, Y=Y, pr=pr, rej=rej, G=G)

def resid(M, G, w):
    X1 = np.column_stack([np.ones(len(G)), G]); WX = X1 * w[:, None]
    return M - X1 @ np.linalg.solve(X1.T @ WX + 1e-9 * np.eye(X1.shape[1]), WX.T @ M)

def wq_bins(x, w, nb=5):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(w[o]); cw /= cw[-1]; b = np.zeros(len(x), int); b[o] = np.minimum(nb - 1, (cw * nb - 1e-9).astype(int)); return b

def pair_stats(c, panel, K, J, w, sel, fwl=True, Yover=None):
    U = c["masks"][panel]; Yd = c["Y"] if Yover is None else Yover
    yK = (Yd[K] if sel is None else Yd[K][sel[K]]).mean(0)[U]; yJ = (Yd[J] if sel is None else Yd[J][sel[J]]).mean(0)[U]
    y = yK - yJ; x = c["pr"][K][U] - c["pr"][J][U]
    if fwl:
        R = resid(np.column_stack([x, y]), c["G"][U], w); x, y = R[:, 0], R[:, 1]
    else:
        x = x - (w * x).sum() / w.sum(); y = y - (w * y).sum() / w.sum()
    sdx = np.sqrt((w * x * x).sum() / w.sum()); x = x / sdx
    slope = (w * x * y).sum() / (w * x * x).sum()
    b = wq_bins(x, w); curve = np.array([(w[b == j] * y[b == j]).sum() / w[b == j].sum() for j in range(5)])
    # actual-verdict crossover on the same units: [y | only K rejects] - [y | only J rejects] (raw y, not FWL)
    yr = yK - yJ; oK = c["rej"][K][U] & ~c["rej"][J][U]; oJ = c["rej"][J][U] & ~c["rej"][K][U]
    wa, wb = w * oK, w * oJ
    X = (yr @ wa / wa.sum() - yr @ wb / wb.sum()) if (oK.sum() >= 5 and oJ.sum() >= 5 and wa.sum() > 0 and wb.sum() > 0) else np.nan
    return slope, curve, X

def cell_stats(C, panel, W=None, SEL=None, fwl=True, YO=None):
    out = {}
    for k, c in C.items():
        n = len(c["masks"][panel]); w = np.ones(n) if W is None else W[k][c["masks"][panel]]
        res = [pair_stats(c, panel, K, J, w, None if SEL is None else SEL[k], fwl, None if YO is None else YO[k])
               for K, J in itertools.combinations([J for J in J5 if J in c["Y"]], 2)]
        out[k] = (np.mean([r[0] for r in res]), np.mean([r[1] for r in res], 0), np.nanmean([r[2] for r in res]) if any(np.isfinite(r[2]) for r in res) else np.nan,
                  len(res), sum(r[0] > 0 for r in res))
    return out

def pooled(S, i, who=None):
    per = []
    for s in ([who] if who else STS):
        v_ = [S[k][i] for k in S if k[0] == s and np.all(np.isfinite(S[k][i]))]
        if v_: per.append(np.mean(v_, 0))
    return np.mean(per, 0) if per else np.nan

def analyse(label, part="new", probe="lin16", panels=("right", "wrong"), fwl=True, NB=1000, NPERM=500, keep=None, dss=DSS):
    C = {(st, ds): prep(st, ds, part, probe) for st in STS for ds in dss}
    P(f"\n== {label} (part {part}, probe {probe}, fwl {fwl}) ({time.time() - t00:.0f}s)")
    for panel in panels:
        S0 = cell_stats(C, panel, fwl=fwl); B = []
        for b in range(NB):
            W = {k: np.bincount(rng.integers(0, 1500, 1500), minlength=1500).astype(float) for k in C}   # question weights (resampling all 1,500 ids)
            SEL = {k: {J: rng.integers(0, len(c["Y"][J]), len(c["Y"][J])) for J in c["Y"]} for k, c in C.items()}
            B.append(cell_stats(C, panel, W, SEL, fwl))
        NUL = []
        for b in range(NPERM):
            YO = {}
            for k, c in C.items():
                labs = [J for J in c["Y"] for _ in range(len(c["Y"][J]))]; rows = np.vstack([c["Y"][J] for J in c["Y"]]); perm = rng.permutation(len(labs)); d = {}
                for lab, j in zip(labs, perm): d.setdefault(lab, []).append(rows[j])
                YO[k] = {J: np.array(x) for J, x in d.items()}
            NUL.append(cell_stats(C, panel, fwl=fwl, YO=YO))
        sl = pooled(S0, 0); slb = np.array([pooled(s, 0) for s in B]); cv = pooled(S0, 1); cvb = np.array([pooled(s, 1) for s in B])
        X = pooled(S0, 2); Xb = np.array([pooled(s, 2) for s in B])
        nul = np.array([pooled(s, 0) for s in NUL]); pp = (np.sum(np.abs(nul) >= abs(sl)) + 1) / (len(nul) + 1)
        lo, hi = np.nanpercentile(cvb, [2.5, 97.5], axis=0)
        npairs = sum(S0[k][3] for k in S0); npos = sum(S0[k][4] for k in S0)
        P(f"  [{panel:11s}] units " + "/".join(str(len(c["masks"][panel])) for c in C.values()) + f" | slope per SD {sl:+.4f} [{np.nanpercentile(slb, 2.5):+.4f}, {np.nanpercentile(slb, 97.5):+.4f}]"
          f" (qwen7 {pooled(S0, 0, 'qwen7'):+.4f}, llama8 {pooled(S0, 0, 'llama8'):+.4f}); pairs with slope > 0: {npos}/{npairs}; permutation p {pp:.4f} (null sd {nul.std():.4f})")
        P(f"      quintile curve of R_K - R_J (refusal points): " + "  ".join(f"Q{j + 1} {cv[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}]" for j in range(5))
          + f" | Q5 - Q1 {cv[4] - cv[0]:+.3f} [{np.nanpercentile(cvb[:, 4] - cvb[:, 0], 2.5):+.3f}, {np.nanpercentile(cvb[:, 4] - cvb[:, 0], 97.5):+.3f}]")
        P(f"      per cell slope: " + " ".join(f"{k[0]}/{k[1]} {S0[k][0]:+.3f} ({S0[k][4]}/{S0[k][3]})" for k in S0)
          + f" | actual-verdict crossover X (only K rejects - only J rejects; pairs with >= 5 each) {X:+.3f} [{np.nanpercentile(Xb, 2.5):+.3f}, {np.nanpercentile(Xb, 97.5):+.3f}]")
        if keep is not None: keep[(label, panel)] = dict(slope=sl, slope_ci=np.nanpercentile(slb, [2.5, 97.5]), curve=cv, curve_lo=lo, curve_hi=hi, X=X, p=pp,
                                                          per_student={s: pooled(S0, 0, s) for s in STS}, per_cell={k: S0[k][0] for k in S0})

WHICH = sys.argv[1]; NBm = int(sys.argv[2]); NPm = int(sys.argv[3])
P.path = f"{HERE}/v08_figure_{WHICH}_out.txt"; FIG = {}
if WHICH == "main": analyse("MAIN: new questions", NB=NBm, NPERM=NPm, keep=FIG)
if WHICH == "var":
    analyse("no FWL", fwl=False, NB=NBm, NPERM=NPm, keep=FIG)
    analyse("known-sure / near-dup removed", panels=("ks", "right_nodup", "wrong_nodup"), NB=NBm, NPERM=NPm, keep=FIG)
if WHICH == "var2":
    analyse("best probe p64", probe="p64", NB=NBm, NPERM=NPm, keep=FIG)
    analyse("HOME: training questions, out-of-fold probes", part="home", NB=NBm, NPERM=NPm, keep=FIG)
if WHICH == "byds":
    analyse("NQ-open only (judges pay 8-16% of wrong answers at a35)", dss=("nq",), NB=NBm, NPERM=NPm, keep=FIG)
    analyse("TriviaQA + PopQA only (judges pay 0-6% of wrong answers at a35)", dss=("h1", "pq"), NB=NBm, NPERM=NPm, keep=FIG)
pickle.dump(FIG, open(f"{CACHE}/v08_fig_{WHICH}.pkl", "wb")); P.save()
