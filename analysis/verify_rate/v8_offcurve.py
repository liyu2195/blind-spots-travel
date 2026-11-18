"""(verify_rate) Off-curve graders. (1) Residual (actual - predicted new-question abstention) from the qwen7-uniform leniency curve, per grader,
with per-(student, dataset) cell values. (2) The offered explanation: false accepts / rejections ALIGNED with what the student knows. Per run, from the
untrained samples on trained questions: AUROC of acceptance among key-wrong samples against the question's confidence (modal-answer share of the 32
samples), and AUROC of rejection among key-right samples against (1 - confidence); 0.5 = unaligned (any uniform grader). Does the residual follow the
alignment across non-uniform graders, and within LLM judges? (3) long80: are the trained answers long (the form the grader pays)? Leniency if every
answer were long. (4) Self-keys: share of trained questions with no key at all.
Usage: nice -n 10 python3 -I v8_offcurve.py > out_v8_offcurve.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, numpy as np, pandas as pd, warnings
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from collections import Counter
warnings.filterwarnings("ignore")
sys.path.insert(0, (_RA + "/verify_rate"))
from vr import base, ans, IDKF, RL, DATA, DSN, jl, C
HERE = (_RA + "/verify_rate")
T = pd.read_pickle(f"{HERE}/cache/table.pkl"); W = pd.read_pickle(f"{HERE}/cache/worth.pkl")
M = W[(W.c == 0.5)].copy()
f = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
QU = M[M.uni & (M.st == "qwen7")]; p0 = curve_fit(f, QU.x, QU.idk_n, p0=[0, -8])[0]
M["res"] = M.idk_n - f(M.x, *p0)
M["Wbs8"] = M.Wbs8.astype(float); QU = M[M.uni & (M.st == "qwen7")]
pw = curve_fit(f, QU.Wbs8.values, QU.idk_n.values, p0=[3, -6], maxfev=50000)[0]; M["res_wbs8"] = M.idk_n - f(M.Wbs8.values, *pw)
print("Wbs8 curve check: params", np.round(pw, 2), "in-sample R2", 1 - np.sum((QU.idk_n - f(QU.Wbs8.values, *pw))**2) / np.sum((QU.idk_n - QU.idk_n.mean())**2))
print("== (1) residual from the qwen7-uniform curve (a %.2f b %.2f) by grader; cell means (seed means per student x dataset)" % tuple(p0))
rng = np.random.default_rng(5)
for sub, g in M[~M.uni].groupby("sub"):
    cells = g.groupby(["st", "ds"]).res.mean(); v = cells.values
    bs = [rng.choice(v, len(v)).mean() for _ in range(4000)] if len(v) > 1 else [np.nan]
    print(f"  {sub:12s} runs {len(g):2d} cells {len(v)}  mean resid {g.res.mean():+.3f} [cell boot {np.nanpercentile(bs, 2.5):+.3f}, {np.nanpercentile(bs, 97.5):+.3f}]  "
          f"(Wbs8 curve {g.res_wbs8.mean():+.3f})  cells: " + " ".join(f"{a[0][0]}{a[1]}:{b:+.2f}" for a, b in cells.items()))
# (2) alignment AUROCs
def auroc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float)
    if y.all() or (~y).all(): return np.nan
    r = pd.Series(s).rank().values; n1 = y.sum(); n0 = (~y).sum(); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
CF = {}
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        B = base(st, ds); m = B["home"] & (B["s"] >= 0); conf = {}
        for i in B["H"]:
            pass
        df = pd.DataFrame(dict(i=B["idx"][m], x=np.array(B["X"], object)[m], ok=(B["valid"] & ~B["idk"])[m]))
        mod = df[df.ok].groupby("i").x.agg(lambda s: Counter(s).most_common(1)[0][1] / 32.0)
        CF[(st, ds)] = (B, mod)
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("v1b", f"{HERE}/v1_build.py")
print("\n== (2) alignment of the grader's verdicts with the student's confidence (untrained samples, trained questions)")
# re-implement acceptance per run quickly from v1's cached per-question arrays is not enough (need per sample): recompute with the same rules
def accept_samples(B, r):
    """acceptance per sample for the grader kinds needed here (judge / api / synth incl. frozen keys, hash, long)."""
    idx, X = B["idx"], B["X"]; app = B["valid"] & ~B["idk"]; p = np.full(len(idx), np.nan); k = r.sp_kind
    if k == "judge":
        pj = B["PJ"].get(r.sp_J);  ok = app & np.isfinite(pj); p[ok] = (pj[ok] > r.sp_t).astype(float); return p
    if k == "api":
        lv = {}; [lv.setdefault(kk, v) for src in ("H", "A") for kk, v in B["SON"].get(src, {}).items()]
        for j in np.where(app)[0]:
            v = lv.get((int(idx[j]), X[j]))
            if v is not None: p[j] = float(v >= 4)
        return p
    if k != "synth": return p
    al = {q["idx"]: set(q["aliases"]) for q in jl(f"{DATA}/{r.sp_alias}")} if isinstance(r.sp_alias, str) else None
    kn = {q["idx"]: bool(q["knows"]) for q in jl(f"{DATA}/{r.sp_knowsf}")}
    from vr import hunit, ANY
    for j in np.where(app)[0]:
        i = int(idx[j]); x = X[j]
        if al is not None and i not in al: continue
        right = ((ANY in al[i]) or C.tq_strict(x, al[i])) if al is not None else bool(B["strict"][j])
        kk = kn.get(i, False)
        fnk = r.sp_fnk if r.sp_fnk == r.sp_fnk and r.sp_fnk is not None else None
        if right:
            fn = r.sp_fn if fnk is None else (r.sp_fnk if kk else r.sp_fnu)
            p[j] = (0.0 if (fn > 0 and hunit("fn", r.seed, i, x) < fn) else 1.0) if r.sp_mode == "hash" else 1.0 - fn
        else:
            fv = r.sp_fpk if kk else r.sp_fpu
            p[j] = (1.0 if hunit("fp", r.seed, i, x) < fv else 0.0) if r.sp_mode == "hash" else ((fv if B["words"][j] >= 5 else 0.0) if r.sp_mode == "long" else fv)
    return p
rows = []
for (st, ds), g in M[~M.uni].groupby(["st", "ds"]):
    B, mod = CF[(st, ds)]; m = B["home"] & (B["s"] >= 0) & B["valid"] & ~B["idk"]
    cq = pd.Series(B["idx"]).map(mod).fillna(0).values
    seen = {}
    for r in g.itertuples():
        key = (r.sub, r.seed if r.sp_mode == "hash" else 0)
        if key not in seen:
            p = accept_samples(B, r); ok = m & np.isfinite(p)
            wr = ok & ~B["strict"]; rt = ok & B["strict"]
            seen[key] = (auroc(cq[wr], p[wr] > 0.5) if r.sp_kind in ("judge", "api") or r.sp_mode == "hash" or isinstance(r.sp_alias, str) else (auroc(cq[wr], p[wr] > 0.5) if r.sp_mode == "long" else 0.5),
                         auroc(1 - cq[rt], p[rt] < 0.5) if (r.sp_kind in ("judge", "api") or r.sp_mode == "hash" or isinstance(r.sp_alias, str)) else 0.5)
        rows.append(dict(run=r.run, st=st, ds=ds, sub=r.sub, fam=r.fam, res=r.res, res_wbs8=r.res_wbs8, auc_fa=seen[key][0], auc_fr=seen[key][1]))
A = pd.DataFrame(rows); A.to_csv(f"{HERE}/offcurve_alignment.csv", index=False)
cells = A.groupby(["st", "ds", "sub"]).agg(fam=("fam", "first"), res=("res", "mean"), res_wbs8=("res_wbs8", "mean"), auc_fa=("auc_fa", "mean"), auc_fr=("auc_fr", "mean")).reset_index()
print(cells.groupby("sub")[["res", "auc_fa", "auc_fr"]].mean().round(3).sort_values("res").to_string())
def sp(d, a, b):
    d = d[np.isfinite(d[a]) & np.isfinite(d[b])]; return spearmanr(d[a], d[b])[0], len(d)
for nm, d in (("all non-uniform cells (det. verdicts only)", cells[cells.fam.isin(["judge", "api", "frz", "hash"])]), ("LLM-judge cells only", cells[cells.fam == "judge"]),
              ("non-judge cells (api, frz, hash)", cells[cells.fam.isin(["api", "frz", "hash"])])):
    a1 = sp(d, "auc_fa", "res"); a2 = sp(d, "auc_fr", "res")
    print(f"  {nm:44s} Spearman(residual, FA alignment) {a1[0]:+.2f}  (residual, FR alignment) {a2[0]:+.2f}  n cells {a1[1]}")
# (3) long80: answer length of the final answers
print("\n== (3) long80: share of answered (non-IDK, valid) final greedy answers with >= 5 words, home / new; untrained greedy for reference")
for (st, ds), g in T[T["std"] & T.st.isin(["qwen7", "llama8"]) & (T["sub"].isin(["long80", "FP00FN00"]) | (T["sub"] == "judgeQt50"))].groupby(["st", "ds"]):
    B, _ = CF[(st, ds)]; gi = B["s"] < 0
    out = []
    for part in ("home", "new"):
        mm = gi & B[part] & B["valid"] & ~B["idk"]; out.append(f"untrained {part} {np.mean(B['words'][mm] >= 5):.2f}")
    for r in g.itertuples():
        G = {q["idx"]: ans(q) for q in map(json.loads, open(f"{RL}/{r.run}/eval_rl_ck120.jsonl")) if q.get("s", -1) < 0}
        for part, items in (("home", B["H"]), ("new", B["NEW"])):
            xs = [G.get(i, "") for i in items]; xs = [x for x in xs if x and not IDKF.match(x)]
            out.append(f"{r['sub'] if False else r.sub}/s{r.seed} {part} {np.mean([len(x.split()) >= 5 for x in xs]):.2f}")
    print(f"  {st} {ds}: " + " | ".join(out))
L8 = M[M["sub"] == "long80"]
print("  long80 leniency as measured (L) vs if every answer were >= 5 words (L_long = acc + (1 - acc) * 0.8):")
for r in L8.itertuples():
    Ll = r.acch + (1 - r.acch) * 0.8
    print(f"    {r.run:36s} L {r.L32h:.3f} -> pred {f(r.x, *p0):.3f} | L_long {Ll:.3f} -> pred {f(Ll - 0.5, *p0):.3f} | actual {r.idk_n:.3f}")
# (4) self-keys: share of trained questions with no key
print("\n== (4) self-keys: share of trained questions whose frozen key accepts nothing; per-question acceptance bimodality (share of questions with acceptance 0 or >= 0.9)")
Q = pickle.load(open(f"{HERE}/cache/qarr.pkl", "rb"))
for sub, g in M[M.fam.isin(["frz"]) | M["sub"].isin(["FP35FN00", "FP00FN00", "judgeQa35"])].groupby("sub"):
    r = g.iloc[0]; qm = Q[r.run]["q_mean"]
    nk = np.nan
    if isinstance(r.sp_alias, str):
        al = jl(f"{DATA}/{r.sp_alias}"); nk = np.mean([a["aliases"] == ["zzznokeyzzz"] for a in al])
    print(f"  {sub:10s} (e.g. {r.run}) no-key share {nk:.2f}  L {g.L32h.mean():.3f}  per-question acceptance: share 0 {np.mean(qm == 0):.2f}, share >= .9 {np.mean(qm >= 0.9):.2f}, W {g.W.mean():.3f}  resid {g.res.mean():+.3f}")
