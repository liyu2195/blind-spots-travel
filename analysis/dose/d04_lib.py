"""(dose) Shared analysis pieces for d04-d06 (own code).
Population ('known-sure'): untrained greedy answer valid, not IDK, audited right (loose key OR Sonnet >= 4) and sure (>= 31/32 samples agree).
Runs: FK20, IDK credit 0.5, final checkpoint; runs whose abstention on all new questions is >= 0.95 are dropped (degenerate).
Arms (grader code -> own predictor target): judge<J><op> -> J_op (J in Q M G O R, op a35 / a50 / t50); sim00fn00 = key -> key; apiS55 = Sonnet ->
  son; planted PopQA arms -> their set (fnB80 / fnB50 / fnA80: relation-type set; fnH80 / fnH60: random half of each type; pl<arm>: copied sets);
  reference pool ('no-location' graders = key plus uniform random false rejects / accepts): sim<FP>fn<FN> and lenQ.
Generic covariates G (grader-free) for predictor kind k: conf, logit P_aud and logit P_sure of the layer-16 linear probes, and logit P_aud of kind k
  (aud = 'the untrained greedy answer is audited right', sure = '>= 31/32 samples agree'; fitted on training questions like every predictor).
  HOME analyses use the out-of-fold versions of every fitted predictor.
FWL: x_perp, y_perp = residuals of x (logit P_reject) and y (refusal) on [1, G] (weighted least squares); the adjusted slope per SD of x_perp is
  sum w x_perp y_perp / sum w x_perp^2 * sd(x_perp); partial correlation = sum w x_perp y_perp / sqrt(sum w x_perp^2 sum w y_perp^2)."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, pickle, numpy as np
sys.path.insert(0, (_RA + "/dose"))
from d00_common import *
KINDS = ("lin4", "lin8", "lin12", "lin16", "lin20", "lin24", "linlast", "lin16p64", "mlp16", "knn16", "tfidf", "knnE")
PLANT = {"fnB80": ("pl_type", 0.8, 0.2), "fnB50": ("pl_type", 0.5, 0.2), "fnA80": ("pl_type", 0.2, 0.8), "fnH80": ("pl_hid", 0.8, 0.2),
         "fnH60": ("pl_hid", 0.6, 0.0), "plrealM": ("pl_realM", 1.0, 0.0), "plrealO": ("pl_realO", 1.0, 0.0), "plscrM": ("pl_scrM", 1.0, 0.0),
         "plsharpM": ("pl_sharpM", 1.0, 0.0), "plsharpO": ("pl_sharpO", 1.0, 0.0)}
REFPAT = re.compile(r"^(sim\d\dfn\d\d|lenQ)$")
TH = thresholds()

def arm_of(g):
    m = re.fullmatch(r"judge([QMGOR])(a35|a50|t50)", g)
    if m: return f"J{m.group(1)}_{m.group(2)}", f"{m.group(1)}_{m.group(2)}"
    if g == "sim00fn00": return "key", "key"
    if g == "apiS55": return "son", "son"
    if g in PLANT: return g, PLANT[g][0]
    return None, None

def load_cell(st, ds, kinds=KINDS):
    B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
    PR = {k: pickle.load(open(f"{CACHE}/pred_{st}_{ds}_{k}.pkl", "rb"))["targets"] for k in kinds}
    N = B["N"]; v = B["valid"][:, 0]; aud = v & (B["kl"][:, 0] | (B["son"][:, 0] >= 4)); sure = v & (B["conf"] >= 31 / 32)
    D = dict(st=st, ds=ds, dsn=DSN[ds], B=B, PR=PR, N=N, v=v, aud=aud, ks=sure & aud, new=B["new"], train=B["train"])
    D["Unew"] = np.where(B["new"] & sure & aud)[0]; D["Uhome"] = np.where(B["train"] & sure & aud)[0]
    arms = {}; ref = []
    for n, r in sorted(R.items()):
        if r["c"] != "05": continue
        y = (r["o"] == 0).astype(float); a = float(y[B["new"]].mean())
        if a >= 0.95: continue
        rec = dict(run=n, seed=r["seed"], g=r["g"], y=y, a=a, o=r["o"])
        A, tgt = arm_of(r["g"])
        if A is not None: arms.setdefault(A, dict(target=tgt, runs=[]))["runs"].append(rec)
        if REFPAT.match(r["g"]): ref.append(rec)
    D["arms"] = arms; D["ref"] = ref
    return D

def verdicts(D, target):
    """actual verdicts of grader 'target' on every question's untrained answers: v_bin (greedy rejected), v_share (share of the 33 answers
    rejected among valid ones), v_py (-logit P(Yes) of the greedy answer; judges only). nan where undefined."""
    B = D["B"]; N = D["N"]; val = B["valid"]
    m = re.fullmatch(r"([QMGOR])_(a35|a50|t50)", target)
    if m:
        J, op = m.groups(); t = 0.5 if op == "t50" else TH[J][f"{D['st']}_{D['dsn']}"][f"t_{op}"]; P = B["PJ"][J]
        rej = (P <= t) & val; vb = np.where(val[:, 0], rej[:, 0], np.nan)
        vs = np.where(val.sum(1) > 0, rej.sum(1) / np.maximum(val.sum(1), 1), np.nan)
        return dict(v_bin=vb.astype(float), v_share=vs, v_py=np.where(val[:, 0], -logit(P[:, 0]), np.nan))
    if target == "key":
        rej = ~B["ks"] & val; return dict(v_bin=np.where(val[:, 0], rej[:, 0], np.nan).astype(float),
                                          v_share=np.where(val.sum(1) > 0, rej.sum(1) / np.maximum(val.sum(1), 1), np.nan))
    if target == "son":
        s = B["son"][:, 0]; return dict(v_bin=np.where(val[:, 0] & (s > 0), s < 4, np.nan).astype(float))
    if target == "pl_type":
        tr = D["train"]; kt = {r["idx"]: bool(r["knows"]) for r in jl(f"{DATA}/pqa_knows_type.jsonl")}
        types = {B["prop"][i] for i in np.where(tr)[0] if kt.get(int(i), False)}
        return dict(v_bin=np.array([B["prop"][i] in types for i in range(N)], float))
    if target in ("pl_realM", "pl_realO"):
        J = target[-1]; t = TH[J][f"{D['st']}_{D['dsn']}"]["t_a35"]
        return dict(v_bin=np.where(val[:, 0], B["PJ"][J][:, 0] <= t, np.nan).astype(float))
    return {}

def gen_cov(D, kind, U, home=False):
    """generic covariates on rows U (columns: conf, logit aud lin16, logit sure lin16, logit aud of kind)."""
    key = "oof" if home else "p"; PR = D["PR"]
    cols = [D["B"]["conf"][U], logit(PR["lin16"]["aud"][key][U]), logit(PR["lin16"]["sure"][key][U])]
    if kind not in ("lin16",) and kind in PR and "aud" in PR[kind]: cols.append(logit(PR[kind]["aud"][key][U]))
    return np.column_stack(cols)

def resid(M, Z, w):
    """weighted residuals of the columns of M (n x m) on [1, Z] (n x p)."""
    X = np.column_stack([np.ones(len(Z)), Z]); WX = X * w[:, None]
    beta = np.linalg.lstsq(X.T @ WX + 1e-9 * np.eye(X.shape[1]), WX.T @ M, rcond=None)[0]
    return M - X @ beta

def fwl(x, Y, G, w):
    """x: n, Y: n x runs, G: n x p, w: n weights -> per run: slope per SD of x_perp, partial correlation; also x_perp and Y_perp."""
    R = resid(np.column_stack([x, Y]), G, w); xp = R[:, 0]; Yp = R[:, 1:]
    sxx = (w * xp * xp).sum(); sd = np.sqrt(sxx / w.sum())
    sxy = (w[:, None] * xp[:, None] * Yp).sum(0); syy = (w[:, None] * Yp * Yp).sum(0)
    return sxy / sxx * sd, sxy / np.sqrt(sxx * np.maximum(syy, 1e-12)), xp, Yp

def wquantile_bins(x, w, nb=5):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(w[o]); cw /= cw[-1]; b = np.zeros(len(x), int)
    b[o] = np.minimum(nb - 1, (cw * nb - 1e-9).astype(int)); return b
