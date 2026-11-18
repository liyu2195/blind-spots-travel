"""(bothsigns) b05: where the readability -> transfer relation holds and where it fails.
(1) mean transfer by readability bin and sign (eligible arms of b04); (2) cell-level test inside the natural / intermediate arms (judges, copies of
judges' rejections, popularity graders, key misses): points = (arm, sign, student, dataset), T and readability per cell, Spearman with cell
bootstrap and measurement bootstrap; (3) same judge, same operating point, same student: false rejections vs false accepts in the same currency
(paired differences in readability and transfer); (4) placebo null check; (5) content channel of false accepts (changed answers only). Inputs: b04_table.csv, cache/b02_boot.pkl, cache/b02_armboot.pkl,
cache/b03_*.pkl. Output: b05_details_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, (_RA + "/bothsigns"))
from b00_common import OUT, STS, DSS

rng = np.random.default_rng(5)
A = pd.read_csv(f"{OUT}/b04_table.csv"); AB = pickle.load(open(f"{OUT}/cache/b02_armboot.pkl", "rb"))
D = pickle.load(open(f"{OUT}/cache/b02_boot.pkl", "rb")); U = D["units"]; B = D["boots"]
RD = []
for st in STS:
    for ds in DSS:
        RD += pickle.load(open(f"{OUT}/cache/b03_{st}_{ds}.pkl", "rb"))
out = open(f"{OUT}/b05_details_out.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
q = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))


def eligible(df, cur):
    return df[(~df.placebo) & np.isfinite(df[f"{cur}_h"]) & np.isfinite(df[f"{cur}_n"]) & (df[f"{cur}_h_lo"] > 0) & (df[f"{cur}_h"] >= 0.05) & np.isfinite(df.R_hidQA)]


# (1) bins
pr("(1) mean transfer T = new/home by readability bin (eligible arms; median T in brackets; n arms)")
bins = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.95), (0.95, 1.01)]
for cur in ("kind", "dec"):
    E = eligible(A, cur).copy(); E["T"] = E[f"{cur}_n"] / E[f"{cur}_h"]
    for sg in (["FR"], ["FA", "form"]):
        cells = []
        for lo, hi in bins:
            d = E[E.sign.isin(sg) & (E.R_hidQA >= lo) & (E.R_hidQA < hi)]
            cells.append(f"[{lo:.2f},{min(hi, 1):.2f}) " + (f"{d['T'].clip(-1, 2).mean():+.2f} ({d['T'].median():+.2f}) n {len(d)}" if len(d) else "--"))
        pr(f"  {cur:5s} {'+'.join(sg):8s} " + " | ".join(cells))

# (2) cell level inside natural arms
NAT = lambda a: a.startswith("judge") or a in ("plrealM", "plrealO", "popH40", "popL40", "keymiss")
rmap = defaultdict(list)
for r in RD:
    rmap[(r["arm"], r["sign"], r["st"], r["ds"])].append(r)
rows = []
for k, u in enumerate(U):
    if not NAT(u["arm"]) or "placebo" in u["sign"]:
        continue
    rr = rmap.get((u["arm"], u["sign"], u["st"], u["ds"]), [])
    rr = [r for r in rr if set(r["runs"]) == set(u["runs"])] or rr
    if not rr or not np.isfinite(rr[0].get("hidQA", np.nan)):
        continue
    for cur in ("kind", "dec"):
        h, n = u["pt"][0][f"{cur}_logit"], u["pt"][1][f"{cur}_logit"]
        bh, bn = B[f"{cur}_logit"][:, k, 0], B[f"{cur}_logit"][:, k, 1]
        lo = np.nanpercentile(bh, 2.5) if np.isfinite(bh).sum() > 20 else np.nan
        rows.append(dict(cur=cur, arm=u["arm"], sign=u["sign"], st=u["st"], ds=u["ds"], h=h, n=n, h_lo=lo, R=rr[0]["hidQA"], Rbs=rr[0]["hidQA_bs"], bh=bh, bn=bn))
C = pd.DataFrame(rows)
pr("\n(2) cell level, natural / intermediate arms only (judges, copies of judges' rejections, popularity graders, key misses); a cell enters if its home"
   " effect CI > 0 and >= 0.05")
for cur in ("kind", "dec"):
    E = C[(C.cur == cur) & np.isfinite(C.h) & np.isfinite(C.n) & (C.h_lo > 0) & (C.h >= 0.05)].reset_index(drop=True)
    for nm, sel in (("pooled", E.index >= 0), ("qwen7", E.st == "qwen7"), ("llama8", E.st == "llama8"), ("FR", E.sign == "FR"), ("FA", E.sign == "FA"),
                    ("PopQA", E.ds == "pq"), ("TriviaQA", E.ds == "h1"), ("NQ-open", E.ds == "nq")):
        d = E[sel].reset_index(drop=True)
        if len(d) < 6:
            pr(f"  {cur:5s} {nm:9s} n {len(d)} (too few)"); continue
        T = (d.n / d.h).values; R = d.R.values; rho = spearmanr(R, T).correlation
        ra = [spearmanr(R[i], T[i]).correlation for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
        rm = []
        for b in range(500):
            Tb = np.array([x[b] for x in d.bn]) / np.array([x[b] for x in d.bh]); Rb = np.array([x[b % 300] for x in d.Rbs])
            m = np.isfinite(Tb) & np.isfinite(Rb)
            if m.sum() >= 6:
                rm.append(spearmanr(Rb[m], Tb[m]).correlation)
        perm = np.array([spearmanr(R, rng.permutation(T)).correlation for _ in range(3000)]); p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / 3001
        pr(f"  {cur:5s} {nm:9s} n {len(d):2d} rho {rho:+.2f} cells [{q(ra)[0]:+.2f},{q(ra)[1]:+.2f}] measurement [{q(rm)[0]:+.2f},{q(rm)[1]:+.2f}] perm p {p:.4f}"
           f" | readability range {R.min():.2f}-{R.max():.2f}, median T {np.median(T):+.2f}")
C.drop(columns=["Rbs", "bh", "bn"]).to_csv(f"{OUT}/b05_cells.csv", index=False)

# (3) same judge, operating point, student: FR vs FA
pr("\n(3) same judge, same operating point, same student: false rejections minus false accepts (arm level, pooled over datasets)")
J = A[A.arm.str.startswith("judge") & A.sign.isin(["FR", "FA"])]
for cur in ("kind", "dec"):
    pairs = []
    for (arm, st), d in J.groupby(["arm", "st"]):
        if set(d.sign) != {"FR", "FA"}:
            continue
        fr = d[d.sign == "FR"].iloc[0]; fa = d[d.sign == "FA"].iloc[0]
        ok = lambda r: np.isfinite(r[f"{cur}_h"]) and r[f"{cur}_h_lo"] > 0 and r[f"{cur}_h"] >= 0.05 and np.isfinite(r[f"{cur}_n"])
        if not (ok(fr) and ok(fa)):
            continue
        pairs.append((arm, st, fr, fa))
    if not pairs:
        pr(f"  {cur}: no pair with both home effects reliable"); continue
    dT = np.array([p[2][f"{cur}_n"] / p[2][f"{cur}_h"] - p[3][f"{cur}_n"] / p[3][f"{cur}_h"] for p in pairs])
    dR = np.array([p[2].R_hidQA - p[3].R_hidQA for p in pairs])
    dH = np.array([p[2][f"{cur}_h"] - p[3][f"{cur}_h"] for p in pairs]); dN = np.array([p[2][f"{cur}_n"] - p[3][f"{cur}_n"] for p in pairs])
    bsT = []; bsR = []
    for b in range(2000):
        ii = rng.integers(0, len(pairs), len(pairs)); j = b % 500
        t = [AB[(p[0], "FR", p[1])][f"{cur}_n"][j] / AB[(p[0], "FR", p[1])][f"{cur}_h"][j] - AB[(p[0], "FA", p[1])][f"{cur}_n"][j] / AB[(p[0], "FA", p[1])][f"{cur}_h"][j] for p in pairs]
        bsT.append(np.nanmean(np.clip(np.array(t)[ii], -2, 2))); bsR.append(np.mean(dR[ii]))
    pr(f"  {cur:5s} pairs {len(pairs):2d}: readability FR - FA {dR.mean():+.3f} [{q(bsR)[0]:+.3f},{q(bsR)[1]:+.3f}] (FR higher in {np.mean(dR > 0):.2f});"
       f" transfer FR - FA {np.clip(dT, -2, 2).mean():+.2f} [{q(bsT)[0]:+.2f},{q(bsT)[1]:+.2f}] (FR higher in {np.mean(dT > 0):.2f}); home FR - FA {dH.mean():+.3f}, new FR - FA {dN.mean():+.3f}")
    for p, a, b_ in zip(pairs, dT, dR):
        pr(f"      {p[0]:10s} {p[1]:6s} T FR {p[2][f'{cur}_n'] / p[2][f'{cur}_h']:+.2f} FA {p[3][f'{cur}_n'] / p[3][f'{cur}_h']:+.2f} | R FR {p[2].R_hidQA:.2f} FA {p[3].R_hidQA:.2f}")

# (5) content channel of false accepts: share of the grader's paid kind among CHANGED wrong answers (final != untrained greedy)
pr("\n(5) FA / form content channel (changed wrong answers only), arm level: home, new, T; Spearman over FA / form arms with reliable home content effect")
E = A[(A.sign.isin(["FA", "form"])) & np.isfinite(A.kindc_h) & np.isfinite(A.kindc_n) & (A.kindc_h_lo > 0) & (A.kindc_h >= 0.05) & np.isfinite(A.R_hidQA)].copy()
E["T"] = E.kindc_n / E.kindc_h
for _, r in E.sort_values(["st", "R_hidQA"]).iterrows():
    pr(f"  {r.arm:10s} {r.st:6s} R {r.R_hidQA:.2f} content home {r.kindc_h:+.3f} [{r.kindc_h_lo:+.3f},{r.kindc_h_hi:+.3f}] new {r.kindc_n:+.3f} [{r.kindc_n_lo:+.3f},{r.kindc_n_hi:+.3f}] T {r['T']:+.2f}")
for nm, d in (("all FA+form", E), ("judges only", E[E.arm.str.startswith("judge")]), ("designed removed", E[~E.designed])):
    if len(d) >= 5:
        T = d["T"].values; R = d.R_hidQA.values; rho = spearmanr(R, T).correlation
        ra = [spearmanr(R[i], T[i]).correlation for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
        perm = np.array([spearmanr(R, rng.permutation(T)).correlation for _ in range(3000)]); p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / 3001
        pr(f"  Spearman {nm:17s} n {len(d):2d} rho {rho:+.2f} [{q(ra)[0]:+.2f},{q(ra)[1]:+.2f}] perm p {p:.4f}")
jj = E[E.arm.str.startswith("judge")]
pr(f"  judges: content new / home pooled = {jj.kindc_n.mean() / jj.kindc_h.mean():+.2f} (mean new {jj.kindc_n.mean():+.3f}, mean home {jj.kindc_h.mean():+.3f}, n {len(jj)}); "
   f"hash: {E[E.arm == 'hash35'].kindc_n.mean():+.3f} new vs {E[E.arm == 'hash35'].kindc_h.mean():+.3f} home")

# (4) placebo null check
pr("\n(4) placebo arms (random-noise graders on a pseudo-random half; reference = the other noise families): effects should be 0")
P = A[A.placebo]
for cur in ("kind", "dec"):
    vals = pd.concat([P[f"{cur}_h"], P[f"{cur}_n"]]).dropna()
    los = pd.concat([P[f"{cur}_h_lo"], P[f"{cur}_n_lo"]]); his = pd.concat([P[f"{cur}_h_hi"], P[f"{cur}_n_hi"]])
    cover = np.mean([(l <= 0 <= h) for l, h in zip(los, his) if np.isfinite(l) and np.isfinite(h)])
    pr(f"  {cur:5s} {len(vals)} placebo estimates (arm x part): mean {vals.mean():+.3f}, max |E| {vals.abs().max():.3f}, 95% CIs covering 0: {cover:.2f}; readability {P.R_hidQA.min():.2f}-{P.R_hidQA.max():.2f}")
out.close()
