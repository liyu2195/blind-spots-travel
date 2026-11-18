"""(bothsigns) b04: does one monotone relation between readability and transfer hold across error signs?
Inputs: b02_arms.csv + cache/b02_armboot.pkl (effects, bootstrap draws), cache/b03_<st>_<ds>.pkl (readability per cell / unit).
Arm readability = mean over the arm's cells (units) of the probe AUC on new questions (cells without enough affected items skipped).
Transfer T = E_new / E_home in the kind currency (primary: defined for FR, FA and form) and the decision currency (FR and FA).
An arm enters a T test only if its home effect is reliably positive and non-trivial (95% CI lower bound > 0 and point >= 0.05); placebos never
enter (no affected set).
Spearman rho with: arm bootstrap (resample arms), measurement bootstrap (each arm's T and readability drawn from their bootstrap draws), both
combined; permutation p (shuffle T over arms). Cross-sign test: isotonic (increasing) fit of T on readability over FR arms, residuals of FA /
form arms (and the reverse). Designed extremes = simulated arms built to be maximally readable or unreadable (visible type sets, type-wide
copies, long80, hash, hidden halves, scrambled copy). Outputs: b04_relation_out.txt, b04_table.csv, b04_transfer_vs_readability.png"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, glob
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, (_RA + "/bothsigns"))
from b00_common import OUT, STS, DSS

rng = np.random.default_rng(11)
A = pd.read_csv(f"{OUT}/b02_arms.csv"); AB = pickle.load(open(f"{OUT}/cache/b02_armboot.pkl", "rb"))
RD = []
for st in STS:
    for ds in DSS:
        RD += pickle.load(open(f"{OUT}/cache/b03_{st}_{ds}.pkl", "rb"))
FEATS = ("hidQA", "hid16", "hid24", "txtQ", "txtQA", "hidQA_strat", "conf_auc")
rg = defaultdict(list)
for r in RD:
    rg[(r["arm"], r["sign"], r["st"])].append(r)
for f in FEATS:
    A[f"R_{f}"] = [np.nanmean([u.get(f, np.nan) for u in rg[(a, s, t)]]) if rg[(a, s, t)] and np.isfinite([u.get(f, np.nan) for u in rg[(a, s, t)]]).any() else np.nan
                   for a, s, t in zip(A.arm, A.sign, A.st)]
A["R_cells"] = [sum(np.isfinite(u.get("hidQA", np.nan)) for u in rg[(a, s, t)]) for a, s, t in zip(A.arm, A.sign, A.st)]
RB = {}
for key, us in rg.items():
    bs = np.array([u["hidQA_bs"] for u in us if np.isfinite(u.get("hidQA", np.nan))])
    RB[key] = np.nanmean(bs, 0) if len(bs) else np.full(300, np.nan)
DESIGNED = {"fnB80", "fnB50", "fnA80", "fnH80", "fnH60", "typeA40", "typeB20", "typeB40", "typeB60", "plsharpM", "plsharpO", "plscrM", "hash35", "long80"}
A["designed"] = A.arm.isin(DESIGNED); A["placebo"] = A.sign.str.contains("placebo")
A["signc"] = A.sign.map({"FR": "FR", "FA": "FA", "form": "form"}).fillna("placebo")
A.to_csv(f"{OUT}/b04_table.csv", index=False)
out = open(f"{OUT}/b04_relation_out.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")


def eligible(df, cur):
    """an arm's transfer ratio is interpretable only with a reliable, non-trivial home effect: 95% CI lower bound > 0 and point >= 0.05"""
    return df[(~df.placebo) & np.isfinite(df[f"{cur}_h"]) & np.isfinite(df[f"{cur}_n"]) & (df[f"{cur}_h_lo"] > 0) & (df[f"{cur}_h"] >= 0.05) & np.isfinite(df.R_hidQA)]


def spearman_tests(df, cur, rcol="R_hidQA", nb=2000):
    if len(df) < 5:
        return None
    T = (df[f"{cur}_n"] / df[f"{cur}_h"]).values; R = df[rcol].values; n = len(df)
    rho = spearmanr(R, T).correlation
    keys = list(zip(df.arm, df.sign, df.st))
    TB = np.array([AB[k][f"{cur}_n"] / AB[k][f"{cur}_h"] for k in keys])        # arms x draws
    RBm = np.array([RB.get(k, np.full(300, np.nan)) for k in keys])
    ra = []; rm = []; rc = []
    for b in range(nb):
        ii = rng.integers(0, n, n)
        if len(set(ii)) >= 4:
            ra.append(spearmanr(R[ii], T[ii]).correlation)
        j = b % TB.shape[1]; jr = b % RBm.shape[1]
        Tb = TB[:, j]; Rb = RBm[:, jr] if rcol == "R_hidQA" else R
        m = np.isfinite(Tb) & np.isfinite(Rb)
        if m.sum() >= 5:
            rm.append(spearmanr(Rb[m], Tb[m]).correlation)
            ii2 = rng.choice(np.where(m)[0], m.sum())
            if len(set(ii2)) >= 4:
                rc.append(spearmanr(Rb[ii2], Tb[ii2]).correlation)
    perm = np.array([spearmanr(R, rng.permutation(T)).correlation for _ in range(3000)])
    p = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1)
    q = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))
    return dict(n=n, rho=rho, arm=q(ra), meas=q(rm), comb=q(rc), p=p)


def line(name, res):
    if res is None:
        pr(f"  {name:44s} n < 5"); return
    pr(f"  {name:44s} n {res['n']:2d}  rho {res['rho']:+.2f}  arms [{res['arm'][0]:+.2f},{res['arm'][1]:+.2f}]  measurement [{res['meas'][0]:+.2f},{res['meas'][1]:+.2f}]  "
       f"combined [{res['comb'][0]:+.2f},{res['comb'][1]:+.2f}]  perm p {res['p']:.4f}")


SPLITS = [("pooled", lambda d: d), ("qwen7", lambda d: d[d.st == "qwen7"]), ("llama8", lambda d: d[d.st == "llama8"]),
          ("FR only", lambda d: d[d.sign == "FR"]), ("FA + form only", lambda d: d[d.sign.isin(["FA", "form"])]),
          ("designed extremes removed", lambda d: d[~d.designed]), ("designed removed, qwen7", lambda d: d[(~d.designed) & (d.st == "qwen7")]),
          ("designed removed, llama8", lambda d: d[(~d.designed) & (d.st == "llama8")]), ("designed removed, FR", lambda d: d[(~d.designed) & (d.sign == "FR")]),
          ("designed removed, FA", lambda d: d[(~d.designed) & (d.sign == "FA")]), ("judges only", lambda d: d[d.arm.str.startswith("judge")]),
          ("designed extremes only", lambda d: d[d.designed])]
for cur, lab in (("kind", "KIND currency (share of the run's refusals / wrong answers of the kind the error rewards): T = new / home"),
                 ("dec", "DECISION currency (extra refusal / answering on affected items): T = new / home")):
    pr(f"\n===== Spearman(readability = hidQA AUC on new questions, transfer) -- {lab}")
    E = eligible(A, cur)
    pr(f"  eligible arms (home effect CI > 0 and >= 0.05, readability defined, not placebo): {len(E)}; excluded non-placebo arms: "
       + ", ".join(f"{a}/{s}/{t}" for a, s, t in A[(~A.placebo) & ~A.index.isin(E.index)][["arm", "sign", "st"]].values))
    for nm, fn in SPLITS:
        line(nm, spearman_tests(fn(E), cur))
    # cross-sign: isotonic fit on one sign, residuals of the other
    pr(f"  cross-sign test ({cur}): isotonic fit of T on readability over one sign's arms, mean residual of the other sign's arms")
    for fit_s, test_s in ((["FR"], ["FA", "form"]), (["FR"], ["FA"]), (["FA", "form"], ["FR"])):
        F = E[E.sign.isin(fit_s)]; Tt = E[E.sign.isin(test_s)]
        if len(F) < 4 or len(Tt) < 3:
            pr(f"    fit {fit_s} -> test {test_s}: too few arms"); continue
        def resid(Fd, Td, j=None):
            if j is None:
                tf = (Fd[f"{cur}_n"] / Fd[f"{cur}_h"]).values; tt = (Td[f"{cur}_n"] / Td[f"{cur}_h"]).values
            else:
                tf = np.array([AB[k][f"{cur}_n"][j] / AB[k][f"{cur}_h"][j] for k in zip(Fd.arm, Fd.sign, Fd.st)])
                tt = np.array([AB[k][f"{cur}_n"][j] / AB[k][f"{cur}_h"][j] for k in zip(Td.arm, Td.sign, Td.st)])
            m = np.isfinite(tf)
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(Fd.R_hidQA.values[m], np.clip(tf[m], -1, 2))
            return np.clip(tt, -1, 2) - iso.predict(Td.R_hidQA.values)
        r0 = resid(F, Tt); bs = []
        for b in range(1000):
            Fi = F.iloc[rng.integers(0, len(F), len(F))]; Ti = Tt.iloc[rng.integers(0, len(Tt), len(Tt))]
            bs.append(np.nanmean(resid(Fi, Ti, j=b % 500)))
        pr(f"    fit {'+'.join(fit_s):8s} -> test {'+'.join(test_s):8s}: n {len(F)}/{len(Tt)}  mean residual {np.nanmean(r0):+.2f} [{np.nanpercentile(bs, 2.5):+.2f},{np.nanpercentile(bs, 97.5):+.2f}]"
           f"  (share of test arms below the curve {np.mean(r0 < 0):.2f})")
# robustness: readability variants and effect-fit variants (pooled, all eligible arms; designed removed)
pr("\n===== robustness: rho (pooled, eligible arms) for readability variants and reference-fit variants")
for cur in ("kind", "dec", "kind_lin", "dec_lin", "kind_knn", "dec_knn"):
    E = eligible(A, cur)
    for rcol in ("R_hidQA", "R_hid16", "R_hid24", "R_txtQ", "R_txtQA", "R_hidQA_strat"):
        Ei = E[np.isfinite(E[rcol])]
        a = spearman_tests(Ei, cur, rcol=rcol, nb=500); b = spearman_tests(Ei[~Ei.designed], cur, rcol=rcol, nb=500)
        pr(f"  {cur:9s} {rcol:14s} all: n {a['n']:2d} rho {a['rho']:+.2f} [{a['arm'][0]:+.2f},{a['arm'][1]:+.2f}] p {a['p']:.4f} | designed removed: "
           + (f"n {b['n']:2d} rho {b['rho']:+.2f} [{b['arm'][0]:+.2f},{b['arm'][1]:+.2f}] p {b['p']:.4f}" if b else "n<5"))
# the per-arm table
pr("\n===== arm table (kind currency first; R = hidQA readability on new questions; T = new/home; placebos: home/new only)")
cols = ["arm", "sign", "st", "cells", "runs", "kind_h", "kind_n", "kind_T", "dec_h", "dec_n", "dec_T", "R_hidQA", "R_hid16", "R_hidQA_strat"]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
pr(A[cols].round(3).to_string(index=False))
out.close()

# ---- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"FR": "#2a78d6", "FA": "#eb6834", "form": "#1baf7a"}
fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), sharex=True, sharey="row")
for row, cur in enumerate(("kind", "dec")):
    E = eligible(A, cur)
    for col, st in enumerate(STS):
        ax = axes[row, col]; d = E[E.st == st]
        ax.axhline(0, color="#c8c7c1", lw=1, zorder=0); ax.axhline(1, color="#c8c7c1", lw=1, ls=(0, (3, 3)), zorder=0)
        for _, r in d.iterrows():
            T = r[f"{cur}_n"] / r[f"{cur}_h"]; lo, hi = r[f"{cur}_T_lo"], r[f"{cur}_T_hi"]
            c = COL[r.sign]; mk = "o" if r.sign == "FR" else ("s" if r.sign == "FA" else "D")
            ax.errorbar(r.R_hidQA, np.clip(T, -0.6, 1.6), yerr=[[np.clip(T, -0.6, 1.6) - np.clip(lo, -0.6, 1.6)], [np.clip(hi, -0.6, 1.6) - np.clip(T, -0.6, 1.6)]],
                        fmt="none", ecolor=c, alpha=0.35, lw=1.2, zorder=1)
            ax.scatter(r.R_hidQA, np.clip(T, -0.6, 1.6), s=64, marker=mk, facecolor=("white" if r.designed else c), edgecolor=c, linewidth=2, zorder=3)
            nm = r.arm.replace("judge", "J").replace("fn20", "")
            ax.annotate(nm, (r.R_hidQA, np.clip(T, -0.6, 1.6)), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#52514e")
        res = spearman_tests(d, cur, nb=500)
        if res:
            ax.text(0.02, 0.97, f"Spearman {res['rho']:+.2f} [{res['arm'][0]:+.2f}, {res['arm'][1]:+.2f}], n {res['n']}", transform=ax.transAxes, va="top", fontsize=9, color="#0b0b0b")
        ax.set_title(f"{st}: {'kind' if cur == 'kind' else 'decision'} currency", fontsize=11, color="#0b0b0b", loc="left")
        ax.set_ylim(-0.65, 1.65); ax.set_xlim(0.38, 1.04)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=9)
        if row == 1:
            ax.set_xlabel("readability: probe AUC on new questions (fitted on training questions)", fontsize=9, color="#52514e")
        if col == 0:
            ax.set_ylabel("transfer = new / home effect", fontsize=9, color="#52514e")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=COL["FR"], markersize=8, label="false rejections (FR)"),
     Line2D([], [], marker="s", ls="", color=COL["FA"], markersize=8, label="false accepts (FA)"),
     Line2D([], [], marker="D", ls="", color=COL["form"], markersize=8, label="answer form (long80)"),
     Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#52514e", markersize=8, label="designed extreme (hollow)")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Transfer vs readability, every arm with a reliable home effect (bars: 95% crossed-bootstrap CI, clipped)", fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(f"{OUT}/b04_transfer_vs_readability.png", dpi=130, facecolor="#fcfcfb")
