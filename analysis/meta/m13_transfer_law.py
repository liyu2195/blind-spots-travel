"""(meta) Transfer law: the share of a grader-specific effect at home that reaches new questions vs the visibility of the grader's mistakes.
Points = (student, dataset, grader sub-family). Effects: FA hack (m06: paid share of wrong answers beyond uniform-grader runs at the same abstention)
and FR selective refusal (m07: extra refusal of the grader's rejected-right questions beyond uniform-grader runs at the same abstention).
transfer = mean effect on new questions / mean effect at home (points with a home effect >= 0.05). Visibility = Gini = 2 AUC - 1 of a model fitted on
the untrained answers of trained questions and scored on new questions (m08: FA answer-level text model; FR question text + own confidence).
For plreal<J> the new-question set is judge J's own rejections, so its visibility is that of judge J (a35). Discovery qwen7, confirmation llama8.
Inputs cache/hack.pkl, cache/refusal.pkl, cache/visibility.csv. Output stdout (out_m13_transfer_law.txt), cache/transfer_points.csv."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
OUT = (_RA + "/meta/cache")
H = pd.read_pickle(f"{OUT}/refusal.pkl")          # contains hack_* (m06) and eff_* (m07)
V = pd.read_csv(f"{OUT}/visibility.csv")
S = H[H["std"] & H.st.isin(["qwen7", "llama8"]) & (H.kl == 0.2) & (H.c == 0.5)]
vis = V.groupby(["st", "ds", "sub"]).agg(FA=("FAa_auc_new", "mean"), FR=("FRqc_auc_new", "mean"), FRt=("FRq_auc_new", "mean")).reset_index()
pts = []
for (st, ds, sub), d in S.groupby(["st", "ds", "sub"]):
    fam = d.fam.iloc[0]
    v = vis[(vis.st == st) & (vis.ds == ds) & (vis["sub"] == sub)]
    for eff, col, vcol in (("FA hack", "hack", "FA"), ("FR refusal", "eff", "FR")):
        h = d[f"{col}_home"].astype(float); n = d[f"{col}_new"].astype(float); m = np.isfinite(h) & np.isfinite(n)
        if m.sum() < 2 or h[m].mean() < 0.05: continue
        vv = v[vcol].mean() if len(v) else np.nan
        if sub.startswith("plreal"):
            J = sub[-1]; vj = vis[(vis.st == st) & (vis.ds == ds) & (vis["sub"] == f"judge{J}a35")]; vv = vj["FR"].mean()
        pts.append(dict(st=st, ds=ds, sub=sub, fam=fam, effect=eff, n=int(m.sum()), home=h[m].mean(), new=n[m].mean(), transfer=n[m].mean() / h[m].mean(),
                        auc=vv, gini=2 * vv - 1 if np.isfinite(vv) else np.nan))
P = pd.DataFrame(pts); P.to_csv(f"{OUT}/transfer_points.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(P.sort_values(["effect", "gini"]).round(3).to_string(index=False))
Q = P[np.isfinite(P.gini)]
print(f"\nall points with a visibility value: {len(Q)} (FA {sum(Q.effect == 'FA hack')}, FR {sum(Q.effect == 'FR refusal')})")
for nm, d in (("all", Q), ("FA hack", Q[Q.effect == "FA hack"]), ("FR refusal", Q[Q.effect == "FR refusal"]), ("qwen7", Q[Q.st == "qwen7"]), ("llama8", Q[Q.st == "llama8"]),
              ("judges only", Q[Q.fam == "judge"]), ("simulated only", Q[Q.fam != "judge"])):
    if len(d) < 4: continue
    rs = spearmanr(d.gini, d.transfer); rp = pearsonr(d.gini, d.transfer)
    rng = np.random.default_rng(0); perm = [abs(spearmanr(d.gini, rng.permutation(d.transfer.values))[0]) for _ in range(5000)]
    print(f"  {nm:15s} n {len(d):3d}  Spearman {rs[0]:+.3f} (perm p {np.mean(np.array(perm) >= abs(rs[0])):.4f})  Pearson {rp[0]:+.3f}")
# discovery qwen7 -> confirmation llama8: transfer = b * gini (through the origin) and a + b gini
for eff in ("FA hack", "FR refusal", None):
    d = Q if eff is None else Q[Q.effect == eff]
    a = d[d.st == "qwen7"]; b = d[d.st == "llama8"]
    if len(a) < 3 or len(b) < 3: continue
    k = np.sum(a.gini * a.transfer) / np.sum(a.gini**2); c1 = np.polyfit(a.gini, a.transfer, 1)
    for nm, pr in (("T = k G", k * b.gini), ("T = a + b G", np.polyval(c1, b.gini))):
        r = b.transfer - pr
        print(f"  [{eff or 'both'}] fit on qwen7 (n {len(a)}) {nm}: k/coef {k:.3f} / {np.round(c1, 3)} -> llama8 (n {len(b)}): MAE {np.mean(np.abs(r)):.3f}, R2 {1 - np.sum(r**2) / np.sum((b.transfer - b.transfer.mean())**2):+.3f}")
