"""(verify_rate) IDK credit under std-normalised group advantages (trainer: adv = (r - mean) / (std_unbiased + 1e-4), adv = 0 if std < 1e-6;
st_grpo.py / v12 / v13 identical) vs unnormalised (ADV=none, Dr. GRPO style).
(A) Mechanism, exact: in a group of 8 with m IDK replies (reward c) and 8 - m answers that all get the SAME reward (all 0 or all 1), the std-normalised
    advantage of every reply is independent of c; c only enters through groups whose answers are mixed. Share of such homogeneous groups and the
    c-sensitivity of the expected IDK advantage per grader, from the untrained student's samples on the trained questions.
(B) Every run set that varies only the credit (same student, dataset, grader, KL, format, advantage type): abstention on new questions vs c, OLS slope.
Usage: python3 -I v6_idk.py > out_v6_idk.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np, pandas as pd
from math import comb
sys.path.insert(0, (_RA + "/verify_rate"))
from vr import base
HERE = (_RA + "/verify_rate")
def adv(r, norm):
    r = np.asarray(r, float); s = r.std(ddof=1)
    if s < 1e-6: return np.zeros_like(r)
    return (r - r.mean()) / ((s + 1e-4) if norm == "std" else 1.0)
print("== (A1) exact check: group of 8, m IDK (credit c) + answers with one common reward; IDK advantage for c = 0.3 / 0.5 / 0.7 / 0.9")
for m in (1, 2, 4):
    for v in (0.0, 1.0):
        a = [adv([c] * m + [v] * (8 - m), "std")[0] for c in (0.3, 0.5, 0.7, 0.9)]; b = [adv([c] * m + [v] * (8 - m), "none")[0] for c in (0.3, 0.5, 0.7, 0.9)]
        print(f"   m {m} answers all {int(v)}: std {np.round(a, 3)}   none {np.round(b, 3)}")
print("   mixed example, m 1, 7 answers with k accepted, c 0.3 / 0.5 / 0.7 / 0.9 (std):")
for k in (1, 3, 5):
    print(f"     k {k}: {np.round([adv([c] + [1] * k + [0] * (7 - k), 'std')[0] for c in (0.3, 0.5, 0.7, 0.9)], 3)}")
def e_adv(p, c, norm):
    """expected advantage of one IDK reply in a group with 7 answers, k ~ Bin(7, p) accepted."""
    w = np.array([comb(7, k) * p ** k * (1 - p) ** (7 - k) for k in range(8)])
    a = np.array([adv([c] + [1] * k + [0] * (7 - k), norm)[0] for k in range(8)]); return float(w @ a)
print("\n== (A2) from the untrained samples (32 per trained question): per-question acceptance a_q of one valid answer; P(7 answers homogeneous);")
print("   mean expected IDK advantage at c = 0.3 / 0.5 / 0.7 / 0.9 (std | none). Deterministic key: a_q = strict-right share; random FP f: a_q = q + (1 - q) f")
for st, ds in (("phi4m", "h1"), ("smol3", "h1"), ("qwen7", "h1"), ("qwen7", "pq"), ("llama8", "h1")):
    B = base(st, ds); m = B["home"] & (B["s"] >= 0) & B["valid"] & ~B["idk"]
    qs = pd.Series(B["strict"][m].astype(float)).groupby(B["idx"][m]).mean()
    grs = [("key", qs.values), ("FP20", qs.values + (1 - qs.values) * 0.2), ("FP35", qs.values + (1 - qs.values) * 0.35)]
    if "Q" in B["PJ"]:
        pj = pd.Series((B["PJ"]["Q"][m] > 0.5).astype(float)).groupby(B["idx"][m]).mean(); grs.append(("judgeQt50 (per-sample verdicts treated iid)", pj.values))
    for nm, a in grs:
        hom = np.mean(a ** 7 + (1 - a) ** 7)
        es = [np.mean([e_adv(p, c, "std") for p in a]) for c in (0.3, 0.5, 0.7, 0.9)]; en = [np.mean([e_adv(p, c, "none") for p in a]) for c in (0.3, 0.5, 0.7, 0.9)]
        print(f"   {st:6s} {ds} {nm:44s} P(homog.) {hom:.2f}  std {np.round(es, 2)} (range {max(es) - min(es):.2f})  none {np.round(en, 2)} (range {max(en) - min(en):.2f})")
print("   note: a deterministic judge's verdicts on one question's samples are not iid; the iid line for judgeQt50 understates its homogeneity.")
T = pd.read_pickle(f"{HERE}/cache/table.pkl")
T = T[~T.run.str.contains("flagc|_T240|T240") & ~T.run.str.contains("f1_") & ~T.fam.str.startswith("other") & T.home_std & (T.force == 0)
      & T.other_keys.apply(lambda k: all(x == "PROMPT" for x in k))]
T["regime"] = np.where(T.tag.str.startswith("FK20"), "FK20 (KL .2, strict fmt)", np.where(T.kl == 0, "old (no KL)", "other"))
print("\n== (B) credit sweeps: abstention on NEW questions (and at home) by credit; slope = OLS d abstention / d c over runs (SE)")
rows = []
for k, d in T.groupby(["st", "ds", "regime", "kl", "fmt", "adv", "sub"]):
    if d.c.nunique() < 2 or k[2] == "other": continue
    b, a = np.polyfit(d.c, d.idk_n, 1); res = d.idk_n - (a + b * d.c); n = len(d)
    from scipy.stats import theilslopes
    ts = theilslopes(d.idk_n.values, d.c.values)[0]
    se = np.sqrt(np.sum(res ** 2) / max(1, n - 2) / np.sum((d.c - d.c.mean()) ** 2)) if n > 2 else np.nan
    by = d.groupby("c").agg(n=("run", "size"), new=("idk_n", "mean"), home=("idk_h", "mean"))
    rows.append(dict(st=k[0], ds=k[1], regime=k[2], adv=k[5], grader=k[6], runs=n, credits=len(by), slope=b, se=se, theil=ts))
    print(f"  {k[0]:6s} {k[1]} {k[2]:26s} adv {k[5]:4s} {k[6]:10s} runs {n:2d}: " + "  ".join(f"c{c:.2f}:{r.new:.2f}(n{int(r.n)})" for c, r in by.iterrows()) + f"   slope {b:+.2f} (SE {se:.2f}; Theil-Sen {ts:+.2f})")
R = pd.DataFrame(rows); R.to_csv(f"{HERE}/idk_sweeps.csv", index=False)
print("\n== summary of slopes (per unit credit) by advantage type and grader kind")
R["kind"] = np.where(R.grader == "FP00FN00", "key", np.where(R.grader.str.startswith("judge"), "judge t50", "random noise"))
print(R.groupby(["adv", "kind"]).agg(sweeps=("slope", "size"), runs=("runs", "sum"), mean_slope=("slope", "mean"), median_theil=("theil", "median"), min_slope=("slope", "min"), max_slope=("slope", "max")).round(2).to_string())
