"""(stats) Collect the primary test of each headline claim from the c*_summary.pkl files and apply Holm's step-down correction
(a) to the pooled primary set (10 claims) and (b) to the confirmation-only set (llama8; for claim 4 the llama8 test set is the confirmation).
Two 'no effect' secondary tests that the draft states as findings (claim 2: no step effect; claim 8: hidden set carries none) are added in (c).
Bootstrap / permutation p-values at their resolution floor are entered at the floor (1 / (draws + 1)). Combined p over the two students' independent
exact permutation tests: Fisher. Output: s99_summary_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
from scipy import stats as SS
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P = Log(f"{HERE}/s99_summary_out.txt"); P(__doc__)
L = lambda n: pickle.load(open(f"{HERE}/{n}", "rb"))
fisher = lambda ps: 1 - SS.chi2.cdf(-2 * np.sum(np.log(ps)), 2 * len(ps))
c1 = L("c1_summary.pkl"); c2 = L("c2_summary.pkl"); c3 = L("c3_summary.pkl"); c4 = L("c4_summary.pkl"); c5 = L("c5_summary.pkl")
c6 = L("c6_summary.pkl"); c78 = L("c78_summary.pkl"); c9 = L("c9_summary.pkl"); c10 = L("c10_summary.pkl")
floor = lambda p, n: max(p, 1.0 / (n + 1))
pooled = {
    "C1 content stays home (TOST S_new, margin 0.05)": c1["RES"]["pooled"]["tost"]["p"],
    "C2 first revisit at fixed step (regime permutation, Fisher)": fisher([floor(c2[st]["p_perm_d21"], 2000) for st in STS]),
    "C3 lenient judge = noise, new abstention (TOST, margin 0.03)": floor(c3[("abst", "pooled")]["tost"]["p"], 4000),
    "C4 leniency curve out of sample (permutation of L)": floor(c4["p_perm"], 2000),
    "C5 crossover fingerprint (judge-label permutation)": floor(c5["PERM"]["pooled"], 10000),
    "C6 readable split interaction (stratified permutation)": floor(c6["MAIN"]["pooled"]["p_perm"], 2000),
    "C7 copy minus scrambled (exact permutation, Fisher)": c78["c7"]["p_fisher"],
    "C8 visible minus hidden transfer ratio (bootstrap)": floor(c78["c8"]["pooled"]["p_dratio"], 2000),
    "C9 FR minus FA ICC at a35 (question bootstrap)": 1.0 / 2001,
    "C10 capitals DiD vs gold key (exact permutation, Fisher)": fisher([c10[st]["p"] for st in STS]),
}
conf = {
    "C1": c1["RES"]["llama8"]["tost"]["p"], "C2": floor(c2["llama8"]["p_perm_d21"], 2000), "C3": floor(c3[("abst", "llama8")]["tost"]["p"], 4000),
    "C4": floor(c4["p_perm"], 2000), "C5": floor(c5["PERM"]["llama8"], 10000), "C6": floor(c6["MAIN"]["llama8"]["p_perm"], 2000),
    "C7": c78["c7"]["p"]["llama8"], "C8": floor(c78["c8"]["llama8"]["p_dratio"], 2000), "C9": 1.0 / 2001, "C10": c10["llama8"]["p"]}
disc = {
    "C1": c1["RES"]["qwen7"]["tost"]["p"], "C2": floor(c2["qwen7"]["p_perm_d21"], 2000), "C3": floor(c3[("abst", "qwen7")]["tost"]["p"], 4000),
    "C5": floor(c5["PERM"]["qwen7"], 10000), "C6": floor(c6["MAIN"]["qwen7"]["p_perm"], 2000), "C7": c78["c7"]["p"]["qwen7"],
    "C8": floor(c78["c8"]["qwen7"]["p_dratio"], 2000), "C9": 1.0 / 2001, "C10": c10["qwen7"]["p"]}
for lab, S in (("(a) pooled primary set", pooled), ("(b) confirmation set (llama8 only)", conf), ("(b') discovery set (qwen7 only)", disc)):
    H = holm(S); P(f"\n== {lab}: raw p and Holm-adjusted p")
    for k in sorted(S, key=lambda k: S[k]): P(f"  {k:62s} raw {S[k]:.2e}  Holm {H[k]:.2e}  {'PASS' if H[k] < 0.05 else 'fail'}")
nof = dict(pooled)
nof["C2b no step effect at fixed visit (TOST +-0.04), qwen7"] = c2["qwen7"]["tost"]["p"]
nof["C2b no step effect at fixed visit (TOST +-0.04), llama8"] = c2["llama8"]["tost"]["p"]
nof["C8b hidden set carries none (TOST +-0.05), pooled"] = c78["c8"]["pooled"]["tost"]["p"]
H = holm(nof); P("\n== (c) pooled primary set plus the two stated 'no effect' secondaries")
for k in sorted(nof, key=lambda k: nof[k]): P(f"  {k:62s} raw {nof[k]:.2e}  Holm {H[k]:.2e}  {'PASS' if H[k] < 0.05 else 'fail'}")
P.save()
