"""(rev_R5) Check: does the judge-paid share of wrong TRAINING samples stay flat over a single pass?
The draft reports +0.007 / -0.012 for one-pass (B1) runs with the lenient Qwen judge (Qt50). Here the same is computed for the
strict cut (Qa35) as well, and the change is split into: paid share among wrong answers (FA side), paid share among key-right
answers (1 - FR), IDK share and reply length, from the trainer's per-step logs only.
paid_all = reward - c * idk (rewards are 0/1 for non-IDK replies); wrong share = 1 - idk - correct_strict - invalid (approx.);
paid_right = paid_all - wrong_paid * wrong share; accept_right = paid_right / correct_strict.
Windows: early = steps 1-5, late1 = steps 19-23 (end of the first pass for FK20; every B1 step is a first visit), late = steps 101-120.
Output: r5_onepass.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, glob, os, numpy as np
R = (_RR + "/run1/rl")
C = 0.5
def load(run):
    return [json.loads(l) for l in open(os.path.join(R, run, "train_log.jsonl"))]
def feats(rows):
    out = []
    for r in rows:
        idk, cs, inv = r["idk"], r["correct_strict"], r.get("invalid") or 0.0
        wp = r["wrong_paid"] if r["wrong_paid"] is not None else np.nan
        wshare = max(1e-9, 1 - idk - cs - inv)
        paid_all = r["reward"] - C * idk
        paid_right = paid_all - (0 if np.isnan(wp) else wp) * wshare
        acc_right = paid_right / cs if cs > 0 else np.nan
        out.append((r["step"], wp, acc_right, idk, r["resp_tokens"], cs))
    return np.array(out, dtype=float)
def win(A, a, b, col):
    m = (A[:, 0] >= a) & (A[:, 0] <= b)
    return np.nanmean(A[m, col])
lines = []
for pat in ["{st}_B1K20judgeQa35_idkc05_s*_h1", "{st}_B1K20judgeQt50_idkc05_s*_h1", "{st}_FK20judgeQa35_idkc05_s*_h1",
            "{st}_FK20judgeQt50_idkc05_s*_h1", "{st}_B1K20sim00fn00_idkc05_s*_h1"]:
    for st in ("qwen7", "llama8"):
        runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(R, pat.format(st=st))))
        if not runs: continue
        lines.append(f"== {pat.format(st=st)}  ({len(runs)} runs)")
        lines.append("   run                                   | wrong_paid early->late1 (->late) | accept_right early->late1 (->late) | idk early->late1 (->late) | tokens early->late")
        for run in runs:
            A = feats(load(run))
            s = lambda col: (win(A, 1, 5, col), win(A, 19, 23, col), win(A, 101, 120, col))
            wp, ar, ik, tk = s(1), s(2), s(3), s(4)
            lines.append(f"   {run:40s}| {wp[0]:.3f} -> {wp[1]:.3f} ({wp[2]:.3f}) | {ar[0]:.3f} -> {ar[1]:.3f} ({ar[2]:.3f}) | {ik[0]:.3f} -> {ik[1]:.3f} ({ik[2]:.3f}) | {tk[0]:.1f} -> {tk[2]:.1f}")
open((_RA + "/rev_R5/r5_onepass.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
