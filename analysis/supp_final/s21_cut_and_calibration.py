"""(supp_final) s21_cut_and_calibration: EXPLORATORY, written AFTER the E14 result (Nov 18). Why does one pair (Qwen, TriviaQA) miss at the fixed cut k = 20,
and is the miss explained by something measurable? Existing runs only.
k* = the student's calibration point: the smallest number of agreeing samples (of 32) at which its untrained greedy answer is right at least half the
     time (isotonic fit on the 750 training questions, audited truth). With reward 1 / 0 / 0.5, paying answers below k* pays answers that are more
     often wrong than right.
A. Per pair: k*; share right among training answers the fixed cut pays (>= 20 agreeing) and among the ones just above the cut (20-25 agreeing);
   label-free stand-ins for that share: what the Qwen and Mistral judges accept at their default cut, and how often the other student's greedy answer agrees.
B. Every own-vote arm that has runs (all k): reward minus the answer key's (95% interval, questions and runs resampled), against k - k*.
C. Summary: arms whose cut lies from 2 below to 12 above k*, and arms outside that window.
D. k* estimated from only 100 labelled training questions (500 random draws): median, 5-95% range, share of draws inside the window [k* - 2, k* + 12]
   of the pair's own k*... (i.e. how often 100 labels put the cut where the runs match the key).
Usage: python3 -I -B s21_cut_and_calibration.py   Output: s21_cut_and_calibration_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_common import *
import aud as audlib
out = [__doc__]; rng = np.random.default_rng(2121); NB = 4000; ARMS = []
def pav(x, y):
    o = np.argsort(x, kind="stable"); xs = np.asarray(x, float)[o]; ys = np.asarray(y, float)[o]; v = list(ys); w = [1.0] * len(ys); i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1] + 1e-12:
            v[i] = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1]); w[i] += w[i + 1]; del v[i + 1]; del w[i + 1]
            if i > 0: i -= 1
        else: i += 1
    fit = np.repeat(v, np.array(w, int)); return lambda q: np.interp(q, xs, fit)
kk = np.arange(0, 33)
def kstar_of(c, y):
    p = pav(c, y)(kk); return int(kk[np.argmax(p >= 0.5)]) if (p >= 0.5).any() else 33
BASES = {(st, ds): pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")) for st in ("qwen7", "llama8") for ds in ("pq", "h1", "nq")}
for st in ("qwen7", "llama8"):
    for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo")):
        Bp = BASES[(st, ds)]; R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn); g0 = lambda a: np.asarray(a)[:, 0]
        tr = np.asarray(Bp["train"], bool); new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]; n = len(ii)
        ok = g0(Bp["valid"]).astype(bool) & ~g0(Bp["idk"]).astype(bool); right = ok & (g0(Bp["kl"]).astype(bool) | (g0(Bp["son"]) >= 4))
        cnt = np.rint(np.asarray(Bp["conf"], float) * 32).astype(int); T = tr & ok; ks = kstar_of(cnt[T], right[T]); cell = f"{st} {dsn}"
        other = BASES[("llama8" if st == "qwen7" else "qwen7", ds)]; peer = np.array([bool(a) and bool(b) and same(a, b) for a, b in zip([x[0] for x in Bp["X"]], [x[0] for x in other["X"]])])
        paid = T & (cnt >= 20); marg = T & (cnt >= 20) & (cnt <= 25); PQ = g0(Bp["PJ"]["Q"]).astype(float); PM = g0(Bp["PJ"]["M"]).astype(float)
        out.append(f"\n==== {cell}: calibration point k* = {ks}; fixed cut 20 is {20 - ks:+d} from it")
        out.append(f"  A. training answers paid at k = 20: {int(paid.sum())}, right {right[paid].mean():.3f}; just above the cut (20-25 agreeing): {int(marg.sum())}, right {right[marg].mean():.3f} | "
                   f"label-free stand-ins on the paid answers: Qwen judge accepts {(PQ[paid] > 0.5).mean():.3f}, Mistral judge accepts {(PM[paid] > 0.5).mean():.3f}, other student agrees {peer[paid].mean():.3f}; "
                   f"just above the cut: {(PQ[marg] > 0.5).mean():.3f}, {(PM[marg] > 0.5).mean():.3f}, {peer[marg].mean():.3f}")
        G = defaultdict(list)
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05": continue
            if not (r["g"] == "sim00fn00" or re.fullmatch(r"frzVOTE\d+", r["g"])): continue
            o = np.asarray(r["o"]); idk = (o == 0)
            if idk[new].mean() >= 0.95: continue
            rr = np.array([(o[i] == 1) or (o[i] == 2 and LV.get((i, r["f"][i]), 0) >= 4) for i in ii]); G[r["g"]].append((rr.astype(float) + 0.5 * idk[ii], float(idk[ii].mean())))
        K = np.array([v for v, _ in G["sim00fn00"]])
        for g in sorted((g for g in G if g != "sim00fn00"), key=lambda g: int(g[7:])):
            A = np.array([v for v, _ in G[g]]); d = np.empty(NB)
            for t in range(NB):
                q = rng.integers(0, n, n); d[t] = A[rng.integers(0, len(A), len(A))][:, q].mean() - K[rng.integers(0, len(K), len(K))][:, q].mean()
            k = int(g[7:]); m = float(A.mean() - K.mean()); lo, hi = np.percentile(d, [2.5, 97.5]); ARMS.append((cell, k, k - ks, m, float(lo), float(hi), len(A), float(np.mean([a for _, a in G[g]]))))
            out.append(f"  B. k = {k:2d} (k - k* = {k - ks:+3d}; {len(A)} runs; abstains {np.mean([a for _, a in G[g]]):.2f}): reward {A.mean():.3f}, minus answer key ({K.mean():.3f}, {len(K)} runs) {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        s = []; Ti = np.where(T)[0]
        for _ in range(500):
            j = rng.choice(Ti, 100, replace=False); s.append(kstar_of(cnt[j], right[j]))
        s = np.array(s); out.append(f"  D. k* from 100 labelled questions: median {int(np.median(s))}, 5-95% {int(np.percentile(s, 5))}-{int(np.percentile(s, 95))}; a cut 4 above it falls in [k*-2, k*+12] in {np.mean((s + 4 >= ks - 2) & (s + 4 <= ks + 12)):.2f} of draws; "
                                f"the fixed cut 20 falls in that window: {'yes' if ks - 2 <= 20 <= ks + 12 else 'NO'}")
out.append("\n==== C. all own-vote arms, by distance of the cut from the calibration point")
inw = [a for a in ARMS if -2 <= a[2] <= 12]; outw = [a for a in ARMS if not (-2 <= a[2] <= 12)]
out.append(f"   inside [k*-2, k*+12]: {len(inw)} arms ({sum(a[6] for a in inw)} runs): reward minus key from {min(a[3] for a in inw):+.3f} to {max(a[3] for a in inw):+.3f}, mean {np.mean([a[3] for a in inw]):+.4f}; arms below -0.015: {sum(a[3] < -0.015 for a in inw)}")
for a in sorted(inw, key=lambda a: a[2]): out.append(f"      {a[0]:11s} k={a[1]:2d} ({a[2]:+3d}): {a[3]:+.3f} [{a[4]:+.3f}, {a[5]:+.3f}] ({a[6]} runs, abstains {a[7]:.2f})")
out.append(f"   outside: {len(outw)} arms ({sum(a[6] for a in outw)} runs): reward minus key from {min(a[3] for a in outw):+.3f} to {max(a[3] for a in outw):+.3f}, mean {np.mean([a[3] for a in outw]):+.4f}; arms below -0.015: {sum(a[3] < -0.015 for a in outw)}")
for a in sorted(outw, key=lambda a: a[2]): out.append(f"      {a[0]:11s} k={a[1]:2d} ({a[2]:+3d}): {a[3]:+.3f} [{a[4]:+.3f}, {a[5]:+.3f}] ({a[6]} runs, abstains {a[7]:.2f})")
x = np.array([a[2] for a in ARMS], float); y = np.array([a[3] for a in ARMS]); over = np.maximum(x - 12, 0); under = np.maximum(-2 - x, 0)
X = np.c_[np.ones(len(x)), over, under]; b = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ b
out.append(f"   least squares over the {len(x)} arms: reward minus key = {b[0]:+.4f} {b[1]:+.4f} per sample beyond k*+12 {b[2]:+.4f} per sample below k*-2; R2 {1 - res.var() / y.var():.2f}")
json.dump(dict(arms=[dict(cell=a[0], k=a[1], dist=a[2], diff=a[3], lo=a[4], hi=a[5], runs=a[6], abst=a[7]) for a in ARMS], fit=[float(x) for x in b], r2=float(1 - res.var() / y.var())), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s21_cut_and_calibration.json"), "w"), indent=1)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s21_cut_and_calibration_out.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
