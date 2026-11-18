"""(rev_F3) Check: does the Qwen judge's imprint survive away from the extreme a35 cut?  Same estimator as verify_map/v07_spec.py
(grader-specific slope per SD of the residualised predictor: judge run minus the mean slope of its 3 nearest-abstention reference runs
sim<FP>fn<FN> / lenQ, computed with the same x on the same answers; units = known-sure, audited-right untrained answers on NEW questions;
G = [conf, logit P_aud, logit P_sure]), applied to judgeQa35, judgeQa50 and judgeQt50 runs (FK20 credit 0.5 as cached; new abstention < 0.95),
with x = the lin16 probe of the Q judge's rejection AT THE RUN'S OWN CUT (rej_Q_<op>, v02) and x = the actual verdict at that cut (v_bin).
Because refusal is rare at t50, the slope is also reported divided by the run's mean refusal rate on the units (relative imprint).
Question bootstrap (200 draws, questions within cell). Inputs read-only from verify_map/cache. Output: z3_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, pickle, json, numpy as np
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import load, logit, thr, CACHE, STS, DSS
out = []; rng = np.random.default_rng(3)
def P(s): print(s, flush=True); out.append(s)
REF = re.compile(r"^(sim\d\dfn\d\d|lenQ)$")
def slopes(G, x, Y, w):
    X1 = np.column_stack([np.ones(len(x)), G]); WX = X1 * w[:, None]; M = np.column_stack([x, Y])
    R = M - X1 @ np.linalg.solve(X1.T @ WX + 1e-9 * np.eye(X1.shape[1]), WX.T @ M); xp = R[:, 0]; Yp = R[:, 1:]
    sxx = (w * xp * xp).sum(); return (w[:, None] * xp[:, None] * Yp).sum(0) / sxx * np.sqrt(sxx / w.sum())
res = {}
for st in STS:
    for ds in DSS:
        D = load(st, ds); P16 = pickle.load(open(f"{CACHE}/p_{st}_{ds}_L16.pkl", "rb"))["T"]
        cd = D["cd"]; v = np.isin(cd[:, 0], (1, 2)); aud = v & ((cd[:, 0] == 1) | (D["son0"] >= 4)); sure = v & (D["conf"] >= 31 / 32)
        U = np.where(D["new"] & aud & sure)[0]
        G = np.column_stack([D["conf"][U], logit(P16["aud"]["p"][U]), logit(P16["sure"]["p"][U])])
        runs = []
        for n, r in sorted(D["runs"].items()):
            a = float((r["o"][D["new"]] == 0).mean())
            if a >= 0.95: continue
            m = re.fullmatch(r"judgeQ(a35|a50|t50)", r["g"]); k = m.group(1) if m else ("ref" if REF.match(r["g"]) else None)
            if k: runs.append((k, a, (r["o"][U] == 0).astype(float)))
        if not runs: continue
        K = np.array([r[0] for r in runs]); A = np.array([r[1] for r in runs]); Y = np.column_stack([r[2] for r in runs]); iref = np.where(K == "ref")[0]
        for op in ("a35", "a50", "t50"):
            ij = np.where(K == op)[0]
            if len(ij) == 0 or f"rej_Q_{op}" not in P16: continue
            xs = {"lin16": logit(P16[f"rej_Q_{op}"]["p"][U]), "v_bin": (D["PJ"]["Q"][U, 0] <= thr("Q", st, ds, op)).astype(float)}
            for kx, x in xs.items():
                if x.std() < 1e-9: continue
                def est(w):
                    sl = slopes(G, x, Y, w); d = []; rel = []
                    for i in ij:
                        nb = iref[np.argsort(np.abs(A[iref] - A[i]), kind="stable")[:3]]
                        d.append(sl[i] - sl[nb].mean()); rate = (w * Y[:, i]).sum() / w.sum(); rel.append((sl[i] - sl[nb].mean()) / max(rate, 0.02))
                    return np.mean(d), np.mean(rel), np.mean(Y[:, ij].mean(0))
                e = est(np.ones(len(U)))
                bs = np.array([est(np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U)).astype(float))[:2] for _ in range(200)])
                res[(st, ds, op, kx)] = (e, bs, len(ij), float(A[ij].mean()))
        del D
P(__doc__)
for op in ("a35", "a50", "t50"):
    for kx in ("lin16", "v_bin"):
        for st in STS:
            cells = [k for k in res if k[0] == st and k[2] == op and k[3] == kx]
            if not cells: continue
            e = np.mean([res[k][0][0] for k in cells]); r_ = np.mean([res[k][0][1] for k in cells])
            n = min(res[k][1].shape[0] for k in cells); bb = np.mean([res[k][1][:n] for k in cells], 0)
            lo, hi = np.percentile(bb[:, 0], [2.5, 97.5]); rlo, rhi = np.percentile(bb[:, 1], [2.5, 97.5])
            det = " ".join(f"{k[1]}:{res[k][0][0]:+.3f}(runs {res[k][2]}, abst {res[k][3]:.2f}, unit IDK {res[k][0][2]:.2f})" for k in cells)
            P(f"Q {op} {kx:6s} {st:6s}: grader-specific slope {e:+.3f} [{lo:+.3f}, {hi:+.3f}] | relative to refusal rate {r_:+.2f} [{rlo:+.2f}, {rhi:+.2f}] | {det}")
open((_RA + "/rev_F3/z3_out.txt"), "w").write("\n".join(out) + "\n")
