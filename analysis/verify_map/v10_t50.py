"""(verify_map) Crossover at the default operating point (judge<J>t50 runs; verdict: P(Yes) > 0.5 pays), both signs, same estimator as v05
(units: new questions, valid untrained greedy; RIGHT = loose key or Sonnet >= 4; WRONG = key-wrong & Sonnet 1..3; cells need |A|, |B| >= 5;
runs with new abstention >= 0.95 dropped). Only cells with >= 2 judges having t50 runs: h1 (Q M G O), nq (Q G O). Also the mean abstention.
Output: v10_t50_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, itertools
sys.path.insert(0, (_RA + "/verify_map"))
from v00_common import *
P = Log(f"{HERE}/v10_t50_out.txt"); P(__doc__); rng = np.random.default_rng(10)
for kind in ("right", "wrong"):
    X = {}; boots = {st: [] for st in STS}
    for st in STS:
        for ds in DSS:
            D = load(st, ds); cd0 = D["cd"][:, 0]; v = np.isin(cd0, (1, 2)); s0 = D["son0"]
            good = ((cd0 == 1) | (s0 >= 4)) if kind == "right" else ((cd0 == 2) & (s0 >= 1) & (s0 <= 3))
            U = np.where(D["new"] & v & good)[0]; acc = {J: D["PJ"][J][U, 0] > 0.5 for J in J5}; Y = {}; AB = {}
            for n, r in sorted(D["runs"].items()):
                m = re.fullmatch(r"judge([QMGOR])t50", r["g"]); a = (r["o"][D["new"]] == 0).mean()
                if not m or a >= 0.95: continue
                Y.setdefault(m.group(1), []).append((r["o"][U] == 0).astype(float)); AB.setdefault(m.group(1), []).append(a)
            for K, J in itertools.combinations([J for J in J5 if J in Y], 2):
                if kind == "right": A, B = (~acc[K]) & acc[J], (~acc[J]) & acc[K]
                else: A, B = acc[K] & ~acc[J], acc[J] & ~acc[K]
                if A.sum() < 5 or B.sum() < 5: continue
                g = lambda J_, w: (np.mean(Y[J_], 0) @ (w * A)) / (w * A).sum() - (np.mean(Y[J_], 0) @ (w * B)) / (w * B).sum()
                w1 = np.ones(len(U)); X[(st, ds, K, J)] = (g(K, w1) - g(J, w1), int(A.sum()), int(B.sum()), np.mean(AB[K]), np.mean(AB[J]))
    for (st, ds, K, J), x in sorted(X.items()):
        P(f"  {kind} {st} {ds} {K}-{J}: X {x[0]:+.3f} (|A| {x[1]}, |B| {x[2]}; abstention {K} {x[3]:.2f}, {J} {x[4]:.2f})")
    per = [np.mean([x[0] for k, x in X.items() if k[0] == st]) for st in STS if any(k[0] == st for k in X)]
    P(f"  {kind}: cells {len(X)}, negative {sum(x[0] < 0 for x in X.values())}; pooled mean {np.mean(per):+.3f} (per student " + ", ".join(f"{p:+.3f}" for p in per) + ")")
P.save()
