"""(rev_R5) Check of claim C6 (readability split). The draft splits rows by whether the sign of the probe contrast p = probe_J - probe_J'
agrees with J's actual verdict rj, and reads a positive 'predictable' effect plus a negative 'unpredictable' effect as 'transfer only where readable'.
If a J-trained run's new-question refusal simply follows the probe direction, that split produces this pattern mechanically. Direct test:
per block (student, dataset, J, J'; the c6 rows: NEW questions with a valid untrained greedy answer on which J and J' disagree at a35), regress the
mean final IDK rate of J's runs on [rj, z(p)] with 5 confidence-bin fixed effects (OLS). b_rj = refusal tracking J's verdict BEYOND the linear
readout; b_p = tracking the readout per SD. Averaged over blocks per student (cell-weighted like c6). Also a dose-response: the rj effect
(mean IDK difference, rj minus not rj) within terciles of |p|, and the IDK difference between p > 0 and p <= 0 within rj = 1 and within rj = 0.
Inputs (read-only): stats/cache/base_*, runs_*, probe_*_L16 pickles and final/thresholds (as in c6_readable.py). Output: r5_c6_regress.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
TH = {J: thresholds(J) for J in J4}
out = []; agg = defaultdict(list)
for st in STS:
    for ds in DSS:
        dsn = DSN[ds]
        B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb"))
        PR = pickle.load(open(f"{CACHE}/probe_{st}_{ds}_L16.pkl", "rb"))["probes"]
        v = B["valid"][:, 0]; rej = {J: B["PJ"][J][:, 0] <= TH[J][f"{st}_{dsn}"]["t_a35"] for J in J4}
        sb = np.minimum(4, (B["conf"] * 5).astype(int)); sc = {J: PR[f"rej_{J}"]["score"] for J in J4}
        Y = defaultdict(list)
        for n, r in sorted(R.items()):
            m = re.fullmatch(r"judge([QMGO])a35", r["g"])
            if not m or r["tag"] != "FK20" or r["c"] != "05": continue
            if (r["o"][B["new"]] == 0).mean() >= 0.95: continue
            Y[m.group(1)].append((r["o"] == 0).astype(float))
        for J in J4:
            if J not in Y: continue
            yJ = np.mean(Y[J], 0)
            for Jp in J4:
                if Jp == J: continue
                ii = np.where(B["new"] & v & (rej[J] != rej[Jp]) & np.isfinite(sc[J]) & np.isfinite(sc[Jp]))[0]
                if len(ii) < 20: continue
                rj = rej[J][ii].astype(float); p = sc[J][ii] - sc[Jp][ii]; z = (p - p.mean()) / (p.std() + 1e-9); y = yJ[ii]
                X = np.column_stack([rj, z] + [(sb[ii] == s).astype(float) for s in range(5)])
                keep = X[:, 2:].sum(0) > 0; X = np.column_stack([X[:, :2], X[:, 2:][:, keep]])
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                ap = np.abs(p); q1, q2 = np.quantile(ap, [1 / 3, 2 / 3]); terc = np.digitize(ap, [q1, q2])
                dose = []
                for t in range(3):
                    m1 = (terc == t) & (rj == 1); m0 = (terc == t) & (rj == 0)
                    dose.append(y[m1].mean() - y[m0].mean() if m1.any() and m0.any() else np.nan)
                pr1 = y[(rj == 1) & (p > 0)].mean() - y[(rj == 1) & (p <= 0)].mean() if ((rj == 1) & (p > 0)).any() and ((rj == 1) & (p <= 0)).any() else np.nan
                pr0 = y[(rj == 0) & (p > 0)].mean() - y[(rj == 0) & (p <= 0)].mean() if ((rj == 0) & (p > 0)).any() and ((rj == 0) & (p <= 0)).any() else np.nan
                agg[(st, ds, J)].append((coef[0], coef[1], *dose, pr1, pr0, len(ii)))
for who in STS:
    cells = [np.nanmean(np.array(L), 0) for k, L in agg.items() if k[0] == who]
    C_ = np.array(cells); m = np.nanmean(C_, 0)
    out.append(f"{who}: cells {len(cells)} | b_rj (verdict beyond readout) {m[0]:+.3f} (cells > 0: {int((C_[:, 0] > 0).sum())}/{len(cells)}) | "
               f"b_p per SD {m[1]:+.3f} (cells > 0: {int((C_[:, 1] > 0).sum())}/{len(cells)}) | rj effect by |p| tercile low/mid/high {m[2]:+.3f} {m[3]:+.3f} {m[4]:+.3f} | "
               f"IDK(p>0) - IDK(p<=0): within rj=1 {m[5]:+.3f}, within rj=0 {m[6]:+.3f} | mean rows per block {m[7]:.0f}")
    for k, L in sorted(agg.items()):
        if k[0] != who: continue
        a = np.nanmean(np.array(L), 0)
        out.append(f"    {k[1]} J={k[2]}: b_rj {a[0]:+.3f}  b_p {a[1]:+.3f}  dose {a[2]:+.3f} {a[3]:+.3f} {a[4]:+.3f}  p-split within rj=1 {a[5]:+.3f} rj=0 {a[6]:+.3f}")
open((_RA + "/rev_R5/r5_c6_regress.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
