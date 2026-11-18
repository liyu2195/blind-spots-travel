"""(meta) Abstention law. For each run: the uniform-grader answer threshold t* = (c - FA) / (1 - FA - FR) (FA, FR measured on the untrained
student's samples on trained questions, strict key as reference). Predicted abstention on a question set = share of its questions whose untrained
success probability (share of the 32 samples that are strictly right) is below t* (ties count half). Compares with the trained run's abstention.
Writes cache/law.pkl (the table with predictions) and prints fit statistics by family."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, glob, numpy as np, pandas as pd, sys
OUT = (_RA + "/meta/cache")
T = pd.read_pickle(f"{OUT}/runs_table.pkl")
QS = {}
for f in glob.glob(f"{OUT}/runs_*_*.pkl"):
    if f.endswith("runs_table.pkl"): continue
    P = pickle.load(open(f, "rb")); Qf = P["Qf"]
    for part, items in (("home", P["HOME"]), ("new", P["NEW"])):
        QS[(P["st"], P["ds"], part)] = dict(qs=np.array([Qf[i]["q_strict"] for i in items]), ql=np.array([Qf[i]["q_loose"] for i in items]),
                                            g=np.array([Qf[i]["gcode"] for i in items]))
def F(q, t):
    return float(np.mean(q < t - 1e-9) + 0.5 * np.mean(np.abs(q - t) <= 1e-9))
def pred_row(r, part, which="qs"):
    k = (r.st, r.ds, part)
    if k not in QS or not np.isfinite(r.tstar): return np.nan
    return F(QS[k][which], r.tstar)
for part in ("home", "new"):
    T[f"pidk_{part}"] = [pred_row(r, part) for r in T.itertuples()]
    T[f"pidk_l_{part}"] = [pred_row(r, part, "ql") for r in T.itertuples()]
    # predicted accuracy if the policy answers exactly the questions with q >= t* and keeps the untrained greedy answer
    T[f"pright_{part}"] = [np.mean((QS[(r.st, r.ds, part)]["qs"] >= r.tstar) * (QS[(r.st, r.ds, part)]["g"] == 1)) if (r.st, r.ds, part) in QS and np.isfinite(r.tstar) else np.nan for r in T.itertuples()]
T.to_pickle(f"{OUT}/law.pkl")
std = (T.home_in_eval == 1) & (T.ck == 120) & (T.fmt == "1") & (T.temp == 0.8) & (T.adv == "std") & T.tag.str.startswith("FK") & (T.force == 0) & (T.quorum == 0)
def stats(d, y, x):
    d = d[np.isfinite(d[y]) & np.isfinite(d[x])]
    if len(d) < 3: return f"n {len(d)}"
    r = np.corrcoef(d[x], d[y])[0, 1]; mae = np.mean(np.abs(d[y] - d[x])); b = np.polyfit(d[x], d[y], 1)
    return f"n {len(d):4d}  r {r:+.3f}  R2(identity) {1 - np.sum((d[y] - d[x])**2) / np.sum((d[y] - d[y].mean())**2):+.3f}  MAE {mae:.3f}  fit y = {b[1]:+.3f} + {b[0]:.3f} x"
U = T[std & T.fam.isin(["gold", "keyFP", "keyFN", "mixU"])]
print("UNIFORM graders (gold / random FA / random FR / mixed), standard runs")
for part in ("new", "home"):
    print(f" [{part}] all         ", stats(U, f"idk_{part}", f"pidk_{part}"))
    for st in sorted(U.st.unique()): print(f" [{part}] {st:8s}    ", stats(U[U.st == st], f"idk_{part}", f"pidk_{part}"))
    for ds in sorted(U.ds.unique()): print(f" [{part}] ds {ds:8s} ", stats(U[U.ds == ds], f"idk_{part}", f"pidk_{part}"))
print(" loose-key q instead of strict:", stats(U, "idk_new", "pidk_l_new"))
print(" by KL:"); [print(f"   kl {k}", stats(U[U.kl == k], "idk_new", "pidk_new")) for k in sorted(U.kl.unique())]
print(" by IDK credit:"); [print(f"   c {k}", stats(U[U.c == k], "idk_new", "pidk_new")) for k in sorted(U.c.unique())]
print("\nOTHER families on the same law (new questions), standard runs")
for fam in sorted(T[std].fam.unique()):
    d = T[std & (T.fam == fam)]
    print(f"  {fam:9s}", stats(d, "idk_new", "pidk_new"), f" mean resid {np.nanmean(d.idk_new - d.pidk_new):+.3f}")
