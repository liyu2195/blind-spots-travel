"""(meta) How selective is the learned abstention? For every run, on NEW questions: AUROC of 'run says IDK' against the untrained student's
success probability (share of 32 samples right; low success should be refused first) and the collateral share = P(IDK | untrained greedy right).
Compared with the ideal selective policy at the same abstention rate (refuse the questions with the lowest untrained success first).
Also: how predictable each grader's rejections are from the student's own confidence alone (home questions), for every FR-structured grader
(AUROC of rej_q >= .5 against -conf among untrained-right questions), which tests why self-generated-key graders (frz) over-abstain.
Inputs cache/runs_<st>_<ds>.pkl, cache/drive.pkl. Output: stdout (out_m12_selectivity.txt) and cache/selectivity.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT = (_RA + "/meta/cache")
T = pd.read_pickle(f"{OUT}/drive.pkl").set_index("run")
rows = []; conf_rows = []
for f in sorted(glob.glob(f"{OUT}/runs_*_*.pkl")):
    if f.endswith("runs_table.pkl"): continue
    P = pickle.load(open(f, "rb")); st, ds = P["st"], P["ds"]; Qf = P["Qf"]
    qn = np.array([Qf[i]["q_loose"] for i in P["NEW"]]); gn = np.array([Qf[i]["gcode"] for i in P["NEW"]])
    qh = np.array([Qf[i]["q_loose"] for i in P["HOME"]]); gh = np.array([Qf[i]["gcode"] for i in P["HOME"]]); ch = np.array([Qf[i]["conf"] for i in P["HOME"]])
    for run, r in P["R"].items():
        o = r["new"]["o"]; idk = (o == 0).astype(int); a = idk.mean()
        auc = roc_auc_score(idk, -qn) if 0 < idk.sum() < len(idk) else np.nan
        # ideal: refuse the round(a n) lowest-q questions (ties by index)
        k = int(round(a * len(qn))); ideal = np.zeros(len(qn), bool); ideal[np.argsort(qn, kind="stable")[:k]] = True
        rows.append(dict(run=run, sel_auc_new=auc, coll_new=float(idk[gn == 1].mean()), coll_ideal_new=float(ideal[gn == 1].mean()),
                         share_refusals_uright=float((idk & (gn == 1)).sum() / max(1, idk.sum())), share_refusals_uright_ideal=float((ideal & (gn == 1)).sum() / max(1, ideal.sum()))))
    if "SPECQ" in P:
        seen = set()
        for run, r in P["R"].items():
            key = r["speckey"]
            if key in seen: continue
            seen.add(key); rej = P["SPECQ"][key]["home"]["rej"]; sel = (gh == 1) & np.isfinite(rej)
            y = (rej[sel] >= 0.5).astype(int)
            if 10 <= y.sum() <= len(y) - 10:
                conf_rows.append(dict(st=st, ds=ds, run=run, auc_conf=roc_auc_score(y, -ch[sel]), auc_q=roc_auc_score(y, -qh[sel]), pos=float(y.mean())))
S = pd.DataFrame(rows).set_index("run"); T = T.join(S, how="left"); T.reset_index().to_pickle(f"{OUT}/selectivity.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
M = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & (T.kl == 0.2) & (T.c == 0.5)].reset_index()
M["abin"] = pd.cut(M.idk_new, [0, .1, .3, .5, .7, .9, 1.0])
print("NEW questions, qwen7 + llama8 standard runs, by abstention bin: actual vs ideal-selective refusal of untrained-right questions")
print(M.groupby(["abin", "st"], observed=True).agg(n=("run", "size"), idk=("idk_new", "mean"), sel_auc=("sel_auc_new", "mean"), coll=("coll_new", "mean"), coll_ideal=("coll_ideal_new", "mean"),
      refusals_uright=("share_refusals_uright", "mean"), refusals_uright_ideal=("share_refusals_uright_ideal", "mean")).round(3).to_string())
print("\nselectivity AUROC by family (runs with 5-95% abstention):")
d = M[(M.idk_new > 0.05) & (M.idk_new < 0.95)]
print(d.groupby("fam").agg(n=("run", "size"), idk=("idk_new", "mean"), sel_auc=("sel_auc_new", "mean"), sd=("sel_auc_new", "std"), coll=("coll_new", "mean"), coll_ideal=("coll_ideal_new", "mean")).round(3).to_string())
C = pd.DataFrame(conf_rows).merge(T.reset_index()[["run", "fam", "sub"]], on="run")
print("\nhow well the student's own confidence predicts each grader's rejections of right answers (home, untrained-right questions)")
print(C.groupby(["fam", "sub", "st"]).agg(n=("ds", "size"), auc_conf=("auc_conf", "mean"), auc_q=("auc_q", "mean"), pos=("pos", "mean")).round(3).to_string())
