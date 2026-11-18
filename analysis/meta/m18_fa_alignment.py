"""(meta) Are a grader's false accepts aligned with what the student knows? For each grader (one spec per sub-family, home questions), AUROC of
'accepted' among the untrained student's key-wrong samples against (a) the student's per-question confidence and (b) how often the student gives that
exact answer among its 32 samples. Explains off-curve graders (Sonnet, frozen self-keys). Inputs cache/runs_<st>_<ds>.pkl, cache/law2.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
sys.path.insert(0, (_RX + "/common"))
from rc import sa, BASE
OUT = (_RA + "/meta/cache"); DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
T = pd.read_pickle(f"{OUT}/law2.pkl"); rows = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        P = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); Qf = P["Qf"]
        U = {}
        for r in map(json.loads, open(f"{BASE}/{st}_{DSN[ds]}_1500_n32_s1.jsonl")): U[(r["idx"], r["s"])] = sa(r)
        freq = {}
        for (i, s), x in U.items():
            if s >= 0 and x: freq[(i, x)] = freq.get((i, x), 0) + 1
        D = T[(T.st == st) & (T.ds == ds) & T["std"] & ~T.fam.isin(["gold", "keyFP", "keyFN", "mixU"])]
        seen = set()
        for run, sub, fam in zip(D.run, D["sub"], D.fam):
            if sub in seen: continue
            seen.add(sub); W = P["SPECQ"][P["R"][run]["speckey"]]["home"]
            y = (W["w_acc"] > 1e-9).astype(int)
            if len(y) < 100 or y.sum() < 20 or (1 - y).sum() < 20: continue
            conf = np.array([Qf[int(i)]["conf"] for i in W["w_idx"]]); fq = np.array([freq.get((int(i), U.get((int(i), int(s)), "")), 0) for i, s in zip(W["w_idx"], W["w_s"])])
            rows.append(dict(st=st, ds=ds, fam=fam, sub=sub, auc_conf=roc_auc_score(y, conf), auc_freq=roc_auc_score(y, fq), FA=float(W["w_acc"].mean())))
A = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(A.groupby(["fam", "sub"]).agg(n=("ds", "size"), FA=("FA", "mean"), auc_conf=("auc_conf", "mean"), auc_freq=("auc_freq", "mean")).round(3).to_string())
A.to_csv(f"{OUT}/fa_alignment.csv", index=False)
