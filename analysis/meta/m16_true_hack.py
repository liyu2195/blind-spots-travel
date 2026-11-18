"""(meta) Check of the FA-hack transfer for strict judges (a35): restrict the new-question wrong answers to those that the Sonnet audit
also rates wrong (level <= 3), then compare the judge's acceptance of them in judge-trained runs vs uniform-grader runs (same student, dataset).
Judge P(Yes) of final answers from the score_<J>_cand_runs files (same joins as m01). Output stdout."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, os, glob, pickle, numpy as np, pandas as pd
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, C, RL, BASE, DATA
OUT = (_RA + "/meta/cache"); API = (_RX + "/apitest"); FIN = (_RX + "/final")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
T = pd.read_pickle(f"{OUT}/refusal.pkl").set_index("run")
def jl(p): return [json.loads(l) for l in open(p)]
rows = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        dsn = DSN[ds]; probs, H1, HELD = setup(dsn); NEW = list(HELD)
        LV = {}
        for f in (f"{API}/auditT_{st}_{dsn}.jsonl", f"{API}/frontierH_{st}_{dsn}.jsonl"):
            for r in map(json.loads, open(f)): LV[(r["idx"], r["norm"])] = r["level"]
        for J in ("Q", "M", "G", "O"):
            d = {}
            for cf, sf in ((f"{DATA}/cand_runs_{dsn}.jsonl", f"{BASE}/score_{J}_cand_runs_{dsn}.jsonl"), (f"{DATA}/cand_runs_{dsn}.jsonl_before_delta", f"{BASE}/score_{J}_cand_runs_{dsn}.jsonl"),
                           (f"{DATA}/cand_runs_{dsn}2.jsonl", f"{BASE}/score_{J}_cand_runs_{dsn}2.jsonl")):
                if not (os.path.exists(cf) and os.path.exists(sf)): continue
                a_, b_ = jl(cf), jl(sf)
                if len(a_) != len(b_): continue
                for a, b in zip(a_, b_): d.setdefault((a["idx"], a["norm"]), b["py"] / (b["py"] + b["pn"]) if b["py"] + b["pn"] > 1e-3 else 0.5)
            t = json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_{dsn}"]["t_a35"]
            runs = [r for r in T.index if T.loc[r, "st"] == st and T.loc[r, "ds"] == ds and T.loc[r, "std"] and T.loc[r, "kl"] == 0.2 and T.loc[r, "c"] == 0.5
                    and (T.loc[r, "sub"] == f"judge{J}a35" or T.loc[r, "fam"] in ("gold", "keyFP", "keyFN", "mixU"))]
            for run in runs:
                ck = int(T.loc[run, "ck"]); g = {}
                for x in map(json.loads, open(f"{RL}/{run}/eval_rl_ck{ck}.jsonl")):
                    if x.get("s", -1) < 0: g[x["idx"]] = sa(x)
                acc_sw, acc_sr, n_sw, n_sr, n_un = [], [], 0, 0, 0
                for i in NEW:
                    x = g.get(i, "")
                    if not x or IDK.match(x) or C.tq_correct(x, set(probs[i]["aliases"])): continue
                    p = d.get((i, x)); l = LV.get((i, x))
                    if p is None or l is None: n_un += 1; continue
                    (acc_sr if l >= 4 else acc_sw).append(p > t)
                rows.append(dict(st=st, ds=ds, J=J, run=run, judge_run=T.loc[run, "sub"] == f"judge{J}a35", idk=T.loc[run, "idk_new"],
                                 acc_sonnet_wrong=np.mean(acc_sw) if len(acc_sw) >= 10 else np.nan, n_sw=len(acc_sw), acc_sonnet_right=np.mean(acc_sr) if len(acc_sr) >= 5 else np.nan,
                                 n_sr=len(acc_sr), unaudited=n_un))
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/true_hack.csv", index=False)
pd.set_option("display.width", 250)
g = A.groupby(["J", "st", "ds", "judge_run"]).agg(runs=("run", "size"), idk=("idk", "mean"), acc_sonnet_wrong=("acc_sonnet_wrong", "mean"), n_sw=("n_sw", "mean"),
                                                  acc_sonnet_right=("acc_sonnet_right", "mean"), n_sr=("n_sr", "mean"), unaudited=("unaudited", "mean")).round(3)
print("new questions: judge (a35) acceptance of key-wrong answers that Sonnet also rates wrong (acc_sonnet_wrong) / rates right, judge-trained vs uniform runs")
print(g.to_string())
w = A.pivot_table(index=["J", "st", "ds"], columns="judge_run", values="acc_sonnet_wrong", aggfunc="mean")
w["diff"] = w[True] - w[False]
print("\njudge-trained minus uniform, acceptance of Sonnet-confirmed wrong answers on new questions:"); print(w.round(3).to_string())
print("mean diff", round(w["diff"].mean(), 3), "positive in", int((w["diff"] > 0).sum()), "of", int(w["diff"].notna().sum()))
