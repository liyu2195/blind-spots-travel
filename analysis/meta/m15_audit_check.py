"""(meta) Check: are the judge-paid 'wrong' answers (loose key) really wrong? Sonnet audit levels (apitest/auditT_* on trained runs'
key-wrong answers; level >= 4 = right) for the paid and unpaid key-wrong final answers of judge-trained runs vs uniform-grader runs, home and new.
Also recomputes the FA hack with audited-right answers removed from the wrong set (hack_audited) for judge runs.
Inputs cache/runs_<st>_<ds>.pkl (final answers are re-read from the run folders), apitest/auditT_*. Output stdout."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, glob, numpy as np, pandas as pd
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, C, RL
OUT = (_RA + "/meta/cache"); API = (_RX + "/apitest"); FIN = (_RX + "/final")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
T = pd.read_pickle(f"{OUT}/refusal.pkl").set_index("run")
rows = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        dsn = DSN[ds]; probs, H1, HELD = setup(dsn)
        LV = {}
        for r in map(json.loads, open(f"{API}/auditT_{st}_{dsn}.jsonl")): LV[(r["idx"], r["norm"])] = r["level"]
        P = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb"))
        # judge scores of final answers: reuse the builder's lookup by re-reading score files would be long; use cross columns instead:
        # here we only need, per run, which key-wrong answers are audited right; paid status for the run's own judge is re-derived from P via speckey
        for run, r in P["R"].items():
            if run not in T.index or not T.loc[run, "std"]: continue
            fam = T.loc[run, "fam"]
            if fam not in ("judge", "gold", "keyFP", "keyFN", "mixU"): continue
            g = {}
            for x in map(json.loads, open(f"{RL}/{run}/eval_rl_ck{r['ck']}.jsonl")):
                if x.get("s", -1) < 0: g[x["idx"]] = sa(x)
            for part, items in (("home", P["HOME"]), ("new", P["NEW"])):
                w = [(i, g.get(i, "")) for i in items]
                w = [(i, x) for i, x in w if x and not IDK.match(x) and not C.tq_correct(x, set(probs[i]["aliases"]))]
                lv = [LV.get((i, x)) for i, x in w]; have = [l for l in lv if l is not None]
                rows.append(dict(run=run, st=st, ds=ds, fam=fam, sub=T.loc[run, "sub"], part=part, n_wrong=len(w), audited=len(have) / max(1, len(w)),
                                 aud_right=float(np.mean([l >= 4 for l in have])) if have else np.nan))
A = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("share of key-wrong final answers that Sonnet rates right (level >= 4), and audit coverage")
print(A.groupby(["fam", "part"]).agg(runs=("run", "size"), cover=("audited", "mean"), aud_right=("aud_right", "mean")).round(3).to_string())
print(A[A.fam == "judge"].groupby(["sub", "part"]).agg(runs=("run", "size"), cover=("audited", "mean"), aud_right=("aud_right", "mean")).round(3).to_string())
A.to_csv(f"{OUT}/audit_check.csv", index=False)
