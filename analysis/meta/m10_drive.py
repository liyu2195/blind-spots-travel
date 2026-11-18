"""(meta) GRPO-aware abstention drive, computed from the UNTRAINED student's samples and the run's grader only.
For a trained question i let p_i = probability that the grader accepts one untrained (valid, non-IDK) sample. In a group of 8 with one
'I don't know' (reward c) and 7 answers with k ~ Binomial(7, p_i) accepted (reward 1, else 0):
   std-normalised advantage of the IDK reply  A_std(k) = (c - m) / sd,   m, sd = mean, population sd of the 8 rewards (sd floor 1e-4 as in GRPO eps)
   unnormalised (Dr. GRPO, ADV=none)           A_none(k) = c - m.
drive D = mean over trained questions of E_k[A(k)] (the push on the shared IDK reply summed over questions, per question).
Also D_pos = share of trained questions with E[A_std] > 0 (per-question 'IDK wins').
Writes cache/drive.pkl (law2 table + D_std, D_none, D_pos, D_used = D_std or D_none by the run's ADV setting)."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, glob, numpy as np, pandas as pd
from math import comb
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, C, BASE
OUT = (_RA + "/meta/cache")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
T = pd.read_pickle(f"{OUT}/law2.pkl")
def adv_table(c):
    a_std, a_none = np.zeros(8), np.zeros(8)
    for k in range(8):
        r = np.array([c] + [1.0] * k + [0.0] * (7 - k)); m = r.mean(); sd = r.std()
        a_std[k] = (c - m) / (sd + 1e-4); a_none[k] = c - m
    return a_std, a_none
BIN = {}
def binom_w(p):
    return np.array([comb(7, k) * p**k * (1 - p)**(7 - k) for k in range(8)])
res = {}
for f in sorted(glob.glob(f"{OUT}/runs_*_*.pkl")):
    if f.endswith("runs_table.pkl"): continue
    P = pickle.load(open(f, "rb")); st, ds = P["st"], P["ds"]
    if "SPECQ" not in P: continue
    dsn = DSN[ds]; probs, H1, HELD = setup(dsn); HOME = P["HOME"]
    # counts of strictly-right and wrong valid samples per trained question
    cnt = {i: [0, 0] for i in HOME}; hs = set(HOME)
    bf = f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl"
    import os
    if not os.path.exists(bf): bf = f"{BASE}/{st}_{dsn}_1500_idk_n32_s1.jsonl"
    for r in map(json.loads, open(bf)):
        if r["s"] < 0 or r["idx"] not in hs: continue
        x = sa(r)
        if not x or IDK.match(x): continue
        cnt[r["idx"]][0 if C.tq_strict(x, set(probs[r["idx"]]["aliases"])) else 1] += 1
    nr = np.array([cnt[i][0] for i in HOME], float); nw = np.array([cnt[i][1] for i in HOME], float)
    for run, r in P["R"].items():
        q = P["SPECQ"].get(r["speckey"])
        if q is None: continue
        rej = np.nan_to_num(q["home"]["rej"], nan=0.0); fa = np.nan_to_num(q["home"]["fa"], nan=0.0)
        tot = nr + nw; ok = tot > 0
        p = np.where(ok, (nr * (1 - rej) + nw * fa) / np.maximum(tot, 1), np.nan)
        a_std, a_none = adv_table(r["c"])
        W = np.array([binom_w(pi) for pi in p[ok]])
        e_std = W @ a_std; e_none = W @ a_none
        res[run] = dict(D_std=float(e_std.mean()), D_none=float(e_none.mean()), D_pos=float((e_std > 0).mean()), p_mean=float(np.nanmean(p)),
                        D_std_med=float(np.median(e_std)))
D = pd.DataFrame.from_dict(res, orient="index"); D.index.name = "run"
T = T.set_index("run").join(D, how="left").reset_index()
T["D_used"] = np.where(T.adv == "none", T.D_none, T.D_std)
T.to_pickle(f"{OUT}/drive.pkl")
print("drive computed for", D.shape[0], "runs")
