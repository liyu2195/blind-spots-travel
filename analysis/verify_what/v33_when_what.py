"""(verify_what) Item 3 companion (dynamics d18-d20 claim): Q-judge FK20 runs vs key-trained FK20 runs, per part (home / new, dedup), on
questions whose untrained greedy answer is key-wrong (well-formed, not IDK) and has a Q verdict (exact first line, else normalised):
  WHEN: sel = P(run answers | greedy judge-paid) - P(run answers | greedy judge-rejected), threshold of the run's level;
  WHAT: chg = Q pay rate of the run's final wrong answer - pay rate of the untrained greedy, on questions wrong in both (verdict available).
Key-trained runs evaluated at each level's threshold. Abstention matching: each judge run vs the key runs of the same student and set whose
refusal rate on new questions is within 0.05 (difference judge - mean matched key). Output v33_when_what.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import re, json, pickle, sys, numpy as np, pandas as pd
sys.path.insert(0, _RC)
import st_common as C
OUT = (_RA + "/verify_what")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
TH = json.load(open((_RX + "/final/thresholds_Q.json")))
def outcome(x, al):
    if x is None or x == "": return 3
    if IDK_FULL.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
rows = []
for st in ("qwen7", "llama8"):
    for ds, dsn in (("h1", "tqa"), ("pq", "pqa"), ("nq", "nqo")):
        Dd = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); Q = Dd["Q"]; B = Dd["B"]; R = Dd["R"]; N = len(Q["train"]); al = Q["aliases"]
        JQ = pickle.load(open(f"{OUT}/jl_{dsn}.pkl", "rb"))["Q"]
        def pyes(i, line, x):
            v = JQ["line"].get((i, line)); return v if v is not None else JQ["norm"].get((i, x))
        thr = {"a35": TH[f"{st}_{dsn}"]["t_a35"], "a50": TH[f"{st}_{dsn}"]["t_a50"], "t50": 0.5}
        go = [outcome(B["g"][i], al[i]) for i in range(N)]; p0 = [pyes(i, B["gl"][i], B["g"][i]) if go[i] == 2 else None for i in range(N)]
        parts = {"home": [i for i in range(N) if Q["train"][i]], "new": [i for i in range(N) if not Q["train"][i] and not Q["dup"][i]]}
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05": continue
            m = re.fullmatch(r"judgeQ(a35|a50|t50)", r["g"]); key = re.fullmatch(r"sim\d\dfn\d\d", r["g"])
            if not (m or key): continue
            o = [outcome(r["ck120"][i], al[i]) for i in range(N)]
            a_new = np.mean([o[i] == 0 for i in parts["new"]])
            for lv in ([m.group(1)] if m else ["a35", "a50", "t50"]):
                t = thr[lv]
                for part, items in parts.items():
                    U = [i for i in items if p0[i] is not None]
                    pa = [o[i] in (1, 2) for i in U if p0[i] > t]; rj = [o[i] in (1, 2) for i in U if p0[i] <= t]
                    w = [(pyes(i, r["ck120_line"][i], r["ck120"][i]), p0[i]) for i in U if o[i] == 2]; w = [(a, b) for a, b in w if a is not None]
                    rows.append(dict(st=st, ds=ds, run=name, kind="judgeQ" if m else "key", lv=lv, part=part, a_new=a_new,
                                     sel=(np.mean(pa) - np.mean(rj)) if pa and rj else np.nan,
                                     chg=np.mean([a > t for a, b in w]) - np.mean([b > t for a, b in w]) if w else np.nan, n_chg=len(w)))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/v33_rows.csv", index=False)
for lv in ("a35", "a50", "t50"):
    for st in ("qwen7", "llama8"):
        for part in ("home", "new"):
            J = T[(T.kind == "judgeQ") & (T.lv == lv) & (T.st == st) & (T.part == part)]; K = T[(T.kind == "key") & (T.lv == lv) & (T.st == st) & (T.part == part)]
            dm = []
            for _, x in J.iterrows():
                mk = K[(K.ds == x.ds) & (np.abs(K.a_new - x.a_new) <= 0.05)]
                if len(mk): dm.append((x.sel - mk.sel.mean(), x.chg - mk.chg.mean()))
            dm = np.array(dm) if dm else np.full((1, 2), np.nan)
            P(f"{lv} {st:6s} {part:4s}: judge runs {len(J):2d} sel {J.sel.mean():+.3f} chg {J.chg.mean():+.3f} | key runs {len(K):3d} sel {K.sel.mean():+.3f} chg {K.chg.mean():+.3f} | "
              f"abstention-matched judge - key ({len(dm) if np.isfinite(dm).all() else 0} runs matched): sel {np.nanmean(dm[:, 0]):+.3f} chg {np.nanmean(dm[:, 1]):+.3f}")
open(f"{OUT}/v33_when_what.txt", "w").write("\n".join(L) + "\n")
