"""Per family (FK20 runs, all datasets): metric by visit number k (steps whose 32 questions are all on their k-th visit).
FP side: wrong_paid; FN side: fn_rate; also idk and reward. Shows which grader errors are learned per question (rise with k) and
which are learned in the first epoch (visit-1 steps already moved, see d04)."""
import numpy as np, pandas as pd, dyn
D = dyn.derive(dyn.logs())
D = D[(D.tag == "FK20") & (D.idkc == "idkc05")]
pvc = {}
def pv(seed):
    if seed not in pvc:
        O = dyn.order(seed, 750); pvc[seed] = np.array([b[0][1] if len({v for i, v in b}) == 1 else 0 for b in O])
    return pvc[seed]
D = D.copy(); D["visit"] = [pv(s)[t - 1] for s, t in zip(D.seed, D.step)]
D["jt"] = [dyn.jinfo(g)[1] if f == "judge" else "" for g, f in zip(D.g, D.fam)]
D["fam2"] = np.where(D.fam == "judge", "judge_" + D.jt.astype(str), D.fam)
pd.set_option("display.width", 250)
for m, fams in (("wrong_paid", ["hash", "judge_t50", "judge_a50", "judge_a35", "sonnet", "long", "typeFP", "keyFP", "lenQ"]),
                ("fn_rate", ["judge_t50", "judge_a50", "judge_a35", "sonnet", "plreal", "plsharp", "typeFN", "plscr", "halfFN", "keyFN"]),
                ("idk", ["gold", "judge_t50", "judge_a50", "judge_a35", "plreal", "plsharp", "plscr", "halfFN", "typeFN", "hash"])):
    T = D[D.fam2.isin(fams) & (D.visit > 0)].groupby(["fam2", "st", "visit"])[m].mean().unstack("visit")
    n = D[D.fam2.isin(fams)].groupby(["fam2", "st"]).run.nunique()
    T.insert(0, "n_runs", n)
    print(f"\n=== {m} by visit number (FK20, pooled datasets)"); print(T.round(3).to_string())
D.to_pickle(f"{dyn.HERE}/d10_logs_visit.pkl")
