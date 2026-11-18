"""Per judge (Q, G, M, O, R, Sonnet) x threshold x student, FK20 runs pooled over datasets: wrong_paid and FN at visit 1 (first 5 and
last 5 steps of epoch 1) and at visits 2-5; epoch-1 change (transfer) vs. visit-1 -> visit-5 change. Seeds split for confirmation:
seeds 0-1 (discovery) vs seeds >= 2 (confirmation), when available."""
import numpy as np, pandas as pd, dyn
D = pd.read_pickle(f"{dyn.HERE}/d10_logs_visit.pkl")
D = D[D.fam.isin(["judge", "sonnet"])].copy()
D["J"] = [("S" if f == "sonnet" else dyn.jinfo(g)[0]) + "_" + ("api" if f == "sonnet" else dyn.jinfo(g)[1] + dyn.jinfo(g)[2]) for g, f in zip(D.g, D.fam)]
rows = []
for (J, st, run), x in D.groupby(["J", "st", "run"]):
    x = x.sort_values("step"); r = dict(J=J, st=st, run=run, ds=x.ds.iloc[0], seed=x.seed.iloc[0])
    for m in ("wrong_paid", "fn_rate", "idk"):
        e = x[x.step <= 23]
        r[f"{m}_s1-5"] = e[e.step <= 5][m].mean(); r[f"{m}_s19-23"] = e[e.step >= 19][m].mean()
        for k in (2, 3, 4, 5): r[f"{m}_v{k}"] = x[x.visit == k][m].mean()
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{dyn.HERE}/d15_judges_runs.csv", index=False)
pd.set_option("display.width", 260)
for m in ("wrong_paid", "fn_rate"):
    T = R.groupby(["J", "st"]).agg(n=("run", "size"), start=(f"{m}_s1-5", "mean"), end_e1=(f"{m}_s19-23", "mean"),
                                    v2=(f"{m}_v2", "mean"), v3=(f"{m}_v3", "mean"), v5=(f"{m}_v5", "mean")).reset_index()
    T["epoch1_change"] = T.end_e1 - T.start; T["revisit_change_v2_to_v5_vs_e1end"] = T.v5 - T.end_e1
    T["share_revisit"] = T["revisit_change_v2_to_v5_vs_e1end"] / (T["revisit_change_v2_to_v5_vs_e1end"].abs() + T["epoch1_change"].abs())
    print(f"\n=== {m}: start (steps 1-5), end of epoch 1 (19-23), visits 2/3/5; epoch-1 change vs later (visit 5 - end of epoch 1)")
    print(T.round(3).to_string(index=False))
# seed split confirmation for the pooled judge family
for m in ("wrong_paid", "fn_rate"):
    print(f"\n=== {m}: seed split (pooled judges, not Sonnet)")
    for lab, y in (("seeds 0-1", R[(R.seed <= 1) & ~R.J.str.startswith("S")]), ("seeds 2+", R[(R.seed >= 2) & ~R.J.str.startswith("S")])):
        for st in ("qwen7", "llama8"):
            z = y[y.st == st]
            e1 = (z[f"{m}_s19-23"] - z[f"{m}_s1-5"]); lt = (z[f"{m}_v5"] - z[f"{m}_s19-23"])
            print(f"  {lab:9s} {st:7s} runs={len(z):3d} epoch1 {e1.mean():+.3f} (runs>0: {np.mean(e1 > 0):.2f})  later {lt.mean():+.3f} (runs>0: {np.mean(lt > 0):.2f})")
