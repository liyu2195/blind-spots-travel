"""Group diversity over training: lower bound on the share of unanimous groups (8 identical answers => zero advantage) = 2 - distinct,
at the start (steps 1-5), end of epoch 1 (19-23) and end (111-120), by family x student (FK20 / FK00); exact unanimity at step 1 from the
untrained 32 samples (probability that 8 draws are identical), and 'IDK collapse' events (sampled IDK share >= 0.95 at some step)."""
import numpy as np, pandas as pd, json, dyn
from collections import Counter
D = dyn.derive(dyn.logs()); D = D[D.idkc == "idkc05"]
pd.set_option("display.width", 250)
W = {"start": (1, 5), "e1end": (19, 23), "end": (111, 120)}
rows = []
for (st, tag, fam), x in D[D.tag.isin(["FK20", "FK00"])].groupby(["st", "tag", "fam"]):
    r = dict(st=st, tag=tag, fam=fam, runs=x.run.nunique())
    for w, (a, b) in W.items():
        y = x[(x.step >= a) & (x.step <= b)]
        r[f"distinct_{w}"] = y.distinct_per_group.mean(); r[f"unanim_lb_{w}"] = y.agree_lb.mean()
    rows.append(r)
T = pd.DataFrame(rows)
print("== distinct answers per group and lower bound on unanimous groups"); print(T[T.runs >= 4].round(3).to_string(index=False))
# exact unanimity of 8 untrained samples
print("\n== untrained: P(8 samples identical) averaged over the 750 training questions (32 samples, with replacement)")
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        dsn = dyn.DSF[ds]; ids = set(dyn.train_ids(f"{dsn}_h1.jsonl")); S = {}
        for r in map(json.loads, open(f"{dyn.BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
            if r["s"] >= 0 and r["idx"] in ids: S.setdefault(r["idx"], []).append(dyn.sa(r))
        p = [sum((c / len(v)) ** 8 for c in Counter(v).values()) for v in S.values()]
        print(f"  {st} {ds}: P(unanimous) = {np.mean(p):.3f}; share of questions with P > 0.9: {np.mean(np.array(p) > 0.9):.3f}")
# IDK collapse events
print("\n== runs whose sampled IDK share reaches >= 0.95 at some step (first step), FK20 / FK00")
E = []
for run, x in D[D.tag.isin(["FK20", "FK00"])].groupby("run"):
    x = x.sort_values("step"); hit = x[x.idk >= 0.95]
    E.append(dict(run=run, st=x.st.iloc[0], ds=x.ds.iloc[0], fam=x.fam.iloc[0], tag=x.tag.iloc[0], first=int(hit.step.iloc[0]) if len(hit) else 0,
                  end_idk=x[x.step > 110].idk.mean()))
E = pd.DataFrame(E); h = E[E["first"] > 0]
print(f"  {len(h)} of {len(E)} runs; first step quartiles {np.percentile(h['first'], [25, 50, 75])}; share before step 24: {np.mean(h['first'] < 24):.2f}")
print(h.groupby(["st", "ds"]).size().to_string())
print("  of these, end-of-training sampled IDK (steps 111-120) mean", round(h.end_idk.mean(), 3), "; still >= 0.95:", int((h.end_idk >= 0.95).sum()))
