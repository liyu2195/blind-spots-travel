"""Does the judge's wrong_paid (and reward) depend on training STEPS or on VISITS per question? For each visit regime, the mean of the
metric over the steps in which every batch question is on its k-th visit (pure epochs; mixed boundary steps dropped). If the curves of
B2v / B5v / FK20 / B10v collapse when plotted against k (but not against steps), what the student learns from the judge is per question.
Also per-run tests: B1 slope over steps (no revisits), and the jump at each regime's first revisit step."""
import numpy as np, pandas as pd, dyn
D = dyn.derive(dyn.logs())
D = D[(D.ds == "h1") & (D.idkc == "idkc05") & D.tag.isin(["B1K20", "B2vK20", "B5vK20", "B10vK20", "FK20"])]
NQ = {"B1K20": 3840, "B2vK20": 1920, "B5vK20": 768, "B10vK20": 384, "FK20": 750}
def pure_visit(tag, seed):
    """visit number of each step's batch if all 32 questions share it, else 0"""
    O = dyn.order(seed, NQ[tag]); return np.array([b[0][1] if len({v for i, v in b}) == 1 else 0 for b in O])
rows = []
for run, x in D[D.g.isin(["judgeQt50", "hash35fn00", "sim00fn00", "judgeQa35"])].groupby("run"):
    x = x.sort_values("step"); info = dyn.parse(run); pv = pure_visit(info["tag"], info["seed"])
    for k in range(1, 11):
        m = pv[: len(x)] == k
        if m.sum() == 0: continue
        sel = x[m]
        rows.append(dict(run=run, st=info["st"], g=info["g"], tag=info["tag"], seed=info["seed"], visit=k, n_steps=int(m.sum()),
                         mid_step=float(sel.step.mean()), wrong_paid=sel.wrong_paid.mean(), reward=sel.reward.mean(), idk=sel.idk.mean(),
                         fn=sel.fn_rate.mean(), correct=sel.correct_strict.mean()))
V = pd.DataFrame(rows); V.to_csv(f"{dyn.HERE}/d09_visit_curves.csv", index=False)
pd.set_option("display.width", 250)
for g in ("judgeQt50", "hash35fn00", "judgeQa35", "sim00fn00"):
    for m in ("wrong_paid", "reward", "fn"):
        if g == "sim00fn00" and m != "reward": continue
        if g == "hash35fn00" and m == "fn": continue
        P = V[V.g == g].groupby(["st", "tag", "visit"])[m].mean().unstack("visit")
        print(f"\n== {g} {m} by visit number (columns), mean over runs"); print(P.round(3).to_string())
# within-regime first-epoch slope (steps of visit 1) and first-revisit jump, per run
print("\n== judgeQt50 per run: slope of wrong_paid over first-visit steps (per 10 steps), and jump = mean(visit-2 steps, first 10) - mean(last 10 visit-1 steps)")
for run, x in D[D.g == "judgeQt50"].groupby("run"):
    x = x.sort_values("step"); info = dyn.parse(run); pv = pure_visit(info["tag"], info["seed"])[: len(x)]
    s1 = x[pv == 1]; b = np.polyfit(s1.step, s1.wrong_paid, 1)[0] * 10
    s2 = x[pv == 2]
    j = (s2.wrong_paid.values[:10].mean() - s1.wrong_paid.values[-10:].mean()) if len(s2) else np.nan
    jr = (s2.reward.values[:10].mean() - s1.reward.values[-10:].mean()) if len(s2) else np.nan
    print(f"  {run:40s} visit-1 steps {len(s1):3d} slope/10 steps {b:+.4f}  jump wrong_paid {j:+.3f}  jump reward {jr:+.3f}")
