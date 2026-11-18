"""Validate the reconstruction of the training batches (random.seed(seed); shuffle) against the logs:
per step, predicted = mean over the 32 batch questions of the untrained strict-right share (32 base samples);
observed = logged correct_strict. If the reconstruction is right, they correlate across steps (and a wrong seed gives ~0)."""
import numpy as np, pandas as pd, dyn
D = dyn.logs()
out = []
for (st, ds), g in D[D.tag == "FK20"].groupby(["st", "ds"]):
    B = dyn.base_stats(st, dyn.DSF[ds]); ids = dyn.train_ids(f"{dyn.DSF[ds]}_h1.jsonl")
    q = B.loc[ids, "q_strict"].values; qi = B.loc[ids, "q_idk"].values
    for seed, gs in g.groupby("seed"):
        res = {}
        for sd in (seed, seed + 100):   # true seed and a wrong seed (placebo)
            O = dyn.order(sd, len(ids))
            pred = np.array([q[[i for i, v in b]].mean() for b in O])
            # step 1 identical across runs; use all runs: within-run correlation over steps 1-120 of correct_strict with pred
            rs = []
            for run, x in gs.groupby("run"):
                x = x.sort_values("step")
                if len(x) < 120 or x.g.iloc[0].startswith(("frz", "blind")): continue
                obs = x.correct_strict.values[:120]
                # remove the slow trend (policy drift) with a 9-step centred moving average before correlating
                k = 9; tr = pd.Series(obs).rolling(k, center=True, min_periods=1).mean().values
                tp = pd.Series(pred).rolling(k, center=True, min_periods=1).mean().values
                rs.append(np.corrcoef(obs - tr, pred - tp)[0, 1])
            res[sd == seed] = (np.median(rs), len(rs), pred[0])
        s1 = gs[gs.step == 1].correct_strict.iloc[0]
        out.append(dict(st=st, ds=ds, seed=seed, n_runs=res[True][1], r_true=round(res[True][0], 3), r_wrong_seed=round(res[False][0], 3),
                        step1_pred=round(res[True][2], 3), step1_obs=round(s1, 3)))
T = pd.DataFrame(out)
print(T.to_string())
print("step-1 predicted vs observed across (st, ds, seed):", np.corrcoef(T.step1_pred, T.step1_obs)[0, 1].round(3))
T.to_csv(f"{dyn.HERE}/d02_order_check.csv", index=False)
