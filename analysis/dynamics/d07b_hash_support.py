"""Hash grader: uptake of a paid wrong answer on TRAINING questions as a function of how often the untrained student samples a paid wrong
answer (share of 32 samples), vs held-out questions (virtual verdicts); by student (qwen7 = discovery, llama8 = confirmation)."""
import numpy as np, pandas as pd, dyn
Q = pd.read_pickle(f"{dyn.HERE}/d07_hash_q.pkl"); H = Q[Q.g == "hash35fn00"].copy()
H["bin"] = pd.cut(H.base_paid_share, [-0.01, 0.0, 1/32 + 1e-9, 4/32 + 1e-9, 12/32 + 1e-9, 1.0], labels=["0", "1/32", "2-4/32", "5-12/32", ">12/32"])
pd.set_option("display.width", 200)
T = H.groupby(["st", "bin", "split"], observed=True).agg(n=("paid", "size"), final_paid=("paid", "mean"), final_idk=("out", lambda o: np.mean(o == 0))).unstack("split")
print(T.round(3).to_string())
