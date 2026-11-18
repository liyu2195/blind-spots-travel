"""No KL (FK00) vs KL 0.2 (FK20), TriviaQA, judgeQt50 and sim35fn00 (random 35% false accepts), both students: per-visit training
metrics and final outcomes on training vs held-out questions. Also the long runs (LK20 / LK50 = 360 steps, ~15 visits) for sim graders."""
import numpy as np, pandas as pd, dyn
from d06_final import final_cache
D = pd.read_pickle(f"{dyn.HERE}/d10_logs_visit.pkl") if False else None
L = dyn.derive(dyn.logs())
L = L[(L.ds == "h1") & (L.idkc == "idkc05") & L.tag.isin(["FK00", "FK20"]) & L.g.isin(["judgeQt50", "sim35fn00"]) & (L.seed <= 1)]
pv = {}
def vis(seed):
    if seed not in pv:
        O = dyn.order(seed, 750); pv[seed] = np.array([b[0][1] if len({v for i, v in b}) == 1 else 0 for b in O])
    return pv[seed]
L = L.copy(); L["visit"] = [vis(s)[t - 1] for s, t in zip(L.seed, L.step)]
pd.set_option("display.width", 250)
for m in ("wrong_paid", "reward", "idk", "distinct_per_group", "kl", "correct_strict"):
    T = L[L.visit > 0].groupby(["g", "st", "tag", "visit"])[m].mean().unstack("visit")
    print(f"\n== {m} by visit (seeds 0-1)"); print(T.round(3).to_string())
print("\n== final greedy outcomes, train vs held-out (TriviaQA)")
for st in ("qwen7", "llama8"):
    F = final_cache(st, "h1"); ids = set(dyn.train_ids("tqa_h1.jsonl")); tr = np.array([i in ids for i in F["idx"]])
    for k, run in enumerate(F["runs"]):
        info = dyn.parse(run)
        if info["g"] not in ("judgeQt50", "sim35fn00") or info["tag"] not in ("FK00", "FK20") or info["seed"] > 1 or info["idkc"] != "idkc05": continue
        o = F["O"][k]
        f = lambda m: f"idk {np.mean(o[m] == 0):.3f} right {np.mean((o[m] == 1) | (o[m] == 2)):.3f} wrong {np.mean(o[m] == 3):.3f} bad {np.mean(o[m] == 4):.3f}"
        print(f"  {run:40s} TRAIN {f(tr)} | HELD {f(~tr)}")
