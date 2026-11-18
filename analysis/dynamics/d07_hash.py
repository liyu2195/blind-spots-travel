"""Question-specific false ACCEPTS (hash35fn00 grader: a wrong answer is paid iff sha-hash(seed, question, answer) < 0.35; no pattern).
Recompute the grader exactly; at the final checkpoint compare training questions with held-out questions (where the same hash is a
virtual verdict nobody trained on). Also: dose (5 vs 6 visits; random by the shuffle), recency of the last visit, and whether the untrained
student ever samples a paid wrong answer on the question (exploration). Placebo: the same virtual hash on the seed-matched gold run."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd, json, dyn, sys
from d06_final import final_cache, visits
sys.path.insert(0, _RC)
from st_grpo import hunit
from collections import defaultdict
rows = []; qrows = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = final_cache(st, ds); idx = F["idx"]; dsn = dyn.DSF[ds]
        ids = dyn.train_ids(f"{dsn}_h1.jsonl"); tpos = {i: k for k, i in enumerate(ids)}
        H1 = set(ids); probs = {r["idx"]: r for r in map(json.loads, open(f"{dyn.DATA}/{dsn}_1500.jsonl"))}
        trq = {probs[i]["question"].strip().lower() for i in H1}
        held = np.array([(i not in H1) and probs[i]["question"].strip().lower() not in trq for i in idx]); train = np.array([i in H1 for i in idx])
        # untrained samples: share of the 32 samples that are wrong and hash-paid (per seed)
        S = defaultdict(list)
        for r in map(json.loads, open(f"{dyn.BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
            if r["s"] >= 0: S[r["idx"]].append(dyn.sa(r))
        for k, run in enumerate(F["runs"]):
            info = dyn.parse(run)
            if info["g"] not in ("hash35fn00", "sim00fn00") or info["tag"] != "FK20": continue
            seed = info["seed"]; o = F["O"][k]; a = F["A"][run]
            paid = np.array([o[j] == 3 and hunit("fp", seed, int(idx[j]), a[j]) < 0.35 for j in range(len(idx))])
            nv, last, first = visits(info, ids)
            for j in range(len(idx)):
                if not (train[j] or held[j]): continue
                i = int(idx[j]); al = set(probs[i]["aliases"]); xs = S[i]
                pw = [x for x in xs if x and not dyn.IDK_FULL.match(x) and not dyn.C.tq_strict(x, al) and hunit("fp", seed, i, x) < 0.35]
                qrows.append(dict(run=run, st=st, ds=ds, g=info["g"], seed=seed, split="train" if train[j] else "held", out=int(o[j]), paid=bool(paid[j]),
                                  nv=int(nv[tpos[i]]) if train[j] else 0, last=int(last[tpos[i]]) if train[j] else 0,
                                  base_paid_share=len(pw) / max(1, len(xs)),
                                  base_wrong_share=np.mean([bool(x) and not dyn.IDK_FULL.match(x) and not dyn.C.tq_strict(x, al) for x in xs])))
Q = pd.DataFrame(qrows); Q.to_pickle(f"{dyn.HERE}/d07_hash_q.pkl")
Q["wrong"] = Q.out == 3
def share(x): return x.paid.sum() / max(1, x.wrong.sum())
print("Share of final WRONG answers that the hash grader pays (0.35 = chance), by grader x split x student (pooled over runs):")
for (g, st, split), x in Q.groupby(["g", "st", "split"]):
    print(f"  {g:10s} {st:7s} {split:5s} runs={x.run.nunique()} wrong={x.wrong.sum():5d} paid/wrong={share(x):.3f}  paid/all={x.paid.mean():.3f}  idk={np.mean(x.out == 0):.3f}")
H = Q[Q.g == "hash35fn00"]
print("\nPer run: paid/wrong train vs held (hash runs)")
for run, x in H.groupby("run"):
    tr, he = x[x.split == "train"], x[x.split == "held"]
    print(f"  {run:40s} train {share(tr):.3f} (n wrong {tr.wrong.sum()})  held {share(he):.3f} (n wrong {he.wrong.sum()})")
T = H[H.split == "train"]
print("\nDose (training questions of hash runs): visits 5 vs 6 (random by the shuffle)")
for nv, x in T.groupby("nv"): print(f"  visits={nv}: n={len(x)} paid/wrong={share(x):.3f} paid/all={x.paid.mean():.3f} wrong/all={x.wrong.mean():.3f}")
print("Recency: step of last visit (questions with 5 visits)")
T5 = T[T.nv == 5].copy(); T5["rec"] = pd.cut(T5["last"], [0, 100, 106, 112, 120])
for r, x in T5.groupby("rec", observed=True): print(f"  last visit {r}: n={len(x)} paid/wrong={share(x):.3f} paid/all={x.paid.mean():.3f}")
print("Exploration: did the untrained student sample a paid wrong answer (32 samples)?")
for e, x in T.groupby(T.base_paid_share > 0):
    print(f"  any paid sample={e}: n={len(x)} paid/wrong={share(x):.3f} paid/all={x.paid.mean():.3f}")
He = H[H.split == "held"]
for e, x in He.groupby(He.base_paid_share > 0):
    print(f"  [held] any paid sample={e}: n={len(x)} paid/wrong={share(x):.3f} paid/all={x.paid.mean():.3f}")
