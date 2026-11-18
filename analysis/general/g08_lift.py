"""(explorer 'general') Judge-specific pay lift of changed wrong answers on the small students (TriviaQA; PopQA where both judges
scored). For a run trained with judge K (Qwen3.5-9B 'Q9' or Gemma-4-12B 'G12'; O = the other judge):
  changed wrong answer = final greedy answer well-formed, not IDK, wrong by the loose key, and different from the run's own
  untrained (ck0) greedy answer to that question;
  lift = [acc_K - acc_O over changed wrong answers] - [acc_K - acc_O over the untrained greedy key-wrong answers of the same
  questions split], acc_J = P_J(Yes) > 0.5 (lookup g02; answers without both scores dropped).
Home = training questions, new = held-out (de-duplicated). Key-trained runs give the placebo (K := Q9, and K := G12).
Question bootstrap (2000) within run; pooled estimate over runs (run-weighted) with a bootstrap over runs and questions.
Output: g08_lift.txt"""
import json, os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import setup, HERE, C, RL, read_eval

rng = np.random.default_rng(8)
out = open(f"{HERE}/g08_lift.txt", "w")
def P(*a): print(*a, file=out); print(*a)
for ds in ("tqa", "pqa"):
    probs, H1, HELD = setup(ds); HOME = np.array(sorted(H1)); NEW = np.array(HELD)
    try: LK = {J: pickle.load(open(f"{HERE}/g02_lookup_{J}_{ds}.pkl", "rb")) for J in ("Q9", "G12")}
    except FileNotFoundError: continue
    P(f"\n######## {ds}")
    for st in ("phi4m", "smol3", "granite"):
        Cc = pickle.load(open(f"{HERE}/cache_{st}.pkl", "rb"))
        res = []
        for run, d in sorted(Cc.items()):
            m = d["meta"]
            if m["ds"] != ds: continue
            is_j = m["fam"] == "judge" and m["jm"] in ("Q9", "G12") and m["jmode"] == "free"
            is_k = m["fam"] == "key" and m["kl"] == 0 and m["pre"] in ("", "F")
            if not (is_j or is_k): continue
            fr, ur = read_eval(run, m["final"])[0], read_eval(run, 0)[0]
            code = d["f"]["cs"] if m["fmt"] == 1 else d["f"]["cf"]
            if (code == 3).mean() > 0.2: continue
            ucode = d["u"]["cs"] if m["fmt"] == 1 else d["u"]["cf"]
            Ks = [m["jm"]] if is_j else ["Q9", "G12"]
            for K in Ks:
                O = "G12" if K == "Q9" else "Q9"
                row = dict(run=run, K=K, kind="judge" if is_j else "key", outlet=m["outlet"], pre=m["pre"] or "old")
                for split, qs in (("home", HOME), ("new", NEW)):
                    ch, ub = [], []
                    for i in qs:
                        fl, ul = C.tq_line(fr[i]["text"]), C.tq_line(ur[i]["text"])
                        if code[i] == 2 and fl != ul:
                            a, b = LK[K].get((i, fl)), LK[O].get((i, fl))
                            if a is not None and b is not None: ch.append(((a > 0.5) - (b > 0.5)))
                        if ucode[i] == 2:
                            a, b = LK[K].get((i, ul)), LK[O].get((i, ul))
                            if a is not None and b is not None: ub.append(((a > 0.5) - (b > 0.5)))
                    ch, ub = np.array(ch, float), np.array(ub, float)
                    row[split] = (ch, ub)
                res.append(row)
        if not res: continue
        P(f"\n=== {st} {ds}")
        for kind in ("judge", "key"):
            R = [r for r in res if r["kind"] == kind]
            for r in R:
                s = f"  {r['run']:36s} K={r['K']:3s}"
                for split in ("home", "new"):
                    ch, ub = r[split]
                    if len(ch) < 5 or len(ub) < 5: s += f" | {split}: n_changed {len(ch)} (too few)"; continue
                    v = ch.mean() - ub.mean()
                    bs = [ch[rng.integers(0, len(ch), len(ch))].mean() - ub[rng.integers(0, len(ub), len(ub))].mean() for _ in range(1000)]
                    s += f" | {split}: lift {v:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (changed wrong {len(ch)}, untrained wrong {len(ub)})"
                P(s)
            for split in ("home", "new"):
                ok = [r for r in R if len(r[split][0]) >= 5 and len(r[split][1]) >= 5]
                if not ok: continue
                v = np.mean([r[split][0].mean() - r[split][1].mean() for r in ok])
                bs = []
                for _ in range(1000):
                    rr = [ok[j] for j in rng.integers(0, len(ok), len(ok))]
                    bs.append(np.mean([r[split][0][rng.integers(0, len(r[split][0]), len(r[split][0]))].mean() - r[split][1][rng.integers(0, len(r[split][1]), len(r[split][1]))].mean() for r in rr]))
                P(f"  POOLED {kind:5s} {split:4s}: {len(ok)} run-judge rows, lift {v:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]; "
                  f"positive in {sum((r[split][0].mean() - r[split][1].mean()) > 0 for r in ok)}/{len(ok)}")
out.close()
