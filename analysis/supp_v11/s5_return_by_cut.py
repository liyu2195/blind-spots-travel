"""(supp_v11) s5_return_by_cut: what a trained model earns on HELD-OUT questions under the training reward itself (1 for a right answer, 0 for a
wrong or malformed one, 0.5 for "I don't know"), with truth = loose key or audit level >= 4 (pooled audit, supp_v11/aud.py; a key-wrong answer
without an audit verdict counts as wrong; its share is printed). Per student, dataset and grader arm (standard recipe FK20, credit 0.5): mean over
runs of the shares right / wrong / IDK and the return = right + 0.5 * IDK; runs that answer IDK on >= 95% of held-out questions are dropped.
Also the untrained model's return (greedy answers) and, for reference, the best return any refusal rule could reach with the untrained answers
(answer exactly the right ones: right + 0.5 * (1 - right)).
Usage: python3 -I -B s5_return_by_cut.py   Output: s5_return_by_cut_out.txt, s5_return_by_cut.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_common import *
import aud as audlib
out = [__doc__]; ROWS = []
ARMS = ["sim00fn00"] + [f"judge{J}{c}" for J in "QMGO" for c in ("t50", "acc", "a50", "a35")] + ["judgeQRt50", "judgeQRa35", "apiS55", "frzMAJ35"]
for st in ("qwen7", "llama8"):
    for ds, dsn in (("pq", "pqa"), ("nq", "nqo"), ("h1", "tqa")):
        Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn)
        new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]; N = Bp["N"]
        gval = np.asarray(Bp["valid"])[:, 0].astype(bool) & ~np.asarray(Bp["idk"])[:, 0].astype(bool)
        gright = gval & (np.asarray(Bp["kl"])[:, 0].astype(bool) | (np.asarray(Bp["son"])[:, 0] >= 4)); gidk = np.asarray(Bp["idk"])[:, 0].astype(bool)
        base = float(gright[ii].mean() + 0.5 * gidk[ii].mean())
        out.append(f"\n== {st} {ds}: held-out {len(ii)}; untrained: right {gright[ii].mean():.3f}, IDK {gidk[ii].mean():.3f}, return {base:.3f}; "
                   f"best possible with the untrained answers {gright[ii].mean() + 0.5 * (1 - gright[ii].mean()):.3f}")
        G = defaultdict(list)
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05" or r["g"] not in ARMS: continue
            o = np.asarray(r["o"])[ii]; idk = (o == 0)
            if idk.mean() >= 0.95: continue
            right = np.array([(r["o"][i] == 1) or (r["o"][i] == 2 and LV.get((i, r["f"][i]), 0) >= 4) for i in ii])
            unaud = np.array([(r["o"][i] == 2) and ((i, r["f"][i]) not in LV) for i in ii])
            G[r["g"]].append((right.mean(), ((o == 2) | (o == 3)).mean() - (right & (o == 2)).mean(), idk.mean(), right.mean() + 0.5 * idk.mean(), unaud.mean()))
        for g in ARMS:
            if g not in G: continue
            a = np.mean(G[g], axis=0)
            out.append(f"   {g:11s} runs {len(G[g]):2d}: right {a[0]:.3f}  wrong {a[1]:.3f}  IDK {a[2]:.3f}  return {a[3]:.3f} (runs {min(x[3] for x in G[g]):.3f}-{max(x[3] for x in G[g]):.3f})"
                       f"  unaudited key-wrong {a[4]:.3f}")
            ROWS.append(dict(st=st, ds=ds, g=g, runs=len(G[g]), right=float(a[0]), wrong=float(a[1]), idk=float(a[2]), ret=float(a[3]), unaud=float(a[4]), base=base))
H = os.path.dirname(os.path.abspath(__file__))
json.dump(ROWS, open(f"{H}/s5_return_by_cut.json", "w"), indent=1)
# pooled over datasets (mean of dataset means), judges averaged per cut
out.append("\n== pooled over the three datasets (mean of dataset means); 'judges' = mean over the Q M G O judges that have runs at the cut")
for st in ("qwen7", "llama8"):
    for label, arms in (("answer key", ["sim00fn00"]), ("Qwen judge default", ["judgeQt50"]), ("Qwen judge key-matched (PopQA only)", ["judgeQacc"]), ("Qwen judge 50% cut", ["judgeQa50"]),
                        ("Qwen judge 35% cut", ["judgeQa35"]), ("judges default", [f"judge{J}t50" for J in "QMGO"]), ("judges 50% cut", [f"judge{J}a50" for J in "QMGO"]),
                        ("judges 35% cut", [f"judge{J}a35" for J in "QMGO"]), ("reference-guided default", ["judgeQRt50"]), ("Sonnet", ["apiS55"])):
        per = defaultdict(list)
        for r in ROWS:
            if r["st"] == st and r["g"] in arms: per[r["ds"]].append(r)
        if not per: continue
        m = lambda k: np.mean([np.mean([x[k] for x in v]) for v in per.values()])
        out.append(f"   {st:7s} {label:36s} datasets {''.join(sorted(per))!s:8s} right {m('right'):.3f} wrong {m('wrong'):.3f} IDK {m('idk'):.3f} return {m('ret'):.3f}")
open(f"{H}/s5_return_by_cut_out.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out[1:]))
