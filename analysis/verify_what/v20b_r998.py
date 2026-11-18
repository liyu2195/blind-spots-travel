"""(verify_what) Item 2 check of the claim's validation statistic: correlation across the 36 (student, set, seed) cells of predicted vs logged
step-1 strict correctness, true seed vs wrong seeds (if a wrong seed also gives r ~ 0.99, r = 0.998 is driven by between-set differences and
does not validate the schedule; the within-set test and v20 V2/V3 do). Output v20b_r998.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, pickle, re, sys, numpy as np
sys.path.insert(0, _RC); sys.path.insert(0, (_RA + "/verify_what"))
import st_common as C
from v20_sched import schedule
OUT = (_RA + "/verify_what"); DATA = _RD; DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
LOG = pickle.load(open(f"{OUT}/logs.pkl", "rb")); CFG = pickle.load(open(f"{OUT}/v02_configs.pkl", "rb"))
cells = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        Dd = pickle.load(open(f"{OUT}/cache_{st}_{ds}.pkl", "rb")); B = Dd["B"]; al = Dd["Q"]["aliases"]
        ids = [json.loads(l)["idx"] for l in open(f"{DATA}/{DSN[ds]}_h1.jsonl")]
        qs = np.array([np.mean([bool(x) and C.tq_strict(x, al[i]) for x in B["s"][i]]) for i in ids])
        for sd in range(6):
            rs = [r for r in LOG if r.startswith(f"{st}_FK20") and r.endswith(f"_s{sd}_{ds}") and r in CFG and CFG[r]["TRAINF"] == f"{DSN[ds]}_h1.jsonl"]
            if not rs: continue
            obs = LOG[rs[0]]["correct_strict"][0]
            pred = {w: qs[[j for j, v in schedule(w, len(ids), steps=1)[0]]].mean() for w in [sd] + [1000 + 37 * k + sd for k in range(50)]}
            cells.append((st, ds, sd, obs, pred))
obs = np.array([c[3] for c in cells]); tr = np.array([c[4][c[2]] for c in cells])
rw = [np.corrcoef(obs, np.array([c[4][1000 + 37 * k + c[2]] for c in cells]))[0, 1] for k in range(50)]
within = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        m = [k for k, c in enumerate(cells) if c[0] == st and c[1] == ds]
        within.append(np.corrcoef(obs[m], tr[m])[0, 1])
out = (f"cells {len(cells)}: r(true seed) = {np.corrcoef(obs, tr)[0, 1]:.3f}; r(wrong seeds) median {np.median(rw):.3f} [min {min(rw):.3f}, max {max(rw):.3f}]; "
       f"within (student, set) r over 6 seeds, true seed: " + " ".join(f"{x:.2f}" for x in within))
print(out); open(f"{OUT}/v20b_r998.txt", "w").write(__doc__ + "\n" + out + "\n")
