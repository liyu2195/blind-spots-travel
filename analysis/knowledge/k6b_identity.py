"""(knowledge) Judge identity fingerprint: for runs trained with judge J at the 35% target (J in Q M G O R), is the probe of J's own
verdicts (fitted on trained questions, untrained hidden state) the best forecaster of the run's refusals on new questions among the five
judges' 35%-target probes?  Chance rank-1 rate = 0.2.  Uses k6_rows.pkl.  Output: k6b_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
from collections import defaultdict
rows = pickle.load(open((_RA + "/knowledge/k6_rows.pkl"), "rb"))
out = []
def P(s): print(s); out.append(s)
P(__doc__)
J5 = ["Q35", "M35", "G35", "O35", "R35"]
for st in ("qwen7", "llama8", None):
    L = [d for d in rows if d["own"] in J5 and (st is None or d["st"] == st)]
    rk = []; gap = []; byj = defaultdict(list)
    for d in L:
        v = {j: d[f"p_{j}"] for j in J5}; order = sorted(J5, key=lambda j: -v[j]); r = order.index(d["own"]) + 1; rk.append(r)
        oth = np.mean([v[j] for j in J5 if j != d["own"]]); gap.append(v[d["own"]] - oth); byj[d["own"]].append((r, v[d["own"]] - oth))
    rk = np.array(rk); gap = np.array(gap)
    P(f"{st or 'both'}: runs {len(L)}  own judge ranked 1st in {np.mean(rk == 1):.2f} (chance 0.20), mean rank {rk.mean():.2f} (chance 3.0); own minus mean of other judges {gap.mean():+.4f}, >0 in {np.mean(gap > 0):.2f}")
    for j, v in sorted(byj.items()): P(f"    trained with {j}: runs {len(v)}, rank-1 share {np.mean([x[0] == 1 for x in v]):.2f}, own - others {np.mean([x[1] for x in v]):+.4f}")
# sign test over runs: own > mean of others
from scipy.stats import binomtest
L = [d for d in rows if d["own"] in J5]
k = sum(d[f"p_{d['own']}"] > np.mean([d[f"p_{j}"] for j in J5 if j != d["own"]]) for d in L)
P(f"sign test own > mean(others): {k}/{len(L)}, p = {binomtest(k, len(L)).pvalue:.2g}")
open((_RA + "/knowledge/k6b_out.txt"), "w").write("\n".join(out) + "\n")
