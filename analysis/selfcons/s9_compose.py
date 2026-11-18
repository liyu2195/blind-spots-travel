"""selfcons s9: composition of the relocated refusals (from s3_rows.pkl), FK20 IDK credit 0.5 runs, by student and grader group.
RL-only = refused by RL, answered by SC at the same coverage; SC-only = answered by RL, refused by SC.  Per 100 new questions:
relocated count; RL-only: mean SC agreement, share the grader rejects (V=0) and its SC-bin expectation, share whose untrained greedy is
wrong (audited); SC-only: mean SC agreement, share with no untrained answer (greedy IDK / malformed), share the grader rejects, RL's own
audited accuracy there, and the untrained greedy's accuracy there (valid greedy only).  Output: s9_compose.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
sys.path.insert(0, (_RA + "/selfcons"))
from sc_lib import *
rows = pickle.load(open(f"{OUT}/s3_rows.pkl", "rb")); out = [__doc__]
for st in STS:
    L = [d for d in rows if d["st"] == st and d["tag"] == "FK20" and d["credit"] == "idkc05"]
    for gname, fams in [("ALL", None)] + GROUPS:
        X = [d for d in L if fams is None or d["fam"] in fams]
        if not X: continue
        m = lambda k: np.nanmean([x[k] for x in X])
        out.append(f"{st:7s} {gname:18s} runs {len(X):3d} | relocated/100 {m('reloc100'):4.1f} ({m('reloc_share'):.2f} of RL refusals) | RL-only: SC {m('conf1'):.2f}, "
                   f"grader rejects {m('p1'):.2f} (exp {m('E1'):.2f}), untrained wrong {m('e1'):.2f} | SC-only: SC {m('conf2'):.2f}, no untrained answer "
                   f"{1-m('sc_only_valid_share'):.2f}, grader rejects {m('p2'):.2f} (exp {m('E2'):.2f}), RL right {m('accRL_on_sc_only'):.2f}, untrained right {1-m('e2'):.2f}")
open(f"{OUT}/s9_compose.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
