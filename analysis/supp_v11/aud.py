"""(supp_v11) pooled audit lookup, the same sources as selfcons/s0_build.sonnet_levels: (question idx, normalised answer) -> median Sonnet level,
pooled over both students' audit files (frontierH, frontierA, auditT) and the Sonnet-reward runs' verdict caches."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, os, glob, numpy as np
from collections import defaultdict
API = (_RX + "/apitest"); RL = (_RR + "/run1/rl")
def levels(dsn):
    L = defaultdict(list)
    for st in ("qwen7", "llama8"):
        for kind in ("frontierH", "frontierA", "auditT"):
            f = f"{API}/{kind}_{st}_{dsn}.jsonl"
            if os.path.exists(f):
                for r in map(json.loads, open(f)): L[(r["idx"], r["norm"])].append(r["level"])
    dsx = {"tqa": "h1", "pqa": "pq", "nqo": "nq"}[dsn]
    for f in glob.glob(f"{RL}/*_FK20apiS55_idkc05_s*_{dsx}/api_verdicts.jsonl"):
        for r in map(json.loads, open(f)): L[(r["idx"], r["norm"])].append(r["level"])
    return {k: float(np.median(v)) for k, v in L.items()}
