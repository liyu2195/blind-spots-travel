"""(bothsigns) b01b: judge P(Yes) lookup for (question, normalised answer) pairs, to tell whether a trained run's final wrong answer is one the
judge pays. Sources: judge scores of the trained runs' answer pool (score_<J>_cand_runs_<set>[2|D|E].jsonl with data/cand_runs_<set>[2|D|E].jsonl;
the (idx, s) keys of the pre-delta score files are a prefix of the current pool) and of the untrained samples (base score files, via the cache).
Pairs with several raw texts get the mean P(Yes). Output: cache/jlook_<st>_<ds>.pkl {J: {(idx, norm): p}} and printed coverage of final
key-wrong answers per run family. Usage: nice -n 10 python3 b01b_judgelookup.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, pickle
from collections import defaultdict
import numpy as np
sys.path.insert(0, (_RA + "/bothsigns"))
from b00_common import *

POOL = {}
def pool(name):
    if name not in POOL:
        f = f"{DATA}/cand_runs_{name}.jsonl"
        POOL[name] = {(r["idx"], r["s"]): r["norm"] for r in map(json.loads, open(f))} if os.path.exists(f) else {}
    return POOL[name]

for ds in DSS:
    dsn = DSN[ds]
    cand = {J: defaultdict(list) for J in JUDGES}
    for J in JUDGES:
        for suf in ("", "2", "D", "E"):
            f = f"{BASE}/score_{J}_cand_runs_{dsn}{suf}.jsonl"
            if not os.path.exists(f):
                continue
            P = pool(dsn + suf)
            for r in map(json.loads, open(f)):
                nm = P.get((r["idx"], r["s"]))
                if nm is None:
                    continue
                cand[J][(r["idx"], nm)].append(r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5)
    for st in STS:
        Pc = pickle.load(open(f"{OUT}/cache/cache_{st}_{ds}.pkl", "rb")); Q, R = Pc["Q"], Pc["R"]
        look = {}
        for J in JUDGES:
            d = {k: float(np.mean(v)) for k, v in cand[J].items()}
            nb = 0
            for k, lst in enumerate(Q["samples"]):
                i = int(Q["idx"][k])
                for s in lst:
                    if (i, s["x"]) not in d and np.isfinite(s["pj"][J]):
                        d[(i, s["x"])] = s["pj"][J]; nb += 1
                if Q["g"][k] and (i, Q["g"][k]) not in d and np.isfinite(Q["pj"][J][k]):
                    d[(i, Q["g"][k])] = float(Q["pj"][J][k]); nb += 1
            look[J] = d
        pickle.dump(look, open(f"{OUT}/cache/jlook_{st}_{ds}.pkl", "wb"))
        # coverage of final key-wrong answers
        line = []
        for J in JUDGES:
            cov = defaultdict(list)
            for name, r in R.items():
                fam = "ref" if is_reference(r | {"cfg": None}) else ("own" if r.get("judge") == J else "other")
                ks = np.where((r["code"] == 3) & (Q["part"] >= 0))[0]
                if len(ks) == 0:
                    continue
                cov[fam].append(np.mean([(int(Q["idx"][k]), r["ans"][k]) in look[J] for k in ks]))
            line.append(f"{J}: pool {len(cand[J])} | " + ", ".join(f"{f} {np.mean(v):.2f} (n {len(v)})" for f, v in sorted(cov.items())))
        print(st, ds, " || ".join(line), flush=True)
