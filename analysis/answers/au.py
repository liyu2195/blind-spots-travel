"""(answers) Shared helpers: load caches, grader families, outcome codes, IDK detection, clustering of answer strings."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, pickle, json, os
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import IDK, C, same
HERE = (_RA + "/answers")
API = (_RX + "/apitest")
SETS = ("tqa", "pqa", "nqo"); STS = ("qwen7", "llama8")
LAB = {"tqa": "TriviaQA", "pqa": "PopQA", "nqo": "NQ-open"}

def load(st, dsn):
    return pickle.load(open(f"{HERE}/cache_{st}_{dsn}.pkl", "rb"))

def jc(dsn, J):
    f = f"{HERE}/jc_{dsn}_{J}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else {}

def fam(g):
    if g == "sim00fn00": return "gold"
    if re.fullmatch(r"sim00fn\d\d", g): return "keyFN"
    if re.fullmatch(r"sim\d\dfn00", g): return "keyFP"
    if re.fullmatch(r"sim\d\dfn\d\d", g): return "mix"
    if g.startswith("hash"): return "hash"
    if g.startswith(("typeA", "typeB")): return "typeprog"
    if g.startswith(("popH", "popL")): return "popprog"
    if g.startswith(("fnH", "fnB", "fnA")): return "fnset"
    if g.startswith("pl"): return "plcopy"
    if g.startswith("blind"): return "blind"
    if g.startswith(("frzVOTE", "voteq")): return "vote"
    if g == "apiS55": return "sonnet"
    if g == "lenQ": return "lenQ"
    if g == "long80": return "long80"
    m = re.fullmatch(r"judge([QMGOR]S?)(a35|a50|t50)(.*)", g)
    if m: return "judge"
    if g.startswith("frz"): return "frzother"
    return "other"

def judge_of(g):
    m = re.fullmatch(r"judge([QMGOR]S?)(a35|a50|t50)(.*)", g)
    return (m.group(1), m.group(2), m.group(3)) if m else None

def is_idk(x):
    return bool(x) and bool(IDK.match(x))

def is_idk_loose(x):
    """training-time abstention test (prefix match, C.is_idk) -- broader than the eval's full-match IDK"""
    return C.is_idk(x)

def code(x, al):
    """0 IDK, 1 right (loose key), 2 wrong, 3 malformed/empty"""
    if not x: return 3
    if is_idk(x): return 0
    return 1 if C.tq_correct(x, al) else 2

def nwords(x):
    return len(x.split()) if x else 0

def thr(st, dsn, J, kind):
    """trainer threshold for judge J at a35 / a50 / t50"""
    if kind == "t50": return 0.5
    d = json.load(open(f"{_RX}/final/thresholds_{J}.json")) if J != "QS" else None
    if d is None: return None
    return d[f"{st}_{dsn}"][f"t_{kind}"]

def sonnet_levels(st, dsn):
    LV = {}
    for f in (f"{API}/frontierH_{st}_{dsn}.jsonl", f"{API}/auditT_{st}_{dsn}.jsonl"):
        if os.path.exists(f):
            for r in map(json.loads, open(f)): LV[(r["idx"], r["norm"])] = r["level"]
    return LV

def runs(P, tag="FK20", fams=None, idkc=5):
    out = []
    for name, r in P["R"].items():
        if r["tag"] != tag or r["idkc"] != idkc: continue
        f = fam(r["g"])
        if fams is not None and f not in fams: continue
        out.append(name)
    return sorted(out)

def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) == 0: return (np.nan, np.nan)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return (np.percentile(b, 2.5), np.percentile(b, 97.5))

def kappa(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); po = np.mean(x == y); px, py = x.mean(), y.mean(); pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 - 1e-12 else np.nan
