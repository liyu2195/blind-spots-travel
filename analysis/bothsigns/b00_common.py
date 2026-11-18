"""(bothsigns) Shared helpers, written for this task. Read-only on project data; writes only under explore_1116/bothsigns/.
Conventions:
  home = the 750 training questions (<set>_h1.jsonl); new = the other questions minus exact duplicates of training questions (text or qid).
  answer parsing = the trainer's strict-format rule: a reply counts only if it stopped, is one line and has no think tag; then st_common.parse_tq.
  IDK = the trainer's full-match IDK list (IDK_FULL in st_grpo).
  truth for (question, answer) items: strict key (st_common.tq_strict, the simulated graders' notion of right), loose key (tq_correct),
  Sonnet audit level (frontierA = untrained samples on training questions, frontierH = untrained greedy answers on new questions,
  auditT = trained runs' key-wrong answers on new questions); audited right = loose key OR Sonnet level >= 4.
"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import os, re, json, glob, hashlib, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, _RC)
import st_common as C

RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
API = (_RX + "/apitest"); FIN = (_RX + "/final")
OUT = (_RA + "/bothsigns")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
STS = ("qwen7", "llama8"); DSS = ("h1", "pq", "nq")
JUDGES = ("Q", "M", "G", "O", "R")
IDK = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
LONG_WORDS = 5


def jl(p):
    return [json.loads(l) for l in open(p)]


def hunit(*parts):
    """identical to st_grpo.hunit (deterministic pseudo-uniform number from the parts)"""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)


def parse(r):
    t = r["text"]
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t:
        return ""
    return C.parse_tq(t)


def setup(dsn):
    allq = jl(f"{DATA}/{dsn}_1500.jsonl"); probs = {p["idx"]: p for p in allq}
    H1 = {p["idx"] for p in jl(f"{DATA}/{dsn}_h1.jsonl")}
    trq = {p["question"].strip().lower() for p in allq if p["idx"] in H1}
    trid = {p.get("qid") for p in allq if p["idx"] in H1 and p.get("qid") is not None}
    HOME = sorted(H1)
    NEW = [p["idx"] for p in allq if p["idx"] not in H1 and not (p["question"].strip().lower() in trq or (p.get("qid") is not None and p.get("qid") in trid))]
    return probs, HOME, NEW


_CFG = None
def configs():
    """run name -> config dict, from every queue line KEY=VALUE ... OUTNAME=<run> (work/q_* and bundle/st/queue_*); last one wins.
    Also returns, per run, the set of distinct configs seen (to flag conflicting duplicates)."""
    global _CFG
    if _CFG is not None:
        return _CFG
    roots = sorted(glob.glob((_RT + "/queues/*"))) + sorted(glob.glob((_RT + "/queue_*")))
    cfg = {}; seen = defaultdict(set)
    for r in roots:
        files = [r] if os.path.isfile(r) else [os.path.join(r, f) for f in sorted(os.listdir(r))]
        for p in files:
            if not os.path.isfile(p):
                continue
            try:
                txt = open(p, errors="ignore").read()
            except Exception:
                continue
            for line in txt.splitlines():
                kv = dict(re.findall(r"([A-Z_0-9]+)=(\S+)", line))
                if "OUTNAME" in kv:
                    n = kv["OUTNAME"]; cfg[n] = kv
                    seen[n].add(tuple(sorted((k, v) for k, v in kv.items() if k not in ("CAP", "MICRO", "GENCHUNK", "STUDENT", "APIWORKERS"))))
    _CFG = (cfg, {k: len(v) for k, v in seen.items()})
    return _CFG


RUNPAT = re.compile(r"^(qwen7|llama8)_FK20(.+)_idkc05_s(\d+)_(h1|pq|nq)$")


def run_spec(name):
    """Parse an FK20 idkc05 run of a main student; returns dict or None. Grader code g, seed s, dataset ds and config fields."""
    m = RUNPAT.match(name)
    if not m:
        return None
    st, g, s, ds = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    cfg, ndup = configs()
    c = cfg.get(name, {})
    sp = dict(run=name, st=st, g=g, s=s, ds=ds, cfg=c, ndup=ndup.get(name, 0))
    sp["kl"] = float(c.get("KLB", "nan")); sp["adv"] = c.get("ADV", "?"); sp["idk"] = float(c.get("IDK", "nan"))
    sp["reward"] = c.get("REWARD", "synth" if "FPK" in c else "?")
    sp["fpmode"] = c.get("FPMODE", "random")
    for k in ("FPK", "FPU", "FN", "FNK", "FNU", "JT"):
        sp[k.lower()] = float(c[k]) if k in c else None
    sp["knowsf"] = c.get("KNOWSF"); sp["force"] = c.get("FORCE"); sp["quorum"] = c.get("QUORUM")
    sp["judge"] = None
    mj = re.match(r"judge([QMGOR])(a35|a50|t50)$", g)
    if mj:
        sp["judge"] = mj.group(1); sp["op"] = mj.group(2)
    return sp


def is_reference(sp):
    """Reference pool: key-family graders whose errors carry no item structure (random false accepts / rejects at fixed rates, or none)."""
    if sp is None or sp["reward"] != "synth" or sp["fpmode"] != "random":
        return False
    if sp["knowsf"] and sp["fnk"] is not None:      # set-dependent false rejects
        return False
    if sp["fpk"] is None or sp["fpu"] is None or abs(sp["fpk"] - sp["fpu"]) > 1e-9:
        return False
    if sp["force"] or sp["quorum"]:
        return False
    return bool(re.fullmatch(r"sim\d\dfn\d\d|lenQ", sp["g"]))


def standard(sp):
    return sp is not None and abs(sp["kl"] - 0.2) < 1e-9 and sp["adv"] == "std" and abs(sp["idk"] - 0.5) < 1e-9
