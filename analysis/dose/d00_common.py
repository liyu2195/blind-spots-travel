"""(dose) Shared helpers, own code. Only the trainer's answer parser / key functions are imported from st_common (parse_tq, tq_correct,
tq_strict), because the trainer itself uses them for rewards and evaluation.
Conventions (fixed before any estimate was computed; they follow the project's rc.py / bm.py definitions):
  answer of a reply = '' (malformed) if fin != 'stop', or a newline is in the stripped text, or 'think>' is in it; else parse_tq(text)
  IDK = full match of the project IDK list on the parsed answer; outcome codes 0 IDK, 1 right (loose key), 2 wrong, 3 malformed
  TRAIN = ids in <dsn>_h1.jsonl; NEW = other ids minus questions whose lower-cased text or qid equals a training question's
  audited right = loose key OR Sonnet level >= 4; conf = share of the 32 untrained samples agreeing with the greedy answer
  ('same' = equal, or whole-word containment of a >= 3-character string)
  judge J rejects an answer at operating point op <=> P(Yes)/(P(Yes)+P(No)) <= t (t = t_a35 / t_a50 from final/thresholds_J.json, or 0.5 = t50)"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, numpy as np
sys.path.insert(0, _RC)
import st_common as C
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
ADE = (_RR + "/run1/adeval"); API = (_RX + "/apitest"); FIN = (_RX + "/final")
HERE = (_RA + "/dose"); CACHE = f"{HERE}/cache"
os.makedirs(CACHE, exist_ok=True)
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}; DSF = {v: k for k, v in DSN.items()}
STS = ("qwen7", "llama8"); DSS = ("h1", "pq", "nq")
IDKF = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
J5 = ("Q", "M", "G", "O", "R")

def jl(p): return [json.loads(l) for l in open(p)]
def ans(r):
    t = r.get("text", "")
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)
def outcome(x, al):
    if not x: return 3
    if IDKF.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
def same(a, b):
    return bool(a == b or (len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b)) or (len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a)))
def py_of(r):
    d = r["py"] + r["pn"]; return r["py"] / d if d > 1e-3 else 0.5
def thresholds():
    return {J: json.load(open(f"{FIN}/thresholds_{J}.json")) for J in J5}
def questions(dsn):
    allq = jl(f"{DATA}/{dsn}_1500.jsonl"); probs = {p["idx"]: p for p in allq}
    H = sorted(p["idx"] for p in jl(f"{DATA}/{dsn}_h1.jsonl")); hs = set(H)
    qt = {probs[i]["question"].strip().lower() for i in H}; qid = {probs[i].get("qid") for i in H} - {None}
    NEW = sorted(p["idx"] for p in allq if p["idx"] not in hs and p["question"].strip().lower() not in qt
                 and not (p.get("qid") is not None and p.get("qid") in qid))
    return probs, H, NEW
def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); return np.log(p / (1 - p))
def sig(x): return 1.0 / (1.0 + np.exp(-x))
def pct(b, q=(2.5, 97.5)):
    b = np.asarray([x for x in b if np.isfinite(x)]); return tuple(np.percentile(b, q)) if len(b) else (np.nan, np.nan)

class Log:
    def __init__(self, path): self.path = path; self.L = []
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); print(s, flush=True); self.L.append(s)
    def save(self): open(self.path, "w").write("\n".join(self.L) + "\n")
