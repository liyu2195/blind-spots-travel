"""(verify_map) Shared helpers, written for this verification from the BRIEF's raw-file descriptions. Only the trainer's own answer parser and key
functions (st_common.parse_tq / tq_correct / tq_strict) are imported, because the trainer used them for rewards and evaluation.
Conventions (fixed before any estimate):
  reply -> answer: '' (malformed) unless fin == 'stop', one line after strip, no 'think>'; else parse_tq(text)
  IDK = full match of the trainer's IDK_FULL list on the parsed answer. Outcome codes: 0 IDK, 1 right (loose key), 2 wrong, 3 malformed.
  TRAIN = ids in <dsn>_h1.jsonl; NEW = the rest minus exact text / qid duplicates of a training question.
  judge J rejects an answer at op <=> P(Yes)/(P(Yes)+P(No)) <= t (t_a35 / t_a50 from final/thresholds_J.json; 0.5 for t50)."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, numpy as np
sys.path.insert(0, _RC)
import st_common as C
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
API = (_RX + "/apitest"); FIN = (_RX + "/final")
HERE = (_RA + "/verify_map"); CACHE = f"{HERE}/cache"
os.makedirs(CACHE, exist_ok=True)
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
STS = ("qwen7", "llama8"); DSS = ("h1", "pq", "nq"); J5 = ("Q", "M", "G", "O", "R")
IDKF = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")

def jl(p): return [json.loads(l) for l in open(p)]
def parse(r):
    t = r.get("text", "")
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)
def code(x, al):
    if not x: return 3
    if IDKF.match(x): return 0
    return 1 if C.tq_correct(x, al) else 2
def agree(a, b):
    return bool(a == b or (len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b)) or (len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a)))
def pyes(r):
    d = r["py"] + r["pn"]; return r["py"] / d if d > 1e-3 else 0.5
def thr(J, st, ds, op):
    if op == "t50": return 0.5
    return json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_{DSN[ds]}"][f"t_{op}"]
def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); return np.log(p / (1 - p))
def load(st, ds):
    return pickle.load(open(f"{CACHE}/c_{st}_{ds}.pkl", "rb"))
class Log:
    def __init__(self, path): self.path = path; self.L = []
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); print(s, flush=True); self.L.append(s)
    def save(self): open(self.path, "w").write("\n".join(self.L) + "\n")
