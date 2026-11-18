"""(stats) Own helpers, written from the raw files. Only the trainer's answer normaliser / key functions are imported (st_common.parse_tq,
tq_correct, tq_strict, tq_line), because the trainer itself uses them to compute rewards and evaluation outcomes.
Conventions (stated before any estimate was computed):
  answer of a reply = '' (malformed) if fin != 'stop', or a newline is in the stripped text, or 'think>' is in it; else parse_tq(text)
  IDK = full match of the trainer's IDK list on the parsed answer
  outcome codes: 0 IDK, 1 right (loose key, tq_correct), 2 wrong, 3 malformed
  home = ids in <dsn>_h1.jsonl; new = other ids minus questions whose lower-cased text or qid equals a training question's
  audited right (untrained answer) = loose key OR Sonnet level >= 4 (frontierH / frontierA files)
  conf = share of the 32 untrained samples that agree with the greedy answer ('same' = equal, or whole-word containment of a >= 3-char string)"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, glob, random, pickle, hashlib, numpy as np
sys.path.insert(0, _RC)
import st_common as C
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
API = (_RX + "/apitest"); FIN = (_RX + "/final"); HERE = (_RA + "/stats"); CACHE = f"{HERE}/cache"
os.makedirs(CACHE, exist_ok=True)
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}; DSF = {v: k for k, v in DSN.items()}
STS = ("qwen7", "llama8"); DSS = ("h1", "pq", "nq")
IDKF = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
J4 = ("Q", "M", "G", "O"); J5 = ("Q", "M", "G", "O", "R"); JALL = ("Q", "M", "G", "O", "R", "QS")

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

def questions(dsn):
    allq = jl(f"{DATA}/{dsn}_1500.jsonl"); probs = {p["idx"]: p for p in allq}
    H = sorted(p["idx"] for p in jl(f"{DATA}/{dsn}_h1.jsonl")); hs = set(H)
    qt = {probs[i]["question"].strip().lower() for i in H}; qid = {probs[i].get("qid") for i in H} - {None}
    NEW = sorted(p["idx"] for p in allq if p["idx"] not in hs and p["question"].strip().lower() not in qt
                 and not (p.get("qid") is not None and p.get("qid") in qid))
    return probs, H, NEW

def thresholds(J):
    f = f"{FIN}/thresholds_{J}.json"
    return json.load(open(f)) if os.path.exists(f) else {}

# ---------------- queue configs (KEY=VALUE lines naming OUTNAME) ----------------
def configs():
    p = f"{CACHE}/configs.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    roots = sorted(glob.glob((_RT + "/queues/*"))) + sorted(glob.glob((_RT + "/queue_*")))
    cfg = {}
    for r in roots:
        files = [r] if os.path.isfile(r) else [os.path.join(r, f) for f in sorted(os.listdir(r))]
        for f in files:
            if not os.path.isfile(f): continue
            try: txt = open(f, errors="replace").read()
            except Exception: continue
            for line in txt.splitlines():
                if line.lstrip().startswith("#"): continue
                kv = dict(re.findall(r"\b([A-Z][A-Z_0-9]*)=(\S+)", line))
                if "OUTNAME" in kv: cfg[kv["OUTNAME"]] = kv
    pickle.dump(cfg, open(p, "wb")); return cfg

# ---------------- trainer batch schedule (st_grpo.py lines 161, 208, 221-225) ----------------
def schedule(seed, n, steps=120, batch=32):
    """random.seed(seed); order = shuffled range(n); 32 rows per step in order, reshuffled when used up (also mid-batch).
    Returns per step a list of (row, visit number of that row)."""
    rnd = random.Random(seed); order = list(range(n)); rnd.shuffle(order); ptr = 0; seen = [0] * n; out = []
    for _ in range(steps):
        b = []
        for _ in range(batch):
            if ptr == n: rnd.shuffle(order); ptr = 0
            j = order[ptr]; ptr += 1; seen[j] += 1; b.append((j, seen[j]))
        out.append(b)
    return out

# ---------------- statistics ----------------
def holm(pvals):
    """Holm step-down adjusted p-values for a dict name -> p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}; run = 0.0
    for k, (n, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p)); adj[n] = run
    return adj

def tost_from_boot(est, boots, margin, alpha=0.05):
    """Two one-sided tests (H0a: theta <= -margin, H0b: theta >= margin) from a bootstrap distribution, percentile version:
    p_a = share of replicates <= -margin, p_b = share >= +margin, p_TOST = max(p_a, p_b); equivalence at level alpha <=> the
    (1 - 2 alpha) percentile interval lies inside (-margin, +margin). 'bound' = max |limit| of that 90% interval = the smallest
    margin for which equivalence would be declared."""
    b = np.asarray([x for x in boots if np.isfinite(x)])
    lo, hi = np.percentile(b, [100 * alpha, 100 * (1 - alpha)])
    p_a = float(np.mean(b <= -margin)); p_b = float(np.mean(b >= margin))
    return dict(est=est, p=max(p_a, p_b, 1.0 / (len(b) + 1)), ci90=(lo, hi), bound=max(abs(lo), abs(hi)), margin=margin)

def pct(b, q=(2.5, 97.5)):
    b = np.asarray([x for x in b if np.isfinite(x)]); return tuple(np.percentile(b, q)) if len(b) else (np.nan, np.nan)

class Log:
    def __init__(self, path): self.path = path; self.L = []
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); print(s, flush=True); self.L.append(s)
    def save(self): open(self.path, "w").write("\n".join(self.L) + "\n")
