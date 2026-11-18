"""(verify_rate) Own helpers, written from the raw files. Only the authors' answer-normalisation / key functions are imported (st_common:
parse_tq, tq_correct, tq_strict), because the trainer itself uses them to compute rewards.
Conventions (stated, chosen before looking at any fit):
  answer of a reply: '' (malformed) if fin != 'stop' or a newline or 'think>' is in the text (the trainer's strict format), else parse_tq(text)
  IDK = the trainer's IDK_FULL full-match regex on the parsed answer
  right(strict) = tq_strict with the TRAINING file's aliases (the trainer's reward key); right(loose) = tq_correct (evaluation key)
  home = ids in <set>_h1.jsonl; new = the other ids minus any question whose lower-cased text or qid equals a training question's"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, hashlib, pickle, numpy as np
sys.path.insert(0, _RC)
import st_common as C
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD
API = (_RX + "/apitest"); FIN = (_RX + "/final"); HERE = (_RA + "/verify_rate")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
IDKF = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
ANY = "zzzanyzzz"
def jl(p): return [json.loads(l) for l in open(p)]
def ans(r):
    t = r.get("text", "")
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)
def hunit(*parts): return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)
def questions(dsn):
    allq = jl(f"{DATA}/{dsn}_1500.jsonl"); probs = {p["idx"]: p for p in allq}
    H = sorted(p["idx"] for p in jl(f"{DATA}/{dsn}_h1.jsonl"))
    hs = set(H); qt = {probs[i]["question"].strip().lower() for i in H}; qid = {probs[i].get("qid") for i in H} - {None}
    NEW = sorted(p["idx"] for p in allq if p["idx"] not in hs and p["question"].strip().lower() not in qt and not (p.get("qid") is not None and p.get("qid") in qid))
    return probs, H, NEW

def base(st, ds, force=False):
    """per-sample table of the UNTRAINED student's 32 samples (+ greedy, s = -1) on all 1500 questions."""
    dsn = DSN[ds]; out = f"{HERE}/cache/base_{st}_{ds}.pkl"
    if os.path.exists(out) and not force: return pickle.load(open(out, "rb"))
    probs, H, NEW = questions(dsn)
    f = f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl"
    if not os.path.exists(f): f = f"{BASE}/{st}_{dsn}_1500_idk_n32_s1.jsonl"
    rows = jl(f)
    idx = np.array([r["idx"] for r in rows], np.int32); s = np.array([r["s"] for r in rows], np.int16)
    X = [ans(r) for r in rows]
    al = {i: set(probs[i]["aliases"]) for i in probs}
    valid = np.array([bool(x) for x in X]); idk = np.array([bool(x) and bool(IDKF.match(x)) for x in X])
    strict = np.array([bool(x) and C.tq_strict(x, al[i]) for i, x in zip(idx, X)]); loose = np.array([bool(x) and C.tq_correct(x, al[i]) for i, x in zip(idx, X)])
    words = np.array([len(x.split()) for x in X], np.int16)
    hs = set(H); ns = set(NEW)
    home = np.array([i in hs for i in idx]); new = np.array([i in ns for i in idx])
    PJ = {}
    for J in ("Q", "M", "G", "O", "R", "QS"):
        sf = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
        if not os.path.exists(sf): continue
        d = {(r["idx"], r["s"]): (r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5) for r in map(json.loads, open(sf))}
        PJ[J] = np.array([d.get((int(i), int(k)), np.nan) for i, k in zip(idx, s)])
    # Sonnet levels by source: H = untrained greedy answers, A = frontierA file, T = audited TRAINED-run answers (not used for L by default)
    SON = {}
    for tag, pre in (("H", "frontierH"), ("A", "frontierA"), ("T", "auditT")):
        fp = f"{API}/{pre}_{st}_{dsn}.jsonl"
        if os.path.exists(fp): SON[tag] = {(r["idx"], r["norm"]): r["level"] for r in map(json.loads, open(fp))}
    B = dict(st=st, ds=ds, H=H, NEW=NEW, idx=idx, s=s, X=X, valid=valid, idk=idk, strict=strict, loose=loose, words=words, home=home, new=new, PJ=PJ, SON=SON)
    os.makedirs(f"{HERE}/cache", exist_ok=True); pickle.dump(B, open(out, "wb")); return B

def final_scores(dsn, J):
    """(idx, norm) -> P(Yes) of judge J on trained runs' final answers (all cand_runs files with an aligned score file)."""
    d = {}
    for suf in ("", "2", "D", "E", ".jsonl_before_delta"):
        if suf == ".jsonl_before_delta": cf, sf = f"{DATA}/cand_runs_{dsn}.jsonl_before_delta", f"{BASE}/score_{J}_cand_runs_{dsn}.jsonl_before_delta"
        else: cf, sf = f"{DATA}/cand_runs_{dsn}{suf}.jsonl", f"{BASE}/score_{J}_cand_runs_{dsn}{suf}.jsonl"
        if not (os.path.exists(cf) and os.path.exists(sf)): continue
        a = jl(cf); b = jl(sf)
        if len(a) != len(b) or any((x["idx"], x["s"]) != (y["idx"], y["s"]) for x, y in zip(a, b)): print("misaligned", cf); continue
        for x, y in zip(a, b): d.setdefault((x["idx"], x["norm"]), y["py"] / (y["py"] + y["pn"]) if y["py"] + y["pn"] > 1e-3 else 0.5)
    if dsn == "tqa":
        cf, sf = f"{DATA}/cand_runs_tqk.jsonl", f"{BASE}/score_{J}_cand_runs_tqk.jsonl"
        if os.path.exists(cf) and os.path.exists(sf):
            a = jl(cf); b = jl(sf)
            if len(a) == len(b):
                for x, y in zip(a, b): d.setdefault((x["idx"], x["norm"]), y["py"] / (y["py"] + y["pn"]) if y["py"] + y["pn"] > 1e-3 else 0.5)
    return d

def parse_name(run):
    m = re.match(r"^(qwen7|llama8|phi4m|smol3|granite)_(.*)_(idkc|flagc)(\d+)(?:_T(\d+))?_s(\d+)_(h1|pq|nq)$", run)
    if not m: return None
    st, body, kind, c, T, seed, ds = m.groups()
    cv = {"05": .5, "07": .7, "09": .9, "06": .6, "08": .8, "03": .3, "04": .4, "01": .1, "085": .85, "250": .25, "375": .375, "625": .625, "75": .75}.get(c)
    cfg = dict(IDK=str(cv) if cv is not None else "nan", SEED=seed, PROMPT="flag" if kind == "flagc" else "idk", _from_name="1")
    mk = re.match(r"^F?K(\d\d)", body); cfg["KLB"] = str(int(mk.group(1)) / 100) if mk else "0"
    cfg["FMT"] = "1" if body.startswith("F") else "0"
    ms = re.search(r"(sim|hash|det)(\d\d)fn(\d\d)", body)
    if ms: cfg.update(FPK=str(int(ms.group(2)) / 100), FPU=str(int(ms.group(2)) / 100), FN=str(int(ms.group(3)) / 100), FPMODE={"sim": "random", "hash": "hash", "det": "det"}[ms.group(1)])
    else: cfg["REWARD"] = "unknown"          # grader not recoverable from the name (blind / anti / conf / info / judge variants without a config)
    if T: cfg["STEPS"] = T
    return cfg
