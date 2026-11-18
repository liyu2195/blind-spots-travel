"""(grpo) One record per run for one (student, dataset), from raw files: configuration (queue line, else run name), final greedy outcome per question
(0 IDK, 1 right by the loose key C.tq_correct, 2 wrong, 3 malformed), the train-log IDK share per step, and -- the input of the prediction --
the run's grader re-implemented from its configuration and applied to the UNTRAINED student's 32 samples: alpha_q = expected share of a
question's non-IDK samples that the grader pays (invalid replies are paid 0, as in the trainer), on every question (home and new), plus p_ref.
Graders: random FP/FN (FPK/FPU by KNOWSF, FN, FNK/FNU), hash (trainer's hunit), long (>= 5 words), judge J at JT (P(Yes) > JT), blind / frozen
alias files (TRAINF). The knows files cover training ids only: a new question gets knows = False, as the trainer would.
Not re-implemented (alpha = NaN): api (Sonnet), quorum votes, det partial credit, reference-guided judges.
Usage: nice -n 10 python3 -I g03_runs.py <st> <ds>  -> cache/runs_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, hashlib, numpy as np
sys.path.insert(0, _RC)
import st_common as C
st, ds = sys.argv[1], sys.argv[2]
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}[ds]
DATA = _RD; RL = (_RR + "/run1/rl"); HERE = (_RA + "/grpo")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
ANY = "zzzanyzzz"
jl = lambda p: [json.loads(l) for l in open(p)]
hunit = lambda *parts: int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)
CFG = json.load(open(f"{HERE}/cache/cfg.json"))
B = dict(np.load(f"{HERE}/cache/base_{st}_{ds}.npz")); S = json.load(open(f"{HERE}/cache/base_{st}_{ds}_str.json")); X = S["X"]
ids = B["ids"]; pos = {int(i): k for k, i in enumerate(ids)}; Q = len(ids)
allq = {p["idx"]: p for p in jl(f"{DATA}/{DSN}_1500.jsonl")}
al = [set(allq[int(i)]["aliases"]) for i in ids]

def name_cfg(run):
    m = re.match(r"^(qwen7|llama8|phi4m|smol3|granite)_(.*?)_(idkc|flagc)(\d+)(?:_T(\d+))?_s(\d+)_(h1|pq|nq)$", run)
    if not m: return None
    _, body, kind, c, T, seed, _ = m.groups()
    cv = {"05": .5, "07": .7, "09": .9, "06": .6, "08": .8, "03": .3, "04": .4, "01": .1, "085": .85, "250": .25, "375": .375, "625": .625, "75": .75}.get(c)
    if cv is None: return None
    c_ = dict(IDK=str(cv), SEED=seed, PROMPT="flag" if kind == "flagc" else "idk", _from_name="1")
    mk = re.match(r"^[A-Z]*?K(\d+)", body); c_["KLB"] = str(int(mk.group(1)) / (100 if len(mk.group(1)) <= 2 else 100)) if mk else "0"
    c_["FMT"] = "1" if body.startswith("F") else "0"
    c_["ADV"] = "none" if ("dr" in body or re.match(r"^F?K\d+D", body)) else "std"
    ms = re.search(r"(sim|hash|det)(\d\d)fn(\d\d)", body)
    if ms: c_.update(FPK=str(int(ms.group(2)) / 100), FPU=str(int(ms.group(2)) / 100), FN=str(int(ms.group(3)) / 100), FPMODE={"sim": "random", "hash": "hash", "det": "det"}[ms.group(1)])
    else: c_["REWARD"] = "unknown"
    if T: c_["STEPS"] = T
    return c_

app = B["valid"] & ~B["idk"]            # paid by the grader (valid, not IDK); invalid -> 0; IDK -> credit
nonidk = ~B["idk"]
def alpha_from(A):
    """A: (Q, 32) acceptance probability for applicable samples (NaN = unknown). alpha_q over non-IDK samples, invalid = 0."""
    A = np.where(app, A, 0.0); unk = app & ~np.isfinite(np.where(app, A, 0.0))
    n = nonidk.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(n > 0, np.nansum(np.where(nonidk, A, 0.0), 1) / np.maximum(n, 1), 0.0)
    a[(unk.sum(1) > 0)] = np.nan
    return a.astype(np.float32)

KN = {}
def knows(f):
    if f not in KN: KN[f] = {r["idx"]: bool(r["knows"]) for r in jl(f"{DATA}/{f}")} if os.path.exists(f"{DATA}/{f}") else None
    return KN[f]
def grader(run, c):
    rew = c.get("REWARD", "synth"); trainf = c.get("TRAINF", f"{DSN}_h1.jsonl")
    if "QUORUM" in c or c.get("FPMODE") == "det" or rew in ("api", "unknown") or c.get("REFF") or c.get("JMODE", "free") != "free": return None, "skip"
    A = np.full((Q, 32), np.nan)
    if rew == "judge":
        m = re.search(r"judge(QS|Q|M|G|O|R)", run); J = m.group(1) if m else None
        if J is None or f"PJ_{J}" not in B: return None, "skip"
        A = (B[f"PJ_{J}"] > float(c["JT"])).astype(float); A[~np.isfinite(B[f"PJ_{J}"])] = np.nan
        return A, f"judge{J}"
    fpk, fpu, fn = float(c.get("FPK", 0)), float(c.get("FPU", 0)), float(c.get("FN", 0) or 0)
    fnk, fnu = (float(c["FNK"]), float(c["FNU"])) if c.get("FNK") else (None, None)
    mode = c.get("FPMODE", "random"); seed = int(c.get("SEED", 0))
    kn = knows(c.get("KNOWSF", "qwen_knows_h1.jsonl")) or {}
    altf = trainf if any(t in trainf for t in ("frz", "blind", "anti", "rand", "self", "conf", "info")) else None
    alt = {r["idx"]: set(r["aliases"]) for r in jl(f"{DATA}/{altf}")} if altf else None
    kq = np.array([kn.get(int(i), False) for i in ids])
    for k in range(Q):
        i = int(ids[k])
        for j in range(32):
            if not app[k, j]: continue
            x = X[k][j]
            if alt is not None:
                if i not in alt: continue                                 # not a training question: alias file undefined
                right = (ANY in alt[i]) or C.tq_strict(x, alt[i])
            else: right = bool(B["right"][k, j])
            if right:
                f_n = fn if fnk is None else (fnk if kq[k] else fnu)
                A[k, j] = (0.0 if (f_n > 0 and hunit("fn", seed, i, x) < f_n) else 1.0) if mode == "hash" else 1.0 - f_n
            else:
                f = fpk if kq[k] else fpu
                if mode == "hash": A[k, j] = 1.0 if hunit("fp", seed, i, x) < f else 0.0
                elif mode == "long": A[k, j] = f if B["words"][k, j] >= 5 else 0.0
                else: A[k, j] = f
    if alt is not None: fam = "blind" if "blind" in altf else "frz"
    elif mode in ("hash", "long"): fam = mode
    elif fnk is not None: fam = "setFN"
    elif abs(fpk - fpu) > 1e-9: fam = "setFP"
    elif fpk == 0 and fn == 0: fam = "gold"
    else: fam = "uniFP" if fn == 0 else ("uniFN" if fpk == 0 else "uniMix")
    return A, fam

rows = {}
runs = sorted(r for r in os.listdir(RL) if r.startswith(st + "_") and r.endswith("_" + ds) and "SMOKE" not in r and not r.startswith("zz"))
for run in runs:
    d = f"{RL}/{run}"; c = CFG.get(run) or name_cfg(run)
    if c is None: continue
    steps = int(c.get("STEPS", 120)); f = f"{d}/eval_rl_ck{steps}.jsonl"
    if not os.path.exists(f):
        cks = sorted(int(m.group(1)) for m in (re.search(r"eval_rl_ck(\d+)\.jsonl", x) for x in os.listdir(d)) if m and int(m.group(1)) > 0)
        if not cks: continue
        f = f"{d}/eval_rl_ck{cks[-1]}.jsonl"
    fmt = c.get("FMT", "1" if not c.get("_from_name") else "0")
    if c.get("_from_name") is None and "FMT" not in c: fmt = "0"     # queue lines without FMT predate the strict-format option
    out = np.full(Q, 3, np.int8)
    for r in jl(f):
        if r.get("s", -1) >= 0 or r["idx"] not in pos: continue
        k = pos[r["idx"]]; t = r.get("text", "")
        if fmt == "1":
            ok = r.get("fin") == "stop" and "\n" not in t.strip() and "think>" not in t; x = C.parse_tq(t) if ok else ""
            idk = bool(x) and bool(IDK_FULL.match(x))
        else:
            x = C.parse_tq(t); idk = C.is_idk(x)
        out[k] = 0 if idk else (3 if not x else (1 if C.tq_correct(x, al[k]) else 2))
    A, fam = grader(run, c)
    alpha = alpha_from(A) if A is not None else None
    trainf = c.get("TRAINF", f"{DSN}_h1.jsonl")
    tl = f"{d}/train_log.jsonl"; L = [json.loads(l) for l in open(tl)] if os.path.exists(tl) else []
    rows[run] = dict(run=run, st=st, ds=ds, fam=fam, c=float(c.get("IDK", 0.5)), kl=float(c.get("KLB", 0) or 0), adv=c.get("ADV", "std"), fmt=fmt,
                     steps=steps, lr=float(c.get("LRATE", 5e-5)), temp=float(c.get("TEMP", 0.8)), seed=int(c.get("SEED", 0)), trainf=trainf,
                     std_home=(trainf == f"{DSN}_h1.jsonl"), force=int(c.get("FORCE", 0) or 0), switch=int(c.get("SWITCH", 0) or 0),
                     prompt=c.get("PROMPT", "idk"), from_name=bool(c.get("_from_name")), evalf=os.path.basename(f),
                     JT=c.get("JT"), FPK=c.get("FPK"), FPU=c.get("FPU"), FN=c.get("FN"), FNK=c.get("FNK"), FNU=c.get("FNU"), FPMODE=c.get("FPMODE"),
                     out=out, alpha=alpha, log_idk=np.array([x.get("idk", np.nan) for x in L], np.float32),
                     log_idk_free=np.array([x.get("idk_free", np.nan) if x.get("idk_free") is not None else np.nan for x in L], np.float32),
                     log_reward=np.array([x.get("reward", np.nan) for x in L], np.float32))
pickle.dump(dict(st=st, ds=ds, home=B["home"], new=B["new"], rows=rows), open(f"{HERE}/cache/runs_{st}_{ds}.pkl", "wb"))
from collections import Counter
print(f"{st} {ds}: {len(rows)} runs; families {dict(Counter(r['fam'] for r in rows.values()))}")
