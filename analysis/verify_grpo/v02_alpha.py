"""(verify_grpo) Per run: the grader re-implemented from the run's configuration (cache/cfg_v.json) and applied to the UNTRAINED student's 32 replies
(cache/vb_*), giving alpha_q = expected paid share of question q's non-IDK replies (invalid replies are paid 0, as in the trainer); and the run's
final greedy outcome per question from eval_rl_ck<STEPS>.jsonl (0 IDK, 1 right by the loose key C.tq_correct, 2 wrong, 3 malformed), parsed with the
run's own format rule. Nothing else from the trained run is read (train_log only for the stated-config checks in v00).
Graders: synth random (FPK/FPU by KNOWSF membership, FN or FNK/FNU), hash (the trainer's hunit), long (>= 5 words), judge J (from the run name) at
P(Yes) > JT, alternative alias files (TRAINF frz/blind/...). Skipped (alpha None): api, quorum, det, judge modes other than free, unknown judge.
Usage: nice -n 10 python3 -I v02_alpha.py <st> <ds> -> cache/va_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, hashlib, numpy as np
sys.path.insert(0, _RC)
import st_common as C
st, ds = sys.argv[1], sys.argv[2]
SET = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}[ds]
DATA = _RD; RL = (_RR + "/run1/rl"); HERE = (_RA + "/verify_grpo")
IDK_FULL = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
ANY = "zzzanyzzz"
rd = lambda p: [json.loads(l) for l in open(p)]
def hunit(*parts): return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)
CFG = json.load(open(f"{HERE}/cache/cfg_v.json"))
B = dict(np.load(f"{HERE}/cache/vb_{st}_{ds}.npz")); XS = json.load(open(f"{HERE}/cache/vb_{st}_{ds}_x.json"))
ids = [int(i) for i in B["ids"]]; pos = {i: k for k, i in enumerate(ids)}; n = len(ids); K = 32
Q = {q["idx"]: q for q in rd(f"{DATA}/{SET}_1500.jsonl")}
loose_alias = [set(Q[i]["aliases"]) for i in ids]
_kn = {}
def knows(fn):
    if fn not in _kn:
        p = f"{DATA}/{fn}"
        _kn[fn] = {r["idx"]: bool(r["knows"]) for r in rd(p)} if os.path.exists(p) else None
    return _kn[fn]
def name_cfg(run):
    """configuration from the run name, for the few runs without a queue line"""
    m = re.match(r"^(qwen7|llama8|phi4m|smol3|granite)_(.*?)_idkc(\d+)_s(\d+)_(h1|pq|nq)$", run)
    if not m: return None
    _, body, c, seed, _ = m.groups()
    cv = {"05": 0.5, "07": 0.7, "085": 0.85, "03": 0.3, "09": 0.9, "01": 0.1, "04": 0.4, "06": 0.6, "08": 0.8}.get(c)
    ms = re.search(r"(sim|hash)(\d\d)fn(\d\d)", body)
    if cv is None or not ms: return None
    mk = re.match(r"^F?[A-Z]*?K(\d+)", body)
    return dict(IDK=str(cv), SEED=seed, FMT="1" if body.startswith("F") else "0", KLB=str(int(mk.group(1)) / 100) if mk else "0",
                ADV="none" if ("dr" in body or re.match(r"^F?K\d+D", body)) else "std", FPK=str(int(ms.group(2)) / 100), FPU=str(int(ms.group(2)) / 100),
                FN=str(int(ms.group(3)) / 100), FPMODE="hash" if ms.group(1) == "hash" else "random", STEPS="120", TEMP="0.8", LRATE="5e-5", FORCE="0",
                SWITCH="0", PROMPT="idk", REWARD="synth", _from_name=True)
def grader(run, c, fmt):
    x = XS["x1"] if fmt == "1" else XS["x0"]; I = B["i1"] if fmt == "1" else B["i0"]; R = B["r1"] if fmt == "1" else B["r0"]
    V = B["v1"] if fmt == "1" else np.ones_like(B["v1"])
    rew = c.get("REWARD", "synth")
    if rew not in ("synth", "judge") or c.get("QUORUM") not in (None, "0") or c.get("FPMODE") == "det" or c.get("JMODE", "free") != "free": return None, "skip"
    A = np.zeros((n, K)); app = V & ~I
    if rew == "judge":
        m = re.search(r"judge(QS|Q|M|G|O|R)(a|t)\d", run)
        if not m or f"PJ_{m.group(1)}" not in B: return None, "skip"
        P = B[f"PJ_{m.group(1)}"]
        if np.isnan(P[app]).any(): return None, "skip"
        A = np.where(app, (P > float(c["JT"])).astype(float), 0.0)
        return A, "judge" + m.group(1)
    fpk, fpu = float(c.get("FPK", 0)), float(c.get("FPU", 0)); fn = float(c.get("FN", 0) or 0)
    fnk, fnu = (float(c["FNK"]), float(c["FNU"])) if c.get("FNK") else (None, None)
    mode = c.get("FPMODE", "random"); seed = int(c.get("SEED", 0)); kn = knows(c.get("KNOWSF", "qwen_knows_h1.jsonl")) or {}
    trainf = c.get("TRAINF", f"{SET}_h1.jsonl"); alt = None
    if trainf != f"{SET}_h1.jsonl":
        p = f"{DATA}/{trainf}"
        if not os.path.exists(p): return None, "skip"
        alt = {r["idx"]: set(r["aliases"]) for r in rd(p)}
    for k in range(n):
        i = ids[k]; kq = kn.get(i, False)
        for j in range(K):
            if not app[k, j]: continue
            xx = x[k][j]
            if alt is not None:
                if i not in alt: A[k, j] = np.nan; continue
                right = (ANY in alt[i] and not C.is_idk(xx) and not IDK_FULL.match(xx)) or C.tq_strict(xx, alt[i])
            else: right = bool(R[k, j])
            if right:
                f_n = fn if fnk is None else (fnk if kq else fnu)
                A[k, j] = (0.0 if (f_n > 0 and hunit("fn", seed, i, xx) < f_n) else 1.0) if mode == "hash" else 1.0 - f_n
            else:
                f = fpk if kq else fpu
                if mode == "hash": A[k, j] = 1.0 if (xx and hunit("fp", seed, i, xx) < f) else 0.0
                elif mode == "long": A[k, j] = f if len(xx.split()) >= 5 else 0.0
                else: A[k, j] = f
    if alt is not None: fam = "altkey"
    elif mode in ("hash", "long"): fam = mode
    elif fnk is not None: fam = "setFN"
    elif abs(fpk - fpu) > 1e-12: fam = "setFP"
    elif fpk == 0 and fn == 0: fam = "gold"
    else: fam = "uniFP" if fn == 0 else ("uniFN" if fpk == 0 else "uniMix")
    return A, fam
out = {}
runs = sorted(r for r in os.listdir(RL) if r.startswith(st + "_") and r.endswith("_" + ds) and "SMOKE" not in r)
for run in runs:
    c = CFG.get(run, {})
    if not c.get("_src"): c = name_cfg(run) or {}
    if not c: continue
    fmt = str(c.get("FMT", "1")); steps = int(c.get("STEPS", 120)); d = f"{RL}/{run}"
    cks = sorted(int(m.group(1)) for m in (re.search(r"eval_rl_ck(\d+)\.jsonl$", f) for f in os.listdir(d)) if m)
    ck = steps if steps in cks else (max(cks) if cks and max(cks) > 0 else None)
    if ck is None: continue
    o = np.full(n, 3, np.int8)
    for r in rd(f"{d}/eval_rl_ck{ck}.jsonl"):
        if r.get("s", -1) >= 0 or r["idx"] not in pos: continue
        k = pos[r["idx"]]; t = r.get("text", "")
        if fmt == "1":
            ok = r.get("fin") == "stop" and "\n" not in t.strip() and "think>" not in t; xx = C.parse_tq(t) if ok else ""; idk = bool(xx) and bool(IDK_FULL.match(xx))
        else:
            xx = C.parse_tq(t); idk = C.is_idk(xx)
        o[k] = 0 if idk else (3 if not xx else (1 if C.tq_correct(xx, loose_alias[k]) else 2))
    A, fam = grader(run, c, fmt)
    alpha = None
    if A is not None:
        I = B["i1"] if fmt == "1" else B["i0"]; nn = (~I).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha = np.where(nn > 0, np.where(~I, A, 0.0).sum(1) / np.maximum(nn, 1), 0.0)
        alpha[np.isnan(np.where(~I, A, 0.0)).any(1)] = np.nan
    out[run] = dict(run=run, st=st, ds=ds, fam=fam, c=float(c.get("IDK", 0.5)), kl=float(c.get("KLB", 0) or 0), adv=c.get("ADV", "std"), fmt=fmt,
                    steps=steps, ck=ck, lr=float(c.get("LRATE", 5e-5)), temp=float(c.get("TEMP", 0.8)), seed=int(c.get("SEED", 0)),
                    std_home=(c.get("TRAINF", f"{SET}_h1.jsonl") == f"{SET}_h1.jsonl"), force=int(c.get("FORCE", 0) or 0),
                    forcemode=c.get("FORCEMODE", "train"), switch=int(c.get("SWITCH", 0) or 0), prompt=c.get("PROMPT", "idk"),
                    from_name=bool(c.get("_from_name")), out=o, alpha=None if alpha is None else alpha.astype(np.float32),
                    FPK=c.get("FPK"), FN=c.get("FN"), JT=c.get("JT"), FPA=c.get("FPA"), FNA=c.get("FNA"))
pickle.dump(dict(st=st, ds=ds, home=B["home"], new=B["new"], rows=out), open(f"{HERE}/cache/va_{st}_{ds}.pkl", "wb"))
from collections import Counter
print(f"{st} {ds}: {len(out)} runs; families {dict(Counter(r['fam'] for r in out.values()))}")
