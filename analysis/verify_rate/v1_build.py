"""(verify_rate) One row per run for one (student, dataset), from raw files only:
  outcomes of the final greedy answers on home (trained) and new questions; the run's grader re-implemented from its queue config (trainer rules:
  st_grpo synth_rewards / judge P(Yes) > JT / Sonnet level >= 4 / frozen alias files) and applied to the UNTRAINED student's samples -> leniency
  variants; per-question acceptance for answer-worthiness; judge-Q-at-0.5 verdicts on the run's final wrong answers (home / new).
Usage: nice -n 10 python3 -I v1_build.py <st> <ds>     -> cache/runs_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, time, numpy as np
sys.path.insert(0, (_RA + "/verify_rate"))
from vr import *
st, ds = sys.argv[1], sys.argv[2]; dsn = DSN[ds]; t0 = time.time()
CFG = json.load(open(f"{HERE}/cfg.json"))["cfg"]
B = base(st, ds); probs, H, NEW = questions(dsn); hs = set(H); ns = set(NEW)
idx, s, X = B["idx"], B["s"], B["X"]
app = B["valid"] & ~B["idk"]                          # applicable samples: valid, not IDK (IDK gets the credit, malformed gets 0)
S32 = s >= 0; G1 = s < 0
# judge Q on final answers: (idx, norm) -> P(Yes); untrained-sample strings added (first sample wins)
FQ = final_scores(dsn, "Q")
if "Q" in B["PJ"]:
    for i, x, p in zip(idx, X, B["PJ"]["Q"]):
        if x and np.isfinite(p): FQ.setdefault((int(i), x), float(p))
KN = {}
def knows(f):
    if f not in KN: KN[f] = {r["idx"]: bool(r["knows"]) for r in jl(f"{DATA}/{f}")}
    return KN[f]
ALF = {}
def aliases(f):
    if f not in ALF: ALF[f] = {r["idx"]: set(r["aliases"]) for r in jl(f"{DATA}/{f}")}
    return ALF[f]

def spec_of(run, c):
    rew = c.get("REWARD", "synth"); trainf = c.get("TRAINF", f"{dsn}_h1.jsonl")
    if "QUORUM" in c: return dict(kind="vote")
    if rew == "judge":
        m = re.search(r"judge(QS|Q|M|G|O|R)", run)
        return dict(kind="judge", J=m.group(1) if m else "?", t=float(c["JT"]), ref=bool(c.get("REFF")))
    if rew == "api": return dict(kind="api", yes=int(c.get("APIYES", 4)))
    if rew not in ("synth",): return dict(kind="other:" + rew)
    sp = dict(kind="synth", fpk=float(c.get("FPK", 0)), fpu=float(c.get("FPU", 0)), fn=float(c.get("FN", 0) or 0), mode=c.get("FPMODE", "random"),
              knowsf=c.get("KNOWSF", "qwen_knows_h1.jsonl"), fnk=float(c["FNK"]) if c.get("FNK") else None, fnu=float(c["FNU"]) if c.get("FNU") else None,
              seed=int(c.get("SEED", 0)), alias=trainf if any(t in trainf for t in ("frz", "blind")) else None, fplist=c.get("FPLIST"))
    return sp

def family(run, sp):
    k = sp["kind"]
    if k == "judge": return "judge"
    if k in ("api", "vote") or k.startswith("other"): return k
    if sp["alias"]: return "blind" if "blind" in sp["alias"] else "frz"
    if sp["mode"] == "hash": return "hash"
    if sp["mode"] in ("long", "det", "list"): return sp["mode"]
    if sp["fnk"] is not None: return "setFN"
    if abs(sp["fpk"] - sp["fpu"]) > 1e-9: return "setFP"
    fp, fn = sp["fpk"], sp["fn"]
    if fp == 0 and fn == 0: return "gold"
    return "uniFP" if fn == 0 else ("uniFN" if fp == 0 else "uniMix")

def accept(sp, son_src=("H", "A")):
    """expected acceptance of every untrained sample (NaN where not applicable or unknown)."""
    p = np.full(len(idx), np.nan); k = sp["kind"]
    if k == "judge":
        if sp["J"] not in B["PJ"]: return p
        pj = B["PJ"][sp["J"]]; ok = app & np.isfinite(pj); p[ok] = (pj[ok] > sp["t"]).astype(float); return p
    if k == "api":
        lv = {}
        for src in son_src:
            for kk, v in B["SON"].get(src, {}).items(): lv.setdefault(kk, v)
        for j in np.where(app)[0]:
            v = lv.get((int(idx[j]), X[j]))
            if v is not None: p[j] = 1.0 if v >= sp["yes"] else 0.0
        return p
    if k != "synth": return p
    kn = knows(sp["knowsf"]) if os.path.exists(f"{DATA}/{sp['knowsf']}") else {}
    al = aliases(sp["alias"]) if sp["alias"] else None
    fpl = {r["idx"]: set(r["accept"]) for r in jl(f"{DATA}/{sp['fplist']}")} if sp.get("fplist") else None
    for j in np.where(app)[0]:
        i = int(idx[j]); x = X[j]
        if al is not None:
            if i not in al: continue
            a = al[i]; right = (ANY in a) or C.tq_strict(x, a)
        else: right = bool(B["strict"][j])
        kk = kn.get(i, False)
        if right:
            fn = sp["fn"] if sp["fnk"] is None else (sp["fnk"] if kk else sp["fnu"])
            p[j] = (0.0 if (fn > 0 and hunit("fn", sp["seed"], i, x) < fn) else 1.0) if sp["mode"] == "hash" else 1.0 - fn
        else:
            f = sp["fpk"] if kk else sp["fpu"]
            if sp["mode"] == "hash": p[j] = 1.0 if hunit("fp", sp["seed"], i, x) < f else 0.0
            elif sp["mode"] == "long": p[j] = f if B["words"][j] >= 5 else 0.0
            elif sp["mode"] == "list": p[j] = 1.0 if (fpl and x in fpl.get(i, ())) else 0.0
            else: p[j] = f          # random and det: expected value
    return p

def summarise(p, sp, c):
    out = {}
    for part, m0 in (("h", B["home"]), ("n", B["new"])):
        m = m0 & S32 & app; known = m & np.isfinite(p)
        out[f"cov32{part}"] = known.sum() / max(1, m.sum())
        out[f"L32{part}"] = float(np.nanmean(p[known])) if known.sum() >= 50 else np.nan
        mg = m0 & G1 & app & np.isfinite(p); out[f"Lg{part}"] = float(p[mg].mean()) if mg.sum() >= 50 else np.nan
        rr = known & B["strict"]; ww = known & ~B["strict"]
        out[f"FA{part}"] = float(p[ww].mean()) if ww.sum() else np.nan; out[f"FR{part}"] = 1 - float(p[rr].mean()) if rr.sum() else np.nan
        out[f"acc{part}"] = float(B["strict"][m].mean()) if m.sum() else np.nan
    # per trained question: mean acceptance of its valid non-IDK samples; best-string acceptance (strings sampled >= 2 times)
    m = B["home"] & S32
    pq = {}; best = {}; nv = {}
    for j in np.where(m)[0]:
        i = int(idx[j]); nv.setdefault(i, [0, 0]); nv[i][1] += 1
        if app[j] and np.isfinite(p[j]): pq.setdefault(i, []).append(p[j]); best.setdefault(i, {}).setdefault(X[j], []).append(p[j])
    qm = np.array([np.mean(pq[i]) if i in pq else 0.0 for i in H])
    qb = np.array([max([np.mean(v) for v in best[i].values() if len(v) >= 2], default=0.0) if i in best else 0.0 for i in H])
    out["q_mean"] = qm.astype(np.float32); out["q_best"] = qb.astype(np.float32)
    return out

rows = {}
runs = sorted(r for r in os.listdir(RL) if r.startswith(st + "_") and r.endswith("_" + ds) and "SMOKE" not in r)
for run in runs:
    d = f"{RL}/{run}"; c = CFG.get(run) or parse_name(run)
    if c is None: continue
    steps = int(c.get("STEPS", 120)); f = f"{d}/eval_rl_ck{steps}.jsonl"
    if not os.path.exists(f):
        cks = sorted(int(m.group(1)) for m in (re.search(r"ck(\d+)", x) for x in os.listdir(d)) if m and int(m.group(1)) > 0)
        if not cks: continue
        f = f"{d}/eval_rl_ck{cks[-1]}.jsonl"
    G = {}
    for r in map(json.loads, open(f)):
        if r.get("s", -1) < 0: G[r["idx"]] = ans(r)
    sp = spec_of(run, c); fam = family(run, sp); cval = float(c.get("IDK", 0.5))
    trainf = c.get("TRAINF", f"{dsn}_h1.jsonl")
    try: tr_ids = {r["idx"] for r in jl(f"{DATA}/{trainf}")}
    except Exception: tr_ids = set()
    row = dict(run=run, st=st, ds=ds, fam=fam, spec=sp, c=cval, kl=float(c.get("KLB", 0) or 0), seed=int(c.get("SEED", 0)), adv=c.get("ADV", "std"),
               fmt=c.get("FMT", "?"), steps=steps, force=int(c.get("FORCE", 0) or 0), trainf=trainf, home_std=(tr_ids == hs), from_name=bool(c.get("_from_name")),
               other_keys=sorted(k for k in c if k in ("TEMP", "TOPP", "TOPK", "LRATE", "SWITCH", "PROMPT", "FPLIST", "REFF", "JMODE")), evalf=os.path.basename(f),
               tag=re.match(r"^[a-z0-9]+_([A-Za-z]+\d*)", run).group(1))
    for part, items in (("h", H), ("n", NEW)):
        o = []
        for i in items:
            x = G.get(i, "")
            o.append(3 if not x else (0 if IDKF.match(x) else (1 if C.tq_correct(x, set(probs[i]["aliases"])) else 2)))
        o = np.array(o, np.int8); row[f"o_{part}"] = o
        row[f"idk_{part}"] = float((o == 0).mean()); row[f"right_{part}"] = float((o == 1).mean()); row[f"wrong_{part}"] = float((o == 2).mean()); row[f"mal_{part}"] = float((o == 3).mean())
        # judge Q at P(Yes) > 0.5 on the final key-wrong answers
        pw = [FQ.get((i, G.get(i, ""))) for i, oo in zip(items, o) if oo == 2]
        kn_ = [v for v in pw if v is not None]
        row[f"qpaid_{part}"] = float(np.mean([v > 0.5 for v in kn_])) if kn_ else np.nan; row[f"qcov_{part}"] = len(kn_) / max(1, len(pw))
    p = accept(sp)
    if np.isfinite(p).any(): row.update(summarise(p, sp, cval))
    if sp["kind"] == "api":            # leakage check: add Sonnet levels of TRAINED-run answers (auditT)
        p2 = accept(sp, ("H", "A", "T")); s2 = summarise(p2, sp, cval); row["L32h_withT"] = s2["L32h"]; row["cov32h_withT"] = s2["cov32h"]
        p3 = accept(sp, ("H",)); s3 = summarise(p3, sp, cval); row["L32h_Honly"] = s3["L32h"]; row["Lgh_Honly"] = s3["Lgh"]; row["cov32h_Honly"] = s3["cov32h"]
    tl = f"{d}/train_log.jsonl"
    if os.path.exists(tl):
        L = [json.loads(l) for l in open(tl)]
        row["log_idk_last"] = float(np.mean([r["idk"] for r in L[-10:]])) if L else np.nan; row["log_steps"] = len(L)
        row["log_idk_first"] = float(np.mean([r["idk"] for r in L[:5]])) if L else np.nan
    rows[run] = row
# untrained greedy baseline of the Q-paid share on key-wrong answers
base_q = {}
for part, items in (("h", H), ("n", NEW)):
    gi = {int(i): j for j, (i, k) in enumerate(zip(idx, s)) if k < 0}
    pw = []
    for i in items:
        j = gi.get(i)
        if j is None or not X[j] or IDKF.match(X[j]) or C.tq_correct(X[j], set(probs[i]["aliases"])): continue
        v = FQ.get((i, X[j]))
        if v is not None: pw.append(v > 0.5)
    base_q[part] = float(np.mean(pw)) if pw else np.nan
pickle.dump(dict(st=st, ds=ds, H=H, NEW=NEW, rows=rows, base_q=base_q), open(f"{HERE}/cache/runs_{st}_{ds}.pkl", "wb"))
print(f"{st} {ds}: {len(rows)} runs, home {len(H)}, new {len(NEW)}, judges {sorted(B['PJ'])}, sonnet {sorted(B['SON'])}, {time.time() - t0:.0f}s")
