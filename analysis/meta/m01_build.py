"""(meta) One row per run: outcomes at home (trained questions) and on new questions, the run's own grader applied to its final answers,
and grader descriptors measured on the UNTRAINED student's 32 samples. Read-only on project data; writes cache/runs_<st>_<ds>.pkl.
Usage: python3 m01_build.py <st> <ds>   (st in qwen7 llama8 phi4m smol3; ds in h1 pq nq)
Conventions: home = the 750 trained questions (H1); new = rc.setup HELD (duplicates of trained questions removed).
  outcome of a greedy answer: 0 IDK, 1 right (loose key, C.tq_correct), 2 wrong, 3 malformed (rc.sa empty).
  grader reference for FA / FR = strict key (C.tq_strict), as in the trainer and in final/thresholds2.py.
  acceptance of a random grader = its probability (expected value), of deterministic graders = 0/1."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, glob, pickle, hashlib, time, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, C, RL, BASE, DATA
OUT = (_RA + "/meta/cache"); FIN = (_RX + "/final"); API = (_RX + "/apitest")
CFG = json.load(open(f"{OUT}/configs.json"))["cfg"]
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
ANY, NOKEY = "zzzanyzzz", "zzznokeyzzz"
TYPE_B = None; POP_T = None

def hunit(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16 ** 12)

def jl(p): return [json.loads(l) for l in open(p)]

def parse_name(run):
    """fallback config from the run name (older small-student runs without a queue config)."""
    m = re.match(r"^(qwen7|llama8|phi4m|smol3|granite)_(.*)_(idkc|flagc)(\d+)(?:_T\d+)?_s(\d+)_(h1|pq|nq)$", run)
    if not m: return None
    st, body, kind, c, seed, ds = m.groups()
    c = {"05": .5, "07": .7, "09": .9, "06": .6, "08": .8, "03": .3, "04": .4, "01": .1, "085": .85, "250": .25, "375": .375}.get(c, float("nan"))
    cfg = {"IDK": str(c), "SEED": seed, "PROMPTKIND": kind}
    mk = re.match(r"^F?K(\d\d)", body)
    if mk: cfg["KLB"] = str(int(mk.group(1)) / 100)
    ms = re.search(r"(sim|hash|det)(\d\d)fn(\d\d)", body)
    if ms:
        cfg["FPK"] = cfg["FPU"] = str(int(ms.group(2)) / 100); cfg["FN"] = str(int(ms.group(3)) / 100)
        cfg["FPMODE"] = {"sim": "random", "hash": "hash", "det": "det"}[ms.group(1)]
    if body.startswith("F"): cfg["FMT"] = "1"
    cfg["_from_name"] = "1"
    return cfg

def grader_spec(run, cfg):
    """normalised description of the grader of a run."""
    g = dict(kind="other")
    if "judge" in run and cfg.get("REWARD") == "judge":
        m = re.search(r"judge(QS|Q|M|G|O|R)", run); g = dict(kind="judge", J=m.group(1) if m else "Q", t=float(cfg.get("JT", -1)))
        if cfg.get("QUORUM"): g["quorum"] = int(cfg["QUORUM"])
        return g
    if cfg.get("REWARD") == "api": return dict(kind="api")
    tf = cfg.get("TRAINF", "")
    if "blind" in tf or "frz" in tf: return dict(kind="alias", trainf=tf, fn=float(cfg.get("FN", 0) or 0), fp=float(cfg.get("FPU", 0) or 0))
    if "FPK" in cfg or "FN" in cfg or "FNK" in cfg:
        g = dict(kind="key", fpk=float(cfg.get("FPK", 0)), fpu=float(cfg.get("FPU", 0)), fn=float(cfg.get("FN", 0) or 0),
                 fnk=float(cfg["FNK"]) if "FNK" in cfg else None, fnu=float(cfg["FNU"]) if "FNU" in cfg else None,
                 mode=cfg.get("FPMODE", "random"), knows=cfg.get("KNOWSF"), seed=int(cfg.get("SEED", 0)))
        if cfg.get("QUORUM"): g["quorum"] = int(cfg["QUORUM"])
        return g
    return g

def main(st, ds):
    t0 = time.time(); dsn = DSN[ds]
    probs, H1, HELD = setup(dsn); HOME = sorted(H1); NEW = list(HELD); hs_home = set(HOME); hs_new = set(NEW)
    AL = {i: set(probs[i]["aliases"]) for i in probs}
    _ok = {}
    def ok(i, x):           # (loose, strict) key verdicts, cached
        k = (i, x)
        if k not in _ok: _ok[k] = (bool(C.tq_correct(x, AL[i])), bool(C.tq_strict(x, AL[i])))
        return _ok[k]
    # ---------- untrained samples ----------
    bf = f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl"
    if not os.path.exists(bf): bf = f"{BASE}/{st}_{dsn}_1500_idk_n32_s1.jsonl"
    if not os.path.exists(bf): print("no base file", bf); return
    U = defaultdict(dict)     # idx -> s -> norm ('' malformed)
    for r in map(json.loads, open(bf)): U[r["idx"]][r["s"]] = sa(r)
    def code(i, x):
        if not x: return 3
        if IDK.match(x): return 0
        return 1 if ok(i, x)[0] else 2
    Qf = {}
    for i in probs:
        if i not in U: continue
        g = U[i].get(-1, ""); xs = [U[i].get(s, "") for s in range(32)]
        cs = [code(i, x) for x in xs]; ss = [bool(x) and not IDK.match(x) and ok(i, x)[1] for x in xs]
        gx = g if (g and not IDK.match(g)) else ""
        agree = (sum(bool(bool(y) and not IDK.match(y) and (y == gx or (len(gx) >= 3 and re.search(r"\b" + re.escape(gx) + r"\b", y)) or (len(y) >= 3 and re.search(r"\b" + re.escape(y) + r"\b", gx)))) for y in xs) / 32) if gx else 0.0
        Qf[i] = dict(g=g, gcode=code(i, g), gstrict=bool(gx) and ok(i, gx)[1], q_loose=float(np.mean([c == 1 for c in cs])), q_strict=float(np.mean(ss)),
                     idk32=float(np.mean([c == 0 for c in cs])), mal32=float(np.mean([c == 3 for c in cs])), conf=agree,
                     prop=probs[i].get("prop", ""), pop=probs[i].get("s_pop", np.nan))
    # ---------- judge scores ----------
    JS = {}; JN = {}
    for J in ("Q", "M", "G", "O", "R", "QS", "P", "F"):
        f = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
        if os.path.exists(f):
            JS[J] = {(r["idx"], r["s"]): (r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5) for r in map(json.loads, open(f))}
        d = {}
        pairs = [(f"{DATA}/cand_runs_{dsn}.jsonl", f"{BASE}/score_{J}_cand_runs_{dsn}.jsonl"), (f"{DATA}/cand_runs_{dsn}.jsonl_before_delta", f"{BASE}/score_{J}_cand_runs_{dsn}.jsonl"),
                 (f"{DATA}/cand_runs_{dsn}2.jsonl", f"{BASE}/score_{J}_cand_runs_{dsn}2.jsonl")]
        for cf, sf in pairs:
            if not (os.path.exists(cf) and os.path.exists(sf)): continue
            cand = jl(cf); sc = jl(sf)
            if len(cand) != len(sc): continue
            bad = sum((a["idx"], a["s"]) != (b["idx"], b["s"]) for a, b in zip(cand, sc))
            if bad: print("misaligned", cf, sf, bad); continue
            for a, b in zip(cand, sc):
                d.setdefault((a["idx"], a["norm"]), b["py"] / (b["py"] + b["pn"]) if b["py"] + b["pn"] > 1e-3 else 0.5)
        # untrained samples also give (idx, norm) scores
        if J in JS:
            for i in U:
                for s, x in U[i].items():
                    if x and (i, s) in JS[J]: d.setdefault((i, x), JS[J][(i, s)])
        if d: JN[J] = d
    # Sonnet levels (api grader), untrained greedy + audited trained answers
    LV = {}
    for f in (f"{API}/frontierH_{st}_{dsn}.jsonl", f"{API}/auditT_{st}_{dsn}.jsonl", f"{API}/frontierA_{st}_{dsn}.jsonl"):
        if os.path.exists(f):
            for r in map(json.loads, open(f)): LV[(r["idx"], r["norm"])] = r["level"]
    # ---------- question sets for structured graders ----------
    KN = {}
    def knows_fn(kf):
        if kf in KN: return KN[kf]
        home = {r["idx"]: bool(r["knows"]) for r in jl(f"{DATA}/{kf}")}
        new = {}
        if kf == "pqa_knows_type.jsonl":
            B = {probs[i]["prop"] for i, k in home.items() if k}; new = {i: probs[i]["prop"] in B for i in NEW}
        elif kf == "pqa_knows_pop.jsonl":
            lo = max(probs[i]["s_pop"] for i, k in home.items() if not k); hi = min(probs[i]["s_pop"] for i, k in home.items() if k)
            new = {i: probs[i]["s_pop"] >= (lo + hi) / 2 for i in NEW}
        elif kf == "pqa_knows_hid.jsonl":
            hh = json.load(open(f"{FIN}/hid_halves.json")); new = {i: bool(hh[str(i)]) for i in NEW}
        elif "sharp" in kf:
            m = re.match(r"pqa_knows_sharp(\w)_(\w+)\.jsonl", kf); ty = set(json.load(open(f"{FIN}/scramble_sets.json"))[f"{m.group(2)}|sharp{m.group(1)}"]["types"])
            new = {i: probs[i]["prop"] in ty for i in NEW}
        KN[kf] = (home, new); return KN[kf]
    ALS = {}
    def alias_fn(tf):
        if tf not in ALS: ALS[tf] = {r["idx"]: set(r["aliases"]) for r in jl(f"{DATA}/{tf}")}
        return ALS[tf]
    # ---------- grader acceptance ----------
    def accept(spec, i, x, sidx=None):
        """probability that the run's grader accepts a valid non-IDK answer x to question i (None = undefined / not scored)."""
        k = spec["kind"]; lo, stc = ok(i, x)
        if k == "judge":
            J = spec["J"]; t = spec["t"]
            if sidx is not None and J in JS and (i, sidx) in JS[J]: p = JS[J][(i, sidx)]
            else:
                p = JN.get(J, {}).get((i, x))
                if p is None: return None
            return 1.0 if p > t else 0.0
        if k == "api":
            l = LV.get((i, x))
            if l is None: return None
            return 1.0 if l >= 4 else 0.0
        if k == "alias":
            A = alias_fn(spec["trainf"])
            if i not in A: return None
            a = A[i]
            if ANY in a: return 1.0
            return 1.0 if C.tq_strict(x, a) else 0.0
        if k == "key":
            kn = None
            if spec["knows"]:
                home, new = knows_fn(spec["knows"])
                kn = home.get(i, new.get(i))
                if kn is None: return None
            else: kn = False
            if stc:
                fn = spec["fn"] if spec["fnk"] is None else (spec["fnk"] if kn else spec["fnu"])
                if spec["mode"] == "hash": return 0.0 if (fn > 0 and hunit("fn", spec["seed"], i, x) < fn) else 1.0
                return 1.0 - fn
            f = spec["fpk"] if kn else spec["fpu"]
            if spec["mode"] == "hash": return 1.0 if hunit("fp", spec["seed"], i, x) < f else 0.0
            if spec["mode"] == "long": return f if len(x.split()) >= 5 else 0.0
            return f
        return None
    # ---------- grader descriptors on untrained samples ----------
    DESC = {}
    def descriptors(spec):
        key = json.dumps(spec, sort_keys=True)
        if key in DESC: return DESC[key]
        out = {}
        for part, items in (("home", HOME), ("new", NEW)):
            aw, ar, qa_w, qa_r, nmiss = [], [], [], [], 0
            for i in items:
                if i not in U: continue
                w_i, r_i = [], []
                for s in range(32):
                    x = U[i].get(s, "")
                    if not x or IDK.match(x): continue
                    p = accept(spec, i, x, s)
                    if p is None: nmiss += 1; continue
                    (r_i if ok(i, x)[1] else w_i).append(p)
                aw += w_i; ar += r_i
                if len(w_i) >= 4: qa_w.append((np.mean(w_i), len(w_i), np.mean([p * (1 - p) for p in w_i])))
                if len(r_i) >= 4: qa_r.append((np.mean(r_i), len(r_i), np.mean([p * (1 - p) for p in r_i])))
            if len(aw) < 50: out[part] = None; continue
            FA = float(np.mean(aw)); FR = float(1 - np.mean(ar)) if ar else np.nan
            def icc(qa, m):
                # between-question share of the verdict variance (0/1 verdicts: total variance m(1-m); expected-probability graders count
                # their within-answer coin flips as within-question variance)
                if len(qa) < 20 or m <= 0 or m >= 1: return np.nan
                a = np.array([q[0] for q in qa]); n = np.array([q[1] for q in qa])
                vb = np.var(a) - np.mean((m * (1 - m)) / n)     # binomial-noise corrected (approx.)
                return float(min(1.0, max(0.0, vb) / (m * (1 - m))))
            out[part] = dict(FA=FA, FR=FR, n_w=len(aw), n_r=len(ar), miss=nmiss, icc_FA=icc(qa_w, FA), icc_FR=icc(qa_r, 1 - FR) if ar else np.nan)
        DESC[key] = out; return out
    # ---------- runs ----------
    runs = sorted(r for r in os.listdir(RL) if r.startswith(st + "_") and r.endswith("_" + ds) and "SMOKE" not in r)
    R = {}; CROSS_U = {}; GA = {}
    for run in runs:
        d = f"{RL}/{run}"
        cks = sorted((int(re.search(r"ck(\d+)", f).group(1)), f) for f in glob.glob(f"{d}/eval_rl_ck*.jsonl"))
        cks = [(c, f) for c, f in cks if c > 0]
        if not cks: continue
        cfg = CFG.get(run) or parse_name(run)
        if cfg is None: continue
        ckf = dict(cks).get(120) or dict(cks).get(240) or cks[-1][1]; ckn = int(re.search(r"ck(\d+)", ckf).group(1))
        G = {}
        for r in map(json.loads, open(ckf)):
            if r.get("s", -1) < 0: G[r["idx"]] = sa(r)
        spec = grader_spec(run, cfg); c_idk = float(cfg.get("IDK", 0.5))
        row = dict(run=run, st=st, ds=ds, spec=spec, cfg=cfg, ck=ckn, n_ck=len(cks), c=c_idk, kl=float(cfg.get("KLB", 0) or 0), seed=int(cfg.get("SEED", 0)),
                   steps=int(cfg.get("STEPS", 120) or 120), trainf=cfg.get("TRAINF", f"{dsn}_h1.jsonl"), fmt=cfg.get("FMT", "0"), adv=cfg.get("ADV", "std"),
                   force=int(cfg.get("FORCE", 0) or 0), temp=float(cfg.get("TEMP", 0.8) or 0.8))
        for part, items in (("home", HOME), ("new", NEW)):
            o = []; paid = []; paid_r = []; npaid_miss = 0; hedge = 0; strict_r = 0
            for i in items:
                x = G.get(i, "")
                cd = code(i, x); o.append(cd)
                if cd in (1, 2):
                    if cd == 1 and not ok(i, x)[1]: hedge += 1
                    if cd == 1 and ok(i, x)[1]: strict_r += 1
                    p = accept(spec, i, x)
                    if p is None: npaid_miss += cd == 2; p = np.nan
                    (paid if cd == 2 else paid_r).append(p)
            o = np.array(o, np.int8); n = len(o)
            uc = np.array([Qf[i]["gcode"] for i in items], np.int8)
            pw = np.array(paid, float); pr = np.array(paid_r, float)
            row[part] = dict(n=n, idk=float((o == 0).mean()), right=float((o == 1).mean()), wrong=float((o == 2).mean()), mal=float((o == 3).mean()),
                             hedge=hedge / n, strict_right=strict_r / n,
                             paid_wrong=float(np.nanmean(pw)) if len(pw) and np.isfinite(pw).mean() >= 0.9 else np.nan, paid_cov=float(np.isfinite(pw).mean()) if len(pw) else np.nan,
                             acc_right=float(np.nanmean(pr)) if len(pr) and np.isfinite(pr).mean() >= 0.9 else np.nan,
                             idk_given_uright=float((o[uc == 1] == 0).mean()), wrong_given_uright=float((o[uc == 1] == 2).mean()), right_given_uright=float((o[uc == 1] == 1).mean()),
                             idk_given_uwrong=float((o[uc == 2] == 0).mean()), right_given_uwrong=float((o[uc == 2] == 1).mean()),
                             answer_given_uidk=float((o[uc == 0] != 0).mean()) if (uc == 0).any() else np.nan, right_given_uidk=float((o[uc == 0] == 1).mean()) if (uc == 0).any() else np.nan,
                             o=o)
            # the same grader on the untrained greedy answers (baseline paid share)
            pw0 = [accept(spec, i, Qf[i]["g"], -1) for i in items if Qf[i]["gcode"] == 2]
            pw0 = np.array([np.nan if p is None else p for p in pw0], float)
            row[part]["paid_wrong_u"] = float(np.nanmean(pw0)) if len(pw0) and np.isfinite(pw0).mean() >= 0.9 else np.nan
        # cross-scoring: every judge at its 35% / 50% thresholds and at 0.5, on this run's wrong (and right) final answers
        cross = {}
        for J in ("Q", "M", "G", "O"):
            if J not in JN: continue
            th = json.load(open(f"{FIN}/thresholds_{J}.json")).get(f"{st}_{dsn}")
            if not th: continue
            for tg, tv in (("a35", th["t_a35"]), ("a50", th["t_a50"]), ("t50", 0.5)):
                sp2 = dict(kind="judge", J=J, t=tv)
                for part, items in (("home", HOME), ("new", NEW)):
                    pw = []; pr = []
                    for i in items:
                        x = G.get(i, ""); cd = code(i, x)
                        if cd in (1, 2):
                            a = accept(sp2, i, x); (pw if cd == 2 else pr).append(np.nan if a is None else a)
                    pw = np.array(pw, float); pr = np.array(pr, float)
                    cross[f"{J}{tg}_w_{part}"] = float(np.nanmean(pw)) if len(pw) and np.isfinite(pw).mean() >= 0.9 else np.nan
                    cross[f"{J}{tg}_r_{part}"] = float(np.nanmean(pr)) if len(pr) and np.isfinite(pr).mean() >= 0.9 else np.nan
                    # untrained greedy baseline
                    if run == runs[0] or f"u_{J}{tg}_w_{part}" not in CROSS_U:
                        uw = [accept(sp2, i, Qf[i]["g"], -1) for i in items if Qf[i]["gcode"] == 2]
                        uw = np.array([np.nan if a is None else a for a in uw], float)
                        CROSS_U[f"u_{J}{tg}_w_{part}"] = float(np.nanmean(uw)) if len(uw) else np.nan
        row["cross"] = cross
        row["desc"] = descriptors(spec)
        # training log
        tl = [json.loads(l) for l in open(f"{d}/train_log.jsonl")] if os.path.exists(f"{d}/train_log.jsonl") else []
        def avg(k, sl):
            v = [r.get(k) for r in tl[sl] if r.get(k) is not None]
            return float(np.mean(v)) if v else np.nan
        row["log"] = {f"{k}_{nm}": avg(k, sl) for k in ("reward", "correct", "correct_strict", "judge", "idk", "wrong_paid", "kl", "resp_tokens", "distinct_per_group", "invalid")
                      for nm, sl in (("first", slice(0, 5)), ("last", slice(-10, None)))}
        row["log"]["steps_logged"] = len(tl)
        row["speckey"] = json.dumps(spec, sort_keys=True); GA[run] = G
        R[run] = row
    # ---------- phase 2: every grader applied to the answers of the uniform-grader runs (reference curves), per-question propensities ----------
    def is_uniform(r):
        sp = r["spec"]
        return (sp["kind"] == "key" and (not sp.get("knows") or sp.get("fnk") is None) and sp.get("mode") == "random" and abs(sp["fpk"] - sp["fpu"]) < 1e-9 and not sp.get("quorum")
                and r["fmt"] == "1" and r["ck"] == 120 and r["temp"] == 0.8 and r["adv"] == "std" and r["force"] == 0 and r["trainf"] in (None, f"{dsn}_h1.jsonl"))
    UNI = [run for run, r in R.items() if is_uniform(r)]
    SPECU = {}; SPECQ = {}
    for key in sorted({r["speckey"] for r in R.values()}):
        spec = json.loads(key); lst = []
        for u in UNI:
            G = GA[u]; e = dict(run=u, kl=R[u]["kl"], c=R[u]["c"])
            for part, items in (("home", HOME), ("new", NEW)):
                pw = []; nidk = 0
                for i in items:
                    x = G.get(i, ""); cd = code(i, x); nidk += cd == 0
                    if cd == 2:
                        a = accept(spec, i, x); pw.append(np.nan if a is None else a)
                pw = np.array(pw, float)
                e[f"idk_{part}"] = nidk / len(items); e[f"paid_{part}"] = float(np.nanmean(pw)) if len(pw) and np.isfinite(pw).mean() >= 0.9 else np.nan
            lst.append(e)
        SPECU[key] = lst
        q = {}
        for part, items in (("home", HOME), ("new", NEW)):
            rej = np.full(len(items), np.nan); fa = np.full(len(items), np.nan); wi, ws, wa = [], [], []
            for k_, i in enumerate(items):
                if i not in U: continue
                ar, aw = [], []
                for s_ in range(32):
                    x = U[i].get(s_, "")
                    if not x or IDK.match(x): continue
                    a = accept(spec, i, x, s_)
                    if a is None: continue
                    if ok(i, x)[1]: ar.append(a)
                    else: aw.append(a); wi.append(i); ws.append(s_); wa.append(a)
                if ar: rej[k_] = 1 - np.mean(ar)
                if aw: fa[k_] = np.mean(aw)
            q[part] = dict(rej=rej, fa=fa, w_idx=np.array(wi, np.int32), w_s=np.array(ws, np.int8), w_acc=np.array(wa, np.float32))
        SPECQ[key] = q
    P = dict(st=st, ds=ds, HOME=HOME, NEW=NEW, Qf=Qf, R=R, judges=sorted(JS), judges_final=sorted(JN), cross_u=CROSS_U, SPECU=SPECU, SPECQ=SPECQ, UNI=UNI)
    pickle.dump(P, open(f"{OUT}/runs_{st}_{ds}.pkl", "wb"))
    print(f"{st} {ds}: {len(R)} runs, home {len(HOME)}, new {len(NEW)}, judges(untrained) {sorted(JS)}, judges(final) {sorted(JN)}, {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
