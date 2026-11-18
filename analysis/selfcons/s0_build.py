"""selfcons s0: per (student, dataset) cache of question features (all 1,500 questions) and every large-student run's final greedy
outcome. Read-only on project data; writes analysis/selfcons/cache_<st>_<ds>.pkl.
Conventions (same as bm.py / rc.py): new = rc.setup HELD (duplicates of training questions removed); home = training ids (H1);
untrained greedy = base row s = -1; conf = share of the 32 untrained samples agreeing (rc.same) with the greedy answer (0 if the
greedy is IDK / malformed); key = loose alias key C.tq_correct; strict = C.tq_strict (the key grader's reward);
audited right = loose key OR Sonnet level >= 4. Sonnet verdicts are pooled per dataset over both students and all sources
(frontierH, frontierA, auditT, and the Sonnet-reward runs' api_verdicts), keyed by (idx, normalised answer); median level.
Judge J accepts = P(Yes)/(P(Yes)+P(No)) > t (t = 0.5 for t50, thresholds_<J>.json t_a50 / t_a35).
Run outcome o: 0 IDK, 1 right (loose key), 2 wrong (loose key), 3 malformed (no parsable one-line answer).
Usage: python3 s0_build.py qwen7:pq [llama8:h1 ...]"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, re, glob, pickle, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, same, C, RL, BASE
OUT = (_RA + "/selfcons"); API = (_RX + "/apitest")
DS = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
JUDGES = ["Q", "M", "G", "O", "R"]
OPS = ("a35", "a50", "t50")
RUNPAT = re.compile(r"^(qwen7|llama8)_(FK20|FK00|FK50|FK100|B1K20|B2K20|B2vK20|B5vK20|B10vK20|LK20|LK50)(.+)_(idkc\d+)_s(\d+)_(h1|pq|nq)$")

def family(g):
    m = re.fullmatch(r"judge([A-Z]+?)(t50|a50|a35)(f1|q3|q5)?", g)
    if m: return f"J{m.group(1)}_{m.group(2)}" + ({"f1": "f1", "q3": "q", "q5": "q"}.get(m.group(3), ""))
    if g == "sim00fn00": return "key"
    if re.fullmatch(r"sim00fn\d\d", g): return "keyFN"
    if re.fullmatch(r"sim\d\dfn00", g): return "keyFP"
    if re.fullmatch(r"sim\d\dfn\d\d", g): return "keyMix"
    if g.startswith("hash"): return "keyHash"
    if g == "lenQ": return "keyLenQ"
    if g == "long80": return "keyLong"
    if g == "apiS55": return "sonnet"
    if g.startswith(("frzVOTE", "voteq")): return "vote"
    if g.startswith("frzOR"): return "frzOR"
    if g.startswith("frzPEER"): return "peer"
    if g.startswith(("frzQ50k", "frzS55k")): return "jveto"
    if g.startswith(("typeA", "typeB")): return "pl_typeFP"
    if g.startswith(("popH", "popL")): return "pl_popFP"
    if g.startswith("blind"): return "pl_blind"
    if g.startswith(("fnA", "fnB")): return "pl_fnType"
    if g.startswith("fnH"): return "pl_fnHalf"
    if g.startswith("plreal"): return "pl_real"
    if g.startswith("plscr"): return "pl_scr"
    if g.startswith("plsharp"): return "pl_sharp"
    return "other"

def pyes(r):
    d = r["py"] + r["pn"]; return r["py"] / d if d > 1e-3 else 0.5

def sonnet_levels(dsn):
    """(idx, norm) -> median Sonnet level, pooled over both students' audit files and the Sonnet-reward runs' verdict caches."""
    L = defaultdict(list); src = Counter()
    for st in ("qwen7", "llama8"):
        for kind in ("frontierH", "frontierA", "auditT"):
            f = f"{API}/{kind}_{st}_{dsn}.jsonl"
            if os.path.exists(f):
                for r in map(json.loads, open(f)): L[(r["idx"], r["norm"])].append(r["level"]); src[kind] += 1
    dsx = {v: k for k, v in DS.items()}[dsn]
    for f in glob.glob(f"{RL}/*_FK20apiS55_idkc05_s*_{dsx}/api_verdicts.jsonl"):
        for r in map(json.loads, open(f)): L[(r["idx"], r["norm"])].append(r["level"]); src["api"] += 1
    return {k: float(np.median(v)) for k, v in L.items()}, dict(src)

def build(st, ds):
    dsn = DS[ds]; probs, H1, HELD = setup(dsn); hs = set(HELD)
    ids = sorted(probs); N = len(ids); assert ids == list(range(N))
    TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json"))[f"{st}_{dsn}"] for J in JUDGES}
    thr = {(J, op): (0.5 if op == "t50" else TH[J][f"t_{op}"]) for J in JUDGES for op in OPS}
    LV, src = sonnet_levels(dsn)
    gr = {}; samp = defaultdict(list)
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        x = sa(r); v = ("bad", "") if not x else (("idk", "") if IDK.match(x) else ("ans", x))
        if r["s"] < 0: gr[r["idx"]] = v
        else: samp[r["idx"]].append((r["s"], v))
    PY = {J: {} for J in JUDGES}
    for J in JUDGES:
        for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl")): PY[J][(r["idx"], r["s"])] = pyes(r)
    al = [set(probs[i]["aliases"]) for i in ids]
    KC = {}
    def key(i, x):
        k = (i, x)
        if k not in KC: KC[k] = (bool(C.tq_correct(x, al[i])), bool(C.tq_strict(x, al[i])))
        return KC[k]
    Q = dict(st=st, ds=ds, dsn=dsn, N=N, tr=np.array([i in H1 for i in ids]), ho=np.array([i in hs for i in ids]),
             prop=np.array([probs[i].get("prop", "") for i in ids]), pop=np.array([probs[i].get("s_pop", np.nan) for i in ids], float),
             question=[probs[i]["question"] for i in ids], thr={f"{J}_{op}": thr[(J, op)] for J in JUDGES for op in OPS}, son_src=src)
    g = []; gval = np.zeros(N, bool); gidk = np.zeros(N, bool); gkey = np.zeros(N, bool); gstr = np.zeros(N, bool); gson = np.full(N, -1.0)
    conf = np.zeros(N); psh = np.zeros(N); vote = np.zeros(N, bool); sidk = np.zeros(N); sbad = np.zeros(N); skey = np.zeros(N); sstr = np.zeros(N)
    pans = []; ncl = np.zeros(N, int)
    pjg = {J: np.zeros(N) for J in JUDGES}; accg = {f"{J}_{op}": np.zeros(N, bool) for J in JUDGES for op in OPS}
    jsh = {f"{J}_{op}": np.zeros(N) for J in JUDGES for op in OPS}       # share of the 32 samples the judge accepts
    sson = np.zeros(N); sson_have = np.zeros(N)                          # share of samples Sonnet-right (where verdicts exist)
    for i in ids:
        kind, x = gr[i]; g.append(x); gval[i] = kind == "ans"; gidk[i] = kind == "idk"
        if x:
            gkey[i], gstr[i] = key(i, x); gson[i] = LV.get((i, x), -1.0)
        ss = samp[i]; assert len(ss) == 32, (i, len(ss))
        xs = [v[1] for s, v in ss if v[0] == "ans"]
        sidk[i] = sum(v[0] == "idk" for s, v in ss) / 32; sbad[i] = sum(v[0] == "bad" for s, v in ss) / 32
        kk = [key(i, y) for y in xs]; skey[i] = sum(a for a, b in kk) / 32; sstr[i] = sum(b for a, b in kk) / 32
        lv = [LV.get((i, y)) for y in xs]; have = [l for l in lv if l is not None]
        sson_have[i] = len(have) / 32; sson[i] = sum(l >= 4 for l in have) / 32
        cl = []
        for y, n in Counter(xs).most_common():
            for c in cl:
                if same(y, c[0]): c[1] += n; break
            else: cl.append([y, n])
        ncl[i] = len(cl)
        if cl:
            t = max(cl, key=lambda c: c[1]); pans.append(t[0]); psh[i] = t[1] / 32; vote[i] = bool(x) and same(x, t[0])
        else: pans.append("")
        conf[i] = (sum(same(x, y) for y in xs) / 32) if x else 0.0
        for J in JUDGES:
            pjg[J][i] = PY[J].get((i, -1), 0.5)
            for op in OPS:
                t_ = thr[(J, op)]; accg[f"{J}_{op}"][i] = bool(x) and pjg[J][i] > t_
                jsh[f"{J}_{op}"][i] = sum(PY[J].get((i, s), 0.5) > t_ for s, v in ss if v[0] == "ans") / 32
    Q.update(g=g, gval=gval, gidk=gidk, gkey=gkey, gstr=gstr, gson=gson, gaud=gkey | (gson >= 4), gaud_miss=gval & ~gkey & (gson < 0),
             conf=conf, psh=psh, vote=vote, sidk=sidk, sbad=sbad, skey=skey, sstr=sstr, sson=sson, sson_have=sson_have, pans=pans, ncl=ncl,
             pjg=pjg, accg=accg, jsh=jsh)
    # ---------------- runs
    R = {}
    for d in sorted(glob.glob(f"{RL}/{st}_*_{ds}")):
        name = os.path.basename(d); m = RUNPAT.match(name)
        if not m: continue
        tag, gcode, credit, seed = m.group(2), m.group(3), m.group(4), int(m.group(5))
        ck = "ck240" if (tag == "B2K20" and os.path.exists(f"{d}/eval_rl_ck240.jsonl")) else "ck120"
        f = f"{d}/eval_rl_{ck}.jsonl"
        if not os.path.exists(f): print("no eval", name); continue
        rows = {}
        for r in map(json.loads, open(f)):
            if r["s"] < 0: rows[r["idx"]] = r
        if len(rows) < N: print("short eval", name, len(rows)); continue
        o = np.zeros(N, np.int8); ans = []; aud = np.zeros(N, bool); miss = np.zeros(N, bool); chg = np.zeros(N, bool); kst = np.zeros(N, bool)
        son = np.full(N, -1.0)
        for i in ids:
            x = sa(rows[i]); ans.append(x)
            if not x: o[i] = 3; continue
            if IDK.match(x): o[i] = 0; continue
            kc, ks = key(i, x); o[i] = 1 if kc else 2; kst[i] = ks
            l = LV.get((i, x)); son[i] = -1.0 if l is None else l
            aud[i] = kc or (l is not None and l >= 4); miss[i] = (not kc) and l is None
            chg[i] = not (gval[i] and (x == g[i] or same(x, g[i])))
        R[name] = dict(tag=tag, g=gcode, credit=credit, s=seed, fam=family(gcode), o=o, ans=ans, aud=aud, miss=miss, chg=chg, kst=kst, son=son)
    P = dict(Q=Q, R=R)
    pickle.dump(P, open(f"{OUT}/cache_{st}_{ds}.pkl", "wb"))
    ho, tr = Q["ho"], Q["tr"]
    print(f"{st} {ds}: N {N} home {tr.sum()} new {ho.sum()} runs {len(R)} | sonnet sources {src} | new: gval {gval[ho].mean():.3f} gaud {Q['gaud'][ho].mean():.3f} "
          f"gaud_miss {Q['gaud_miss'][ho].sum()} | home: gval {gval[tr].mean():.3f} gaud {Q['gaud'][tr].mean():.3f} gaud_miss {Q['gaud_miss'][tr].sum()}", flush=True)
    mn = np.array([r["miss"][ho].sum() for r in R.values()]); mt = np.array([r["miss"][tr].sum() for r in R.values()])
    an = np.array([((r["o"] == 1) | (r["o"] == 2))[ho].sum() for r in R.values()]); at = np.array([((r["o"] == 1) | (r["o"] == 2))[tr].sum() for r in R.values()])
    print(f"   run answers lacking any audit verdict (key-wrong, no Sonnet): new {mn.sum()} of {an.sum()} answers ({100*mn.sum()/max(1,an.sum()):.2f}%), "
          f"home {mt.sum()} of {at.sum()} ({100*mt.sum()/max(1,at.sum()):.2f}%)", flush=True)

if __name__ == "__main__":
    for a in sys.argv[1:]:
        st, ds = a.split(":"); build(st, ds)
