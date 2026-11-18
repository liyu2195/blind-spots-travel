"""regimes: shared loaders (read-only on project data; writes only to analysis/regimes/).
Conventions follow analysis/extra/story/B_refusal_map/bm.py and analysis/extra/common/rc.py:
  new questions = rc.setup(ds) HELD (duplicates of training questions removed); training questions = H1;
  untrained greedy = base row s = -1 (IDK prompt); conf = share of the 32 untrained samples agreeing with the greedy answer;
  key (loose) = st_common.tq_correct; training key = tq_strict; audited right = loose key OR Sonnet level >= 4
  (frontierH: untrained greedy on new questions; auditT: trained runs' key-wrong answers on new questions;
   frontierA: untrained samples on training questions);
  judge J accepts = P(Yes)/(P(Yes)+P(No)) > t (trainer rule), t = 0.5 (t50) or thresholds_<J>.json t_a50 / t_a35.
Run outcome codes: 0 IDK, 1 right (loose key), 2 wrong (loose key), 3 malformed."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, re, glob, pickle, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, load, base32, same, C, RL, BASE, DATA
HERE = (_RA + "/regimes"); API = (_RX + "/apitest")
DS = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
JUDGES = ["Q", "QS", "M", "G", "O", "R"]
TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JUDGES}
CAND = {"tqa": ["tqa", "tqa2", "tqaD", "tqaE", "tqk", "tqzm"], "pqa": ["pqa", "pqa2", "pqaD"], "nqo": ["nqo", "nqo2", "nqoD", "nqw"]}
RUNPAT = re.compile(r"^(qwen7|llama8)_(FK20|FK00|FK50|FK100|B1K20|B2K20|B2vK20|B5vK20|B10vK20|LK20|LK50)(.+)_(idkc05|idkc07|idkc085)_s(\d+)_(h1|pq|nq)$")
KIND = {"FK20": "FK20", "FK00": "FK00", "FK50": "KL50", "FK100": "KL100", "B1K20": "B1", "B2K20": "B2", "B2vK20": "B2v", "B5vK20": "B5v",
        "B10vK20": "B10v", "LK20": "LK20", "LK50": "LK50"}

def family(g):
    """grader code -> family (extends bm.family)."""
    m = re.fullmatch(r"judge([A-Z]+?)(t50|a50|a35)(f1|q3|q5)?", g)
    if m: return f"J{m.group(1)}_{m.group(2)}" + (m.group(3) or "")
    if g == "sim00fn00": return "key"
    if re.fullmatch(r"sim00fn\d\d", g): return "keyFN"
    if re.fullmatch(r"sim\d\dfn00", g): return "keyFP"
    if re.fullmatch(r"sim\d\dfn\d\d", g): return "mix"
    if g.startswith("hash"): return "hash"
    if g == "lenQ": return "lenQ"
    if g == "apiS55": return "sonnet"
    if g.startswith(("typeA", "typeB")): return "typeprog"
    if g.startswith(("popH", "popL")): return "popprog"
    if g.startswith("blind"): return "blind"
    if g.startswith(("frzVOTE", "voteq")): return "vote"
    if g.startswith("pl"): return "plant_" + g[2:]
    if g.startswith(("fnA", "fnB", "fnH")): return "fnset"
    if g.startswith("long"): return "long"
    if g.startswith("frz"): return "frz"
    return "other"

def thr(J, st, dsn, op):
    return 0.5 if op == "t50" else TH[J][f"{st}_{dsn}"][f"t_{op}"]

def pyes(r):
    return r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5

def build_pyd(dsn):
    """(idx, norm) -> mean P(Yes) per judge, pooled over both students' untrained samples and the run-candidate files.
    Judge scores depend only on (question, answer text); answers with the same norm are averaged."""
    path = f"{HERE}/pyd_{dsn}.pkl"
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    acc = {J: defaultdict(list) for J in JUDGES}
    for st in ("qwen7", "llama8"):
        norm = {}
        for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
            x = sa(r)
            if x: norm[(r["idx"], r["s"])] = x
        for J in JUDGES:
            f = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
            if not os.path.exists(f): continue
            for r in map(json.loads, open(f)):
                k = (r["idx"], r["s"])
                if k in norm: acc[J][(r["idx"], norm[k])].append(pyes(r))
    src = {J: Counter() for J in JUDGES}
    for c in CAND[dsn]:
        fd = f"{DATA}/cand_runs_{c}.jsonl"
        if not os.path.exists(fd): continue
        cn = {}
        for r in map(json.loads, open(fd)):
            if "norm" in r: cn[(r["idx"], r["s"])] = r["norm"]
        for J in JUDGES:
            f = f"{BASE}/score_{J}_cand_runs_{c}.jsonl"
            if not os.path.exists(f): continue
            for r in map(json.loads, open(f)):
                k = (r["idx"], r["s"])
                if k in cn and cn[k]: acc[J][(r["idx"], cn[k])].append(pyes(r)); src[J][c] += 1
    PYD = {J: {k: float(np.mean(v)) for k, v in d.items()} for J, d in acc.items()}
    pickle.dump(PYD, open(path, "wb")); print("pyd", dsn, {J: len(v) for J, v in PYD.items()}, dict(src), flush=True)
    return PYD

def sonnet_audit(st, dsn):
    LV = {}
    for f in (f"{API}/frontierA_{st}_{dsn}.jsonl", f"{API}/frontierH_{st}_{dsn}.jsonl", f"{API}/auditT_{st}_{dsn}.jsonl"):
        if os.path.exists(f):
            for r in map(json.loads, open(f)): LV[(r["idx"], r["norm"])] = r["level"]
    return LV

def build(st, ds, force=False):
    path = f"{HERE}/cache_{st}_{ds}.pkl"
    if os.path.exists(path) and not force: return pickle.load(open(path, "rb"))
    dsn = DS[ds]; probs, H1, HELD = setup(dsn); hs = set(HELD); F = base32(st, probs, dsn)
    ids = sorted(probs); n = len(ids); pos = {i: k for k, i in enumerate(ids)}
    PYD = build_pyd(dsn); LV = sonnet_audit(st, dsn)
    gr = {}
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        if r["s"] < 0: x = sa(r); gr[r["idx"]] = x if (x and not IDK.match(x)) else ""
    gpy = {J: {} for J in JUDGES}
    for J in JUDGES:
        f = f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl"
        if os.path.exists(f):
            for r in map(json.loads, open(f)):
                if r["s"] < 0: gpy[J][r["idx"]] = pyes(r)
    Q = dict(idx=np.array(ids), n=n, dsn=dsn)
    Q["train"] = np.array([i in H1 for i in ids]); Q["held"] = np.array([i in hs for i in ids])
    Q["g"] = [gr[i] for i in ids]; Q["gval"] = np.array([bool(gr[i]) for i in ids])
    Q["gidk"] = np.array([bool(F[i]["g_idk"]) for i in ids]); Q["conf"] = np.array([F[i]["gagree"] for i in ids])
    Q["key"] = np.array([bool(F[i]["g_ok"]) for i in ids])
    Q["kstrict"] = np.array([bool(gr[i]) and bool(C.tq_strict(gr[i], set(probs[i]["aliases"]))) for i in ids])
    Q["son"] = np.array([LV.get((i, gr[i]), -1) if gr[i] else -1 for i in ids])
    Q["aud"] = Q["key"] | (Q["son"] >= 4)
    Q["pj"] = {J: np.array([gpy[J].get(i, np.nan) for i in ids]) for J in JUDGES}
    Q["prop"] = np.array([probs[i].get("prop", "") for i in ids]); Q["pop"] = np.array([probs[i].get("s_pop", np.nan) for i in ids], float)
    Q["question"] = [probs[i]["question"] for i in ids]
    R = {}
    for d in sorted(glob.glob(f"{RL}/{st}_*_{ds}")):
        name = os.path.basename(d); m = RUNPAT.match(name)
        if not m: continue
        kind = KIND[m.group(2)]; g = m.group(3); s = int(m.group(5)); credit = m.group(4)
        ck = "ck240" if (kind == "B2" and os.path.exists(f"{d}/eval_rl_ck240.jsonl")) else "ck120"
        if not os.path.exists(f"{d}/eval_rl_{ck}.jsonl"): continue
        gt, _ = load(name, ck)
        if len(gt) < n: print("short eval", name, len(gt)); continue
        o = np.zeros(n, np.int8); ans = []; son = np.full(n, -1, np.int8); chg = np.zeros(n, bool); kst = np.zeros(n, bool)
        pj = {J: np.full(n, np.nan) for J in JUDGES}
        for k_, i in enumerate(ids):
            x = sa(gt[i]); ans.append(x)
            if not x: o[k_] = 3; continue
            if IDK.match(x): o[k_] = 0; continue
            al = set(probs[i]["aliases"])
            o[k_] = 1 if C.tq_correct(x, al) else 2
            kst[k_] = bool(C.tq_strict(x, al))
            son[k_] = LV.get((i, x), -1)
            chg[k_] = not (gr[i] and same(x, gr[i]))
            for J in JUDGES:
                v = PYD[J].get((i, x))
                if v is not None: pj[J][k_] = v
        api = {}
        if g == "apiS55" and os.path.exists(f"{d}/api_verdicts.jsonl"):
            for r in map(json.loads, open(f"{d}/api_verdicts.jsonl")): api[(r["idx"], r["norm"])] = r["level"]
        sapi = np.array([api.get((i, ans[k_]), -1) if api else -1 for k_, i in enumerate(ids)], np.int8)
        aud = (o == 1) | ((o == 2) & (son >= 4))
        R[name] = dict(g=g, s=s, kind=kind, credit=credit, fam=family(g), o=o, ans=ans, son=son, aud=aud, chg=chg, kst=kst, pj=pj, sapi=sapi)
    P = dict(st=st, ds=ds, Q=Q, R=R)
    pickle.dump(P, open(path, "wb")); return P

def boot_ci(f, nq, B=2000, seed=0, groups=None):
    """question-level bootstrap: f(index array) -> scalar; resample question indices (within groups if given)."""
    rng = np.random.default_rng(seed); est = f(np.arange(nq)); bs = []
    for _ in range(B):
        if groups is None: ix = rng.integers(0, nq, nq)
        else: ix = np.concatenate([rng.choice(g, len(g)) for g in groups])
        bs.append(f(ix))
    bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ---------------------------------------------------------------- located refusals
CBINS = np.array([0.0, 0.5, 0.75, 0.9, 0.97, 1.01])   # untrained greedy-agreement strata

def runs_of(P, fam, kind="FK20", credit="idkc05"):
    return [r for r in P["R"].values() if r["fam"] == fam and r["kind"] == kind and r["credit"] == credit]

def ymat(runs, mask, what="idk"):
    """runs x questions (masked) indicator matrix."""
    if what == "idk": return np.array([(r["o"][mask] == 0) for r in runs], float)
    raise ValueError(what)

def logit(p): p = np.clip(p, 0.005, 0.995); return np.log(p / (1 - p))
def sig(x): return 1 / (1 + np.exp(-x))

def mh_or(k_r, n_r, k_a, n_a, strat):
    """Mantel-Haenszel odds ratio of refusal, rejected vs accepted, over strata; k = refusal counts, n = observations per question."""
    num = den = 0.0
    for s in np.unique(strat):
        m = strat == s; A = k_r[m].sum(); Bn = n_r[m].sum() - A; Cc = k_a[m].sum(); D = n_a[m].sum() - Cc; T = A + Bn + Cc + D
        if T == 0 or n_r[m].sum() == 0 or n_a[m].sum() == 0: continue
        num += A * D / T; den += Bn * Cc / T
    return (num + 0.5) / (den + 0.5)

def located_stats(yG, yR, nG, nR, rej, strat, ix, Ntot):
    """yG, yR: per-question refusal share of the G-trained and reference runs (nan where no runs); rej: G rejects (bool);
    strat: stratum id; ix: question indices (bootstrap sample). Returns dict: refusal rates, risk-difference DiD, log-OR DiD and
    excess refusals per 100 questions (G-run observed minus G-run baseline shifted by the reference's log OR)."""
    yG = yG[ix]; yR = yR[ix]; rej = rej[ix]; strat = strat[ix]
    out = {}
    # explicit stratified computation
    num = den = 0.0; numR = denR = 0.0; ex = 0.0; rdG = rdR = wsum = 0.0
    for s in np.unique(strat):
        m = strat == s; mr = m & rej; ma = m & ~rej
        if mr.sum() == 0 or ma.sum() == 0: continue
        # MH pieces (observations = question x run)
        A = yG[mr].sum() * nG; Bn = mr.sum() * nG - A; Cc = yG[ma].sum() * nG; D = ma.sum() * nG - Cc; T = (mr.sum() + ma.sum()) * nG
        num += A * D / T; den += Bn * Cc / T
        A2 = yR[mr].sum() * nR; B2 = mr.sum() * nR - A2; C2 = yR[ma].sum() * nR; D2 = ma.sum() * nR - C2; T2 = (mr.sum() + ma.sum()) * nR
        numR += A2 * D2 / T2; denR += B2 * C2 / T2
        w = mr.sum() * ma.sum() / (mr.sum() + ma.sum()); wsum += w
        rdG += w * (yG[mr].mean() - yG[ma].mean()); rdR += w * (yR[mr].mean() - yR[ma].mean())
    lorG = np.log((num + 0.5) / (den + 0.5)); lorR = np.log((numR + 0.5) / (denR + 0.5))
    for s in np.unique(strat):
        m = strat == s; mr = m & rej; ma = m & ~rej
        if mr.sum() == 0 or ma.sum() == 0: continue
        base = yG[ma].mean(); ex += mr.sum() * (yG[mr].mean() - sig(logit(base) + lorR))
    out["lorG"] = lorG; out["lorR"] = lorR; out["dlor"] = lorG - lorR
    out["rdG"] = rdG / wsum if wsum else np.nan; out["rdR"] = rdR / wsum if wsum else np.nan; out["drd"] = out["rdG"] - out["rdR"]
    out["excess100"] = 100 * ex / Ntot
    exk = 0.0
    for s in np.unique(strat):
        if s % 10 < 4: continue          # conf strata 4, 5 = untrained agreement >= 0.9 ("known")
        m = strat == s; mr = m & rej; ma = m & ~rej
        if mr.sum() == 0 or ma.sum() == 0: continue
        exk += mr.sum() * (yG[mr].mean() - sig(logit(yG[ma].mean()) + lorR))
    out["excess100_known"] = 100 * exk / Ntot
    out["yG_rej"] = yG[rej].mean(); out["yG_acc"] = yG[~rej].mean(); out["yR_rej"] = yR[rej].mean(); out["yR_acc"] = yR[~rej].mean()
    out["n_rej100"] = 100 * rej.sum() / Ntot
    return out

def placebo_runs(P, U, target):
    """no-location placebo: the key-plus-random-noise grader (key, random false rejects / accepts; FK20) whose mean IDK on U is
    closest to target. None of these graders rejects the judge's rejection set specifically."""
    best = None
    for g in ("sim00fn00", "sim00fn20", "sim00fn40", "sim00fn50", "sim00fn60", "sim00fn70", "sim20fn00", "sim35fn00", "lenQ",
              "sim10fn20", "sim20fn20", "sim25fn20", "sim30fn20", "sim40fn20"):
        rr = [r for r in P["R"].values() if r["g"] == g and r["kind"] == "FK20" and r["credit"] == "idkc05"]
        if not rr: continue
        a = np.mean([(r["o"][U] == 0).mean() for r in rr])
        if best is None or abs(a - target) < best[0]: best = (abs(a - target), g, rr, a)
    return best

def boot_located(yG, yR, nG, nR, rej, strat, dsid, Ntot, B=1000, seed=0, keys=("excess100", "dlor", "drd", "excess100_known")):
    rng = np.random.default_rng(seed); n = len(yG); est = located_stats(yG, yR, nG, nR, rej, strat, np.arange(n), Ntot)
    groups = [np.where(dsid == d)[0] for d in np.unique(dsid)]; bs = {k: [] for k in keys}
    for _ in range(B):
        ix = np.concatenate([rng.choice(g, len(g)) for g in groups])
        o = located_stats(yG, yR, nG, nR, rej, strat, ix, Ntot)
        for k in keys: bs[k].append(o[k])
    ci = {k: (np.nanpercentile(bs[k], 2.5), np.nanpercentile(bs[k], 97.5)) for k in keys}
    return est, ci

if __name__ == "__main__":
    for st in ("qwen7", "llama8"):
        for ds in ("h1", "pq", "nq"):
            P = build(st, ds, force=len(sys.argv) > 1 and sys.argv[1] == "force")
            print(st, ds, "questions", P["Q"]["n"], "train", P["Q"]["train"].sum(), "new", P["Q"]["held"].sum(), "runs", len(P["R"]), flush=True)
