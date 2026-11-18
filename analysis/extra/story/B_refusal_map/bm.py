"""Shared loaders for my own recomputations. Read-only on project data.
Builds one cache per (student, dataset) with held-out question features and the outcome of every trained run on them.
Conventions (same as the project's rc.py / audit_score.py):
  held-out = rc.setup(ds) HELD (duplicates of trained questions removed); untrained greedy = base file row s = -1 (IDK prompt);
  conf = share of the 32 untrained samples that agree with the greedy answer (rc.base32 'gagree');
  key_ok (loose) = st_common.tq_correct; key_strict = tq_strict (the trainer's gold-key reward);
  audited right = loose key OR Sonnet (no key) level >= 4 (frontierH for untrained answers, auditT for trained runs' key-wrong answers);
  judge accepts = P(Yes)/(P(Yes)+P(No)) > t (trainer rule).
Run outcome codes: 0 = IDK, 1 = right (loose key), 2 = wrong (loose key), 3 = malformed."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, re, glob, pickle, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, load, base32, same, C, RL, BASE, DATA
HERE = (_RX + "/story/B_refusal_map"); API = (_RX + "/apitest")
DS = {"h1": ("tqa", "TriviaQA"), "pq": ("pqa", "PopQA"), "nq": ("nqo", "NQ-open")}
TH = json.load(open((_RX + "/confirm/thresholds.json")))
PAIRS = [(st, ds) for st in ("qwen7", "llama8") for ds in ("h1", "pq", "nq")]

def family(g):
    if g == "sim00fn00": return "gold"
    if re.fullmatch(r"sim00fn\d\d", g): return "keyFN"
    if re.fullmatch(r"sim\d\dfn00", g): return "keyFP"
    if re.fullmatch(r"sim\d\dfn\d\d", g): return "mix"
    if g.startswith(("typeA", "typeB")): return "typeprog"
    if g.startswith(("popH", "popL")): return "popprog"
    if g.startswith("blind"): return "blind"
    if g.startswith("frzVOTE"): return "vote"
    if g.startswith("frzOR"): return "orkey"
    if g.startswith("frzPEER"): return "peer"
    if g.startswith(("frzQ50k", "frzS55k")): return "jveto"
    if g == "judgeQa35": return "judge35"
    if g == "judgeQa50": return "judge50"
    if g == "judgeQt50": return "judge_t50"
    if g.startswith("judgeQt50q"): return "quorum"
    if g == "apiS55": return "sonnet"
    return "other"

def clusters(xs):
    cl = []
    for x, n in Counter(xs).most_common():
        for c in cl:
            if same(x, c[0]): c[1] += n; c[2].append(x); break
        else: cl.append([x, n, [x]])
    return cl

def build(st, ds, force=False):
    path = f"{HERE}/cache_{st}_{ds}.pkl"
    if os.path.exists(path) and not force: return pickle.load(open(path, "rb"))
    dsn, lab = DS[ds]; probs, H1, HELD = setup(dsn); hs = set(HELD); n = len(HELD); F = base32(st, probs, dsn)
    LV = {}
    for f in (f"{API}/frontierH_{st}_{dsn}.jsonl", f"{API}/auditT_{st}_{dsn}.jsonl"):
        if os.path.exists(f):
            for r in map(json.loads, open(f)): LV[(r["idx"], r["norm"])] = r["level"]
    gr = {}; samp = defaultdict(list)
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        if r["idx"] not in hs: continue
        x = sa(r); x = x if (x and not IDK.match(x)) else ""
        if r["s"] < 0: gr[r["idx"]] = x
        else: samp[r["idx"]].append((r["s"], x))
    PY = {}
    for r in map(json.loads, open(f"{BASE}/score_Q_{st}_{dsn}_1500_n32_s1.jsonl")):
        if r["idx"] in hs: PY[(r["idx"], r["s"])] = r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
    Q = dict(idx=np.array(HELD), n=n, lab=lab)
    Q["conf"] = np.array([F[i]["gagree"] for i in HELD]); Q["gval"] = np.array([bool(gr[i]) for i in HELD])
    Q["gidk"] = np.array([bool(F[i]["g_idk"]) for i in HELD])
    Q["key"] = np.array([bool(F[i]["g_ok"]) for i in HELD])
    Q["kstrict"] = np.array([bool(gr[i]) and bool(C.tq_strict(gr[i], set(probs[i]["aliases"]))) for i in HELD])
    Q["son"] = np.array([LV.get((i, gr[i]), 0) if gr[i] else 0 for i in HELD]); Q["son_have"] = np.array([((i, gr[i]) in LV) if gr[i] else False for i in HELD])
    Q["aud"] = Q["key"] | (Q["son"] >= 4)
    Q["pj"] = np.array([PY.get((i, -1), 0.5) for i in HELD])
    Q["prop"] = np.array([probs[i].get("prop", "") for i in HELD]); Q["pop"] = np.array([probs[i].get("s_pop", np.nan) for i in HELD], float)
    Q["question"] = [probs[i]["question"] for i in HELD]; Q["gans"] = [gr[i] for i in HELD]; Q["gold"] = [probs[i].get("gold", "") for i in HELD]
    # acceptance-set features on HELD-OUT questions (untrained candidate clusters of the 32 samples)
    t = TH[f"{st}_{dsn}"]; thr = {"a35": t["t_a35"], "a50": t["t_a50"], "t50": 0.5}
    none = {k: np.zeros(n, bool) for k in thr}; knone = np.zeros(n, bool); topacc = {k: np.zeros(n) for k in thr}
    for k_, i in enumerate(HELD):
        xs = [(s, x) for s, x in samp[i] if x]
        cl = clusters([x for s, x in xs]); al = set(probs[i]["aliases"])
        if not cl: knone[k_] = True
        else: knone[k_] = not any(C.tq_strict(c[0], al) for c in cl)
        for k, tv in thr.items():
            acc_any = False; best = 0
            for c in cl:
                members = set(c[2]); ss = [s for s, x in xs if x in members]
                ok = [PY.get((i, s), 0.5) > tv for s in ss]
                if any(ok): acc_any = True; best = max(best, c[1])
            none[k][k_] = not acc_any; topacc[k][k_] = best / 32
    Q["none"] = none; Q["key_none"] = knone; Q["thr"] = thr
    # runs
    R = {}
    pats = [(rf"{st}_FK20(.+)_idkc05_s(\d+)_{ds}$", "FK20", "ck120"), (rf"{st}_B1K20(.+)_idkc05_s(\d+)_{ds}$", "B1", "ck120"),
            (rf"{st}_B2K20(.+)_idkc05_s(\d+)_{ds}$", "B2", "ck240"), (rf"{st}_FK(50|100)(.+)_idkc05_s(\d+)_{ds}$", "FKx", "ck120")]
    for d in sorted(glob.glob(f"{RL}/{st}_*_{ds}")):
        name = os.path.basename(d)
        for pat, kind, ck in pats:
            m = re.match(pat, name)
            if not m: continue
            if kind == "FKx": g = f"KL{m.group(1)}_" + m.group(2); s = int(m.group(3))
            else: g, s = m.group(1), int(m.group(2))
            if not os.path.exists(f"{d}/eval_rl_{ck}.jsonl"):
                if kind == "B2" and os.path.exists(f"{d}/eval_rl_ck120.jsonl"): ck = "ck120"
                else: continue
            gt, _ = load(name, ck); o = np.zeros(n, np.int8); ans = []; aud = np.zeros(n, bool); miss = 0
            for k_, i in enumerate(HELD):
                x = sa(gt[i]); ans.append(x)
                if not x: o[k_] = 3
                elif IDK.match(x): o[k_] = 0
                elif C.tq_correct(x, set(probs[i]["aliases"])): o[k_] = 1; aud[k_] = True
                else:
                    o[k_] = 2; l = LV.get((i, x))
                    if l is None: miss += 1
                    elif l >= 4: aud[k_] = True
            R[name] = dict(g=g, s=s, kind=kind, fam=family(g), o=o, ans=ans, aud=aud, miss=miss, a=float((o == 0).mean()))
            break
    P = dict(st=st, ds=ds, Q=Q, R=R)
    pickle.dump(P, open(path, "wb")); return P

def rule(conf, a):
    n = len(conf); r = np.zeros(n, bool); r[np.argsort(conf, kind="stable")[:round(a * n)]] = True; return r

def kappa(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); po = np.mean(x == y); px, py = x.mean(), y.mean(); pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 - 1e-12 else np.nan

def mh(y, rej, strata, mask):
    """Mantel-Haenszel weighted difference of mean y (rej minus not rej) within strata, over mask."""
    num = den = 0.0
    for s in np.unique(strata[mask]):
        m = mask & (strata == s); a = m & rej; b = m & ~rej; na, nb = a.sum(), b.sum()
        if na == 0 or nb == 0: continue
        w = na * nb / (na + nb); num += w * (y[a].mean() - y[b].mean()); den += w
    return num / den if den > 0 else np.nan

if __name__ == "__main__":
    for st, ds in PAIRS:
        P = build(st, ds, force=len(sys.argv) > 1 and sys.argv[1] == "force")
        print(st, ds, "held-out", P["Q"]["n"], "runs", len(P["R"]))
