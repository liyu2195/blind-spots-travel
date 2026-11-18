"""(knowledge) Per-question features of the UNTRAINED student on all 1,500 questions of a set (trained + new).
Read-only on project data; writes feat_<st>_<ds>.pkl in this folder.
Conventions follow bm.py / rc.py: greedy = base row s=-1 (IDK prompt), samples s=0..31; conf = share of the 32 samples that
agree (rc.same) with the greedy answer; key = loose alias key (C.tq_correct); strict = C.tq_strict; Sonnet level >= 4 = right.
Judge acceptance = P(Yes)/(P(Yes)+P(No)) > threshold (final/thresholds_<J>.json: t_a35, t_a50; t50 = 0.5)."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, math, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import setup, sa, IDK, same, C, BASE
OUT = (_RA + "/knowledge"); API = (_RX + "/apitest")
DS = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}
JUDGES = ["Q", "M", "G", "O", "R", "QS"]

def build(st, ds):
    dsn = DS[ds]; probs, H1, HELD = setup(dsn); hs = set(HELD)
    idxs = sorted(probs); N = len(idxs); assert idxs == list(range(N))
    TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json"))[f"{st}_{dsn}"] for J in JUDGES}
    gr = {}; samp = defaultdict(list)
    for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
        x = sa(r)
        if not x: v = ("bad", "")
        elif IDK.match(x): v = ("idk", "")
        else: v = ("ans", x)
        if r["s"] < 0: gr[r["idx"]] = v
        else: samp[r["idx"]].append((r["s"], v))
    LV = {}
    for f in (f"{API}/frontierH_{st}_{dsn}.jsonl", f"{API}/frontierA_{st}_{dsn}.jsonl"):
        if os.path.exists(f):
            for r in map(json.loads, open(f)): LV[(r["idx"], r["norm"])] = r["level"]
    PY = {J: {} for J in JUDGES}
    for J in JUDGES:
        for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl")):
            d = r["py"] + r["pn"]; PY[J][(r["idx"], r["s"])] = r["py"] / d if d > 1e-3 else 0.5
    F = dict(st=st, ds=ds, N=N, tr=np.array([i in H1 for i in idxs]), ho=np.array([i in hs for i in idxs]),
             prop=np.array([probs[i].get("prop", "") for i in idxs]), pop=np.array([probs[i].get("s_pop", np.nan) for i in idxs], float),
             question=[probs[i]["question"] for i in idxs], aliases=[sorted(probs[i]["aliases"]) for i in idxs])
    g = []; g_valid = np.zeros(N, bool); g_idk = np.zeros(N, bool); g_key = np.zeros(N, bool); g_strict = np.zeros(N, bool)
    son_g = np.full(N, -1); conf = np.zeros(N); pshare = np.zeros(N); p_ok = np.zeros(N, bool); q = np.zeros(N); q_strict = np.zeros(N)
    pass32 = np.zeros(N, bool); s_idk = np.zeros(N); s_bad = np.zeros(N); ent = np.zeros(N); ncl = np.zeros(N, int); top = []
    samples = []
    jg = {J: np.zeros(N) for J in JUDGES}; jacc = {(J, k): np.zeros(N) for J in JUDGES for k in ("a35", "a50", "t50")}
    jg_acc = {(J, k): np.zeros(N, bool) for J in JUDGES for k in ("a35", "a50", "t50")}
    for i in idxs:
        al = set(probs[i]["aliases"]); kind, x = gr[i]; g.append(x)
        g_valid[i] = kind == "ans"; g_idk[i] = kind == "idk"
        if x:
            g_key[i] = bool(C.tq_correct(x, al)); g_strict[i] = bool(C.tq_strict(x, al)); son_g[i] = LV.get((i, x), -1)
        ss = samp[i]; assert len(ss) == 32, (i, len(ss))
        xs = [v[1] for s, v in ss if v[0] == "ans"]; samples.append(xs)
        s_idk[i] = sum(v[0] == "idk" for s, v in ss) / 32; s_bad[i] = sum(v[0] == "bad" for s, v in ss) / 32
        q[i] = sum(bool(C.tq_correct(y, al)) for y in xs) / 32; q_strict[i] = sum(bool(C.tq_strict(y, al)) for y in xs) / 32
        pass32[i] = q[i] > 0
        cl = []
        for y, n in Counter(xs).most_common():
            for c in cl:
                if same(y, c[0]): c[1] += n; break
            else: cl.append([y, n])
        ncl[i] = len(cl)
        if cl:
            t = max(cl, key=lambda c: c[1]); top.append(t[0]); pshare[i] = t[1] / 32; p_ok[i] = bool(C.tq_correct(t[0], al))
            ps = [c[1] / 32 for c in cl] + ([s_idk[i]] if s_idk[i] > 0 else []); ent[i] = -sum(p * math.log(p) for p in ps if p > 0)
        else: top.append("")
        conf[i] = (sum(same(x, y) for y in xs) / 32) if x else 0.0
        for J in JUDGES:
            jg[J][i] = PY[J].get((i, -1), 0.5)
            for k, tv in (("a35", TH[J]["t_a35"]), ("a50", TH[J]["t_a50"]), ("t50", 0.5)):
                jg_acc[(J, k)][i] = bool(x) and jg[J][i] > tv
                jacc[(J, k)][i] = sum(PY[J].get((i, s), 0.5) > tv for s, v in ss if v[0] == "ans") / 32
    F.update(g=g, g_valid=g_valid, g_idk=g_idk, g_key=g_key, g_strict=g_strict, son_g=son_g, g_aud=g_key | (son_g >= 4),
             conf=conf, pshare=pshare, p_ok=p_ok, q=q, q_strict=q_strict, pass32=pass32, s_idk=s_idk, s_bad=s_bad, ent=ent, ncl=ncl,
             top=top, samples=samples, jg=jg, jacc=jacc, jg_acc=jg_acc, TH={J: (TH[J]["t_a35"], TH[J]["t_a50"]) for J in JUDGES})
    pickle.dump(F, open(f"{OUT}/feat_{st}_{ds}.pkl", "wb"))
    ho = F["ho"]
    print(st, ds, "N", N, "trained", F["tr"].sum(), "held", ho.sum(), "| held: g_valid %.3f g_idk %.3f key %.3f aud %.3f son_have %.3f pass32 %.3f conf %.3f"
          % (g_valid[ho].mean(), g_idk[ho].mean(), g_key[ho].mean(), F["g_aud"][ho].mean(), ((son_g >= 0) | ~g_valid)[ho].mean(), pass32[ho].mean(), conf[ho].mean()), flush=True)

if __name__ == "__main__":
    for a in sys.argv[1:]:
        st, ds = a.split(":"); build(st, ds)
