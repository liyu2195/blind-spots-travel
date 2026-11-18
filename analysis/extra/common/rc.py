"""Independent helpers (re-implemented loading; grading functions imported from the authors' st_common)."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, os, re, sys, bisect
from collections import Counter, defaultdict
sys.path.insert(0, _RC)
import st_common as C
RL = (_RR + "/run1/rl"); BASE = (_RR + "/run1/base"); DATA = _RD; FILT = (_RR + "/run1/filter")
IDK = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
def jl(p): return [json.loads(l) for l in open(p)]
def setup(ds="tqa"):
    allq = jl(f"{DATA}/{ds}_1500.jsonl"); probs = {p["idx"]: p for p in allq}
    H1 = {p["idx"] for p in jl(f"{DATA}/{ds}_h1.jsonl")}
    trq = {p["question"].strip().lower() for p in allq if p["idx"] in H1}
    trid = {p.get("qid") for p in allq if p["idx"] in H1}
    HELD = [p["idx"] for p in allq if p["idx"] not in H1 and not (p["question"].strip().lower() in trq or (p.get("qid") is not None and p.get("qid") in trid))]
    return probs, H1, HELD
def sa(r):
    t = r["text"]
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)
def load(run, ck):
    g, s = {}, defaultdict(list)
    for r in map(json.loads, open(f"{RL}/{run}/eval_rl_{ck}.jsonl")):
        if r["s"] < 0: g[r["idx"]] = r
        else: s[r["idx"]].append(r)
    return g, s
def outcome(run, probs, items, ck="ck120"):
    g, _ = load(run, ck); o = {}
    for i in items:
        x = sa(g[i])
        o[i] = "bad" if not x else ("idk" if IDK.match(x) else ("c" if C.tq_correct(x, set(probs[i]["aliases"])) else "w"))
    return o
def same(a, b):
    return bool(a == b or (len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b)) or (len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a)))
def base32(st, probs, ds="tqa"):
    """per question: greedy answer ok?, plurality answer ok?, plurality share, share of samples correct, greedy-agreement share"""
    rows = defaultdict(list); gr = {}
    for r in map(json.loads, open(f"{BASE}/{st}_{ds}_1500_n32_s1.jsonl")):
        if r["s"] < 0: gr[r["idx"]] = r
        else: rows[r["idx"]].append(r)
    F = {}
    for i in gr:
        al = set(probs[i]["aliases"]); xa = [x for x in (sa(r) for r in rows[i]) if x and not IDK.match(x)]
        cl = []
        for x, n in Counter(xa).most_common():
            for c in cl:
                if same(x, c[0]): c[1] += n; break
            else: cl.append([x, n])
        top = max(cl, key=lambda c: c[1]) if cl else None
        gx = sa(gr[i]); gok = bool(gx) and not IDK.match(gx)
        F[i] = dict(g_ok=gok and C.tq_correct(gx, al), p_ok=bool(top) and C.tq_correct(top[0], al), pshare=(top[1] / 32) if top else 0.0,
                    q=sum(C.tq_correct(x, al) for x in xa) / 32, gagree=(sum(same(gx, y) for y in xa) / 32) if gok else 0.0, g_idk=bool(gx) and bool(IDK.match(gx)))
    return F
def curve(F, items, conf="pshare", ok="p_ok"):
    n = len(items); srt = sorted(items, key=lambda i: -F[i][conf]); lv = sorted({F[i][conf] for i in items}, reverse=True)
    pts = [(1.0, 0.0)]; wrong = 0; k = 0
    for L in lv:
        while k < n and F[srt[k]][conf] >= L: wrong += not F[srt[k]][ok]; k += 1
        pts.append((1 - k / n, wrong / n))
    return sorted(pts)
def at(pts, a):
    xs = [p[0] for p in pts]; j = bisect.bisect_left(xs, a)
    if j == 0: return pts[0][1]
    if j >= len(pts): return pts[-1][1]
    (a0, h0), (a1, h1) = pts[j - 1], pts[j]
    return h0 + (h1 - h0) * (a - a0) / max(1e-9, a1 - a0)
def rule_set(F, items, rate, conf="pshare"):
    """the round(rate*n) least-confident questions (ties broken by index)"""
    srt = sorted(items, key=lambda i: (F[i][conf], i)); return set(srt[:round(rate * len(items))])
