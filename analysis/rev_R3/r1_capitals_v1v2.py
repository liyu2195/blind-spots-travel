"""(rev_R3) Capitals probes: for sure & right untrained greedy answers to capital questions, split by probe version (v1 = pqp PopQA-derived,
v2 = pqq hand-written) and class (F = former state, P = present day): how often the strict Qwen judge (a35 cut of the student's PopQA cell)
accepts them, and the median P(Yes). Read-only on project data. Usage: nice -n 10 timeout 280 python3 r1_capitals_v1v2.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, re, numpy as np
sys.path.insert(0, _RC)
import st_common as C
BASE = (_RR + "/run1/base"); DATA = _RD; API = (_RX + "/apitest")
IDKRE = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure|unknown)$")
TH = json.load(open((_RX + "/final/thresholds_Q.json")))
def jl(p): return [json.loads(l) for l in open(p)]
def parse(r):
    t = r["text"]
    if r.get("fin") != "stop" or "\n" in t.strip() or "think>" in t: return ""
    return C.parse_tq(t)
def py_of(r): return r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
for st in ("qwen7", "llama8"):
    thr = TH[f"{st}_pqa"]["t_a35"]
    for tag, ver in (("pqp", "v1"), ("pqq", "v2")):
        P = {p["idx"]: p for p in jl(f"{DATA}/{tag}_1500.jsonl")}
        LV = {(r["idx"], r["norm"]): r["level"] for r in map(json.loads, open(f"{API}/frontierH_{st}_{tag}.jsonl"))}
        g = {}; smp = {}
        for r in map(json.loads, open(f"{BASE}/{st}_{tag}_1500_n32_s1.jsonl")):
            x = parse(r)
            if r["s"] < 0: g[r["idx"]] = x
            else: smp.setdefault(r["idx"], []).append(x)
        pj = {}
        for r in map(json.loads, open(f"{BASE}/score_Q_{st}_{tag}_1500_n32_s1.jsonl")):
            if r["s"] < 0: pj[r["idx"]] = py_of(r)
        for cls in ("F", "P"):
            ids = [i for i, p in P.items() if p.get("prop") == "capital" and p.get("cls") == cls]
            ok = []
            for i in ids:
                x = g.get(i, "")
                if not x or IDKRE.match(x): continue
                right = C.tq_correct(x, set(P[i]["aliases"])) or LV.get((i, x), 0) >= 4
                sure = np.mean([s == x for s in smp.get(i, [])]) >= 31 / 32 if smp.get(i) else False
                if right and sure and i in pj: ok.append(pj[i])
            ok = np.array(ok)
            if len(ok):
                print(f"{st} {ver} cls {cls}: capital questions {len(ids)}, sure&right {len(ok)}, Q accepts at a35 ({thr}) {np.mean(ok > thr):.3f}, at 0.5 {np.mean(ok > 0.5):.3f}, median P(Yes) {np.median(ok):.3f}")
            else:
                print(f"{st} {ver} cls {cls}: capital questions {len(ids)}, sure&right 0")
