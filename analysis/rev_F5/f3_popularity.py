"""(rev_F5) f3: do reference-free judges' rejections of RIGHT, sure untrained answers (a35) track entity popularity on PopQA (a proxy for the
judge's own closed-book knowledge)? All 1,500 PopQA questions, untrained greedy valid, key-right, sure (>= 31/32 agree). Per s_pop quintile:
share of judges rejecting; and the pairwise disagreement rate. Output: f3_popularity_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, re
sys.path.insert(0, _RC)
import numpy as np, st_common as C
BASE = (_RR + "/run1/base"); DATA = _RD; FIN = (_RX + "/final")
L = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)
P(__doc__)
def jl(p): return [json.loads(l) for l in open(p)]
def parse(r):
    t = r.get("text", "")
    return "" if r.get("fin") != "stop" or "\n" in t.strip() else C.parse_tq(t)
def agree(a, b):
    if not a or not b: return False
    return a == b or (len(a) >= 3 and re.search(r"\b" + re.escape(a) + r"\b", b) is not None) or (len(b) >= 3 and re.search(r"\b" + re.escape(b) + r"\b", a) is not None)
Q = {r["idx"]: r for r in jl(f"{DATA}/pqa_1500.jsonl")}; N = len(Q)
pop = np.array([Q[i]["s_pop"] for i in range(N)], float)
for st in ("qwen7", "llama8"):
    g = [""] * N; samp = [[] for _ in range(N)]
    for r in jl(f"{BASE}/{st}_pqa_1500_n32_s1.jsonl"):
        x = parse(r)
        if r["s"] == -1: g[r["idx"]] = x
        else: samp[r["idx"]].append(x)
    ok = np.array([bool(g[i]) and not C.IDK_RE.match(g[i]) and C.tq_correct(g[i], Q[i]["aliases"]) and sum(agree(g[i], s) for s in samp[i]) >= 31 for i in range(N)])
    rej = {}
    for J in "QMGOR":
        t = json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_pqa"]["t_a35"]; v = np.full(N, np.nan)
        for r in jl(f"{BASE}/score_{J}_{st}_pqa_1500_n32_s1.jsonl"):
            if r["s"] == -1: d = r["py"] + r["pn"]; v[r["idx"]] = r["py"] / d if d > 1e-3 else 0.5
        rej[J] = v <= t
    U = np.where(ok)[0]; lp = np.log10(pop[U] + 1); qs = np.quantile(lp, [0.2, 0.4, 0.6, 0.8]); b = np.searchsorted(qs, lp)
    nrej = np.sum([rej[J][U] for J in rej], 0)
    P(f"{st}: right-sure PopQA units {len(U)}; mean share of 5 judges rejecting {nrej.mean() / 5:.3f}")
    for k in range(5):
        m = b == k; pairs = [(rej[a][U][m] != rej[c][U][m]).mean() for i, a in enumerate("QMGOR") for c in "QMGOR"[i + 1:]]
        P(f"   s_pop quintile {k + 1} (median s_pop {np.median(pop[U][m]):.0f}): judges rejecting {nrej[m].mean() / 5:.3f} | all 5 reject {np.mean(nrej[m] == 5):.3f} | none reject {np.mean(nrej[m] == 0):.3f} | pairwise disagreement {np.mean(pairs):.3f}")
    rho = np.corrcoef(np.argsort(np.argsort(lp)), np.argsort(np.argsort(nrej)))[0, 1]
    P(f"   Spearman(log popularity, number of judges rejecting) {rho:+.3f}")
open((_RA + "/rev_F5/f3_popularity_out.txt"), "w").write("\n".join(L) + "\n")
