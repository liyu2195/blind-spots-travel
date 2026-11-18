"""(arch_v4b) Independent re-implementation of the judge crossover in both directions (rev_R4 r4_fa_crossover_b; stats c5), own code.
Units: held-out (duplicate-free, bm.py cache) questions with a valid untrained greedy answer.
  FA units: greedy answer audited wrong (not loose-key right, Sonnet level 1..3); FR units: audited right (loose key or Sonnet >= 4).
Verdicts of judges Q M G O R at their a35 cuts (analysis/extra/final/thresholds_<J>.json, t_a35) on the untrained greedy answer
  (base score files, s = -1; P(Yes) = py / (py + pn), 0.5 if py + pn <= 1e-3, as in bm.py).
For a judge pair (K, J): A = K accepts & J rejects, B = J accepts & K rejects (FA) or A = K rejects & J accepts, B = J rejects & K accepts (FR);
  |A|, |B| >= 5. R_K(S) = mean over K's FK20 judge<K>a35 idkc05 runs with new-question abstention < 0.95 of the IDK rate on S.
  X = [R_K(A) - R_K(B)] - [R_J(A) - R_J(B)] (double difference; per direction = X / 2). Cells = (student, dataset, unordered pair), mean.
Uncertainty: bootstrap (1000) of questions within cell and runs within judge. Output: a03_fa_mirror.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import json, pickle, itertools, numpy as np
BASE = (_RR + "/run1/base"); CACHE = (_RX + "/story/B_refusal_map/cache_{}_{}.pkl")
OUT = open((_RA + "/arch_v4b/a03_fa_mirror.txt"), "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n")
DSN = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}; JS = ["Q", "M", "G", "O", "R"]
TH = {J: json.load(open(f"{_RX}/final/thresholds_{J}.json")) for J in JS}
rng = np.random.default_rng(7); NB = 1000
res = {}
for kind in ("FA", "FR"):
    for st in ("qwen7", "llama8"):
        for ds in ("h1", "pq", "nq"):
            P = pickle.load(open(CACHE.format(st, ds), "rb")); Q = P["Q"]; idx = Q["idx"]; pos = {i: k for k, i in enumerate(idx)}
            unit = Q["gval"] & ((~Q["key"] & (Q["son"] >= 1) & (Q["son"] <= 3)) if kind == "FA" else Q["aud"])
            acc = {}
            for J in JS:
                py = np.full(len(idx), np.nan)
                for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{DSN[ds]}_1500_n32_s1.jsonl")):
                    if r["s"] == -1 and r["idx"] in pos: py[pos[r["idx"]]] = r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5
                acc[J] = py > TH[J][f"{st}_{DSN[ds]}"]["t_a35"]
            runs = {J: [r for r in P["R"].values() if r["kind"] == "FK20" and r["g"] == f"judge{J}a35" and r["a"] < 0.95] for J in JS}
            for K, J in itertools.combinations(JS, 2):
                if not runs[K] or not runs[J]: continue
                if kind == "FA": A = unit & acc[K] & ~acc[J]; B = unit & acc[J] & ~acc[K]
                else: A = unit & ~acc[K] & acc[J]; B = unit & ~acc[J] & acc[K]
                if A.sum() < 5 or B.sum() < 5: continue
                IK = np.array([r["o"] == 0 for r in runs[K]], float); IJ = np.array([r["o"] == 0 for r in runs[J]], float)
                def X(wq, wk, wj):
                    RK = (wk[:, None] * IK).sum(0) / wk.sum(); RJ = (wj[:, None] * IJ).sum(0) / wj.sum()
                    m = lambda R, S: (wq * R * S).sum() / (wq * S).sum()
                    return (m(RK, A) - m(RK, B)) - (m(RJ, A) - m(RJ, B))
                x0 = X(np.ones(len(idx)), np.ones(len(IK)), np.ones(len(IJ)))
                bs = [X(rng.multinomial(len(idx), np.ones(len(idx)) / len(idx)).astype(float),
                        rng.multinomial(len(IK), np.ones(len(IK)) / len(IK)).astype(float) + 1e-9,
                        rng.multinomial(len(IJ), np.ones(len(IJ)) / len(IJ)).astype(float) + 1e-9) for _ in range(NB)]
                res.setdefault(kind, []).append((st, ds, K, J, int(A.sum()), int(B.sum()), x0, np.array(bs)))
for kind in ("FR", "FA"):
    rows = res.get(kind, [])
    pr(f"\n== {kind}: cells {len(rows)}; sign as expected ({'+' if kind == 'FR' else '-'}) in {sum((r[6] > 0) if kind == 'FR' else (r[6] < 0) for r in rows)}")
    for grp, sel in (("qwen7", lambda r: r[0] == "qwen7"), ("llama8", lambda r: r[0] == "llama8"), ("pooled", lambda r: True)):
        for ds in ("h1", "pq", "nq", "all"):
            rr = [r for r in rows if sel(r) and (ds == "all" or r[1] == ds)]
            if not rr: continue
            pt = np.mean([r[6] for r in rr]); bs = np.nanmean(np.array([r[7] for r in rr]), axis=0); lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            pr(f"  {grp:6s} {ds:3s} cells {len(rr):2d}: X {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  per direction {pt/2:+.3f}  (median |A| {int(np.median([r[4] for r in rr]))}, |B| {int(np.median([r[5] for r in rr]))})")
