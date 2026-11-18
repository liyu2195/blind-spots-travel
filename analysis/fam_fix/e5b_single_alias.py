"""Check: does the former-state-capitals refusal survive on unambiguous items?
Split former-state (F) probe items into single-alias (one accepted capital) vs multi-alias (often several historical capitals).
Same definitions as fam_fix/e5_capitals.py (known = untrained greedy right by loose key; refusal = trained run's greedy answer is IDK).
Read-only: imports the project's helpers; writes fam_fix/e5b_single_alias_out.txt."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, glob, os, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import ans, IDKF, C, py_of, thresholds, STS, BASE, DATA
P2 = (_RR + "/run1/adeval/probe2")
CACHE = (_RA + "/stats/cache")
probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probs)
cls = np.array([probs[i]["cls"] for i in ids]); nal = np.array([len(probs[i]["aliases"]) for i in ids])
sub = {"F1": (cls == "F") & (nal == 1), "F2+": (cls == "F") & (nal > 1), "P": cls == "P", "P1": (cls == "P") & (nal == 1)}
def fam(g):
    if g == "sim00fn00": return "gold"
    if re.fullmatch(r"sim00fn[1-9]\d", g): return "keyFN"
    if re.fullmatch(r"judge[QMGO]a35", g): return g
    if re.fullmatch(r"judge[QMGO]t50", g): return g
    return None
out = []
for st in STS:
    g = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        x = ans(r); ok = bool(x) and not IDKF.match(x) and C.tq_correct(x, set(probs[r["idx"]]["aliases"]))
        if r["s"] < 0: g[r["idx"]] = ok
    known = np.array([g[i] for i in ids])
    out.append(f"== {st}: known counts " + ", ".join(f"{k} {int((m & known).sum())}/{int(m.sum())}" for k, m in sub.items()))
    # judge acceptance at a35 of known answers by subset
    for J in "QMGO":
        th = thresholds(J); t35 = th[f"{st}_pqa"]["t_a35"]
        S = {(r["idx"], r["s"]): py_of(r) for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_pqq_1500_n32_s1.jsonl"))}
        acc = {k: np.mean([S[(i, -1)] > t35 for i, m, kn in zip(ids, msk, known) if m and kn]) for k, msk in sub.items()}
        out.append(f"  judge {J} a35 accepts known: " + "  ".join(f"{k} {v:.2f}" for k, v in acc.items()))
    RR = pickle.load(open(f"{CACHE}/runs_{st}_pq.pkl", "rb")); Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb"))
    runs = defaultdict(list)
    for f in sorted(glob.glob(f"{P2}/{st}_FK20*_pq.jsonl")):
        run = os.path.basename(f)[:-6]; m = re.match(rf"^{st}_FK20(.+)_idkc05_s(\d+)_pq$", run)
        if not m or fam(m.group(1)) is None: continue
        o = {}
        for r in map(json.loads, open(f)):
            if r["s"] >= 0: continue
            x = ans(r); o[r["idx"]] = bool(x) and bool(IDKF.match(x))
        y = np.array([o[i] for i in ids], float)
        a_new = float((RR[run]["o"][Bp["new"]] == 0).mean()) if run in RR else np.nan
        runs[fam(m.group(1))].append(dict(run=run, y=y, a=a_new))
    def R(x, k): mk = sub[k] & known; return x["y"][mk].mean()
    ref = runs["keyFN"] + runs["gold"]; ra = np.array([x["a"] for x in ref])
    for arm in ["gold", "judgeQa35", "judgeMa35", "judgeGa35", "judgeOa35", "judgeQt50"]:
        if arm not in runs: continue
        rr = runs[arm]
        line = f"  {arm:10s} n={len(rr)} abst {np.mean([x['a'] for x in rr]):.2f} | refusal of known: " + "  ".join(f"{k} {np.mean([R(x, k) for x in rr]):.2f}" for k in sub)
        if arm != "gold":
            # nearest-3 key+random-FN reference by PopQA new abstention, per run, contrast F-sub minus P
            d1 = []; d2 = []
            for x in rr:
                nn = np.argsort(np.abs(ra - x["a"]))[:3]
                d1.append((R(x, "F1") - R(x, "P")) - np.mean([R(ref[j], "F1") - R(ref[j], "P") for j in nn]))
                d2.append((R(x, "F2+") - R(x, "P")) - np.mean([R(ref[j], "F2+") - R(ref[j], "P") for j in nn]))
            line += f" | vs key+FN nn3: F1-P {np.mean(d1):+.2f}  F2+-P {np.mean(d2):+.2f}"
        out.append(line)
open((_RA + "/fam_fix/e5b_single_alias_out.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
