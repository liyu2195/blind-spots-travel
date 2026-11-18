"""(fam_fix) e7_capitals.py: FROZEN scorer for E7 (abstention taught by the IDK credit under unnormalised advantages, graders at their default
operating point). Capitals probe (pqq, never trained): known = untrained greedy answer right (loose key); refusal = trained run's greedy answer is IDK
(adeval/probe2). Arms are run-name patterns <st>_FK20D<grader>_idkc<credit>_s<seed>_pq: judge arms (grader judgeQt50 / judgeOt50 / ...) and the
key reference (grader sim00fn00) at several credits. For each judge run: G = R_F - R_P; reference G_ref = linear interpolation of the key runs' G at
the judge run's PopQA new-question abstention (key runs of the same student, sorted by abstention; outside their range the nearest key run);
DiD = G - G_ref. Also DiD_same = G minus the mean G of the key runs with the same credit. Per (grader, credit, student): mean over runs; interval =
bootstrap (2000) over probe questions within class (same draw for all runs of a student). Usage: python3 e7_capitals.py [PREFIX default FK20D]
Output: e7_capitals_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, glob, os, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
PRE = sys.argv[1] if len(sys.argv) > 1 else "FK20D"; KEYG = sys.argv[2] if len(sys.argv) > 2 else "sim00fn00"
P2 = (_RR + "/run1/adeval/probe2"); RLD = (_RR + "/run1/rl"); rng = np.random.default_rng(707)
probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probs); cls = np.array([probs[i]["cls"] for i in ids])
out = [__doc__]
for st in STS:
    g = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        if r["s"] < 0:
            x = ans(r); g[r["idx"]] = bool(x) and not IDKF.match(x) and C.tq_correct(x, set(probs[r["idx"]]["aliases"]))
    known = np.array([g[i] for i in ids]); mF = known & (cls == "F"); mP = known & (cls == "P")
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); new = np.asarray(Bp["new"], bool)
    runs = defaultdict(list)
    for f in sorted(glob.glob(f"{P2}/{st}_{PRE}*_pq.jsonl")):
        run = os.path.basename(f)[:-6]; m = re.match(rf"^{st}_{PRE}(.+)_idkc(\d+)_s(\d+)_pq$", run)
        if not m: continue
        o = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(f)) if r["s"] < 0}
        y = np.array([o[i] for i in ids], float)
        ef = f"{RLD}/{run}/eval_rl_ck120.jsonl"
        if not os.path.exists(ef): continue
        idk = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(ef)) if r["s"] < 0}
        a_new = float(np.mean([idk[i] for i in range(len(new)) if new[i] and i in idk]))
        runs[(m.group(1), m.group(2))].append(dict(run=run, y=y, a=a_new))
    keys = sorted([x for (gname, c), L in runs.items() if gname == KEYG for x in L], key=lambda x: x["a"])
    out.append(f"\n=== {st}: known F {int(mF.sum())}, known P {int(mP.sum())}; key reference runs {len(keys)}: " + ", ".join(f"{x['run'].split('_idkc')[1].split('_')[0]}:abst {x['a']:.2f}" for x in keys))
    def G(y, w): return (w * mF * y).sum() / (w * mF).sum() - (w * mP * y).sum() / (w * mP).sum()
    def stat(w):
        res = {}
        ka = np.array([x["a"] for x in keys]); kg = np.array([G(x["y"], w) for x in keys])
        for (gname, c), L in runs.items():
            if gname == KEYG: res[(gname, c)] = (float(np.mean([G(x["y"], w) for x in L])), np.nan, np.nan); continue
            d = []; ds = []
            for x in L:
                gref = float(np.interp(x["a"], ka, kg)) if len(ka) else np.nan
                same = [G(k["y"], w) for k in runs.get((KEYG, c), [])]
                d.append(G(x["y"], w) - gref); ds.append(G(x["y"], w) - (np.mean(same) if same else np.nan))
            res[(gname, c)] = (float(np.mean([G(x["y"], w) for x in L])), float(np.mean(d)), float(np.mean(ds)))
        return res
    est = stat(np.ones(len(ids))); boots = []
    for _ in range(2000):
        w = np.zeros(len(ids))
        for cl in ("F", "P"):
            I = np.where(cls == cl)[0]; np.add.at(w, rng.choice(I, len(I)), 1)
        boots.append(stat(w))
    for k in sorted(est):
        L = runs[k]; ci = lambda j: tuple(np.nanpercentile([b[k][j] for b in boots], [2.5, 97.5]))
        RF = np.mean([(mF * x["y"]).sum() / mF.sum() for x in L]); RP = np.mean([(mP * x["y"]).sum() / mP.sum() for x in L])
        line = f"  {k[0]:12s} credit .{k[1]:3s} runs {len(L)} abst {np.mean([x['a'] for x in L]):.2f} | refuses F {RF:.3f} P {RP:.3f} gap {est[k][0]:+.3f} [{ci(0)[0]:+.3f}, {ci(0)[1]:+.3f}]"
        if k[0] != KEYG: line += f" | DiD vs key at matched abstention {est[k][1]:+.3f} [{ci(1)[0]:+.3f}, {ci(1)[1]:+.3f}] | vs key at the same credit {est[k][2]:+.3f} [{ci(2)[0]:+.3f}, {ci(2)[1]:+.3f}]"
        out.append(line)
open(f"{_RA}/fam_fix/e7_capitals_{PRE}_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
