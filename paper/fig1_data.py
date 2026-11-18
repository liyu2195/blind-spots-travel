"""Data for Figure 1 of v13.1 (Nov 18): per grader and student, on PopQA: share of the known former-state / present-day probe capitals refused (mean
over runs), held-out abstention, and the same refusal for students trained with the answer key plus random rejections at the same abstention (per
run: the three such runs nearest in held-out abstention, as in fam_fix/e15_count_dose.py; then the mean over the grader's runs).
Runs: FK20, credit 0.5, PopQA, not collapsed, with a capitals-probe result. Writes fig1_data.json. Usage: python3 -I -B fig1_data.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, re, json, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P2 = (_RR + "/run1/adeval/probe2"); J = {}
probe = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probe); pc = np.array([probe[i]["cls"] for i in ids])
for st in ("qwen7", "llama8"):
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_pq.pkl", "rb")); new = np.asarray(Bp["new"], bool); gk = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        if r["s"] < 0: x = ans(r); gk[r["idx"]] = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probe[r["idx"]]["aliases"])))
    kn = np.array([gk[i] for i in ids]); mF = kn & (pc == "F"); mP = kn & (pc == "P"); runs = defaultdict(list)
    for name, r in R.items():
        if r["tag"] != "FK20" or r["c"] != "05": continue
        a = float((np.asarray(r["o"]) == 0)[new].mean()); f = f"{P2}/{name}.jsonl"
        if a >= 0.95 or not os.path.exists(f): continue
        o = {q["idx"]: bool(ans(q)) and bool(IDKF.match(ans(q))) for q in map(json.loads, open(f)) if q["s"] < 0}; y = np.array([o[i] for i in ids], float); runs[r["g"]].append((a, float(y[mF].mean()), float(y[mP].mean())))
    ctrl = [x for g, v in runs.items() if re.fullmatch(r"sim00fn\d\d", g) for x in v]
    near = lambda a: float(np.mean([c[1] for c in sorted(ctrl, key=lambda c: abs(c[0] - a))[:3]]))
    J[st] = {g: dict(runs=len(v), abst=float(np.mean([x[0] for x in v])), F=float(np.mean([x[1] for x in v])), P=float(np.mean([x[2] for x in v])), ctrlF=float(np.mean([near(x[0]) for x in v])),
                     Fmin=float(min(x[1] for x in v)), Fmax=float(max(x[1] for x in v))) for g, v in runs.items()}
    J[st]["_known"] = dict(F=int(mF.sum()), P=int(mP.sum()), ctrl_runs=len(ctrl))
json.dump(J, open("fig1_data.json", "w"), indent=1)
for st in J:
    print(st, J[st]["_known"])
    for g in ("sim00fn00", "apiS55", "judgeQRt50", "frzVOTE20", "judgeGacc", "judgeOacc", "judgeMacc", "judgeQacc", "judgeGa50", "judgeOa50", "judgeMa50", "judgeQa50", "judgeOa35", "judgeGa35", "judgeMa35", "judgeQa35", "frzMAJ35"):
        if g in J[st]: d = J[st][g]; print(f"   {g:11s} runs {d['runs']} abstention {d['abst']:.2f} former {d['F']:.2f} present {d['P']:.2f} | random rejections at the same abstention {d['ctrlF']:.2f}")
