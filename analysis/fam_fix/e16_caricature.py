"""(fam_fix) e16_caricature (descriptive; existing runs only): does a trained model imitate its judge's verdicts on unseen questions, or exaggerate
them by kind? Capitals probe, known answers (untrained greedy right). For each judge x cut x student: the judge's own verdicts on the untrained
greedy answers of the UNSEEN probe (accept / reject at the cut), and the trained models' refusal split by that verdict and by kind:
 F accepted by the judge -> refused by the model? (model stricter than its judge);  P rejected by the judge -> refused by the model? (model more
 lenient than its judge). Also the judge's rejection rate and the model's refusal rate per kind. Output: e16_caricature_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, glob, os, pickle, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
P2 = (_RR + "/run1/adeval/probe2"); RLD = (_RR + "/run1/rl"); out = [__doc__]; ROWS = []
TA = json.load(open((_RX + "/final/thresholds_acc.json")))
probs = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probs); cls = np.array([probs[i]["cls"] for i in ids])
for st in ("qwen7", "llama8"):
    g = {}; nr = {i: 0 for i in ids}
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        x = ans(r); ok = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probs[r["idx"]]["aliases"])))
        if r["s"] < 0: g[r["idx"]] = ok
        else: nr[r["idx"]] += ok
    known = np.array([g[i] for i in ids]); sure = known & np.array([nr[i] == 32 for i in ids]); F = known & (cls == "F"); Pm = known & (cls == "P")
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); new = np.asarray(Bp["new"], bool)
    for J in "QMGO":
        S = {r["idx"]: py_of(r) for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_pqq_1500_n32_s1.jsonl")) if r["s"] < 0}; pv = np.array([S[i] for i in ids])
        for cut in ("acc", "a50", "a35"):
            L = []
            for f in sorted(glob.glob(f"{P2}/{st}_FK20judge{J}{cut}_idkc05_s*_pq.jsonl")):
                run = os.path.basename(f)[:-6]; ef = f"{RLD}/{run}/eval_rl_ck120.jsonl"
                if not os.path.exists(ef): continue
                o = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(f)) if r["s"] < 0}
                idk = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(ef)) if r["s"] < 0}
                if np.mean([idk[i] for i in range(len(new)) if new[i] and i in idk]) >= 0.95: continue
                L.append(np.array([o[i] for i in ids], float))
            if not L: continue
            Y = np.mean(L, axis=0); t = TA[J][f"{st}_pqa"]["t_acc"] if cut == "acc" else thresholds(J)[f"{st}_pqa"]["t_" + cut]; acc = pv > t
            m = lambda M: (float(Y[M].mean()) if M.sum() else float("nan"), int(M.sum()))
            row = dict(st=st, J=J, cut=cut, jrejF=float((~acc)[F].mean()), jrejP=float((~acc)[Pm].mean()), mF=float(Y[F].mean()), mP=float(Y[Pm].mean()),
                       F_acc=m(F & acc), F_rej=m(F & ~acc), P_acc=m(Pm & acc), P_rej=m(Pm & ~acc), sureF=m(F & sure), sureF_acc=m(F & sure & acc)); ROWS.append(row)
            out.append(f"  {st:7s} {J} {cut:3s}: former-state: judge rejects {row['jrejF']:.2f}, model refuses {row['mF']:.2f} | of the {row['F_acc'][1]} the judge ACCEPTS the model refuses {row['F_acc'][0]:.2f}; of the {row['sureF_acc'][1]} given 32/32 and accepted {row['sureF_acc'][0]:.2f} || present-day: judge rejects {row['jrejP']:.2f}, model refuses {row['mP']:.2f} | of the {row['P_rej'][1]} the judge REJECTS the model refuses {row['P_rej'][0]:.2f}")
json.dump(ROWS, open("e16_caricature.json", "w"), indent=1)
open("e16_caricature_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
