"""(fix_planted) p05_score_qacc: FROZEN scorer for the six runs (written before the runs start).
Runs llama8_FK20plQaccK{0,12}_idkc05_s{0,1,2}_pq: Llama trained on PopQA with the copy of the Qwen judge's rejections at the key-matched cut in which
the twelve known former-state capital training answers are all accepted (K0) or all rejected (K12). Nothing else differs between the two graders.
Per run: RF / RP = share of the probe's former-state / present-day capitals that the untrained student answers right (187 / 65) and the trained
student refuses (adeval probe2, the paper's rule); HF = the same on the 11 known former-state capitals among the regular held-out questions;
abstention, reward by the answer key (right 1, "I don't know" 0.5) and wrong answers per 100 on the 750 held-out PopQA questions.
LINES:
 Q1  mean RF at K12 minus mean RF at K0 is at least +0.40, and every K12 run is above every K0 run.
 Q2  mean RP at K12 is at most 0.10.
 Q3  reward by the answer key differs by at most 1.5 points and held-out abstention by at most 5 points between K12 and K0.
Usage: python3 -I -B p05_score_qacc.py   Output: p05_score_qacc_out.txt, p05_score_qacc.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
H = os.path.dirname(os.path.abspath(__file__)); P2 = (_RR + "/run1/adeval/probe2"); RLD = (_RR + "/run1/rl"); st = "llama8"; out = [__doc__]
FORMER_NEW = {"Ukrainian Soviet Socialist Republic", "Tang Empire", "Union of South Africa", "Russian Empire", "Soviet Union", "Kingdom of Italy", "German Democratic Republic", "United Kingdom of Great Britain and Ireland",
              "Kingdom of England", "Han dynasty", "Byzantine Empire", "Weimar Republic", "Empire of Japan", "Bourbon Restoration"}
probe = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probe); cls = np.array([probe[i]["cls"] for i in ids]); g = {}
for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
    if r["s"] < 0: x = ans(r); g[r["idx"]] = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probe[r["idx"]]["aliases"])))
kn = np.array([g[i] for i in ids]); mF = kn & (cls == "F"); mP = kn & (cls == "P"); assert mF.sum() == 187 and mP.sum() == 65
Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]; g0 = lambda a: np.asarray(a)[:, 0]
known = g0(Bp["valid"]).astype(bool) & ~g0(Bp["idk"]).astype(bool) & g0(Bp["kl"]).astype(bool)
ent = np.array([q.replace("What is the capital of ", "").rstrip("?") if q.startswith("What is the capital of ") else "" for q in Bp["question"]]); HFm = new & known & np.isin(ent, list(FORMER_NEW)); assert HFm.sum() == 11
R = {}
for k in (0, 12):
    R[k] = []
    for s in (0, 1, 2):
        run = f"{st}_FK20plQaccK{k}_idkc05_s{s}_pq"; ef = f"{RLD}/{run}/eval_rl_ck120.jsonl"; pf = f"{P2}/{run}.jsonl"
        if not (os.path.exists(ef) and os.path.exists(pf)): out.append(f"   {run}: missing ({'evaluation' if not os.path.exists(ef) else 'probe'})"); continue
        o = {r["idx"]: ans(r) for r in map(json.loads, open(pf)) if r["s"] < 0}; y = np.array([bool(o[i]) and bool(IDKF.match(o[i])) for i in ids], float)
        e = {r["idx"]: ans(r) for r in map(json.loads, open(ef)) if r["s"] < 0}; idk = np.array([bool(e[i]) and bool(IDKF.match(e[i])) for i in range(len(new))])
        right = np.array([bool(e[i]) and not IDKF.match(e[i]) and bool(C.tq_correct(e[i], set(Bp["aliases"][i]))) for i in range(len(new))])
        d = dict(run=run, RF=float(y[mF].mean()), RP=float(y[mP].mean()), HF=float(idk[HFm].mean()), abst=float(idk[new].mean()), reward=float(100 * (right[new].mean() + 0.5 * idk[new].mean())), wrong=float(100 * (~right & ~idk)[new].mean()))
        R[k].append(d); out.append(f"   {run}: probe former refused {d['RF']:.3f}, present-day {d['RP']:.3f} | held-out former {d['HF']:.2f} | abstention {d['abst']:.3f}, reward by the key {d['reward']:.1f}, wrong per 100 {d['wrong']:.1f}")
m = lambda k, f: float(np.mean([d[f] for d in R[k]])) if R[k] else float("nan"); JS = dict(runs=R)
if len(R[0]) == 3 and len(R[12]) == 3:
    d1 = m(12, "RF") - m(0, "RF"); sep = min(d["RF"] for d in R[12]) > max(d["RF"] for d in R[0])
    out.append(f"\n   K0: former refused {m(0, 'RF'):.3f} (runs {', '.join(f'{d['RF']:.2f}' for d in R[0])}), present-day {m(0, 'RP'):.3f}, abstention {m(0, 'abst'):.3f}, reward {m(0, 'reward'):.1f}")
    out.append(f"   K12: former refused {m(12, 'RF'):.3f} (runs {', '.join(f'{d['RF']:.2f}' for d in R[12])}), present-day {m(12, 'RP'):.3f}, abstention {m(12, 'abst'):.3f}, reward {m(12, 'reward'):.1f}")
    q1 = d1 >= 0.40 and sep; q2 = m(12, "RP") <= 0.10; q3 = abs(m(12, "reward") - m(0, "reward")) <= 1.5 and abs(m(12, "abst") - m(0, "abst")) <= 0.05
    out.append(f"   LINE Q1 (K12 minus K0 {d1:+.3f} >= +0.40, runs separated {sep}): {'inside' if q1 else 'OUTSIDE'}")
    out.append(f"   LINE Q2 (present-day refused at K12 {m(12, 'RP'):.3f} <= 0.10): {'inside' if q2 else 'OUTSIDE'}")
    out.append(f"   LINE Q3 (reward {m(12, 'reward') - m(0, 'reward'):+.2f} points, abstention {100 * (m(12, 'abst') - m(0, 'abst')):+.1f} points): {'inside' if q3 else 'OUTSIDE'}")
    JS.update(Q1=bool(q1), Q2=bool(q2), Q3=bool(q3), d_RF=d1, separated=bool(sep))
else: out.append("\n   not all six runs are there yet")
open(f"{H}/p05_score_qacc_out.txt", "w").write("\n".join(out) + "\n"); json.dump(JS, open(f"{H}/p05_score_qacc.json", "w"), indent=1); print("\n".join(out[1:]))
