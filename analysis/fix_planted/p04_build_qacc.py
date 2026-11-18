"""(fix_planted) p04_build_qacc (Nov 18; before any run): the copy of the Qwen judge's rejections at the KEY-MATCHED cut for the Llama student on PopQA,
with the twelve known former-state capital training answers all accepted (K0) or all rejected (K12). Same construction as the copies
(final/build_scramble.py) and the hand-set graders, with the key-matched threshold instead of the 35% cut:
  right answer of a training question = the untrained greedy answer if right (loose key, or Sonnet level >= 4), else the most common right sampled
  answer, else none; the question is in the rejection set if the Qwen judge's P(Yes) for that right answer is at or below the threshold.
  K0  = that set without the twelve; K12 = that set with all twelve. The two differ on the twelve questions that the copy itself does not treat alike.
Training with these files: right answers earn 0 on the questions of the set and 1 elsewhere; wrong answers are paid at random at the judge's rate at
this cut (FP_acc); "I don't know" earns 0.5; no judge in the loop.
Writes st/data/pqa_knows_QaccK0_llama8.jsonl and pqa_knows_QaccK12_llama8.jsonl and prints the checks. Usage: python3 -I -B p04_build_qacc.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import *
API = (_RR + "/run1/apitest"); FIN = (_RX + "/final"); dsn = "pqa"; st = "llama8"; J = "Q"; out = [__doc__]
FORMER = {"Sultanate of Rum", "Kingdom of Mysore", "Confederate States of America", "Commonwealth of the Philippines", "Tokugawa Shogunate", "Hyderabad State", "Spanish Empire", "Kievan Rus'", "Kingdom of Jerusalem",
          "Austrian Empire", "Roman Republic", "Republic of China 1912–1949", "Lodhi dynasty", "Kingdom of France", "Czechoslovakia", "British India", "occupation of Japan"}
probs, H1, HELD = setup(dsn); TR = sorted(H1); LV = {}
for f in (f"{API}/frontierH_{st}_{dsn}.jsonl", f"{API}/auditT_{st}_{dsn}.jsonl"):
    if os.path.exists(f):
        for r in map(json.loads, open(f)): LV[(r["idx"], r["norm"])] = r["level"]
ok = lambda i, x: bool(x) and not IDK.match(x) and (C.tq_correct(x, set(probs[i]["aliases"])) or LV.get((i, x), 0) >= 4)
gr = {}; smp = defaultdict(list)
for r in map(json.loads, open(f"{BASE}/{st}_{dsn}_1500_n32_s1.jsonl")):
    if r["idx"] not in H1: continue
    x = sa(r)
    if r["s"] < 0: gr[r["idx"]] = (r["s"], x)
    elif ok(r["idx"], x): smp[r["idx"]].append((r["s"], x))
rid = {}
for i in TR:
    s, x = gr.get(i, (None, ""))
    if ok(i, x): rid[i] = (s, x)
    elif smp[i]:
        top = Counter(x for _, x in smp[i]).most_common(1)[0][0]; rid[i] = next((s, x) for s, x in smp[i] if x == top)
RQ = sorted(rid); stored = json.load(open(f"{FIN}/scramble_sets.json"))[f"{st}|RQ"]
out.append(f"right-answer training questions {len(RQ)} (the same list as for the copies: {sorted(stored) == RQ})")
T = json.load(open(f"{FIN}/thresholds_acc.json"))[J][f"{st}_{dsn}"]; thr = T["t_acc"]; fp = round(float(T["FP_acc"]), 3)
PY = {(r["idx"], r["s"]): (r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5) for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl")) if r["idx"] in H1}
real = {i for i in RQ if not PY.get((i, rid[i][0]), 0.5) > thr}
cap = [i for i in TR if probs[i]["question"].startswith("What is the capital of ")]; ent = lambda i: probs[i]["question"].replace("What is the capital of ", "").rstrip("?")
known = {i for i in TR if gr.get(i) and gr[i][1] and not IDK.match(gr[i][1]) and C.tq_correct(gr[i][1], set(probs[i]["aliases"]))}
F12 = sorted(i for i in cap if ent(i) in FORMER and i in known); O = sorted(i for i in cap if ent(i) not in FORMER and i in known); assert len(F12) == 12 and set(F12) <= set(RQ)
K0 = real - set(F12); K12 = real | set(F12)
out.append(f"threshold (key-matched cut) {thr}; wrong answers paid at random with probability {fp} (the judge's rate at this cut: accepts {T['A_acc']:.3f} of all answers, rejects {T['FN_acc']:.3f} of key-right answers)")
out.append(f"copy of the judge's rejections: {len(real)} of {len(RQ)} right-answer questions ({len(real) / len(RQ):.3f}); of the {len(known)} questions with a key-right greedy answer {len(real & known)} ({len(real & known) / len(known):.3f})")
out.append(f"the twelve: the copy itself rejects {len(real & set(F12))}: " + ", ".join(ent(i) for i in F12 if i in real) + " | accepts: " + ", ".join(ent(i) for i in F12 if i not in real))
out.append(f"other known capital training answers: {len(O)}, in the copy's set {len(real & set(O))} ({len(real & set(O)) / len(O):.3f}); the same in K0 and K12: {(K0 & set(O)) == (K12 & set(O))}")
out.append(f"K0: {len(K0)} questions, twelve in it {len(K0 & set(F12))}; K12: {len(K12)} questions, twelve in it {len(K12 & set(F12))}; K12 minus K0 = the twelve: {K12 - K0 == set(F12)}; K0 differs from the copy on {len(real ^ K0)}, K12 on {len(real ^ K12)} of 750 questions")
other_former = sorted(i for i in cap if ent(i) in FORMER and i not in known); out.append(f"former-state training questions the student does not answer right in its greedy answer: {len(other_former)} ({', '.join(ent(i) for i in other_former)}); in the copy's set: {len(real & set(other_former))} (left as the copy has them)")
for name, S in (("QaccK0", K0), ("QaccK12", K12)):
    fn = f"pqa_knows_{name}_{st}.jsonl"
    with open(f"{_RD}/{fn}", "w") as g:
        for i in TR: g.write(json.dumps({"idx": i, "knows": i in S}) + "\n")
    out.append(f"wrote st/data/{fn}: {len(TR)} rows, {len(S)} in the rejection set")
json.dump(dict(thr=thr, fp=fp, real=sorted(int(i) for i in real), K0=sorted(int(i) for i in K0), K12=sorted(int(i) for i in K12), F12=[int(i) for i in F12], rejected_of_12_by_copy=len(real & set(F12)), rO=len(real & set(O)) / len(O), rAll=len(real & known) / len(known)),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p04_build_qacc.json"), "w"), indent=1)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p04_build_qacc_out.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
