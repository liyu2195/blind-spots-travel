"""planted copies of real judges' false rejects on PopQA, before any run. Writes st/data/pqa_knows_<arm>_<student>.jsonl
(750 trained questions, knows = true -> right answers rejected) and final/scramble_sets.json (sizes, overlaps, predictability). Read-only on all scores.
Per student s and judge J (M = Mistral-Nemo, O = OLMo-2-13B), on the 750 TRAINED PopQA questions:
  right answer of question i = the untrained greedy answer if right (loose key, or Sonnet level >= 4), else the most common right sampled answer (32 samples), else none.
  REAL_J  : i is in the set if J's P(Yes) for that right answer is at or below J's 35%-target threshold (J rejects a right answer of i).
  SCR_J   : the same number of questions, assigned at random (hash) among the questions that have a right answer: same size, no relation to the question.
  SHARP_J : whole relation types ("prop"): types are added in order of REAL_J's rejection rate among their right-answer questions (highest first) until the set
            is at least as large as REAL_J: same size (or just above), as predictable from the question as a set can be.
Predictability of a set: AUC of a logistic regression on TF-IDF of the question text (word 1-2-grams, min_df 2), 5-fold cross-validation over the right-answer
trained questions (fixed folds). Usage: python3 build_scramble.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, hashlib, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, (_RX + "/common"))
from rc import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
API = (_RR + "/run1/apitest"); FIN = (_RX + "/final"); dsn = "pqa"
def hu(*p): return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:12], 16) / float(16 ** 12)
probs, H1, HELD = setup(dsn); TR = sorted(H1); out = {}
def pred_cv(members, items):
    y = np.array([i in members for i in items], int)
    if y.min() == y.max(): return float("nan")
    X = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform([probs[i]["question"] for i in items])
    p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))
for st in ("qwen7", "llama8"):
    LV = {}
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
    RQ = sorted(rid); out[f"{st}|RQ"] = [int(i) for i in RQ]
    for J in ("M", "O"):
        thr = json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_{dsn}"]["t_a35"]; fp = json.load(open(f"{FIN}/thresholds_{J}.json"))[f"{st}_{dsn}"]["FP_a35"]
        PY = {(r["idx"], r["s"]): (r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5) for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_{dsn}_1500_n32_s1.jsonl")) if r["idx"] in H1}
        real = {i for i in RQ if not PY.get((i, rid[i][0]), 0.5) > thr}; k = len(real)
        scr = set(sorted(RQ, key=lambda i: hu("scr", J, st, i))[:k])
        rate = {}; bytype = defaultdict(list)
        for i in RQ: bytype[probs[i]["prop"]].append(i)
        for t, v in bytype.items(): rate[t] = np.mean([i in real for i in v])
        sharp = set(); used = []
        for t in sorted(bytype, key=lambda t: (-rate[t], t)):
            if len(sharp) >= k: break
            sharp |= set(bytype[t]); used.append(t)
        sets = {"real": real, "scr": scr, "sharp": sharp}
        for arm, S in sets.items():
            fn = f"pqa_knows_{arm}{J}_{st}.jsonl"
            with open(f"{_RD}/{fn}", "w") as g:
                for i in TR: g.write(json.dumps({"idx": i, "knows": i in S}) + "\n")
            out[f"{st}|{arm}{J}"] = dict(file=fn, size=len(S), right_q=len(RQ), fp=round(float(fp), 3), pred=pred_cv(S, RQ), overlap_real=len(S & real), types=used if arm == "sharp" else None)
            print(f"{st:6s} {arm:5s}{J}: set {len(S):3d} of {len(RQ)} right-answer trained questions (750 trained); overlap with REAL {len(S & real):3d}; predictability (5-fold AUC) {out[f'{st}|{arm}{J}']['pred']:.3f}; false accepts {fp:.3f}" + (f"; types {', '.join(used)}" if arm == "sharp" else ""))
json.dump(out, open(f"{FIN}/scramble_sets.json", "w"), indent=1)
