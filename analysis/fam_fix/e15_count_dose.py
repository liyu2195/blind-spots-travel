"""(fam_fix) e15_count_dose (descriptive; existing runs only): for every grader whose verdicts on the training capital questions are known, the number
of the 12 KNOWN former-state training answers (untrained greedy answer key-right) it rejects, the share of the other known capital training answers
it rejects, and what its models do on the never-trained probe: refusal of known former-state (R_F) and present-day (R_P) capitals, and the mean R_F,
R_P of the 3 runs trained with the key plus random rejections nearest in PopQA new-question abstention (judge-free, matched).
Graders: judges Q M G O at each cut (P(Yes) of the untrained greedy answer); frozen-key graders (key file aliases: copies of a judge's verdicts,
scrambled copy, type-wide copy, majority of five judges, own vote). Output: e15_count_dose_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, re, json, glob, os, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
FORMER = {"Sultanate of Rum", "Kingdom of Mysore", "Confederate States of America", "Commonwealth of the Philippines", "Tokugawa Shogunate",
          "Hyderabad State", "Spanish Empire", "Kievan Rus'", "Kingdom of Jerusalem", "Austrian Empire", "Roman Republic",
          "Republic of China 1912–1949", "Lodhi dynasty", "Kingdom of France", "Czechoslovakia", "British India", "occupation of Japan"}
P2 = (_RR + "/run1/adeval/probe2"); RLD = (_RR + "/run1/rl"); out = [__doc__]; ROWS = []
TA = json.load(open((_RX + "/final/thresholds_acc.json")))
pq = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqa_1500.jsonl"))}; h1 = [json.loads(l) for l in open(f"{DATA}/pqa_h1.jsonl")]; TR = [r["idx"] for r in h1]
probe = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probe); cls = np.array([probe[i]["cls"] for i in ids])
for st in ("qwen7", "llama8"):
    greedy = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqa_1500_n32_s1.jsonl")):
        if r["s"] < 0: greedy[r["idx"]] = ans(r)
    known = {i: bool(greedy.get(i)) and not IDKF.match(greedy[i]) and bool(C.tq_correct(greedy[i], set(pq[i]["aliases"]))) for i in TR}
    cap = [i for i in TR if pq[i]["question"].startswith("What is the capital of ")]
    F = [i for i in cap if pq[i]["question"].replace("What is the capital of ", "").rstrip("?") in FORMER and known[i]]; O = [i for i in cap if i not in F and known[i] and pq[i]["question"].replace("What is the capital of ", "").rstrip("?") not in FORMER]
    g = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
        if r["s"] < 0:
            x = ans(r); g[r["idx"]] = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probe[r["idx"]]["aliases"])))
    kn = np.array([g[i] for i in ids]); mF = kn & (cls == "F"); mP = kn & (cls == "P")
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); new = np.asarray(Bp["new"], bool); runs = defaultdict(list)
    for f in sorted(glob.glob(f"{P2}/{st}_FK20*_idkc05_s*_pq.jsonl")):
        run = os.path.basename(f)[:-6]; gname = re.match(rf"^{st}_FK20(.+)_idkc05_s(\d+)_pq$", run).group(1); ef = f"{RLD}/{run}/eval_rl_ck120.jsonl"
        if not os.path.exists(ef): continue
        o = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(f)) if r["s"] < 0}
        idk = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(ef)) if r["s"] < 0}
        a = float(np.mean([idk[i] for i in range(len(new)) if new[i] and i in idk]))
        if a >= 0.95: continue
        y = np.array([o[i] for i in ids], float); runs[gname].append((a, y[mF].mean(), y[mP].mean()))
    ctrl = [x for k, v in runs.items() if re.fullmatch(r"sim00fn\d\d", k) for x in v]
    def model(gname):
        L = runs.get(gname)
        if not L: return None
        near = [np.mean([[c[1], c[2]] for c in sorted(ctrl, key=lambda c: abs(c[0] - x[0]))[:3]], axis=0) for x in L]; near = np.mean(near, axis=0)
        return len(L), np.mean([x[0] for x in L]), np.mean([x[1] for x in L]), np.mean([x[2] for x in L]), near[0], near[1]
    def add(label, gname, rejF, rejO):
        m = model(gname)
        if m is None: return
        ROWS.append(dict(st=st, label=label, kF=rejF, nF=len(F), rO=rejO / len(O), runs=m[0], abst=m[1], RF=m[2], RP=m[3], cF=m[4], cP=m[5]))
    for J in "QMGO":
        PJ = {r["idx"]: py_of(r) for r in map(json.loads, open(f"{BASE}/score_{J}_{st}_pqa_1500_n32_s1.jsonl")) if r["s"] < 0}; TH = thresholds(J)
        for cut in ("t50", "acc", "a50", "a35"):
            t = 0.5 if cut == "t50" else (TA[J][f"{st}_pqa"]["t_acc"] if cut == "acc" else TH[f"{st}_pqa"]["t_" + cut])
            add(f"judge {J} {cut}", f"judge{J}{cut}", sum(PJ[i] <= t for i in F), sum(PJ[i] <= t for i in O))
    for gname in ("plrealM", "plrealO", "plscrM", "plsharpM", "plsharpO", "frzMAJ35", "frzQ35", "frzVOTE20", "frzVOTE21", "frzVOTE29", "frzVOTE32"):
        kf = f"{DATA}/pqa_h1_{gname}_{st}.jsonl"
        if not os.path.exists(kf): continue
        K = {r["idx"]: set(r["aliases"]) for r in map(json.loads, open(kf))}
        rej = lambda i: not C.tq_correct(greedy[i], K[i])
        add(f"frozen key {gname}", gname, sum(rej(i) for i in F), sum(rej(i) for i in O))
    add("answer key", "sim00fn00", 0, 0)
for r in sorted(ROWS, key=lambda r: (r["st"], r["kF"])):
    out.append(f"  {r['st']:7s} {r['label']:24s}: rejects {r['kF']:2d} of {r['nF']} known former-state training answers, {r['rO']:.2f} of other capitals | runs {r['runs']} abstention {r['abst']:.2f} | models refuse F {r['RF']:.2f} P {r['RP']:.2f} | matched judge-free models F {r['cF']:.2f} P {r['cP']:.2f}")
out.append("")
for lo, hi in ((0, 2), (3, 6), (7, 12)):
    S = [r for r in ROWS if lo <= r["kF"] <= hi and not r["label"].startswith("answer key")]
    if S: out.append(f"  graders rejecting {lo}-{hi} of the 12: cells {len(S)}; models refuse F {min(r['RF'] for r in S):.2f}-{max(r['RF'] for r in S):.2f} (median {np.median([r['RF'] for r in S]):.2f}); gap beyond matched judge-free models {min((r['RF'] - r['RP']) - (r['cF'] - r['cP']) for r in S):+.2f} to {max((r['RF'] - r['RP']) - (r['cF'] - r['cP']) for r in S):+.2f} (median {np.median([(r['RF'] - r['RP']) - (r['cF'] - r['cP']) for r in S]):+.2f})")
json.dump(ROWS, open("e15_count_dose.json", "w"), indent=1)
open("e15_count_dose_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
