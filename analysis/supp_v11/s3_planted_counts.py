"""(supp_v11) s3_planted_counts: the count-dose table (fam_fix/e15_count_dose.py) for the PLANTED graders, which e15 skips because their rejection
sets are stored as st/data/pqa_knows_<arm>_<student>.jsonl (knows = true: right answers to this training question are rejected) and not as key files.
Arms: realM / realO (copy of the Mistral / OLMo judge's rejections at the 35% cut), scrM (the same number of questions, scrambled), sharpM / sharpO
(type-wide copy: whole relation types, most-rejected first, until the set is as large).
For each arm and student, with exactly e15's definitions: of the 12 KNOWN former-state capital training answers, how many fall in the rejection set;
the share of the other known capital training answers in the set; the share of ALL known training answers in the set; and what the arm's models do on
the never-trained probe (known former-state and present-day capitals refused), with the three random-rejection models nearest in held-out abstention.
Also: how many of the 750 training questions use the probe's template "What is the capital of ...".
Usage: python3 -I -B s3_planted_counts.py   Output: s3_planted_counts_out.txt, s3_planted_counts.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, glob, os, re, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
FORMER = {"Sultanate of Rum", "Kingdom of Mysore", "Confederate States of America", "Commonwealth of the Philippines", "Tokugawa Shogunate",
          "Hyderabad State", "Spanish Empire", "Kievan Rus'", "Kingdom of Jerusalem", "Austrian Empire", "Roman Republic",
          "Republic of China 1912–1949", "Lodhi dynasty", "Kingdom of France", "Czechoslovakia", "British India", "occupation of Japan"}
P2 = (_RR + "/run1/adeval/probe2"); RLD = (_RR + "/run1/rl"); out = [__doc__]; ROWS = []
pq = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqa_1500.jsonl"))}; TR = [json.loads(l)["idx"] for l in open(f"{DATA}/pqa_h1.jsonl")]
probe = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}; ids = sorted(probe); cls = np.array([probe[i]["cls"] for i in ids])
name = lambda i: pq[i]["question"].replace("What is the capital of ", "").rstrip("?")
capall = [i for i in TR if pq[i]["question"].startswith("What is the capital of ")]
out.append(f"training questions {len(TR)}; with the template 'What is the capital of ...' {len(capall)}; of these about a former state (hand list) "
           f"{sum(name(i) in FORMER for i in capall)}")
for st in ("qwen7", "llama8"):
    greedy = {}
    for r in map(json.loads, open(f"{BASE}/{st}_pqa_1500_n32_s1.jsonl")):
        if r["s"] < 0: greedy[r["idx"]] = ans(r)
    known = {i: bool(greedy.get(i)) and not IDKF.match(greedy[i]) and bool(C.tq_correct(greedy[i], set(pq[i]["aliases"]))) for i in TR}
    F = [i for i in capall if name(i) in FORMER and known[i]]; O = [i for i in capall if name(i) not in FORMER and known[i]]; K = [i for i in TR if known[i]]
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
        y = np.array([o[i] for i in ids], float); runs[gname].append((a, y[mF].mean(), y[mP].mean()))
    ctrl = [x for k, v in runs.items() if re.fullmatch(r"sim00fn\d\d", k) for x in v if x[0] < 0.95]
    out.append(f"\n== {st}: known training answers {len(K)}; known former-state capital answers {len(F)}; other known capital answers {len(O)}")
    for arm in ("realM", "scrM", "sharpM", "realO", "sharpO"):
        kf = f"{DATA}/pqa_knows_{arm}_{st}.jsonl"
        if not os.path.exists(kf): continue
        S = {r["idx"] for r in map(json.loads, open(kf)) if r["knows"]}      # flag true = the question is in the rejection set (checked: the copy of Mistral then rejects the same 10 of 12 as the Mistral judge in e15)
        L = runs.get("pl" + arm, [])
        allr = len(L); L = [x for x in L if x[0] < 0.95]
        row = dict(st=st, arm=arm, set_size=len(S & set(TR)), kF=sum(i in S for i in F), nF=len(F), rO=float(np.mean([i in S for i in O])),
                   rAll=float(np.mean([i in S for i in K])), runs=len(L), dropped=allr - len(L))
        if L:
            near = np.mean([np.mean([[c[1], c[2]] for c in sorted(ctrl, key=lambda c: abs(c[0] - x[0]))[:3]], axis=0) for x in L], axis=0)
            row.update(abst=float(np.mean([x[0] for x in L])), RF=float(np.mean([x[1] for x in L])), RP=float(np.mean([x[2] for x in L])),
                       RFmin=float(min(x[1] for x in L)), RFmax=float(max(x[1] for x in L)), cF=float(near[0]), cP=float(near[1]))
            out.append(f"   {arm:7s}: set {row['set_size']} training questions | rejects {row['kF']:2d} of {row['nF']} known former-state answers, {row['rO']:.2f} of other "
                       f"known capitals, {row['rAll']:.2f} of all known training answers | runs {row['runs']} (dropped {row['dropped']}) abstention {row['abst']:.2f} | "
                       f"models refuse F {row['RF']:.2f} (runs {row['RFmin']:.2f}-{row['RFmax']:.2f}) P {row['RP']:.2f} | matched random-rejection models F {row['cF']:.2f} P {row['cP']:.2f}")
        ROWS.append(row)
H = os.path.dirname(os.path.abspath(__file__))
json.dump(ROWS, open(f"{H}/s3_planted_counts.json", "w"), indent=1)
open(f"{H}/s3_planted_counts_out.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out[1:]))
