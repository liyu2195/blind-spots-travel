"""(supp_final) s6_jury_offline (no runs): the five judges and their majority as CLASSIFIERS of the untrained student's answers.
Pool: the student's 32 sampled answers (valid, not IDK) to the PopQA TRAINING questions; right = loose key or audit level >= 4.
1. AUROC of each judge's P(Yes) for right vs wrong, and of the jury score (mean of the five judges' rank-normalised P(Yes)).
2. Single judges at their 50% and 35% cuts: share of all answers accepted, of right answers rejected, of wrong answers paid.
3. The 3-of-5 majority, every judge cut at the same acceptance quantile q, with q chosen so that the majority
   (a) accepts 35% of the answers (the jury trained in the paper), (b) pays as many wrong answers as the single judges do on average at their 35% cut,
   (c) pays as many wrong answers as the single judges do on average at their 50% cut.
   For each: acceptance, right rejected, wrong paid, and how many of the twelve known former-state capital training answers (untrained greedy
   answers, as fam_fix/e15_count_dose.py) the majority rejects, next to the single judges' counts.
Usage: python3 -I -B s6_jury_offline.py   Output: s6_jury_offline_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, numpy as np
from scipy.stats import rankdata
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
FORMER = {"Sultanate of Rum", "Kingdom of Mysore", "Confederate States of America", "Commonwealth of the Philippines", "Tokugawa Shogunate",
          "Hyderabad State", "Spanish Empire", "Kievan Rus'", "Kingdom of Jerusalem", "Austrian Empire", "Roman Republic",
          "Republic of China 1912–1949", "Lodhi dynasty", "Kingdom of France", "Czechoslovakia", "British India", "occupation of Japan"}
out = [__doc__]; JS = "QMGOR"
def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
pq = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqa_1500.jsonl"))}
for st in ("qwen7", "llama8"):
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); tr = np.asarray(Bp["train"], bool)
    val = np.asarray(Bp["valid"]).astype(bool) & ~np.asarray(Bp["idk"]).astype(bool)
    right = val & (np.asarray(Bp["kl"]).astype(bool) | (np.asarray(Bp["son"]) >= 4))
    P = {J: np.asarray(Bp["PJ"][J], float) for J in JS}
    m = val.copy(); m[:, 0] = False; m[~tr] = False                      # the 32 samples of training questions
    ok = m & np.all([~np.isnan(P[J]) for J in JS], axis=0)
    y = right[ok]; S = {J: P[J][ok] for J in JS}
    # the twelve known former-state capital training answers (greedy, column 0)
    name = lambda i: pq[i]["question"].replace("What is the capital of ", "").rstrip("?")
    F = [i for i in np.where(tr)[0] if pq[i]["question"].startswith("What is the capital of ") and name(i) in FORMER and val[i, 0] and bool(Bp["kl"][i][0])]
    out.append(f"\n== {st}: pool {int(ok.sum())} answers, right {y.mean():.3f}; known former-state training answers {len(F)}")
    jury = np.mean([rankdata(S[J]) / len(y) for J in JS], axis=0)
    out.append("   AUROC: " + "  ".join(f"{J} {auc(S[J], y):.3f}" for J in JS) + f"  | jury (mean rank) {auc(jury, y):.3f}")
    def at(q):      # every judge accepts its top share q; majority of five
        thr = {J: np.quantile(S[J], 1 - q) for J in JS}
        votes = np.sum([S[J] > thr[J] for J in JS], axis=0); acc = votes >= 3
        fr = [i for i in F if sum(P[J][i, 0] > thr[J] for J in JS) < 3]
        return acc.mean(), (~acc[y]).mean(), acc[~y].mean(), len(fr)
    single = {}
    for cut in ("a50", "a35"):
        rows = []
        for J in JS:
            t = thresholds(J)[f"{st}_pqa"]["t_" + cut]; a = S[J] > t
            rows.append((a.mean(), (~a[y]).mean(), a[~y].mean(), sum(P[J][i, 0] <= t for i in F)))
        single[cut] = rows
        out.append(f"   single judges at the {cut[1:]}% cut: accept " + " ".join(f"{r[0]:.2f}" for r in rows) + " | right rejected " + " ".join(f"{r[1]:.2f}" for r in rows)
                   + " | wrong paid " + " ".join(f"{r[2]:.2f}" for r in rows) + f" | of the {len(F)} former-state answers rejected " + " ".join(str(r[3]) for r in rows) + f"  (order {JS})")
    grid = np.linspace(0.05, 0.95, 181); G = [at(q) for q in grid]
    def pick(col, target):
        j = int(np.argmin([abs(g[col] - target) for g in G])); return grid[j], G[j]
    for label, col, target in (("accepts 35% of answers", 0, 0.35), ("pays wrong answers like the single judges' 35% cut", 2, np.mean([r[2] for r in single["a35"]])),
                               ("pays wrong answers like the single judges' 50% cut", 2, np.mean([r[2] for r in single["a50"]]))):
        q, g = pick(col, target)
        out.append(f"   majority of five, {label} (each judge accepts its top {q:.2f}): accepts {g[0]:.2f}, right rejected {g[1]:.2f}, wrong paid {g[2]:.3f}, "
                   f"former-state answers rejected {g[3]} of {len(F)}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s6_jury_offline_out.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out[1:]))
