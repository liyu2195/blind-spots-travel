"""(supp_final) s1_judge_knows: does a judge reject right answers because it cannot answer the question itself?
Data: TriviaQA. The Qwen3.5-9B judge's own closed-book greedy answers to the 1,500 questions (st/res/selfans/self_q359_tqa_1500.jsonl), scored by
the loose key or the Sonnet audit as every answer in the paper. Units: questions whose untrained STUDENT greedy answer is valid, not IDK and right
(loose key or audit >= 4), i.e. the right answers the judge grades. For each student, cut (default P(Yes) > 0.5, 50%, 35%) and question set
(training / all 1,500): among right answers the Qwen judge REJECTS, the share of questions the judge itself answers right; the same among right
answers it ACCEPTS; and the judge's closed-book accuracy on all questions. Also the same with the older file st/data/qwen_knows_h1.jsonl
(training questions only), if its indices match.
Usage: python3 -I -B s1_judge_knows.py   Output: s1_judge_knows_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, os, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
out = [__doc__]
SELF = (_RR + "/selfans/self_q359_tqa_1500.jsonl")
for st in ("qwen7", "llama8"):
    Bp = pickle.load(open(f"{CACHE}/base_{st}_h1.pkl", "rb"))
    g = lambda a: (np.asarray(a)[:, 0] if np.asarray(a).ndim == 2 else np.asarray(a))      # column 0 = the greedy answer
    N = Bp["N"]; valid = g(Bp["valid"]).astype(bool); idk = g(Bp["idk"]).astype(bool)
    right = valid & ~idk & (g(Bp["kl"]).astype(bool) | (g(Bp["son"]).astype(float) >= 4))
    train = np.asarray(Bp["train"], bool)
    PQ = g(Bp["PJ"]["Q"]).astype(float)
    th = thresholds("Q").get(f"{st}_tqa", {})
    # the judge's own answers, scored like any answer: loose key or audit
    LV = {}                                   # audit levels of (question, normalised answer) pairs, from both students' audit files
    for s2 in ("qwen7", "llama8"):
        for src in ("frontierH", "frontierA"):
            f = f"{API}/{src}_{s2}_tqa.jsonl"
            if os.path.exists(f):
                for r in map(json.loads, open(f)): LV.setdefault((r["idx"], r["norm"]), r["level"])
    own = {}; ownk = {}; unaud = 0; wrongk = 0
    for r in map(json.loads, open(SELF)):
        if r.get("s", -1) < 0:
            x = ans(r); i = r["idx"]
            v = bool(x) and not IDKF.match(x)
            k = v and bool(C.tq_correct(x, set(Bp["aliases"][i])))
            ownk[i] = k
            if v and not k:
                wrongk += 1; unaud += (i, x) not in LV
            own[i] = k or (v and LV.get((i, x), 0) >= 4)
    jk = np.array([own.get(i, False) for i in range(N)]); jkk = np.array([ownk.get(i, False) for i in range(N)])
    have = np.array([i in own for i in range(N)])
    out.append(f"\n== {st}: right untrained answers {int(right.sum())} of {N} (training {int((right & train).sum())}); judge's own answers found for "
               f"{int(have.sum())} questions; judge closed-book accuracy by the key {jkk[have].mean():.3f}, by key or audit {jk[have].mean():.3f} "
               f"({unaud} of its {wrongk} key-wrong answers have no audit verdict and count as wrong)")
    for cut, t in (("default", 0.5), ("50% cut", th.get("t_a50")), ("35% cut", th.get("t_a35"))):
        if t is None:
            continue
        for name, m in (("training", train), ("all 1,500", np.ones(N, bool))):
            rej = right & m & have & (PQ <= t); acc = right & m & have & (PQ > t)
            if rej.sum() == 0:
                continue
            out.append(f"   {cut:8s} (P(Yes) > {t:.3f}) {name:9s}: judge rejects {int(rej.sum())} right answers; it answers the question right itself on "
                       f"{jk[rej].mean():.3f} of them (by the key alone {jkk[rej].mean():.3f}) | on {jk[acc].mean():.3f} of the {int(acc.sum())} it accepts")
    f = (_RD + "/qwen_knows_h1.jsonl")
    if os.path.exists(f):
        kn = {r["idx"]: bool(r["knows"]) for r in map(json.loads, open(f))}
        k2 = np.array([kn.get(i, False) for i in range(N)]); h2 = np.array([i in kn for i in range(N)])
        out.append(f"   older file qwen_knows_h1.jsonl: {int(h2.sum())} questions, of which training {int((h2 & train).sum())}; share known {k2[h2].mean():.3f}; "
                   f"agreement with the key-scored own answers {np.mean(k2[h2] == jk[h2]):.3f}")
        for cut, t in (("default", 0.5), ("50% cut", th.get("t_a50")), ("35% cut", th.get("t_a35"))):
            if t is None:
                continue
            rej = right & h2 & (PQ <= t)
            if rej.sum():
                out.append(f"   {cut:8s} with the older file: rejects {int(rej.sum())}; knows {k2[rej].mean():.3f}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s1_judge_knows_out.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out[1:]))
