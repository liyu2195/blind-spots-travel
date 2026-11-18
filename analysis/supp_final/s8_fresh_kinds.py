"""(supp_final) s8_fresh_kinds: FROZEN scorer for the fresh-question test (written before any model output on these questions exists).
Question: does the pre-training audit by kind predict which kinds a judge's models give up on questions nobody has seen?
Fresh questions: st/data/pqx_fresh.jsonl = the 2,569 questions of popqa_3000.jsonl that are not among our 1,500 PopQA questions (by text and id).
Kinds = PopQA relation types (the question template).
AUDIT (training questions only): among training questions whose untrained greedy answer is right (loose key) and sure (at least 31 of the 32
  untrained samples right by the loose key), the share each judge rejects at the cut; mean over the four judges Q M G O.
OUTCOME (fresh questions): among fresh questions whose untrained greedy answer is right (loose key) and sure (same rule; samples from
  run1/base/<student>_pqx_fresh_n32_s1.jsonl), the share refused (greedy answer is "I don't know") by saved models (run1/adeval/fresh/<run>.jsonl):
  the judge's models at the cut; models trained with the key plus random rejections (per judge run the three sim00fnXX runs nearest in abstention
  on the standard 750 held-out questions); key-trained models; models trained with the student's own unanimous vote.
Runs with abstention >= 0.95 on the standard held-out questions are dropped. Kinds enter with >= 5 sure right training answers and >= 10 sure
right fresh answers.
LINES:
  A1: at the 35% cut, Spearman rank correlation over kinds between the audit and the refusal by the judges' models (both averaged over the four
      judges) is >= +0.60 in each student.
  A2: kind 'country', Llama, 35% cut: refusal by the judges' models minus refusal by the matched random-rejection models, averaged over the four
      judges, is >= +0.25 with its 95% interval (bootstrap over fresh questions, 2000) above 0. Qwen: reported, no line.
  A3: at the cut where the student earns most under the training reward (Llama: key-matched 'acc'; Qwen: 50% 'a50'), the same rank correlation as A1
      is >= +0.50.
Usage: python3 -I -B s8_fresh_kinds.py   Output: s8_fresh_kinds_out.txt, s8_fresh_kinds.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, glob, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
FR = (_RR + "/run1/adeval/fresh"); out = [__doc__]; ROWS = []; rng = np.random.default_rng(3737)
TA = json.load(open((_RX + "/final/thresholds_acc.json")))
fresh = [json.loads(l) for l in open(f"{DATA}/pqx_fresh.jsonl")]; NF = len(fresh); fprop = np.array([p["prop"] for p in fresh])
for st in ("llama8", "qwen7"):
    f = f"{BASE}/{st}_pqx_fresh_n32_s1.jsonl"
    if not os.path.exists(f): out.append(f"\n== {st}: untrained samples on the fresh questions missing"); continue
    gr = np.zeros(NF, bool); nr = np.zeros(NF, int)
    for r in map(json.loads, open(f)):
        x = ans(r); ok = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(fresh[r["idx"]]["aliases"])))
        if r["s"] < 0: gr[r["idx"]] = ok
        else: nr[r["idx"]] += ok
    fsure = gr & (nr >= 31)
    Bp = pickle.load(open(f"{CACHE}/base_{st}_pq.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_pq.pkl", "rb"))
    tr = np.asarray(Bp["train"], bool); new = np.asarray(Bp["new"], bool); kl = np.asarray(Bp["kl"]).astype(bool)
    val0 = np.asarray(Bp["valid"])[:, 0].astype(bool) & ~np.asarray(Bp["idk"])[:, 0].astype(bool)
    tsure = tr & val0 & kl[:, 0] & (kl[:, 1:].sum(1) >= 31); prop = np.array([str(x) for x in Bp["prop"]])
    P = {J: np.asarray(Bp["PJ"][J])[:, 0].astype(float) for J in "QMGO"}
    runs = defaultdict(list); nofile = 0
    for name, r in R.items():
        if r["tag"] != "FK20" or r["c"] != "05": continue
        a = float((np.asarray(r["o"]) == 0)[new].mean())
        if a >= 0.95: continue
        ff = f"{FR}/{name}.jsonl"
        if not os.path.exists(ff): nofile += 1; continue
        y = np.zeros(NF, bool)
        for q in map(json.loads, open(ff)):
            if q["s"] < 0: x = ans(q); y[q["idx"]] = bool(x) and bool(IDKF.match(x))
        runs[r["g"]].append((a, y))
    ctrl = [x for k, v in runs.items() if re.fullmatch(r"sim00fn\d\d", k) and k != "sim00fn00" for x in v]
    kinds = [p for p in sorted(set(fprop)) if (tsure & (prop == p)).sum() >= 5 and (fsure & (fprop == p)).sum() >= 10]
    out.append(f"\n==== {st}: fresh questions {NF}, sure right {int(fsure.sum())}; training sure right {int(tsure.sum())}; kinds {len(kinds)}; saved models with "
               f"fresh answers {sum(len(v) for v in runs.values())} (without {nofile}); random-rejection runs {len(ctrl)}")
    for cut in ("acc", "a50", "a35"):
        agg = defaultdict(list); nearJ = {}
        for J in "QMGO":
            arm = f"judge{J}{cut}"
            if arm not in runs or not ctrl: continue
            t = TA[J][f"{st}_pqa"]["t_acc"] if cut == "acc" else thresholds(J)[f"{st}_pqa"]["t_" + cut]
            L = runs[arm]; near = [c for x in L for c in sorted(ctrl, key=lambda c: abs(c[0] - x[0]))[:3]]; nearJ[J] = (L, near)
            for p in kinds:
                mt = tsure & (prop == p); mn = fsure & (fprop == p)
                row = dict(st=st, cut=cut, J=J, kind=p, n_train=int(mt.sum()), n_fresh=int(mn.sum()), audit=float((P[J][mt] <= t).mean()),
                           judge=float(np.mean([x[1][mn].mean() for x in L])), free=float(np.mean([c[1][mn].mean() for c in near])),
                           key=float(np.mean([x[1][mn].mean() for x in runs["sim00fn00"]])) if "sim00fn00" in runs else float("nan"),
                           vote=float(np.mean([x[1][mn].mean() for x in runs["frzVOTE32"]])) if "frzVOTE32" in runs else float("nan"),
                           runs=len(L), abst=float(np.mean([x[0] for x in L])))
                ROWS.append(row); agg[p].append(row)
        if not agg: continue
        out.append(f"\n== {st}, cut {cut} ({len(nearJ)} judges averaged): kind | sure right train / fresh | AUDIT rejected in training | FRESH refused by: judge's models, "
                   f"random rejections (matched), key, own unanimous vote | excess")
        tab = sorted(((np.mean([r['audit'] for r in v]), p, v) for p, v in agg.items()), reverse=True)
        for a, p, v in tab:
            m = lambda k: np.mean([r[k] for r in v])
            out.append(f"   {p:14s} {v[0]['n_train']:3d} / {v[0]['n_fresh']:3d} | audit {a:.2f} | {m('judge'):.2f}  {m('free'):.2f}  {m('key'):.2f}  {m('vote'):.2f} | {m('judge') - m('free'):+.2f}")
        rho = spearmanr([t[0] for t in tab], [np.mean([r['judge'] for r in t[2]]) for t in tab]).correlation
        rex = spearmanr([t[0] for t in tab], [np.mean([r['judge'] - r['free'] for r in t[2]]) for t in tab]).correlation
        out.append(f"   rank correlation over {len(tab)} kinds: audit vs refusal by the judges' models {rho:+.2f}; audit vs excess over random rejections {rex:+.2f}")
        if cut == "a35": out.append(f"   LINE A1 ({st}): {rho:+.2f} >= +0.60 -> {'inside' if rho >= 0.60 else 'outside'}")
        if (st, cut) in (("llama8", "acc"), ("qwen7", "a50")): out.append(f"   LINE A3 ({st}): {rho:+.2f} >= +0.50 -> {'inside' if rho >= 0.50 else 'outside'}")
        if cut == "a35" and "country" in agg:
            mn = np.where(fsure & (fprop == "country"))[0]
            def exc(w):
                return np.mean([np.mean([(x[1][mn] * w).sum() / w.sum() for x in L]) - np.mean([(c[1][mn] * w).sum() / w.sum() for c in near]) for L, near in nearJ.values()])
            est = exc(np.ones(len(mn))); bs = [exc(rng.multinomial(len(mn), np.ones(len(mn)) / len(mn)).astype(float)) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            per = ", ".join(f"{J} {np.mean([x[1][mn].mean() for x in L]) - np.mean([c[1][mn].mean() for c in near]):+.2f}" for J, (L, near) in nearJ.items())
            out.append(f"   country ({len(mn)} sure right fresh answers): excess of the judges' models over matched random rejections {est:+.3f} [{lo:+.3f}, {hi:+.3f}]; per judge: {per}")
            if st == "llama8": out.append(f"   LINE A2 (llama8): {est:+.3f} >= +0.25 and interval above 0 -> {'inside' if est >= 0.25 and lo > 0 else 'outside'}")
H = os.path.dirname(os.path.abspath(__file__))
json.dump(ROWS, open(f"{H}/s8_fresh_kinds.json", "w"), indent=1)
open(f"{H}/s8_fresh_kinds_out.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out[1:]))
