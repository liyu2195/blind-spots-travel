"""(knowledge) CONFIRMATION of the pre-training forecast (k6) on run conditions not used in discovery: B1 (3,840 fresh TriviaQA training
questions, each seen once), FK00 (no KL), B2v / B5v / B10v (2 / 5 / 10 visits per question), LK20 / LK50 at step 360 (long runs), IDK credit
0.7 / 0.85, and FK20 runs of tags B2K20.  Probes are the k6 probes (fitted on the 750 standard trained questions to each grader's verdicts on the
untrained greedy answers; untrained hidden state); evaluation on the standard NEW questions.
Criteria fixed before this script was first run (2026-11-15):
  C1  own-grader probe forecast AUROC >= 0.85 in >= 80% of confirmation runs with 0.05 <= a <= 0.95
  C2  judge-trained confirmation runs: own-judge probe AUROC > key-probe AUROC in >= 75% of runs
  C3  key-trained confirmation runs: key probe >= Qwen-35% probe in >= 60% of runs
Own probe: sim00fnXX / simXXfn00 / hash -> key (random or hashed noise is not decodable); judgeQt50 -> Qt; judgeQa35 -> Q35.
Output: k8_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, numpy as np, warnings
from collections import defaultdict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
OUT = (_RA + "/knowledge")
def auc(y, s):
    y = np.asarray(y, bool); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
def own(g):
    if re.fullmatch(r"sim\d\dfn\d\d", g) or g.startswith("hash"): return "key"
    m = re.fullmatch(r"judge(Q|M|G|O|R)(a35|a50|t50)", g)
    if m: return {"a35": f"{m.group(1)}35", "a50": f"{m.group(1)}50", "t50": f"{m.group(1)}t"}[m.group(2)]
    return None
lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
res = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); S = pickle.load(open(f"{OUT}/probe6_{st}_{ds}.pkl", "rb"))
        ho = F["ho"]
        for name, r in sorted(R.items()):
            conf = r["tag"] != "FK20" or r["c"] != "05"
            if not conf: continue
            o = own(r["g"]); y = (r["o"] == 0)[ho]; a = y.mean()
            if o is None or not (0.05 <= a <= 0.95):
                P(f"  skip {name} (own probe {o}, a {a:.3f})"); continue
            d = dict(st=st, ds=ds, name=name, tag=r["tag"], g=r["g"], own=o, a=a, p_own=auc(y, -S[o][ho]), p_key=auc(y, -S["key"][ho]), p_Q35=auc(y, -S["Q35"][ho]),
                     p_sure=auc(y, -S["sure"][ho]), p_son=auc(y, -S["son"][ho]), p_conf=auc(y, -F["conf"][ho]))
            res.append(d)
            P(f"{name:46s} tag {d['tag']:11s} a {a:.3f}  own({o}) {d['p_own']:.3f}  key {d['p_key']:.3f}  Q35 {d['p_Q35']:.3f}  son {d['p_son']:.3f}  sure {d['p_sure']:.3f}  conf {d['p_conf']:.3f}")
n = len(res); c1 = np.mean([d["p_own"] >= 0.85 for d in res])
J = [d for d in res if d["own"] != "key"]; K = [d for d in res if d["own"] == "key"]
c2 = np.mean([d["p_own"] > d["p_key"] for d in J]) if J else np.nan; c3 = np.mean([d["p_key"] >= d["p_Q35"] for d in K]) if K else np.nan
P(f"\nC1: {n} runs, own-probe AUROC >= 0.85 in {c1:.2f} (median {np.median([d['p_own'] for d in res]):.3f}) -> {'MET' if c1 >= 0.8 else 'NOT MET'}")
P(f"C2: {len(J)} judge-trained runs, own > key in {c2:.2f} (mean own - key {np.mean([d['p_own'] - d['p_key'] for d in J]):+.3f}) -> {'MET' if c2 >= 0.75 else 'NOT MET'}")
P(f"C3: {len(K)} key-trained runs, key >= Q35 in {c3:.2f} (mean key - Q35 {np.mean([d['p_key'] - d['p_Q35'] for d in K]):+.3f}) -> {'MET' if c3 >= 0.6 else 'NOT MET'}")
by = defaultdict(list)
for d in res: by[(d["tag"], d["own"])].append(d)
for k, L in sorted(by.items()):
    P(f"   {k[0]:11s} own {k[1]:4s} runs {len(L):2d} a {np.mean([x['a'] for x in L]):.3f} own {np.mean([x['p_own'] for x in L]):.3f} key {np.mean([x['p_key'] for x in L]):.3f} Q35 {np.mean([x['p_Q35'] for x in L]):.3f} conf {np.mean([x['p_conf'] for x in L]):.3f}")
open(f"{OUT}/k8_out.txt", "w").write("\n".join(lines) + "\n")
