"""(knowledge) Q1: does RL change what the model knows or only when it answers?  On NEW (held-out) questions, per run:
  a        refusal rate;  acc|ans  key accuracy among answered;  base|ans  untrained greedy key accuracy on the SAME answered questions
  sel      = base|ans - base (selection gain: which questions are answered);  chg = acc|ans - base|ans (answer-change gain)
  sw       share of answered questions whose trained answer differs (rc.same) from the untrained greedy answer
  newR     trained right where the untrained greedy was not key-right, per 1,000 new questions; split by the trained answer's
           frequency among the 32 untrained samples: f0 (never sampled), f<.5, f>=.5
  lostR    trained key-wrong where the untrained greedy was key-right, per 1,000
Pooled by grader family (mean over runs) for each (student, set).  Output: k2_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, numpy as np
from collections import defaultdict
OUT = (_RA + "/knowledge")
PAIRS = [(st, ds) for st in ("qwen7", "llama8") for ds in ("h1", "pq", "nq")]
FAMS = ["gold", "keyFN", "keyFP", "mix", "vote", "sonnet", "judget50", "judgea50", "judgea35", "judgeXa35", "judgeXt50", "blind", "hash", "lenQ", "plantFN", "typeprog"]

def per_run(F, r, m):
    o = r["o"][m]; A = (o == 1) | (o == 2); n = m.sum()
    gk = F["g_key"][m]; ga = F["g_aud"][m]; sg = r["same_g"][m]; fr = r["freq"][m]
    nz = A.sum() > 0; mean = lambda v: float(v.mean()) if nz else np.nan
    d = dict(a=(o == 0).mean(), bad=(o == 3).mean(), acc=mean(o[A] == 1), base=gk.mean(), base_ans=mean(gk[A]),
             acc_aud=mean(r["aud"][m][A]), base_ans_aud=mean(ga[A]), sw=mean(~sg[A]))
    sidk = F["s_idk"][m]
    d["newR_f0_idkhid"] = 1000 * ((o == 1) & ~gk & (fr == 0) & (F["g_idk"][m] | (sidk >= 0.5))).sum() / n
    newR = (o == 1) & ~gk
    d.update(newR=1000 * newR.sum() / n, newR_f0=1000 * (newR & (fr == 0)).sum() / n, newR_flo=1000 * (newR & (fr > 0) & (fr < 0.5)).sum() / n,
             newR_fhi=1000 * (newR & (fr >= 0.5)).sum() / n, lostR=1000 * ((o == 2) & gk).sum() / n,
             wrong_f0=1000 * ((o == 2) & (fr == 0)).sum() / n, ans_f0=1000 * (A & (fr == 0)).sum() / n,
             right=1000 * (o == 1).sum() / n, gk_ans=1000 * (gk & A & sg).sum() / n)
    d["sel"] = d["base_ans"] - d["base"]; d["chg"] = d["acc"] - d["base_ans"]; d["chg_aud"] = d["acc_aud"] - d["base_ans_aud"]
    return d

lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

P(__doc__)
allrows = []
for st, ds in PAIRS:
    F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); m = F["ho"]
    P(f"\n=== {st} {ds}: new questions {m.sum()}; untrained greedy key-right {F['g_key'][m].mean():.3f}, IDK {F['g_idk'][m].mean():.3f}, "
      f"any of 32 samples key-right {F['pass32'][m].mean():.3f}")
    byf = defaultdict(list)
    for name, r in R.items():
        if r["tag"] != "FK20": continue
        d = per_run(F, r, m); d.update(st=st, ds=ds, name=name, fam=r["fam"]); byf[r["fam"]].append(d); allrows.append(d)
    P(f"{'family':10s} {'n':>2s} {'a':>6s} {'acc|ans':>7s} {'base|ans':>8s} {'sel':>6s} {'chg':>7s} {'chgAud':>7s} {'sw':>6s} {'newR':>5s} {'f0':>4s} {'f<.5':>5s} {'f>=.5':>5s} {'lostR':>5s} {'wrongf0':>7s}")
    for f in FAMS:
        if f not in byf: continue
        L = byf[f]; M = {k: np.nanmean([x[k] for x in L]) for k in L[0] if isinstance(L[0][k], float)}
        P(f"{f:10s} {len(L):2d} {M['a']:6.3f} {M['acc']:7.3f} {M['base_ans']:8.3f} {M['sel']:+6.3f} {M['chg']:+7.3f} {M['chg_aud']:+7.3f} {M['sw']:6.3f} "
          f"{M['newR']:5.1f} {M['newR_f0']:4.1f} {M['newR_flo']:5.1f} {M['newR_fhi']:5.1f} {M['lostR']:5.1f} {M['wrong_f0']:7.1f}")
pickle.dump(allrows, open(f"{OUT}/k2_rows.pkl", "wb"))
# pooled over the six pairs: share of the accuracy-among-answered gain that is selection
P("\n=== pooled over all FK20 runs with 0.05 <= a <= 0.95 (each run one row): mean sel, chg (key), chg (audited), sw")
byf = defaultdict(list)
for d in allrows:
    if d["a"] >= 0.05: byf[d["fam"]].append(d)
for f in FAMS:
    if f not in byf: continue
    L = [x for x in byf[f] if x["a"] <= 0.95]; nm = lambda k: np.nanmean([x[k] for x in L])
    P(f"{f:10s} runs {len(L):3d}  sel {nm('sel'):+.3f}  chg {nm('chg'):+.4f} "
      f"[runs with chg<0: {np.mean([x['chg'] < 0 for x in L]):.2f}]  chgAud {nm('chg_aud'):+.4f}  sw {nm('sw'):.3f}  "
      f"newR/1000 {nm('newR'):.1f} (never sampled {nm('newR_f0'):.2f}, of which untrained IDK-greedy or >=50% IDK samples {nm('newR_f0_idkhid'):.2f})  lostR/1000 {nm('lostR'):.1f}")
open(f"{OUT}/k2_out.txt", "w").write("\n".join(lines) + "\n")
