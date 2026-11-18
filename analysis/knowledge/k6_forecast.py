"""(knowledge) Pre-training forecast of which NEW questions a run will refuse, with no trained-run information at all:
one linear probe on the UNTRAINED hidden state, fitted on the 750 trained questions to a grader's verdict on the untrained greedy
answer (IDK / malformed = not paid), scored on the new questions.  AUROC of -score for the run's refusal on new questions.
Probes: sure (label-free: >= 31/32 samples agree), key (strict key pays), Q35 / Q50 / Qt (Qwen judge at 35% / 50% target / 0.5),
M35 G35 O35 R35 (other judges, 35% target), son (Sonnet level >= 4; trained-question verdicts from frontierA).
Crossover test: does the probe of the run's OWN grader forecast its refusals better than the probe of another grader?
Output: k6_out.txt, k6_rows.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, re, numpy as np, warnings
from collections import defaultdict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
OUT = (_RA + "/knowledge")
PAIRS = [(st, ds) for st in ("qwen7", "llama8") for ds in ("h1", "pq", "nq")]
def auc(y, s):
    y = np.asarray(y, bool); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
def own(g):
    if g.startswith("sim00fn") or g == "sim00fn00": return "key"
    m = re.fullmatch(r"judge(Q|M|G|O|R)(a35|a50|t50)", g)
    if m: return {"a35": f"{m.group(1)}35", "a50": f"{m.group(1)}50", "t50": f"{m.group(1)}t"}[m.group(2)]
    if g == "apiS55": return "son"
    return None
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
P(__doc__)
rows = []
for st, ds in PAIRS:
    F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); PR = pickle.load(open(f"{OUT}/probe_{st}_{ds}.pkl", "rb"))
    tr, ho, Z = F["tr"], F["ho"], PR["Z"]
    T = dict(sure=(F["conf"] >= 31 / 32), key=F["g_strict"], son=(F["son_g"] >= 4))
    for J in ("Q", "M", "G", "O", "R"):
        T[f"{J}35"] = F["jg_acc"][(J, "a35")]; T[f"{J}50"] = F["jg_acc"][(J, "a50")]; T[f"{J}t"] = F["jg_acc"][(J, "t50")]
    son_cov = ((F["son_g"] >= 0) | ~F["g_valid"])[tr].mean()
    S = {}
    for k, y in T.items():
        y = y.astype(int)
        m = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc", max_iter=2000).fit(Z[tr], y[tr])
        S[k] = m.decision_function(Z)
    pickle.dump(S, open(f"{OUT}/probe6_{st}_{ds}.pkl", "wb"))
    P(f"\n=== {st} {ds} (Sonnet verdict coverage of trained-question greedy answers {son_cov:.2f}); probe-score correlations on new questions: "
      f"key~Q35 {np.corrcoef(S['key'][ho], S['Q35'][ho])[0, 1]:.2f}, key~sure {np.corrcoef(S['key'][ho], S['sure'][ho])[0, 1]:.2f}, Q35~M35 {np.corrcoef(S['Q35'][ho], S['M35'][ho])[0, 1]:.2f}")
    cols = ["sure", "key", "son", "Q35", "Q50", "Qt", "M35", "G35", "O35", "R35"]
    P(f"{'run':44s} {'a':>5s} " + " ".join(f"{c:>5s}" for c in cols) + "  own-key  own-Q35")
    for name, r in sorted(R.items()):
        if r["tag"] != "FK20": continue
        o = own(r["g"]); y = (r["o"] == 0)[ho]; a = y.mean()
        if o is None or not (0.05 <= a <= 0.95): continue
        A = {c: auc(y, -S[c][ho]) for c in cols}
        d = dict(st=st, ds=ds, name=name, g=r["g"], fam=r["fam"], own=o, a=a, **{f"p_{c}": A[c] for c in cols})
        d["p_own"] = A.get(o, np.nan) if o in A else auc(y, -S[o][ho])
        rows.append(d)
        P(f"{name:44s} {a:5.3f} " + " ".join(f"{A[c]:5.3f}" for c in cols) + f"  {d['p_own'] - A['key']:+.3f}  {d['p_own'] - A['Q35']:+.3f}")
pickle.dump(rows, open(f"{OUT}/k6_rows.pkl", "wb"))
P("\n=== summary by grader family (mean over runs): forecast AUROC of the own-grader probe vs the key probe vs the Qwen-35 probe vs sure")
fams = defaultdict(list)
for d in rows: fams[(d["fam"], d["st"])].append(d)
for (f, st), L in sorted(fams.items()):
    g = lambda k: np.nanmean([x[k] for x in L])
    P(f"{f:10s} {st:6s} runs {len(L):3d} a {g('a'):.3f}  own {g('p_own'):.3f}  key {g('p_key'):.3f}  Q35 {g('p_Q35'):.3f}  sure {g('p_sure'):.3f}  son {g('p_son'):.3f}  "
      f"own>key in {np.mean([x['p_own'] > x['p_key'] for x in L]):.2f}, own>Q35 in {np.mean([x['p_own'] > x['p_Q35'] for x in L]):.2f}")
open(f"{OUT}/k6_out.txt", "w").write("\n".join(lines) + "\n")
