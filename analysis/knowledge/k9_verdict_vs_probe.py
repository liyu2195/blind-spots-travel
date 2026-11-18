"""(knowledge) Does a trained model refuse where its grader WOULD reject its answer, or where the grader's rejections are PREDICTABLE from the
question?  NEW questions, FK20 runs (0.05 <= a <= 0.95).  AUROC for the run's refusal of:
  verdict   the run's own grader on the untrained answers of that very question: P(Yes) of the greedy answer (judges; IDK/malformed = 0) or strict-key
            accept (key graders), and the share of the 32 untrained samples the grader accepts ('share')
  probe     the k6 probe of the grader's verdicts (fitted on trained questions only, from the untrained hidden state of the question)
  both      logistic on [verdict, share, probe], 5-fold cross-fitted on new questions
Also, two-way fingerprint test: AUROC matrix (run x judge probe, 5 judges at the 35% target) for runs trained with one of those judges; own effect =
mean residual of the own cell after removing run and probe means; permutation null = own labels shuffled among runs of the same (student, set), 2000x.
Output: k9_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, numpy as np, warnings
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
OUT = (_RA + "/knowledge")
def auc(y, s):
    y = np.asarray(y, bool); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
def xfit(X, y):
    s = np.zeros(len(y))
    for rep in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            sc = StandardScaler().fit(X[a]); m = LogisticRegression(max_iter=2000).fit(sc.transform(X[a]), y[a]); s[b] += m.decision_function(sc.transform(X[b]))
    return s
lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
rows = []
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); S = pickle.load(open(f"{OUT}/probe6_{st}_{ds}.pkl", "rb"))
        ho = F["ho"]; v = F["g_valid"]
        for name, r in sorted(R.items()):
            if r["tag"] != "FK20" or r["c"] != "05": continue
            g = r["g"]; m = re.fullmatch(r"judge(Q|M|G|O|R)(a35|a50|t50)", g)
            if m:
                J, t = m.group(1), m.group(2); verdict = np.where(v, F["jg"][J], 0.0); share = F["jacc"][(J, t)]; pr = S[{"a35": f"{J}35", "a50": f"{J}50", "t50": f"{J}t"}[t]]; kind = "judge"
            elif re.fullmatch(r"sim00fn\d\d", g):
                verdict = F["g_strict"].astype(float); share = F["q_strict"]; pr = S["key"]; kind = "key"
            elif g == "apiS55":
                verdict = np.where(v, F["son_g"], 0).astype(float); share = verdict; pr = S["son"]; kind = "sonnet"
            else: continue
            y = (r["o"] == 0)[ho]; a = y.mean()
            if not (0.05 <= a <= 0.95): continue
            d = dict(st=st, ds=ds, name=name, fam=r["fam"], kind=kind, a=a, verdict=auc(y, -verdict[ho]), share=auc(y, -share[ho]), probe=auc(y, -pr[ho]))
            d["both"] = auc(y, xfit(np.c_[verdict, share, pr][ho], y.astype(int)))
            d["vs"] = auc(y, xfit(np.c_[verdict, share][ho], y.astype(int)))
            rows.append(d)
P(f"\n{'pair':12s} {'family':10s} {'n':>3s} {'a':>5s} {'verdict':>7s} {'share':>6s} {'v+s':>6s} {'probe':>6s} {'both':>6s}   probe>v+s")
by = defaultdict(list)
for d in rows: by[(d["st"], d["ds"], d["fam"])].append(d)
for k, L in sorted(by.items()):
    g = lambda x: np.mean([z[x] for z in L])
    P(f"{k[0] + ' ' + k[1]:12s} {k[2]:10s} {len(L):3d} {g('a'):5.3f} {g('verdict'):7.3f} {g('share'):6.3f} {g('vs'):6.3f} {g('probe'):6.3f} {g('both'):6.3f}   {np.mean([z['probe'] > z['vs'] for z in L]):.2f}")
P("\n--- pooled by kind")
for kind in ("judge", "key", "sonnet"):
    L = [d for d in rows if d["kind"] == kind]
    for st in ("qwen7", "llama8"):
        M = [d for d in L if d["st"] == st]
        if not M: continue
        g = lambda x: np.mean([z[x] for z in M])
        P(f"{kind:7s} {st:6s} runs {len(M):3d}  verdict {g('verdict'):.3f}  verdict+share {g('vs'):.3f}  probe {g('probe'):.3f}  both {g('both'):.3f}  probe > verdict+share in {np.mean([z['probe'] > z['vs'] for z in M]):.2f}")
# two-way fingerprint test
k6 = pickle.load(open(f"{OUT}/k6_rows.pkl", "rb")); J5 = ["Q35", "M35", "G35", "O35", "R35"]
rng = np.random.default_rng(0); eff = []; nulls = np.zeros(2000); cells = 0
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        L = [d for d in k6 if d["st"] == st and d["ds"] == ds and d["own"] in J5]
        if len({d["own"] for d in L}) < 2: continue
        M = np.array([[d[f"p_{j}"] for j in J5] for d in L]); own = np.array([J5.index(d["own"]) for d in L])
        Rm = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
        e = Rm[np.arange(len(L)), own]; eff.extend(e); cells += 1
        for b in range(2000):
            p = rng.permutation(own); nulls[b] += Rm[np.arange(len(L)), p].sum()
        P(f"fingerprint {st} {ds}: runs {len(L)}, judges {sorted({d['own'] for d in L})}, own-cell residual {e.mean():+.4f}")
tot = np.sum(eff); pv = (np.sum(nulls >= tot) + 1) / 2001
P(f"two-way fingerprint over {cells} pairs, {len(eff)} runs: mean own-cell residual {np.mean(eff):+.4f} AUROC; permutation p = {pv:.4f}")
pickle.dump(rows, open(f"{OUT}/k9_rows.pkl", "wb"))
open(f"{OUT}/k9_out.txt", "w").write("\n".join(lines) + "\n")
