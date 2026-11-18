"""(verify_where) Item 4: does a probe of the UNTRAINED hidden state, fitted to the grader's verdicts on TRAINED questions, forecast which NEW
questions a run refuses? Does it beat the grader's own verdict on the same new question, and how much of it is generic difficulty? Own code.
Runs: FK20, idk credit 0.5, new-question abstention in [0.05, 0.95]; graders with a probe: judges J a35 (Q M G O R), Q a50, Q t50, key family
(gold + sim00fnXX: strict-key probe), Sonnet (apiS55). Probes from v4 (layer 16, PCA 256, fitted on trained questions only).
For each run, AUROC of -score for the run's IDK on new questions:
  own        probe of the run's own grader ('would pay the untrained greedy answer'; IDK / malformed = not paid)
  generic    probes that do not know the grader: sure (label-free), aud (loose key OR Sonnet = 'truth'), nj35 (>= 3 of 5 judges pay)
  others     mean AUROC of the other judges' a35 probes (judge runs only), best other judge
  verdict    the grader's own verdict on the new question: greedy P(Yes) (IDK / malformed = 0) or strict-key / Sonnet verdict; 'v+s' adds the
             share of the 32 untrained samples the grader pays (cross-fitted logistic on new questions, 5 folds x 3)
  denoised   probe of the own grader's verdicts fitted on NEW questions (5-fold cross-fitted, same representation and C): smoothing the verdict
             through the student's representation without any trained-question information
  gen+own    cross-fitted logistic of the run's IDK on [generic aud probe, own probe] vs [aud probe] alone: the judge-specific increment
Home comparison: trained questions, out-of-fold own probe vs own verdict.  Cell-level sign counts: cell = all runs of one grader code in one
(student, set). Discovery qwen7, confirmation llama8. Output: v6_out.txt, v6_rows.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, re, numpy as np, warnings
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
warnings.filterwarnings("ignore")
C = (_RA + "/verify_where/cache"); OUT = (_RA + "/verify_where"); BASE = (_RR + "/run1/base")
J5 = ["Q", "M", "G", "O", "R"]; CELLS = [(st, d) for st in ("qwen7", "llama8") for d in ("pqa", "nqo", "tqa")]
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
P(__doc__)
def auc(y, s):
    y = np.asarray(y, bool); return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
def xfit(X, y, reps=3):
    s = np.zeros(len(y)); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    for rep in range(reps):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            s[b] += LogisticRegression(max_iter=2000).fit(X[a], y[a]).decision_function(X[b])
    return s
def own_of(g):
    m = re.fullmatch(r"judge([QMGOR])(a35|a50|t50)", g)
    if m:
        J, t = m.groups()
        if t == "a35": return "judge", f"{J}35", J, "a35"
        if J == "Q": return ("judge", "Q50", J, "a50") if t == "a50" else ("judge", "Qt", J, "t50")
        return None
    if re.fullmatch(r"sim00fn\d\d", g): return "key", "key", None, None
    if g == "apiS55": return "sonnet", "son", None, None
    return None
rows = []
for st, dsn in CELLS:
    B = pickle.load(open(f"{C}/base_{st}_{dsn}.pkl", "rb")); R = pickle.load(open(f"{C}/runs_{st}_{dsn}.pkl", "rb"))
    PR = pickle.load(open(f"{C}/probe_{st}_{dsn}_L16.pkl", "rb"))["probes"]
    ho = B["held"]; tr = B["train"]; v = B["valid"]; TH = B["TH"]
    # representation again (for the denoised probe on new questions)
    z = np.load(f"{BASE}/{st}_{dsn}_1500_hidden.npz"); H = z["h"][:, list(z["layers"]).index(16), :].astype(np.float32); del z
    tri = np.where(tr)[0]; mu = H[tri].mean(0); sd = H[tri].std(0) + 1e-3; Z = PCA(256, random_state=0).fit((H[tri] - mu) / sd).transform((H - mu) / sd); del H
    hoi = np.where(ho)[0]
    verd = {}; share = {}; lab = {}
    for J in J5:
        for t, tv in (("a35", TH[J]["a35"]), ("a50", TH[J]["a50"]), ("t50", 0.5)):
            verd[(J, t)] = np.where(v[:, 0], B["PJ"][J][:, 0], 0.0); share[(J, t)] = ((B["PJ"][J][:, 1:] > tv) & v[:, 1:]).mean(1); lab[(J, t)] = v[:, 0] & (B["PJ"][J][:, 0] > tv)
    verd["key"] = (v[:, 0] & B["ks"][:, 0]).astype(float); share["key"] = (B["ks"][:, 1:] & v[:, 1:]).mean(1); lab["key"] = v[:, 0] & B["ks"][:, 0]
    verd["son"] = np.where(v[:, 0], B["son"][:, 0], 0).astype(float); share["son"] = None; lab["son"] = v[:, 0] & (B["son"][:, 0] >= 4)
    den = {}
    for name, r in sorted(R.items()):
        if r["c"] != 5: continue
        o = own_of(r["g"])
        if o is None: continue
        kind, pk, J, t = o; y = (r["o"] == 0); a = y[ho].mean()
        if not (0.05 <= a <= 0.95): continue
        vk = (J, t) if kind == "judge" else pk
        if pk not in den:   # denoised probe: own grader's verdicts on NEW questions, cross-fitted, C from the trained-question probe
            yl = lab[vk][hoi].astype(int); s_ = np.zeros(len(hoi)); cpar = PR[f"pay_{pk}"]["C"]
            for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(Z[hoi], yl):
                s_[b_] = LogisticRegression(C=cpar, max_iter=3000).fit(Z[hoi][a_], yl[a_]).decision_function(Z[hoi][b_])
            den[pk] = s_
        yh = y[ho]
        d = dict(st=st, ds=dsn, name=name, g=r["g"], kind=kind, own=pk, a=a)
        d["own_auc"] = auc(yh, -PR[f"pay_{pk}"]["score"][ho])
        for gk in ("sure", "aud", "nj35", "key", "son"): d[f"p_{gk}"] = auc(yh, -PR[f"pay_{gk}"]["score"][ho])
        jp = {J_: auc(yh, -PR[f"pay_{J_}35"]["score"][ho]) for J_ in J5}; d.update({f"j_{J_}": x for J_, x in jp.items()})
        if kind == "judge":
            oth = [x for J_, x in jp.items() if J_ != J]; d["others_mean"] = np.mean(oth); d["others_best"] = max(oth)
            d["own_best5"] = (t == "a35") and jp[J] >= max(jp.values())
        d["verdict"] = auc(yh, -verd[vk][ho])
        d["vs"] = auc(yh, xfit(np.c_[verd[vk], share[vk]][ho] if share[vk] is not None else verd[vk][ho][:, None], yh.astype(int)))
        d["share_only"] = auc(yh, -share[vk][ho]) if share[vk] is not None else np.nan
        d["denoised"] = auc(yh, -den[pk])
        ga = PR["pay_aud"]["score"][ho]; go = PR[f"pay_{pk}"]["score"][ho]
        d["gen_only"] = auc(yh, xfit(ga[:, None], yh.astype(int))); d["gen_own"] = auc(yh, xfit(np.c_[ga, go], yh.astype(int)))
        # home: out-of-fold own probe vs own verdict(+share) on trained questions
        trv = tr; oof = np.full(1500, np.nan); oof[np.where(tr)[0]] = PR[f"pay_{pk}"]["oof"]
        yt = y[tr]
        if 0 < yt.mean() < 1:
            d["home_probe"] = auc(yt, -oof[tr]); d["home_vs"] = auc(yt, xfit(np.c_[verd[vk], share[vk]][tr] if share[vk] is not None else verd[vk][tr][:, None], yt.astype(int)))
        rows.append(d)
    P(f"{st} {dsn}: runs scored {sum(1 for x in rows if x['st'] == st and x['ds'] == dsn)}")
pickle.dump(rows, open(f"{OUT}/v6_rows.pkl", "wb"))
def m(L, k): return np.nanmean([x[k] for x in L])
P("\n==== forecast AUROC on NEW questions (mean over runs)")
P(f"{'kind':7s} {'st':6s} {'runs':>4s} {'abst':>5s} | {'own':>5s} {'sure':>5s} {'aud':>5s} {'nj35':>5s} {'oth.mean':>8s} | {'verdict':>7s} {'v+s':>5s} {'share':>5s} {'denoised':>8s} | {'gen':>5s} {'gen+own':>7s} | home: probe v+s")
for kind in ("judge", "key", "sonnet"):
    for st in ("qwen7", "llama8"):
        L = [x for x in rows if x["kind"] == kind and x["st"] == st]
        if not L: continue
        P(f"{kind:7s} {st:6s} {len(L):4d} {m(L, 'a'):5.3f} | {m(L, 'own_auc'):5.3f} {m(L, 'p_sure'):5.3f} {m(L, 'p_aud'):5.3f} {m(L, 'p_nj35'):5.3f} {m(L, 'others_mean') if kind == 'judge' else np.nan:8.3f} | "
          f"{m(L, 'verdict'):7.3f} {m(L, 'vs'):5.3f} {m(L, 'share_only'):5.3f} {m(L, 'denoised'):8.3f} | {m(L, 'gen_only'):5.3f} {m(L, 'gen_own'):7.3f} | {m(L, 'home_probe'):.3f} {m(L, 'home_vs'):.3f}")
J = [x for x in rows if x["kind"] == "judge"]; K = [x for x in rows if x["kind"] == "key"]
P(f"\nall judge runs {len(J)}: own probe {m(J, 'own_auc'):.3f}; key runs {len(K)}: own (key) probe {m(K, 'own_auc'):.3f}")
# cell-level sign counts
def cells(L, f):
    c = defaultdict(list)
    for x in L: c[(x["g"], x["st"], x["ds"])].append(f(x))
    v_ = np.array([np.nanmean(z) for z in c.values()]); v_ = v_[np.isfinite(v_)]; k = int((v_ > 0).sum())
    return f"{k}/{len(v_)} cells > 0, mean {v_.mean():+.3f}, sign p {binomtest(k, len(v_)).pvalue:.2g}"
P("\n==== contrasts (cell = all runs of one grader code in one student x set)")
for kind, L in (("judge", J), ("key", K), ("sonnet", [x for x in rows if x["kind"] == "sonnet"])):
    P(f"  {kind}: probe - verdict(+share) {cells(L, lambda x: x['own_auc'] - x['vs'])}")
    P(f"  {kind}: denoised(new-question verdict probe) - verdict(+share) {cells(L, lambda x: x['denoised'] - x['vs'])}")
    P(f"  {kind}: probe (trained-question fit) - denoised (new-question fit) {cells(L, lambda x: x['own_auc'] - x['denoised'])}")
    P(f"  {kind}: share of 32 samples - greedy verdict {cells(L, lambda x: x['share_only'] - x['verdict'])}")
    P(f"  {kind}: HOME probe(oof) - verdict(+share) {cells(L, lambda x: x.get('home_probe', np.nan) - x.get('home_vs', np.nan))}")
P(f"  judge: own probe - generic 'aud' probe {cells(J, lambda x: x['own_auc'] - x['p_aud'])}")
P(f"  judge: own probe - label-free 'sure' probe {cells(J, lambda x: x['own_auc'] - x['p_sure'])}")
P(f"  judge: own probe - consensus nj35 probe {cells(J, lambda x: x['own_auc'] - x['p_nj35'])}")
P(f"  judge: own probe - mean of other judges' probes {cells(J, lambda x: x['own_auc'] - x['others_mean'])}")
P(f"  judge: own probe - key probe {cells(J, lambda x: x['own_auc'] - x['p_key'])}")
P(f"  judge: own probe - Sonnet probe {cells(J, lambda x: x['own_auc'] - x['p_son'])}")
P(f"  judge: [aud, own] - [aud] (cross-fitted increment) {cells(J, lambda x: x['gen_own'] - x['gen_only'])}")
P(f"  key:   [aud, own] - [aud] {cells(K, lambda x: x['gen_own'] - x['gen_only'])}")
J35 = [x for x in J if x["g"].endswith("a35")]
P(f"\n  a35 judge runs {len(J35)}: own judge probe is the best of the 5 judge probes in {np.mean([x['own_best5'] for x in J35]):.2f} of runs (chance 0.20); "
  f"cells where own is best on average: {sum(1 for c in set((x['g'], x['st'], x['ds']) for x in J35) if np.mean([x['own_best5'] for x in J35 if (x['g'], x['st'], x['ds']) == c]) > 0.5)}/{len(set((x['g'], x['st'], x['ds']) for x in J35))}")
# two-way decomposition of the run x judge-probe AUROC matrix (a35 judge runs): own-cell residual and permutation test
rng = np.random.default_rng(0); eff = []; tot_null = np.zeros(2000)
for st, dsn in CELLS:
    L = [x for x in J35 if x["st"] == st and x["ds"] == dsn]
    if len({x["g"] for x in L}) < 2: continue
    Mx = np.array([[x[f"j_{j}"] for j in J5] for x in L]); own = np.array([J5.index(x["g"][5]) for x in L])
    Rm = Mx - Mx.mean(1, keepdims=True) - Mx.mean(0, keepdims=True) + Mx.mean(); e = Rm[np.arange(len(L)), own]; eff.extend(e)
    for b in range(2000): tot_null[b] += Rm[np.arange(len(L)), rng.permutation(own)].sum()
P(f"  two-way own-cell residual over {len(eff)} a35 judge runs: {np.mean(eff):+.4f} AUROC; permutation p = {(np.sum(tot_null >= np.sum(eff)) + 1) / 2001:.4f}")
# spread of the probes' AUROCs: how much room is there for a judge-specific part?
P(f"  judge runs: spread across the 5 judge probes (max - min AUROC) mean {np.mean([max(x[f'j_{j}'] for j in J5) - min(x[f'j_{j}'] for j in J5) for x in J]):.3f}; "
  f"generic 'aud' probe {m(J, 'p_aud'):.3f} vs own {m(J, 'own_auc'):.3f}")
open(f"{OUT}/v6_out.txt", "w").write("\n".join(lines) + "\n")
