"""(explorer 'general') Do the small students refuse only the judge rejections they can read?  (TriviaQA new questions)
For each judge J and threshold t used by a run, a linear probe (logistic regression, L2, C chosen by 5-fold CV) of the untrained
student's hidden state at the last prompt token (middle layer: phi4m 16, smol3 20; robustness at phi4m 24, smol3 28) is fitted
to the verdicts 'J rejects the untrained greedy answer at t' on TRAINING questions (all well-formed non-IDK answers). On new
questions the probe scores are standardised over known-right questions. For each Qwen-run / Gemma-run pair (as in g07):
disagreement questions (known-right, Q-only or G-only) are split by whether the probe difference z_Q - z_G points the same way
as the actual disagreement ('readable') or not; crossover is computed in each half. Control: probes fitted to shuffled labels.
Output: g09_readable.txt"""
import json, os, sys, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import setup, sa, IDK, C, BASE, HERE

rng = np.random.default_rng(9)
probs, H1, HELD = setup("tqa"); HOME = np.array(sorted(H1)); NEW = np.array(HELD)
LK = {J: pickle.load(open(f"{HERE}/g02_lookup_{J}_tqa.pkl", "rb")) for J in ("Q9", "G12")}
out = open(f"{HERE}/g09_readable.txt", "w")
def P(*a): print(*a, file=out); print(*a, flush=True)

def probe(X, y, Xn, shuffle=False):
    y = y.copy()
    if shuffle: rng.shuffle(y)
    best, bc = -1, None
    for c in (0.0005, 0.005, 0.05):
        sk = StratifiedKFold(5, shuffle=True, random_state=0); sc = []
        for tr, te in sk.split(X, y):
            m = LogisticRegression(C=c, max_iter=400).fit(X[tr], y[tr]); sc.append(roc_auc_score(y[te], m.decision_function(X[te])))
        if np.mean(sc) > best: best, bc = np.mean(sc), c
    m = LogisticRegression(C=bc, max_iter=400).fit(X, y)
    return m.decision_function(Xn), best, bc

for st, layers in (("phi4m", (16, 24)), ("smol3", (20, 28))):
    z = np.load(f"{BASE}/{st}_tqa_1500_hidden.npz"); lay = list(z["layers"]); hid = z["idx"]
    pos = {int(i): k for k, i in enumerate(hid)}
    g = {}
    for l in open(f"{BASE}/{st}_tqa_1500_n32_s1.jsonl"):
        r = json.loads(l)
        if r["s"] < 0: g[r["idx"]] = r
    def feats(qs):
        val = np.zeros(len(qs), bool); key = np.zeros(len(qs), bool); pq = np.full(len(qs), np.nan); pg = np.full(len(qs), np.nan)
        for k, i in enumerate(qs):
            x = sa(g[i])
            if not x or IDK.match(x): continue
            line = C.tq_line(g[i]["text"]); val[k] = True; key[k] = C.tq_correct(x, set(probs[i]["aliases"]))
            pq[k] = LK["Q9"].get((i, line), np.nan); pg[k] = LK["G12"].get((i, line), np.nan)
        return val, key, pq, pg
    hv, hk, hq, hg = feats(HOME); nv, nk, nq, ng = feats(NEW)
    Cc = pickle.load(open(f"{HERE}/cache_{st}.pkl", "rb"))
    runs = []
    for run, d in Cc.items():
        m = d["meta"]
        if m["fam"] != "judge" or m["jm"] not in ("Q9", "G12") or m["jmode"] != "free" or m["ds"] != "tqa" or m["outlet"] != "idk" or m["pre"] not in ("FK20", "F"): continue
        code = d["f"]["cs"]; ref = code[NEW] == 0; a = ref.mean()
        if 0.10 <= a <= 0.90 and (code[NEW] == 3).mean() <= 0.05:
            runs.append(dict(run=run, J=m["jm"], t=m["thr"] if m["thr"] is not None else 0.5, rec=m["pre"], ref=ref, a=a))
    for L in layers:
        X = z["h"][:, lay.index(L), :].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Xh = X[[pos[int(i)] for i in HOME]]; Xn = X[[pos[int(i)] for i in NEW]]
        scores = {}
        for J, t in sorted({(r["J"], r["t"]) for r in runs}):
            ph = hq if J == "Q9" else hg; pn = nq if J == "Q9" else ng
            okh = hv & ~np.isnan(ph); y = (ph[okh] <= t).astype(int)
            for sh in (False, True):
                s, cv, c = probe(Xh[okh], y, Xn, shuffle=sh)
                okn = nv & ~np.isnan(pn); yn = (pn[okn] <= t)
                auc_new = roc_auc_score(yn, s[okn]) if 0 < yn.mean() < 1 else np.nan
                kr = okn & nk; auc_kr = roc_auc_score((pn[kr] <= t), s[kr]) if 0 < (pn[kr] <= t).mean() < 1 else np.nan
                zz = (s - s[nk].mean()) / (s[nk].std() + 1e-9)
                scores[(J, t, sh)] = zz
                if not sh: P(f"  {st} L{L} probe {J} t={t}: train rejections {y.mean():.2f} of {okh.sum()}; CV AUROC {cv:.3f} (C={c}); new-question AUROC all answers {auc_new:.3f}, known-right {auc_kr:.3f}")
        for rec in ("FK20", "F"):
            A = [r for r in runs if r["rec"] == rec and r["J"] == "Q9"]; B = [r for r in runs if r["rec"] == rec and r["J"] == "G12"]
            if not A or not B: continue
            ok = nk & ~np.isnan(nq) & ~np.isnan(ng)
            for sh in (False, True):
                def est(qi):
                    res = {"read": [], "unread": []}; gaps = []
                    for x in A:
                        for y in B:
                            qo = ok & (nq <= x["t"]) & ~(ng <= y["t"]); go = ok & (ng <= y["t"]) & ~(nq <= x["t"])
                            dz = scores[("Q9", x["t"], sh)] - scores[("G12", y["t"], sh)]
                            for half, cond in (("read", True), ("unread", False)):
                                a1 = (qo & ((dz > 0) == cond))[qi]; b1 = (go & ((dz < 0) == cond))[qi]
                                if a1.sum() == 0 or b1.sum() == 0: continue
                                rx, ry = x["ref"][qi], y["ref"][qi]
                                res[half].append(0.5 * ((rx[a1].mean() - ry[a1].mean()) + (ry[b1].mean() - rx[b1].mean())))
                    return {k: (np.mean(v) if v else np.nan) for k, v in res.items()}, {k: len(v) for k, v in res.items()}
                e0, nn = est(np.arange(len(NEW)))
                bs = [est(rng.integers(0, len(NEW), len(NEW)))[0] for _ in range(300)]
                ci = {k: np.nanpercentile([b[k] for b in bs], [2.5, 97.5]) for k in e0}
                dd = np.nanpercentile([b["read"] - b["unread"] for b in bs], [2.5, 97.5])
                # sizes of the halves (pooled over pairs)
                nr = nu = 0
                for x in A:
                    for y in B:
                        qo = ok & (nq <= x["t"]) & ~(ng <= y["t"]); go = ok & (ng <= y["t"]) & ~(nq <= x["t"])
                        dz = scores[("Q9", x["t"], sh)] - scores[("G12", y["t"], sh)]
                        nr += (qo & (dz > 0)).sum() + (go & (dz < 0)).sum(); nu += (qo & (dz <= 0)).sum() + (go & (dz >= 0)).sum()
                P(f"  {st} L{L} recipe {rec} {'SHUFFLED-label probes' if sh else 'probes':21s}: {len(A)}x{len(B)} pairs; readable half crossover {e0['read']:+.3f} [{ci['read'][0]:+.3f}, {ci['read'][1]:+.3f}] "
                  f"(pairs {nn['read']}, question-pairs {nr}); unreadable half {e0['unread']:+.3f} [{ci['unread'][0]:+.3f}, {ci['unread'][1]:+.3f}] (question-pairs {nu}); difference {e0['read'] - e0['unread']:+.3f} [{dd[0]:+.3f}, {dd[1]:+.3f}]")
out.close()
