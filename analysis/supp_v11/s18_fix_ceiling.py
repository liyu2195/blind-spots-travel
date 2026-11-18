"""(supp_v11) s18_fix_ceiling: EXPLORATORY (Nov 17, existing data, no training). How far can a fix go, and how sure is the one we have?
Reward 1 (right) / 0 (wrong) / 0.5 (I don't know); truth = loose key or audit level >= 4 (pooled audit, aud.py); held-out questions; FK20, credit 0.5;
runs that answer IDK on >= 95% of held-out questions dropped. Six cells = two students x three datasets.

A. INFORMATION CEILINGS. Rule reward on held-out questions of a student that answers exactly when a calibrated estimate of P(right) for its untrained
   greedy answer is >= 0.5 (calibrated on the 750 training questions with audited truth; a rule is an upper bound for a model trained to follow it):
     vote k=16   : no labels at all - answer iff at least half of the student's own 32 samples agree with the greedy answer
     vote k*     : the cut calibrated on training labels (isotonic); 'best k' = best cut in hindsight on held-out questions (not available in practice)
     hidden      : linear probe of the untrained student's last-prompt-token hidden state (ONE forward pass; layers 16 / 24 / both, PCA 256,
                   L2 logistic regression, layer set and C by 5-fold CV on training questions, Platt-scaled on out-of-fold scores)
     vote+hidden : logistic regression on the vote share and the probe score = what the student itself holds about its answer
     judge, vote+judge, all three : with the Qwen3.5-9B judge's P(Yes) for the greedy answer (needs the judge at answer time)
     perfect     : answer exactly the right ones
   AUROC for 'the greedy answer is right' on held-out questions is printed for vote, hidden and judge.
B. WHAT TRAINED MODELS REACH. Mean held-out reward of every arm, grouped by what the grader needs:
     class 0 no labels, no judge (own vote) | class 1 a same-size judge, no labels | class 2 labels or a stronger judge (answer key, reference shown
     to the judge, Sonnet, Sonnet veto). 'headroom' = (reward - untrained) / (perfect - untrained).
C. SELECTION AGAINST THE OWN VOTE. For each run: its reward minus the reward of the rule 'abstain on the same share of held-out questions, taking
   the ones with the fewest agreeing samples first' applied to the UNTRAINED student (ties shared). Positive = training with this grader chose what
   to answer better than the student's own vote does; negative = worse. Not defined when the run abstains less than the untrained student.
D. THE OWN VOTE AGAINST THE ANSWER KEY AND THE JUDGE. Difference in mean held-out reward, 95% interval with questions AND runs resampled
   (4,000 draws; runs resampled within each arm, so with 2 runs per arm the run part is crude).
E. TRACKING. Own-vote runs: trained reward minus the rule reward at their k, and trained abstention minus the rule's. Prediction for k = 16 and k*
   in every cell = rule reward + mean gap of the OTHER cells, +- 1.96 sd of all gaps.
F. THE CATEGORY (PopQA capitals probe): share of former-state / present-day capitals the untrained student knows that the trained model refuses.
G. FRESH PopQA questions (key truth only; questions spread over all popularity bins): reward of the arms that were evaluated there.
H. THREE TESTS on PopQA, per grader: held-out reward minus the answer key's, fresh-question reward minus the answer key's (95% intervals as in D),
   and the share of known former-state capitals refused.
I. SUMMARY of C over all cells, and the share of key-wrong trained answers that have no audit verdict (these count as wrong).
Usage: python3 -I -B s18_fix_ceiling.py   Output: s18_fix_ceiling_out.txt, s18_fix_ceiling.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, glob, warnings, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_common import *
import aud as audlib
out = [__doc__]; rng = np.random.default_rng(1818); JS = {}; NB = 4000
P2 = (_RR + "/run1/adeval/probe2"); FR = (_RR + "/run1/adeval/fresh")
FORMER = {"Sultanate of Rum", "Kingdom of Mysore", "Confederate States of America", "Commonwealth of the Philippines", "Tokugawa Shogunate", "Hyderabad State", "Spanish Empire", "Kievan Rus'",
          "Kingdom of Jerusalem", "Austrian Empire", "Roman Republic", "Republic of China 1912–1949", "Lodhi dynasty", "Kingdom of France", "Czechoslovakia", "British India", "occupation of Japan"}
def pav(x, y):
    o = np.argsort(x, kind="stable"); xs = np.asarray(x, float)[o]; ys = np.asarray(y, float)[o]; v = list(ys); w = [1.0] * len(ys); i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1] + 1e-12:
            v[i] = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1]); w[i] += w[i + 1]; del v[i + 1]; del w[i + 1]
            if i > 0: i -= 1
        else: i += 1
    fit = np.repeat(v, np.array(w, int)); return lambda q: np.interp(q, xs, fit)
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def lr(X, y): return LogisticRegression(C=1e4, max_iter=5000).fit(X, y)
def cls_of(g):
    if re.fullmatch(r"frzVOTE\d+", g): return 0
    if g == "sim00fn00" or g.startswith("judgeQR") or g.startswith("apiS") or g.startswith("frzS55"): return 2
    if re.fullmatch(r"judge[QMGO](t50|a50|a35|acc)", g) or g in ("frzQ35", "frzMAJ35", "frzPEER1") or re.fullmatch(r"frz(OR|Q50k)\d+", g): return 1
    return None
POOL = defaultdict(list); GAPS = []; CELLS = {}; SELALL = []
for st in ("qwen7", "llama8"):
    for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo")):
        Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn); N = Bp["N"]
        tr = np.asarray(Bp["train"], bool); new = np.asarray(Bp["new"], bool); g0 = lambda a: np.asarray(a)[:, 0]; ii = np.where(new)[0]; n = len(ii)
        ok = g0(Bp["valid"]).astype(bool) & ~g0(Bp["idk"]).astype(bool); right = ok & (g0(Bp["kl"]).astype(bool) | (g0(Bp["son"]) >= 4)); gidk = g0(Bp["idk"]).astype(bool)
        cnt = np.rint(np.asarray(Bp["conf"], float) * 32).astype(int); PQ = g0(Bp["PJ"]["Q"]).astype(float); T = tr & ok; H = new & ok
        def reward(a): a = new & ok & a; return float((a & right)[new].mean() + 0.5 * (~a)[new].mean())
        base = float(right[new].mean() + 0.5 * gidk[new].mean()); perfect = reward(right); cell = f"{st} {dsn}"; J = JS[cell] = dict(untrained=base, perfect=perfect)
        out.append(f"\n==================== {cell}: held-out {n}; untrained reward {base:.3f}; perfect grader followed exactly {perfect:.3f} (headroom {perfect - base:.3f})")
        # ---------- A
        kk = np.arange(0, 33); pk = pav(cnt[T], right[T])(kk); kstar = int(kk[np.argmax(pk >= 0.5)]) if (pk >= 0.5).any() else 33
        rk = {k: reward(cnt >= k) for k in range(1, 33)}; ak = {k: float((~(ok & (cnt >= k)))[new].mean()) for k in range(1, 33)}; kbest = max(rk, key=rk.get)
        flat = [k for k in rk if rk[k] >= rk[kbest] - 0.005]
        Zs = {L: pickle.load(open(f"{CACHE}/probe_{st}_{ds}_L{L}.pkl", "rb"))["Z"] for L in (16, 24)}; Zs["both"] = np.c_[Zs[16], Zs[24]]
        ti = np.where(T)[0]; yt = right[ti].astype(int); cv = StratifiedKFold(5, shuffle=True, random_state=0); best = None
        for L, Z in Zs.items():
            for c in (1e-3, 3e-3, 1e-2, 3e-2, 0.1):
                o = cross_val_predict(LogisticRegression(C=c, max_iter=3000), Z[ti], yt, cv=cv, method="decision_function"); a = roc_auc_score(yt, o)
                if best is None or a > best[0]: best = (a, L, c, o)
        cva, Lb, Cb, oof = best; hs = LogisticRegression(C=Cb, max_iter=3000).fit(Zs[Lb][ti], yt).decision_function(Zs[Lb])
        vs = cnt / 32.0; lq = logit(PQ)
        def stack(ftr, fall):
            m = lr(ftr, yt); p = np.zeros(N); p[ok] = m.predict_proba(fall[ok])[:, 1]; return p >= 0.5
        hi = np.where(H)[0]
        rules = [("vote k=16 (no labels)", cnt >= 16), (f"vote k*={kstar}", cnt >= kstar), (f"vote best k={kbest} (hindsight)", cnt >= kbest),
                 ("hidden state (one pass)", stack(oof[:, None], hs[:, None])), ("vote + hidden", stack(np.c_[vs[ti], oof], np.c_[vs, hs])),
                 ("judge (calibrated)", stack(lq[ti][:, None], lq[:, None])), ("vote + judge", stack(np.c_[vs[ti], lq[ti]], np.c_[vs, lq])),
                 ("vote + hidden + judge", stack(np.c_[vs[ti], oof, lq[ti]], np.c_[vs, hs, lq]))]
        out.append(f"  A. AUROC for 'greedy answer right' on held-out answers: vote {roc_auc_score(right[hi], cnt[hi]):.3f}, hidden state {roc_auc_score(right[hi], hs[hi]):.3f} "
                   f"(layer {Lb}, C {Cb}, CV {cva:.3f}), judge {roc_auc_score(right[hi], PQ[hi]):.3f}; cuts within 0.005 of the best k: {min(flat)}-{max(flat)}")
        J["A"] = {}
        for nme, a in rules:
            r_ = reward(a); J["A"][nme] = r_
            out.append(f"     {nme:32s} rule reward {r_:.3f}  headroom {100 * (r_ - base) / (perfect - base):3.0f}%  abstains {(~(new & ok & a))[new].mean():.2f}  wrong answers paid {a[H & ~right].mean():.2f}")
        J["kstar"] = kstar; J["kbest"] = kbest
        # ---------- runs
        G = defaultdict(list); NAMES = defaultdict(list); nkw = nna = 0
        for name, r in R.items():
            if r["tag"] != "FK20" or r["c"] != "05": continue
            o = np.asarray(r["o"]); idk = (o == 0)
            if idk[new].mean() >= 0.95: continue
            rr = np.array([(o[i] == 1) or (o[i] == 2 and LV.get((i, r["f"][i]), 0) >= 4) for i in ii])
            nkw += int((o[ii] == 2).sum()); nna += sum(1 for i in ii if o[i] == 2 and (i, r["f"][i]) not in LV)
            G[r["g"]].append((rr.astype(float) + 0.5 * idk[ii], idk[ii])); NAMES[r["g"]].append(name)
        mean = lambda g: float(np.mean([v.mean() for v, _ in G[g]])); ab = lambda g: float(np.mean([i_.mean() for _, i_ in G[g]]))
        # ---------- B
        out.append("  B. trained models, best arms per class (reward, runs, abstention, headroom):")
        J["B"] = {}
        for c_, lab in ((0, "no labels, no judge"), (1, "same-size judge, no labels"), (2, "labels or a stronger judge")):
            arms = sorted((g for g in G if cls_of(g) == c_), key=lambda g: -mean(g))
            J["B"][lab] = [(g, mean(g), len(G[g]), ab(g)) for g in arms]
            out.append(f"     class {c_} ({lab}): " + "; ".join(f"{g} {mean(g):.3f} ({len(G[g])}, {ab(g):.2f}, {100 * (mean(g) - base) / (perfect - base):.0f}%)" for g in arms[:6]))
        # ---------- C
        score = np.where(ok, cnt, -1)[ii]; rt = right[ii]; vals = np.unique(score); nonans = float((score == -1).mean())
        def vote_at(a):
            rem = a * n; tot = 0.0
            for v in vals:
                m = score == v; sz = m.sum()
                if rem >= sz: rem -= sz; continue
                tot += (1 - rem / sz) * rt[m].sum(); rem = 0
            return tot / n + 0.5 * a
        SEL = {}
        for g in G:
            s = [v.mean() - vote_at(i_.mean()) for v, i_ in G[g] if i_.mean() >= nonans - 1e-9]
            if s: SEL[g] = (float(np.mean(s)), len(s), float(min(s)), float(max(s)))
        show = ["sim00fn00"] + sorted((g for g in SEL if cls_of(g) == 0), key=lambda g: int(g[7:])) + [g for g in ("judgeQt50", "judgeQa50", "judgeQa35", "judgeMa50", "judgeMa35", "judgeQRt50", "apiS55") if g in SEL]
        out.append("  C. reward minus the own-vote rule at the same abstention (runs; range): " + "; ".join(f"{g} {SEL[g][0]:+.3f} ({SEL[g][1]}; {SEL[g][2]:+.3f} to {SEL[g][3]:+.3f})" for g in show if g in SEL))
        allm = sorted(((v[0], g, v[1]) for g, v in SEL.items() if v[1] >= 2), reverse=True)
        out.append(f"     of {len(allm)} arms with >= 2 such runs: highest " + ", ".join(f"{g} {m:+.3f}" for m, g, _ in allm[:4]) + "; lowest " + ", ".join(f"{g} {m:+.3f}" for m, g, _ in allm[-3:])
                   + f"; arms above +0.010: {sum(m > 0.010 for m, _, _ in allm)}")
        J["C"] = {g: SEL[g] for g in SEL}; J["audit_missing"] = (nna, nkw)
        for g in G:
            for v, i_ in G[g]:
                if i_.mean() >= nonans - 1e-9: SELALL.append((cell, g, float(v.mean() - vote_at(i_.mean()))))
        out.append(f"     key-wrong trained answers without an audit verdict (counted wrong): {nna} of {nkw} ({100 * nna / max(nkw, 1):.1f}%)")
        # ---------- D
        def boot(a, b):
            A = np.array([v for v, _ in G[a]]); Bm = np.array([v for v, _ in G[b]]); d = np.empty(NB)
            for t in range(NB):
                q = rng.integers(0, n, n); d[t] = A[rng.integers(0, len(A), len(A))][:, q].mean() - Bm[rng.integers(0, len(Bm), len(Bm))][:, q].mean()
            return float(A.mean() - Bm.mean()), d
        votes = sorted((g for g in G if cls_of(g) == 0), key=lambda g: int(g[7:]))
        J["D"] = {}
        if votes:
            near = min(votes, key=lambda g: abs(int(g[7:]) - kstar)); J["vote_near"] = near
            out.append(f"  D. own vote against other graders (difference in reward, 95% interval; vote arm nearest the calibrated cut: {near}):")
            for v in votes:
                row = []
                for b in ("sim00fn00", "judgeQt50", "judgeQa50", "judgeQa35", "apiS55", "judgeQRt50"):
                    if b not in G: continue
                    m, d = boot(v, b); lo, hi_ = np.percentile(d, [2.5, 97.5]); row.append(f"{b} {m:+.3f} [{lo:+.3f}, {hi_:+.3f}]"); J["D"][f"{v}-{b}"] = (m, float(lo), float(hi_))
                    if v == near: POOL[b].append((cell, m, d))
                out.append(f"     {v} ({len(G[v])} runs, abstains {ab(v):.2f}) minus: " + "; ".join(row))
        # ---------- E
        J["E"] = []
        for v in votes:
            k = int(g_ := v[7:]); GAPS.append((cell, k, mean(v) - rk[k], ab(v) - ak[k], mean(v), rk[k])); J["E"].append((k, mean(v), rk[k], ab(v), ak[k]))
        CELLS[cell] = dict(rk=rk, kstar=kstar, key=mean("sim00fn00") if "sim00fn00" in G else float("nan"), keyn=len(G.get("sim00fn00", [])), base=base,
                           bestj=max((mean(g) for g in G if re.fullmatch(r"judge[QMGO](t50|a50|a35|acc)", g) and len(G[g]) >= 2), default=float("nan")))
        # ---------- F, G (PopQA only)
        if ds == "pq":
            pq = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqa_1500.jsonl"))}; probe = {p["idx"]: p for p in map(json.loads, open(f"{DATA}/pqq_1500.jsonl"))}
            ids = sorted(probe); pc = np.array([probe[i]["cls"] for i in ids]); gk = {}
            for r in map(json.loads, open(f"{BASE}/{st}_pqq_1500_n32_s1.jsonl")):
                if r["s"] < 0: x = ans(r); gk[r["idx"]] = bool(x) and not IDKF.match(x) and bool(C.tq_correct(x, set(probe[r["idx"]]["aliases"])))
            kn = np.array([gk[i] for i in ids]); mF = kn & (pc == "F"); mP = kn & (pc == "P"); PR = defaultdict(list)
            for f in sorted(glob.glob(f"{P2}/{st}_FK20*_idkc05_s*_pq.jsonl")):
                name = os.path.basename(f)[:-6]
                if name not in R: continue
                a = float((np.asarray(R[name]["o"]) == 0)[new].mean())
                if a >= 0.95: continue
                o = {r["idx"]: bool(ans(r)) and bool(IDKF.match(ans(r))) for r in map(json.loads, open(f)) if r["s"] < 0}; y = np.array([o[i] for i in ids], float)
                PR[R[name]["g"]].append((a, y[mF].mean(), y[mP].mean()))
            out.append(f"  F. capitals probe: known former-state capitals {int(mF.sum())}, known present-day capitals {int(mP.sum())}. Refused by trained models "
                       f"(arm: runs, held-out abstention | former-state, runs | present-day):")
            J["F"] = {}
            order = ["sim00fn00"] + sorted((g for g in PR if cls_of(g) == 0), key=lambda g: int(g[7:])) + [g for g in ("judgeQt50", "judgeQacc", "judgeQa50", "judgeQa35", "judgeMacc", "judgeMa50", "judgeMa35", "frzQ35", "frzMAJ35", "judgeQRt50", "judgeQRa35", "apiS55") if g in PR]
            for g in order:
                if g not in PR: continue
                L = PR[g]; J["F"][g] = [list(map(float, x)) for x in L]
                out.append(f"     {g:11s}: {len(L)}, {np.mean([x[0] for x in L]):.2f} | {np.mean([x[1] for x in L]):.2f} ({', '.join(f'{x[1]:.2f}' for x in L)}) | {np.mean([x[2] for x in L]):.2f}")
            ctrl = [x for g, L in PR.items() if re.fullmatch(r"sim\d\dfn\d\d", g) and g != "sim00fn00" for x in L]
            if ctrl:
                for v in sorted((g for g in PR if cls_of(g) == 0), key=lambda g: int(g[7:])):
                    a = np.mean([x[0] for x in PR[v]]); nearc = sorted(ctrl, key=lambda c: abs(c[0] - a))[:5]
                    out.append(f"     random-rejection runs nearest in abstention to {v} ({a:.2f}): abstention {np.mean([c[0] for c in nearc]):.2f}, former-state {np.mean([c[1] for c in nearc]):.2f}, present-day {np.mean([c[2] for c in nearc]):.2f}")
            fresh = [json.loads(l) for l in open(f"{DATA}/pqx_fresh.jsonl")]; NF = len(fresh)
            def fresh_reward(path):
                v = np.zeros(NF)
                for q in map(json.loads, open(path)):
                    if q["s"] < 0:
                        x = ans(q); v[q["idx"]] = 0.5 if (x and IDKF.match(x)) else float(bool(x) and bool(C.tq_correct(x, set(fresh[q["idx"]]["aliases"]))))
                return v
            fb = f"{BASE}/{st}_pqx_fresh_n32_s1.jsonl"; FRW = defaultdict(list)
            for name, r in R.items():
                if r["tag"] != "FK20" or r["c"] != "05" or (np.asarray(r["o"]) == 0)[new].mean() >= 0.95: continue
                ff = f"{FR}/{name}.jsonl"
                if os.path.exists(ff): FRW[r["g"]].append(fresh_reward(ff))
            if os.path.exists(fb) and FRW:
                ub = fresh_reward(fb).mean(); J["G"] = dict(untrained=float(ub))
                order = ["sim00fn00"] + sorted((g for g in FRW if cls_of(g) == 0), key=lambda g: int(g[7:])) + [g for g in ("judgeQt50", "judgeQacc", "judgeQa50", "judgeQa35", "judgeMa50", "judgeMa35", "judgeQRt50", "apiS55") if g in FRW]
                out.append(f"  G. fresh questions ({NF}, key truth): untrained {ub:.3f}; " + "; ".join(f"{g} {np.mean([v.mean() for v in FRW[g]]):.3f} ({len(FRW[g])})" for g in order if g in FRW))
                for g in order:
                    if g in FRW: J["G"][g] = (float(np.mean([v.mean() for v in FRW[g]])), len(FRW[g]))
                out.append("  H. three tests (arm: held-out reward minus key | fresh reward minus key | former-state capitals refused, present-day refused):")
                J["H"] = {}
                for g in sorted((g for g in G if cls_of(g) == 0), key=lambda g: int(g[7:])) + [g for g in ("judgeQt50", "judgeQacc", "judgeQa50", "judgeQa35", "judgeMt50", "judgeMacc", "judgeMa50", "judgeMa35", "judgeQRt50", "apiS55") if g in G]:
                    m, d = boot(g, "sim00fn00"); t1 = f"{m:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] ({len(G[g])})"; t2 = "no fresh answers"; t3 = "no probe answers"; fm = None
                    if g in FRW:
                        A = np.array(FRW[g]); Bm = np.array(FRW["sim00fn00"]); d = np.empty(NB)
                        for t in range(NB):
                            q = rng.integers(0, NF, NF); d[t] = A[rng.integers(0, len(A), len(A))][:, q].mean() - Bm[rng.integers(0, len(Bm), len(Bm))][:, q].mean()
                        fm = float(A.mean() - Bm.mean()); t2 = f"{fm:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] ({len(A)})"
                    if g in PR: t3 = f"{np.mean([x[1] for x in PR[g]]):.2f}, {np.mean([x[2] for x in PR[g]]):.2f} ({len(PR[g])})"
                    J["H"][g] = (m, fm, float(np.mean([x[1] for x in PR[g]])) if g in PR else None)
                    out.append(f"     {g:11s}: {t1} | {t2} | {t3}")
                vv = [g for g in FRW if cls_of(g) == 0]
                if vv and "sim00fn00" in FRW:
                    v = min(vv, key=lambda g: abs(int(g[7:]) - kstar)); A = np.array(FRW[v]); Bm = np.array(FRW["sim00fn00"]); d = np.empty(NB)
                    for t in range(NB):
                        q = rng.integers(0, NF, NF); d[t] = A[rng.integers(0, len(A), len(A))][:, q].mean() - Bm[rng.integers(0, len(Bm), len(Bm))][:, q].mean()
                    out.append(f"     {v} minus answer key on fresh questions: {A.mean() - Bm.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
# ---------- pooled D
out.append("\n==================== POOLED over cells (own-vote arm nearest the calibrated cut; cells weighted equally; cells resampled independently)")
J = JS["pooled"] = {}
for b, L in POOL.items():
    d = np.mean([x[2] for x in L], axis=0); m = float(np.mean([x[1] for x in L])); lo, hi_ = np.percentile(d, [2.5, 97.5]); J[b] = (m, float(lo), float(hi_), len(L))
    out.append(f"   own vote minus {b:11s}: {m:+.3f} [{lo:+.3f}, {hi_:+.3f}] over {len(L)} cells ({'; '.join(f'{c} {x:+.3f}' for c, x, _ in L)}); share of draws within +-0.015: {np.mean(np.abs(d) <= 0.015):.2f}; above -0.010: {np.mean(d > -0.010):.2f}")
    if b == "sim00fn00":
        L5 = [x for x in L if x[0] != "llama8 tqa"]; d5 = np.mean([x[2] for x in L5], axis=0)
        out.append(f"      without llama8 tqa (only strict votes exist there): {np.mean([x[1] for x in L5]):+.3f} [{np.percentile(d5, 2.5):+.3f}, {np.percentile(d5, 97.5):+.3f}]; within +-0.015: {np.mean(np.abs(d5) <= 0.015):.2f}")
# ---------- E
out.append("\n==================== E. TRACKING of the own-vote runs (cell, k: trained reward | rule reward | gap; trained abstention minus rule abstention)")
for c, k, gp, da, m, r_ in GAPS: out.append(f"   {c:11s} k={k:2d}: {m:.3f} | {r_:.3f} | {gp:+.3f}; abstention {da:+.3f}")
gp = np.array([x[2] for x in GAPS]); sd = float(gp.std(ddof=1))
out.append(f"   gap over {len(gp)} (cell, k) pairs: mean {gp.mean():+.4f}, sd {sd:.4f}, range {gp.min():+.3f} to {gp.max():+.3f}; correlation of trained and rule reward across pairs {np.corrcoef([x[4] for x in GAPS], [x[5] for x in GAPS])[0, 1]:.3f}")
loo = []
for c in CELLS:
    oth = np.array([x[2] for x in GAPS if x[0] != c]); loo += [abs(x[4] - (x[5] + oth.mean())) for x in GAPS if x[0] == c]
out.append(f"   leave-one-cell-out: mean absolute error of 'rule reward + mean gap of the other cells' {np.mean(loo):.4f} (max {np.max(loo):.3f})")
out.append("   predicted trained reward (prediction +- 1.96 sd of gaps) | answer key trained (runs) | best same-size judge arm with >= 2 runs:")
JS["E_pred"] = {}
for c, D in CELLS.items():
    oth = np.array([x[2] for x in GAPS if x[0] != c]).mean(); row = []
    for lab, k in (("k=16", 16), (f"k*={D['kstar']}", D["kstar"])):
        p = D["rk"].get(k, D["base"]) + oth; row.append(f"{lab}: {p:.3f} [{p - 1.96 * sd:.3f}, {p + 1.96 * sd:.3f}]"); JS["E_pred"][f"{c} {lab}"] = (float(p), float(p - 1.96 * sd), float(p + 1.96 * sd))
    out.append(f"   {c:11s} " + "; ".join(row) + f" | key {D['key']:.3f} ({D['keyn']}) | judge {D['bestj']:.3f}")
out.append("\n==================== I. SUMMARY of C: reward minus the own-vote rule at the same abstention, all cells")
S = np.array([x[2] for x in SELALL]); arms = defaultdict(list)
for c, g, v in SELALL: arms[(c, g)].append(v)
am = {k: np.mean(v) for k, v in arms.items() if len(v) >= 2}
out.append(f"   runs {len(S)}: mean {S.mean():+.4f}, median {np.median(S):+.4f}; above 0: {np.mean(S > 0):.2f}; above +0.010: {np.mean(S > 0.010):.3f}; above +0.020: {np.mean(S > 0.020):.3f}; max {S.max():+.3f}")
out.append(f"   arms with >= 2 runs (arm = grader in one cell) {len(am)}: above 0: {sum(v > 0 for v in am.values())}; above +0.010: {sum(v > 0.010 for v in am.values())}; above +0.020: {sum(v > 0.020 for v in am.values())}; "
           + "the arms above +0.010: " + "; ".join(f"{c} {g} {v:+.3f} ({len(arms[(c, g)])})" for (c, g), v in sorted(am.items(), key=lambda kv: -kv[1]) if v > 0.010))
for lab, pat in (("answer key", r"sim00fn00"), ("own vote", r"frzVOTE\d+"), ("Sonnet", r"apiS55"), ("judge shown the reference, default", r"judgeQRt50"), ("same-size judges, default cut", r"judge[QMGO]t50"),
                 ("same-size judges, key-matched / 50% cut", r"judge[QMGO](a50|acc)"), ("same-size judges, 35% cut", r"judge[QMGO]a35"), ("key with random rejections", r"sim00fn[1-9]\d")):
    v = np.array([x[2] for x in SELALL if re.fullmatch(pat, x[1])]); cm = [np.mean([x[2] for x in SELALL if re.fullmatch(pat, x[1]) and x[0] == c]) for c in CELLS if any(re.fullmatch(pat, x[1]) and x[0] == c for x in SELALL)]
    out.append(f"   {lab:42s}: runs {len(v):3d}, mean {v.mean():+.4f}, range {v.min():+.3f} to {v.max():+.3f}; cell means " + ", ".join(f"{x:+.3f}" for x in cm))
JS["I"] = dict(runs=len(S), mean=float(S.mean()), share_pos=float(np.mean(S > 0)), share_gt1=float(np.mean(S > 0.010)), arms=len(am), arms_gt1=int(sum(v > 0.010 for v in am.values())))
H_ = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(H_, "s18_fix_ceiling_out.txt"), "w").write("\n".join(out) + "\n"); json.dump(JS, open(os.path.join(H_, "s18_fix_ceiling.json"), "w"), indent=1, default=str)
print("\n".join(out[1:]))
