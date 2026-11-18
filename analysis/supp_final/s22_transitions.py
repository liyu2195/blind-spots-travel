"""(supp_final) s22_transitions: item-level transitions on held-out questions (Nov 18; saved answers and audit verdicts only, no training).
Question: does "no grader beats the student's own vote" hold item by item, or do graders that know the answers fix or drop some wrong answers and
lose some right ones, cancelling in the average?
For each held-out question the UNTRAINED student's greedy answer falls in one class (truth = loose key or audit level >= 4):
  Wc  wrong, at least 20 of its 32 samples agree with it (a confident error: the own vote at a cut of 20 would pay it)
  Wu  wrong, fewer than 20 agree
  Rc  right, at least 20 agree
  Ru  right, fewer than 20 agree
  N   no answer ("I don't know" or malformed)
For each trained run (FK20, credit 0.5; collapsed runs dropped) the trained greedy answer is right / IDK / wrong (malformed counts as wrong).
Next to every run stands the RULE: the untrained student abstaining on the same share of held-out questions, fewest agreeing samples first (class N
first; ties shared). The rule cannot turn a wrong answer into a right one; it can only drop answers.
A. Transition shares within each class, per grader, mean over runs and then over the six student-dataset pairs; the rule's shares next to them.
B. Where the difference in true reward (trained minus rule, points) comes from: corrected answers (W -> right), wrong answers dropped beyond the
   rule (W -> IDK, half a point each), right answers lost beyond the rule (R -> IDK, half a point; R -> wrong, one point), and questions the
   untrained student did not answer.
C. The same without near-duplicates of training questions (TF-IDF cosine >= 0.5 to a training question, or the same gold answer; as stats/c1b).
D. PopQA: corrected answers (W -> right) by relation type, share of them that are near-duplicates, and the length of corrected answers.
Graders: answer key, Sonnet as judge, judge shown the reference (default threshold), OLMo judge (default threshold and 35% cut), the student's own
vote (cut 20, all seeds), Qwen judge at the 35% cut.
Usage: python3 -I -B s22_transitions.py   Output: s22_transitions_out.txt, s22_transitions.json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, re, numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_common import *
import aud as audlib
out = [__doc__]; JS = {}
ARMS = [("sim00fn00", "Answer key"), ("apiS55", "Sonnet as judge"), ("judgeQRt50", "Judge shown the reference"), ("frzVOTE20", "Student's own vote"), ("judgeOt50", "OLMo judge, default"),
        ("judgeOa35", "OLMo judge, 35% cut"), ("judgeQa35", "Qwen judge, 35% cut")]
CL = ("Wc", "Wu", "Rc", "Ru", "N"); CELLS = [(st, ds, dsn) for st in ("qwen7", "llama8") for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo"))]
ACC = {sub: {g: defaultdict(list) for g, _ in ARMS} for sub in ("all", "nodup")}; KIND = defaultdict(lambda: defaultdict(list))
for st, ds, dsn in CELLS:
    Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn)
    new = np.asarray(Bp["new"], bool); tr = np.where(Bp["train"])[0]; ii = np.where(new)[0]; n = len(ii); g0 = lambda a: np.asarray(a)[:, 0]
    ok = g0(Bp["valid"]).astype(bool) & ~g0(Bp["idk"]).astype(bool); right0 = ok & (g0(Bp["kl"]).astype(bool) | (g0(Bp["son"]) >= 4))
    cnt = np.rint(np.asarray(Bp["conf"], float) * 32).astype(int); gre = [x[0] for x in Bp["X"]]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(Bp["question"]); M = vec.transform(Bp["question"])
    sim = (M[ii] @ M[tr].T).toarray().max(1); trg = {Bp["gold"][i].strip().lower() for i in tr}; dup = (sim >= 0.5) | np.array([Bp["gold"][i].strip().lower() in trg for i in ii])
    cls = np.array(["N" if not ok[i] else (("R" if right0[i] else "W") + ("c" if cnt[i] >= 20 else "u")) for i in ii]); score = np.where(ok, cnt, -1)[ii]
    prop = np.array([str(x) for x in Bp["prop"]])[ii] if ds == "pq" else None; cell = f"{st} {dsn}"
    out.append(f"\n== {cell}: held-out {n}; classes " + ", ".join(f"{c} {int((cls == c).sum())}" for c in CL) + f"; near-duplicates of training questions {int(dup.sum())}")
    def rule_idk(a, mask):
        """expected share of the questions in mask that the rule abstains on when it abstains on share a of all n questions"""
        rem = a * n; ab = np.zeros(n)
        for v in np.unique(score):
            m = score == v; sz = m.sum()
            if rem >= sz: ab[m] = 1.0; rem -= sz
            else: ab[m] = rem / sz; rem = 0; break
        return ab
    for g, lab in ARMS:
        runs = [r for r in R.values() if r["tag"] == "FK20" and r["c"] == "05" and r["g"] == g and (np.asarray(r["o"]) == 0)[new].mean() < 0.95]
        if not runs: continue
        for sub, keep in (("all", np.ones(n, bool)), ("nodup", ~dup)):
            rows = []
            for r in runs:
                o = np.asarray(r["o"])[ii]; f = r["f"]; tri = np.array([(o[k] == 1) or (o[k] == 2 and LV.get((i, f[i]), 0) >= 4) for k, i in enumerate(ii)]); tid = (o == 0); twr = ~tri & ~tid
                ab = rule_idk(float(tid.mean()), None); d = dict(abst=float(tid[keep].mean()), reward=float((tri + 0.5 * tid)[keep].mean()))
                rr = (right0[ii] & (1 - ab > 0)).astype(float) * (1 - ab); d["rule_reward"] = float((rr + 0.5 * ab)[keep].mean())
                for c in CL:
                    m = (cls == c) & keep; k_ = max(int(m.sum()), 1)
                    d[c] = dict(n=int(m.sum()), right=float(tri[m].sum() / k_), idk=float(tid[m].sum() / k_), wrong=float(twr[m].sum() / k_), rule_idk=float(ab[m].sum() / k_))
                nk = keep.sum(); W = np.isin(cls, ("Wc", "Wu")) & keep; Rm = np.isin(cls, ("Rc", "Ru")) & keep; Nm = (cls == "N") & keep
                d["dec"] = dict(corrected=float(100 * tri[W].sum() / nk), w_dropped=float(100 * 0.5 * (tid[W].sum() - ab[W].sum()) / nk), r_lost_idk=float(-100 * 0.5 * (tid[Rm].sum() - ab[Rm].sum()) / nk),
                                r_lost_wrong=float(-100 * twr[Rm].sum() / nk), n_class=float(100 * ((tri[Nm] + 0.5 * tid[Nm]).sum() - 0.5 * ab[Nm].sum()) / nk),
                                total=float(100 * ((tri + 0.5 * tid)[keep].mean() - (rr + 0.5 * ab)[keep].mean())))
                rows.append(d)
                if sub == "all" and ds == "pq" and g in ("sim00fn00", "apiS55", "judgeQRt50", "frzVOTE20"):
                    Wm = np.isin(cls, ("Wc", "Wu"))
                    for p in sorted(set(prop)):
                        m = Wm & (prop == p)
                        if m.sum() >= 5: KIND[(cell, g)][p].append((int(m.sum()), float(tri[m].mean())))
                    fx = Wm & tri
                    KIND[(cell, g)]["_dup_share_of_corrected"].append((int(fx.sum()), float(dup[fx].mean()) if fx.sum() else float("nan")))
                    KIND[(cell, g)]["_len_corrected_vs_untrained"].append((float(np.mean([len(f[i]) for i in ii[fx]])) if fx.sum() else float("nan"), float(np.mean([len(gre[i]) for i in ii[fx]])) if fx.sum() else float("nan")))
            agg = dict(runs=len(rows), abst=float(np.mean([x["abst"] for x in rows])), reward=float(np.mean([x["reward"] for x in rows])), rule_reward=float(np.mean([x["rule_reward"] for x in rows])))
            for c in CL: agg[c] = {k: float(np.mean([x[c][k] for x in rows])) for k in ("n", "right", "idk", "wrong", "rule_idk")}
            agg["dec"] = {k: float(np.mean([x["dec"][k] for x in rows])) for k in rows[0]["dec"]}
            ACC[sub][g][cell] = agg
        a = ACC["all"][g][cell]
        out.append(f"   {lab:26s} ({a['runs']} runs, abstains {a['abst']:.2f}): " + " | ".join(f"{c}: right {a[c]['right']:.2f} idk {a[c]['idk']:.2f} (rule {a[c]['rule_idk']:.2f}) wrong {a[c]['wrong']:.2f}" for c in CL)
                   + f" || reward minus rule {a['dec']['total']:+.1f} = corrected {a['dec']['corrected']:+.1f}, wrong dropped {a['dec']['w_dropped']:+.1f}, right lost to IDK {a['dec']['r_lost_idk']:+.1f}, right turned wrong {a['dec']['r_lost_wrong']:+.1f}, unanswered {a['dec']['n_class']:+.1f}")
for sub, title in (("all", "A / B. POOLED over the student-dataset pairs that have the grader (pairs weighted equally)"), ("nodup", "C. The same WITHOUT near-duplicates of training questions")):
    out.append(f"\n==================== {title}")
    out.append("   transition shares: class: -> right / -> IDK (rule's IDK) / -> wrong")
    JS[sub] = {}
    for g, lab in ARMS:
        cells = ACC[sub][g]
        if not cells: continue
        P = {c: {k: float(np.mean([v[c][k] for v in cells.values()])) for k in ("right", "idk", "wrong", "rule_idk")} for c in CL}; D = {k: float(np.mean([v["dec"][k] for v in cells.values()])) for k in next(iter(cells.values()))["dec"]}
        JS[sub][g] = dict(label=lab, pairs=len(cells), runs=int(sum(v["runs"] for v in cells.values())), shares=P, dec=D, per_pair={c: v["dec"] for c, v in cells.items()})
        out.append(f"   {lab:26s} ({len(cells)} pairs, {JS[sub][g]['runs']} runs): " + " | ".join(f"{c}: {P[c]['right']:.3f} / {P[c]['idk']:.3f} ({P[c]['rule_idk']:.3f}) / {P[c]['wrong']:.3f}" for c in CL))
    out.append("   where the reward difference to the rule comes from (points): total = corrected + wrong dropped beyond the rule + right lost to IDK beyond the rule + right turned wrong + unanswered class")
    for g, lab in ARMS:
        if g not in JS[sub]: continue
        D = JS[sub][g]["dec"]; out.append(f"   {lab:26s}: {D['total']:+.2f} = {D['corrected']:+.2f} {D['w_dropped']:+.2f} {D['r_lost_idk']:+.2f} {D['r_lost_wrong']:+.2f} {D['n_class']:+.2f}")
    out.append("   the same on the four pairs that every one of answer key, Sonnet, reference judge and own vote has (PopQA and NQ-open, both students):")
    for g, lab in ARMS:
        if g not in JS[sub]: continue
        pp = [v for c, v in JS[sub][g]["per_pair"].items() if c.split()[1] in ("pqa", "nqo")]
        if pp: out.append(f"   {lab:26s}: " + " ".join(f"{k} {np.mean([v[k] for v in pp]):+.2f}" for k in ("total", "corrected", "w_dropped", "r_lost_idk", "r_lost_wrong", "n_class")))
out.append("\n==================== D. PopQA: where corrected answers (untrained wrong -> trained right) come from")
JS["D"] = {}
for (cell, g), d in sorted(KIND.items()):
    ks = sorted(((np.mean([x[1] for x in v]), p, v[0][0]) for p, v in d.items() if not p.startswith("_")), reverse=True)
    dp = d["_dup_share_of_corrected"]; ln = d["_len_corrected_vs_untrained"]
    JS["D"][f"{cell} {g}"] = dict(top=[(p, float(m), n_) for m, p, n_ in ks[:4]], corrected_per_run=float(np.mean([x[0] for x in dp])), dup_share=float(np.nanmean([x[1] for x in dp])))
    out.append(f"   {cell} {g:11s}: corrected per run {np.mean([x[0] for x in dp]):.1f}, of them near-duplicates {np.nanmean([x[1] for x in dp]):.2f}; length of the corrected answers {np.nanmean([x[0] for x in ln]):.1f} chars (their untrained answers {np.nanmean([x[1] for x in ln]):.1f}); "
               "W -> right by relation (share, n wrong): " + ", ".join(f"{p} {m:.2f} ({n_})" for m, p, n_ in ks[:5]) + " ... " + ", ".join(f"{p} {m:.2f} ({n_})" for m, p, n_ in ks[-2:]))
H_ = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(H_, "s22_transitions_out.txt"), "w").write("\n".join(out) + "\n"); json.dump(JS, open(os.path.join(H_, "s22_transitions.json"), "w"), indent=1)
print("\n".join(out[1:]))
