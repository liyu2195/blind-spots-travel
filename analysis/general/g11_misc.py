"""(explorer 'general') Two further checks on the small students.
 (A) Knowledge does not change: of the key-right final answers on NEW questions (all strict-format runs with 5-95% abstention),
     the share that is the untrained greedy answer (normalised string equal, or rc.same match), another untrained sample
     (32-sample base file), or never sampled; and the gain in accuracy among answered questions split into which questions are
     answered vs changed answers (answered-set accuracy of the untrained greedy answers on the run's answered questions).
 (B) LLM judges on the leniency curve: for each strict-format judge run (KL 0.2 thresholds; no-KL thresholds), L = share of the
     untrained student's well-formed non-IDK answers to TRAINING questions (greedy + 8 samples of the IDK-prompt ck0 rows that
     both judges scored) with P_J(Yes) > t; predicted abstention from the same student's uniform-grader curve of the same recipe
     (fitted on g03 points); residual = observed - predicted.
Output: g11_misc.txt"""
import json, os, sys, pickle
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcommon import setup, sa, IDK, C, BASE, HERE, read_eval, same, RL

out = open(f"{HERE}/g11_misc.txt", "w")
def P(*a): print(*a, file=out); print(*a, flush=True)
probs, H1, HELD = setup("tqa"); NEW = np.array(HELD)
P("(A) key-right final answers on new questions: same as untrained greedy / other untrained sample / never sampled")
for st in ("phi4m", "smol3"):
    g, samp = {}, defaultdict(set)
    for l in open(f"{BASE}/{st}_tqa_1500_n32_s1.jsonl"):
        r = json.loads(l); x = sa(r)
        if r["s"] < 0: g[r["idx"]] = x
        elif x: samp[r["idx"]].add(x)
    Cc = pickle.load(open(f"{HERE}/cache_{st}.pkl", "rb"))
    tot = np.zeros(4); gains = []; nr = 0
    for run, d in Cc.items():
        m = d["meta"]
        if m["ds"] != "tqa" or m["fmt"] != 1 or m["outlet"] != "idk": continue
        cs, xs = d["f"]["cs"], d["f"]["xs"]; a = (cs[NEW] == 0).mean()
        if not (0.05 <= a <= 0.95) or (cs[NEW] == 3).mean() > 0.05: continue
        nr += 1
        for i in NEW:
            if cs[i] != 1: continue
            x = xs[i]; gx = g.get(i, "")
            if x == gx: tot[0] += 1
            elif gx and same(x, gx): tot[1] += 1
            elif x in samp[i]: tot[2] += 1
            else: tot[3] += 1
        ans = [i for i in NEW if cs[i] in (1, 2)]
        if ans:
            acc_run = np.mean([cs[i] == 1 for i in ans])
            ub = [C.tq_correct(g[i], set(probs[i]["aliases"])) if g.get(i) and not IDK.match(g[i]) else False for i in ans]
            acc_u_same = np.mean(ub)
            allv = [i for i in NEW if g.get(i) and not IDK.match(g[i])]
            acc_u_all = np.mean([C.tq_correct(g[i], set(probs[i]["aliases"])) for i in allv])
            gains.append((acc_run - acc_u_all, acc_u_same - acc_u_all))
    s = tot.sum(); gains = np.array(gains)
    P(f"  {st}: {nr} runs, {int(s)} key-right final answers: identical to untrained greedy {tot[0] / s:.3f}, same answer other form {tot[1] / s:.3f}, "
      f"another untrained sample {tot[2] / s:.3f}, never sampled {tot[3] / s:.3f}; accuracy among answered +{gains[:, 0].mean():.3f} over the untrained greedy, "
      f"of which selection of questions +{gains[:, 1].mean():.3f} ({gains[:, 1].mean() / gains[:, 0].mean():.2f})")

P("\n(B) LLM judges at their thresholds on the uniform-grader leniency curve (strict-format recipes)")
pts = json.load(open(f"{HERE}/g03_points.json"))
def sig(L, a, b): return 1 / (1 + np.exp(-(a + b * L)))
LK = {J: pickle.load(open(f"{HERE}/g02_lookup_{J}_tqa.pkl", "rb")) for J in ("Q9", "G12")}
SRC = {"phi4m": "phi4m_judgeQ_idk_s1_h1", "smol3": "smol3_judgeG_idk_h1"}   # runs whose ck0 greedy + 8 samples both judges scored
for st in ("phi4m", "smol3"):
    u = defaultdict(list)
    for l in open(f"{RL}/{SRC[st]}/eval_rl_ck0.jsonl"):
        r = json.loads(l)
        if r["idx"] in H1:
            x = sa(r)
            if x and not IDK.match(x): u[r["idx"]].append(C.tq_line(r["text"]))
    sc = {J: np.array([LK[J].get((i, ln), np.nan) for i in sorted(u) for ln in u[i]]) for J in LK}
    Cc = pickle.load(open(f"{HERE}/cache_{st}.pkl", "rb"))
    for rec in ("FK20", "F"):
        base = [p for p in pts if p["st"] == st and p["rec"] == rec and p["fam"] in ("key", "sim") and p["credit"] == 0.5 and p["ds"] == "tqa" and p["bad"] <= 0.2]
        if len(base) < 5: P(f"  {st} {rec}: {len(base)} uniform runs, no curve"); continue
        (a, b), _ = curve_fit(sig, [p["L"] for p in base], [p["a"] for p in base], p0=(0, -5), maxfev=20000)
        res = []
        for run, d in sorted(Cc.items()):
            m = d["meta"]
            if m["fam"] != "judge" or m["jm"] not in ("Q9", "G12") or m["jmode"] != "free" or m["pre"] != rec or m["ds"] != "tqa" or m["thr"] is None: continue
            v = sc[m["jm"]]; v = v[~np.isnan(v)]; L = float(np.mean(v > m["thr"]))
            obs = float((d["f"]["cs"][NEW] == 0).mean()); pr = float(sig(L, a, b)); res.append(obs - pr)
            P(f"   {run:32s} L {L:.3f} predicted {pr:.3f} observed {obs:.3f} residual {obs - pr:+.3f}")
        if res:
            res = np.array(res); P(f"  {st} {rec}: {len(res)} judge runs, mean residual {res.mean():+.3f}, mean |residual| {np.abs(res).mean():.3f}, "
                                   f"share within +-0.10: {np.mean(np.abs(res) <= 0.10):.2f} (uniform-curve runs n={len(base)})")
out.close()
