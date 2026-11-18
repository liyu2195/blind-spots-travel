"""(supp_v11) s22b_transition_checks: checks on the item-level transitions of s22 (Nov 18; saved answers and audit verdicts only).
Same classes, runs and rule as s22_transitions.py. For every trained run the difference in true reward to the rule is split into
  changed  = wrong -> right (corrected) plus right -> wrong (points; the second term is negative)
  dropped  = wrong answers dropped beyond the rule minus right answers dropped beyond the rule (half a point each)
  rest     = questions the untrained student did not answer
E. Per student-dataset pair: corrected, right -> wrong, changed, dropped, total, for each grader; the share of all wrong answers corrected.
F. Graders that know the answers minus the student's own vote, pair by pair, with the number of pairs in which the difference is positive, and
   two intervals: (t) mean of pairs +- t * s.e. over pairs; (b) bootstrap that resamples pairs and, within each pair, runs (10,000 draws, seed 0).
G. Each grader against the rule: the same two intervals for total, changed and dropped (these are the intervals for Figure 2).
H. The same with truth = the loose answer key only (no audit verdicts), so that Sonnet is not both grader and auditor.
I. Transition shares on the four pairs that the answer key, Sonnet, the reference judge and the own vote all have (PopQA, NQ-open).
Usage: python3 -I -B s22b_transition_checks.py   Output: s22b_transition_checks_out.txt, .json"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, pickle, numpy as np
from collections import defaultdict
from scipy import stats as sps
sys.path.insert(0, (_RA + "/stats")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_common import *
import aud as audlib
out = [__doc__]; JS = {}
ARMS = [("sim00fn00", "Answer key"), ("apiS55", "Sonnet as judge"), ("judgeQRt50", "Judge shown the reference"), ("judgeOt50", "OLMo judge, default"), ("frzVOTE20", "Student's own vote")]
CL = ("Wc", "Wu", "Rc", "Ru", "N"); CELLS = [(st, ds, dsn) for st in ("qwen7", "llama8") for ds, dsn in (("pq", "pqa"), ("h1", "tqa"), ("nq", "nqo"))]
RUNS = {t: {g: defaultdict(list) for g, _ in ARMS} for t in ("audit", "key")}; SH = {g: defaultdict(list) for g, _ in ARMS}
for st, ds, dsn in CELLS:
    Bp = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{CACHE}/runs_{st}_{ds}.pkl", "rb")); LV = audlib.levels(dsn)
    new = np.asarray(Bp["new"], bool); ii = np.where(new)[0]; n = len(ii); g0 = lambda a: np.asarray(a)[:, 0]
    ok = g0(Bp["valid"]).astype(bool) & ~g0(Bp["idk"]).astype(bool); cnt = np.rint(np.asarray(Bp["conf"], float) * 32).astype(int); score = np.where(ok, cnt, -1)[ii]; cell = f"{st} {dsn}"
    def rule_idk(a):
        rem = a * n; ab = np.zeros(n)
        for v in np.unique(score):
            m = score == v; sz = m.sum()
            if rem >= sz: ab[m] = 1.0; rem -= sz
            else: ab[m] = rem / sz; rem = 0; break
        return ab
    for truth in ("audit", "key"):
        right0 = ok & (g0(Bp["kl"]).astype(bool) | ((g0(Bp["son"]) >= 4) if truth == "audit" else False))
        cls = np.array(["N" if not ok[i] else (("R" if right0[i] else "W") + ("c" if cnt[i] >= 20 else "u")) for i in ii])
        W = np.isin(cls, ("Wc", "Wu")); Rm = np.isin(cls, ("Rc", "Ru")); Nm = cls == "N"
        for g, lab in ARMS:
            for r in R.values():
                if not (r["tag"] == "FK20" and r["c"] == "05" and r["g"] == g and (np.asarray(r["o"]) == 0)[new].mean() < 0.95): continue
                o = np.asarray(r["o"])[ii]; f = r["f"]
                tri = np.array([(o[k] == 1) or (truth == "audit" and o[k] == 2 and LV.get((i, f[i]), 0) >= 4) for k, i in enumerate(ii)]); tid = (o == 0); twr = ~tri & ~tid; ab = rule_idk(float(tid.mean()))
                rr = right0[ii].astype(float) * (1 - ab)
                d = dict(corrected=100 * tri[W].sum() / n, r2w=-100 * twr[Rm].sum() / n, w_dropped=100 * 0.5 * (tid[W].sum() - ab[W].sum()) / n, r_dropped=-100 * 0.5 * (tid[Rm].sum() - ab[Rm].sum()) / n,
                         rest=100 * ((tri[Nm] + 0.5 * tid[Nm]).sum() - 0.5 * ab[Nm].sum()) / n, total=100 * ((tri + 0.5 * tid).mean() - (rr + 0.5 * ab).mean()), abst=float(tid.mean()),
                         corr_share=float(tri[W].mean()), r2w_share=float(twr[Rm].mean()))
                d["changed"] = d["corrected"] + d["r2w"]; d["dropped"] = d["w_dropped"] + d["r_dropped"]
                assert abs(d["changed"] + d["dropped"] + d["rest"] - d["total"]) < 1e-9
                RUNS[truth][g][cell].append(d)
                if truth == "audit":
                    SH[g][cell].append({c: dict(right=float(tri[cls == c].mean()), idk=float(tid[cls == c].mean()), wrong=float(twr[cls == c].mean()), rule=float(ab[cls == c].mean())) for c in CL})
KEYS = ("total", "changed", "dropped", "corrected", "r2w", "rest")
def pm(g, truth, k): return {c: float(np.mean([x[k] for x in v])) for c, v in RUNS[truth][g].items() if v}
def ivals(vals_fn, cells, B=10000, seed=0):
    """vals_fn(cell, rng or None) -> value for that pair (rng: resample runs). Returns mean, t interval over pairs, hierarchical bootstrap interval, pairs positive."""
    v = np.array([vals_fn(c, None) for c in cells]); m = float(v.mean()); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k) if k > 1 else float("nan"); tq = sps.t.ppf(0.975, k - 1) if k > 1 else float("nan")
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        cc = rng.integers(0, k, k); bs.append(np.mean([vals_fn(cells[j], rng) for j in cc]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=m, t=(float(m - tq * se), float(m + tq * se)), boot=(float(lo), float(hi)), pos=int((v > 0).sum()), pairs=k, per_pair={c: float(x) for c, x in zip(cells, v)})
def mk(g, truth, k):
    def fn(c, rng):
        xs = [x[k] for x in RUNS[truth][g][c]]
        return float(np.mean(xs)) if rng is None else float(np.mean([xs[j] for j in rng.integers(0, len(xs), len(xs))]))
    return fn
def mkdiff(g, h, truth, k):
    a, b = mk(g, truth, k), mk(h, truth, k); return lambda c, rng: a(c, rng) - b(c, rng)
fmt = lambda r: f"{r['mean']:+.2f}  t [{r['t'][0]:+.2f}, {r['t'][1]:+.2f}]  boot [{r['boot'][0]:+.2f}, {r['boot'][1]:+.2f}]  positive in {r['pos']} of {r['pairs']} pairs"
for truth, title in (("audit", "truth = loose key or audit level >= 4 (as in the paper)"), ("key", "H. truth = loose answer key only")):
    out.append(f"\n################ {title}"); JS[truth] = dict(per_pair={}, vs_rule={}, vs_vote={})
    out.append("E. per pair (points of true reward against the rule): corrected, right->wrong, changed, dropped, total; share of wrong answers corrected, of right answers turned wrong")
    for g, lab in ARMS:
        out.append(f"   {lab}")
        JS[truth]["per_pair"][g] = {}
        for c, v in RUNS[truth][g].items():
            if not v: continue
            a = {k: float(np.mean([x[k] for x in v])) for k in KEYS + ("abst", "corr_share", "r2w_share")}; JS[truth]["per_pair"][g][c] = dict(a, runs=len(v))
            out.append(f"      {c:11s} ({len(v)} runs, abstains {a['abst']:.2f}): corrected {a['corrected']:+.2f}  right->wrong {a['r2w']:+.2f}  changed {a['changed']:+.2f}  dropped {a['dropped']:+.2f}  total {a['total']:+.2f}   corrected share {100 * a['corr_share']:.1f}%  right turned wrong {100 * a['r2w_share']:.1f}%")
        pp = JS[truth]["per_pair"][g]; out.append(f"      mean of pairs: corrected share {100 * np.mean([x['corr_share'] for x in pp.values()]):.1f}%, right turned wrong {100 * np.mean([x['r2w_share'] for x in pp.values()]):.1f}%")
    out.append("G. each grader against the rule (mean of pairs; t interval over pairs; bootstrap over pairs and runs)")
    for g, lab in ARMS:
        cells = [c for c, v in RUNS[truth][g].items() if v]; JS[truth]["vs_rule"][g] = {}
        for k in ("total", "changed", "dropped", "corrected", "r2w"):
            r = ivals(mk(g, truth, k), cells); JS[truth]["vs_rule"][g][k] = r; out.append(f"   {lab:26s} {k:9s}: {fmt(r)}")
    out.append("F. grader minus the student's own vote, pair by pair")
    for g, lab in ARMS[:4]:
        cells = [c for c, v in RUNS[truth][g].items() if v and RUNS[truth]["frzVOTE20"][c]]; JS[truth]["vs_vote"][g] = {}
        for k in ("total", "changed", "dropped", "corrected", "r2w"):
            r = ivals(mkdiff(g, "frzVOTE20", truth, k), cells); JS[truth]["vs_vote"][g][k] = r
            out.append(f"   {lab:26s} {k:9s}: {fmt(r)}   per pair: " + ", ".join(f"{c.replace('qwen7', 'Q').replace('llama8', 'L')} {x:+.2f}" for c, x in r["per_pair"].items()))
out.append("\n################ I. transition shares on the four pairs all four main graders have (PopQA and NQ-open, both students); audit truth")
JS["four"] = {}
for g, lab in ARMS:
    cs = [c for c in SH[g] if c.split()[1] in ("pqa", "nqo") and SH[g][c]]
    P = {cl: {k: float(np.mean([np.mean([x[cl][k] for x in SH[g][c]]) for c in cs])) for k in ("right", "idk", "wrong", "rule")} for cl in CL}; JS["four"][g] = P
    out.append(f"   {lab:26s} ({len(cs)} pairs): " + " | ".join(f"{cl}: {P[cl]['right']:.3f} / {P[cl]['idk']:.3f} ({P[cl]['rule']:.3f}) / {P[cl]['wrong']:.3f}" for cl in CL))
H_ = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(H_, "s22b_transition_checks_out.txt"), "w").write("\n".join(out) + "\n"); json.dump(JS, open(os.path.join(H_, "s22b_transition_checks.json"), "w"), indent=1)
print("\n".join(out[1:]))
