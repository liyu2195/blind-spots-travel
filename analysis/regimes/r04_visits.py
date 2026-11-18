"""r04: one pass vs repeated passes (TriviaQA). B1 = every training question seen once (3,840 questions, 120 steps);
B2v / B5v / B10v = 2 / 5 / 10 visits (1,920 / 768 / 384 questions); FK20 = the main recipe (750 questions, ~5 visits).
The B-runs' training questions are a separate pool (idx >= 100000), so all 1,500 evaluation questions are new for them;
comparisons with FK20 use the 750 held-out (new for every run) questions.
A. Training logs (home): share of key-wrong sampled answers that the grader pays ('wrong_paid'), by visit number
   (visit of a step = floor((step-1) * 32 / pool) + 1); for B1, early (steps 1-24) vs late (97-120) first visits.
   Seed-level mean and range.
B. Final greedy answers on new questions (question-level bootstrap within run family, seeds averaged per question):
   IDK, wrong given, right withheld, content hacks (judge-specific, baseline-corrected as in r02), and the located refusal
   of the grader's own rejections (rg.located_stats; reference = no-location grader closest in abstention among FK20 and B1 runs).
Output: r04_visits.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, numpy as np
sys.path.insert(0, (_RA + "/regimes"))
from rg import *
import r02_costs as R2   # reuses per_question / base_pay definitions

out = open(f"{HERE}/r04_visits.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")

POOL = {"B1": 3840, "B2v": 1920, "B5v": 768, "B10v": 384, "FK20": 750}
FAMS = [("Qt50", "JQ_t50", "Q", "t50"), ("Qa35", "JQ_a35", "Q", "a35"), ("hash35", "hash", None, "hash"), ("key", "key", None, "key")]
NOLOC = ("sim00fn00", "sim00fn20", "sim00fn40", "sim00fn50", "sim00fn60", "sim00fn70", "sim20fn00", "sim35fn00", "lenQ", "sim25fn20")

def group(P, fam, kind):
    return [r for r in P["R"].values() if r["fam"] == fam and r["kind"] == kind and r["credit"] == "idkc05"
            and (fam != "hash" or r["g"] == "hash35fn00")]

for st in ("qwen7", "llama8"):
    P = build(st, "h1"); Q = P["Q"]; dsn = Q["dsn"]
    pr(f"\n===== {st} ({'discovery' if st == 'qwen7' else 'confirmation'}) =====")
    # ---------------- A. training logs
    pr("A. home: grader-paid share of key-wrong samples, mean over seeds (range)")
    for label, fam, J, rule in FAMS:
        for kind in ("B1", "B2v", "B5v", "B10v", "FK20"):
            runs = [n for n, r in P["R"].items() if r["fam"] == fam and r["kind"] == kind and r["credit"] == "idkc05" and (fam != "hash" or r["g"] == "hash35fn00")]
            if not runs: continue
            per = []
            for n in runs:
                L = [json.loads(l) for l in open(f"{RL}/{n}/train_log.jsonl")]
                wp = np.array([x["wrong_paid"] if x["wrong_paid"] is not None else np.nan for x in L]); stp = np.arange(1, len(L) + 1)
                vis = (stp - 1) * 32 // POOL[kind] + 1
                if kind == "B1": per.append([np.nanmean(wp[:24]), np.nanmean(wp[96:120])])
                else: per.append([np.nanmean(wp[vis == v]) for v in range(1, 6)])
            per = np.array(per)
            if kind == "B1":
                d = per[:, 1] - per[:, 0]
                pr(f"  {label:6s} {kind:4s} n={len(runs)} first visits early {per[:,0].mean():.3f} -> late {per[:,1].mean():.3f}; change {d.mean():+.3f} (seeds {d.min():+.3f}..{d.max():+.3f})")
            else:
                d = per[:, 1] - per[:, 0]
                pr(f"  {label:6s} {kind:4s} n={len(runs)} by visit 1..5: " + " ".join(f"{x:.3f}" for x in np.nanmean(per, 0)) +
                   f"; visit2 - visit1 {np.nanmean(d):+.3f} (seeds {np.nanmin(d):+.3f}..{np.nanmax(d):+.3f})")
    # ---------------- B. final answers on new questions
    pr("B. new (held-out) questions, per 100: IDK | wrong given | right withheld | judge-specific content hacks | located refusal of own rejections [95% CI]")
    nl_pool = [r for r in P["R"].values() if r["g"] in NOLOC and r["kind"] in ("FK20", "B1") and r["credit"] == "idkc05"]
    for label, fam, J, rule in FAMS:
        for kind in ("B1", "B2v", "B5v", "B10v", "FK20"):
            runs = group(P, fam, kind)
            if not runs: continue
            held = Q["held"]; N = held.sum(); g = [np.arange(N)]
            if J is not None:
                pq = R2.per_question(P, runs, J, rule, st, dsn)
            else:
                pq = R2.per_question(P, runs, None, "key", st, dsn)
            res = {k: boot_ci(lambda ix, k=k: 100 * np.nanmean(pq[k][held][ix]), N, B=1000) for k in ("idk", "wrong", "RW", "Hj")}
            line = f"  {label:6s} {kind:4s} n={len(runs)} | IDK {res['idk'][0]:5.1f} [{res['idk'][1]:5.1f},{res['idk'][2]:5.1f}] | wrong {res['wrong'][0]:5.1f} | withheld {res['RW'][0]:5.1f}"
            if J is not None:
                line += f" | CHj {res['Hj'][0]:+5.2f} [{res['Hj'][1]:+5.2f},{res['Hj'][2]:+5.2f}]"
                U = held & Q["gval"] & Q["aud"] & Q["kstrict"] & np.isfinite(Q["pj"][J]); rej = Q["pj"][J] <= thr(J, st, dsn, rule)
                yG = np.mean([r["o"][U] == 0 for r in runs], axis=0)
                cand = defaultdict(list)
                for r in nl_pool: cand[(r["g"], r["kind"])].append(r)
                best = min(cand.items(), key=lambda kv: abs(np.mean([(r["o"][U] == 0).mean() for r in kv[1]]) - yG.mean()))
                yR = np.mean([r["o"][U] == 0 for r in best[1]], axis=0); strat = np.digitize(Q["conf"][U], CBINS)
                est, ci = boot_located(yG, yR, len(runs), len(best[1]), rej[U], strat, np.zeros(U.sum(), int), N, B=500)
                line += (f" | LOC {est['excess100']:+5.2f} [{ci['excess100'][0]:+5.2f},{ci['excess100'][1]:+5.2f}] dlogOR {est['dlor']:+.2f} "
                         f"[{ci['dlor'][0]:+.2f},{ci['dlor'][1]:+.2f}] (rej {est['n_rej100']:.1f}/100; refuse rej/acc {est['yG_rej']:.2f}/{est['yG_acc']:.2f}; ref {best[0][1]}:{best[0][0]})")
            pr(line)
    # B1 runs: all 1,500 questions are new -> more power for the gate
    pr("B'. B1 vs B10v runs on ALL 1,500 evaluation questions (all new for B-runs): located refusal of the Qwen judge's rejections")
    for label, fam, J, rule in (("Qa35", "JQ_a35", "Q", "a35"), ("Qt50", "JQ_t50", "Q", "t50")):
        for kind in ("B1", "B10v"):
            runs = group(P, fam, kind)
            if not runs: continue
            allq = np.ones(Q["n"], bool); U = allq & Q["gval"] & Q["aud"] & Q["kstrict"] & np.isfinite(Q["pj"][J]); rej = Q["pj"][J] <= thr(J, st, dsn, rule)
            yG = np.mean([r["o"][U] == 0 for r in runs], axis=0)
            cand = defaultdict(list)
            for r in P["R"].values():
                if r["g"] in NOLOC and r["kind"] == "B1": cand[(r["g"], r["kind"])].append(r)
            best = min(cand.items(), key=lambda kv: abs(np.mean([(r["o"][U] == 0).mean() for r in kv[1]]) - yG.mean()))
            yR = np.mean([r["o"][U] == 0 for r in best[1]], axis=0); strat = np.digitize(Q["conf"][U], CBINS)
            est, ci = boot_located(yG, yR, len(runs), len(best[1]), rej[U], strat, np.zeros(U.sum(), int), Q["n"], B=500)
            pr(f"  {label} {kind:4s} n={len(runs)} | IDK on universe {yG.mean():.2f} (ref {best[0][1]} {yR.mean():.2f}) | LOC {est['excess100']:+5.2f} "
               f"[{ci['excess100'][0]:+5.2f},{ci['excess100'][1]:+5.2f}] dlogOR {est['dlor']:+.2f} [{ci['dlor'][0]:+.2f},{ci['dlor'][1]:+.2f}] | refuse rej/acc {est['yG_rej']:.2f}/{est['yG_acc']:.2f} (ref {est['yR_rej']:.2f}/{est['yR_acc']:.2f})")
out.close()
