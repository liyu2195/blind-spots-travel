"""r02c: per-dataset cost rows, and each operating point against the student's own confidence ranking at the same abstention.
Confidence rule = on each dataset's new questions, refuse the same number of questions as the run family does (mean over seeds),
choosing the lowest untrained agreement first (untrained IDK first), and give the untrained greedy answer elsewhere.
Excess = family's (wrong given + right withheld) minus the rule's, per 100 new questions (question-level bootstrap CI).
Output: r02c_frontier.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, numpy as np
sys.path.insert(0, (_RA + "/regimes"))
from rg import *
import r02_costs as R2

OPS = [("Q_t50", "JQ_t50", "Q", "t50"), ("Q_a50", "JQ_a50", "Q", "a50"), ("Q_a35", "JQ_a35", "Q", "a35"), ("M_a35", "JM_a35", "M", "a35"),
       ("G_a35", "JG_a35", "G", "a35"), ("O_a35", "JO_a35", "O", "a35"), ("G_t50", "JG_t50", "G", "t50"), ("O_t50", "JO_t50", "O", "t50"),
       ("sonnet", "sonnet", None, "son"), ("key", "key", None, "key"), ("lenQ", "lenQ", "Q", "t50"), ("keyFN40", None, None, "fn40")]
out = open(f"{HERE}/r02c_frontier.txt", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")

for st in ("qwen7", "llama8"):
    pr(f"\n===== {st} ({'discovery' if st == 'qwen7' else 'confirmation'}) ===== per 100 new questions")
    PS = {ds: build(st, ds) for ds in ("h1", "pq", "nq")}
    for ds in ("h1", "pq", "nq"):
        P = PS[ds]; Q = P["Q"]; held = Q["held"]; N = held.sum()
        conf = np.where(Q["gval"], Q["conf"], -1.0)[held]; uright = (Q["gval"] & Q["aud"])[held]; uwrong = (Q["gval"] & ~Q["aud"])[held]
        pr(f"--- {ds}: untrained right {100*uright.mean():.1f} wrong {100*uwrong.mean():.1f} IDK/invalid {100*(1-uright.mean()-uwrong.mean()):.1f}")
        for label, fam, J, rule in OPS:
            runs = R2.fam_runs(P, fam, label)
            if not runs: continue
            pq = R2.per_question(P, runs, J, rule if rule != "fn40" else "key", st, Q["dsn"])
            idk = pq["idk"][held]; wrong = pq["wrong"][held]; rw = pq["RW"][held]
            def rule_cost(ix):
                k = int(round(idk[ix].mean() * len(ix))); order = np.argsort(conf[ix] + 1e-6 * np.arange(len(ix)), kind="stable")
                ref = np.zeros(len(ix), bool); ref[order[:k]] = True
                return 100 * ((uwrong[ix] & ~ref).mean() + (uright[ix] & ref).mean())
            def excess(ix): return 100 * (wrong[ix].mean() + rw[ix].mean()) - rule_cost(ix)
            e = boot_ci(excess, N, B=800)
            pr(f"  {label:8s} runs {len(runs)} | IDK {100*idk.mean():5.1f} | wrong {100*wrong.mean():5.1f} | withheld {100*rw.mean():5.1f} | "
               f"conf rule at same IDK: wrong+withheld {rule_cost(np.arange(N)):5.1f} | run wrong+withheld {100*(wrong.mean()+rw.mean()):5.1f} | excess {e[0]:+5.1f} [{e[1]:+5.1f},{e[2]:+5.1f}]")
out.close()
