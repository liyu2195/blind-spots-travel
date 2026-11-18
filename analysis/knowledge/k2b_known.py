"""(knowledge) Q1 in one number per pair: of the trained models' key-right answers on NEW questions (all FK20 runs, idkc 0.5), the share that (i) agree with
the untrained greedy answer, (ii) appear among the 32 untrained samples, (iii) appear in neither; and the share of the accuracy-among-answered gain that is
selection (k2 rows: sel / (sel + chg), runs with 0.05 <= a <= 0.95).  Output: k2b_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, numpy as np
OUT = (_RA + "/knowledge"); lines = []
def P(s): print(s, flush=True); lines.append(s)
P(__doc__)
tot = np.zeros(3)
for st in ("qwen7", "llama8"):
    for ds in ("h1", "pq", "nq"):
        F = pickle.load(open(f"{OUT}/feat_{st}_{ds}.pkl", "rb")); R = pickle.load(open(f"{OUT}/runs_{st}_{ds}.pkl", "rb")); ho = F["ho"]; c = np.zeros(3); nr = 0
        for r in R.values():
            if r["tag"] != "FK20" or r["c"] != "05": continue
            ok = (r["o"] == 1) & ho; nr += 1
            c += [(ok & r["same_g"]).sum(), (ok & ~r["same_g"] & (r["freq"] > 0)).sum(), (ok & ~r["same_g"] & (r["freq"] == 0)).sum()]
        tot += c; P(f"{st} {ds}: {nr} runs, {int(c.sum())} right answers: same as untrained greedy {c[0] / c.sum():.3f}, other untrained sample {c[1] / c.sum():.3f}, never sampled {c[2] / c.sum():.4f}")
P(f"all: {int(tot.sum())} right answers: same as untrained greedy {tot[0] / tot.sum():.3f}, other untrained sample {tot[1] / tot.sum():.3f}, never sampled {tot[2] / tot.sum():.4f}")
rows = [d for d in pickle.load(open(f"{OUT}/k2_rows.pkl", "rb")) if 0.05 <= d["a"] <= 0.95 and not np.isnan(d["chg"])]
sel = np.array([d["sel"] for d in rows]); chg = np.array([d["chg"] for d in rows])
P(f"accuracy-among-answered gain over the untrained greedy accuracy, {len(rows)} runs: selection {sel.mean():+.3f}, answer change {chg.mean():+.4f}; selection share of the gain {sel.sum() / (sel.sum() + chg.sum()):.3f}; runs with |chg| < 0.03: {np.mean(np.abs(chg) < 0.03):.2f}")
open(f"{OUT}/k2b_out.txt", "w").write("\n".join(lines) + "\n")
