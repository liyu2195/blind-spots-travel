"""(arch_v4a) Pool the zero-constant theory's predictions (grpo/eval_table_k1.0.csv, variant conf_half, kappa 1) over every run whose
RECIPE differs from the design recipe (credit 0.5, KL 0.2, std-normalised advantages, 120 steps, temperature 0.8, lr 5e-5), and compare with the
leniency curve fitted on qwen7 main-grid random-error graders (column law_n, from grpo/g06_eval.py). Read-only re-aggregation; no new modelling.
Bootstrap over grader configurations (cfg = all seeds of one grader setting) for R2 / MAE intervals.
Usage: nice -n 10 python3 -I a01_pool_oor.py > a01_pool_oor_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import numpy as np, pandas as pd
T = pd.read_csv((_RA + "/grpo/eval_table_k1.0.csv"))
T = T[T.prompt == "idk"].copy()
def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
def summ(D, name):
    y, m, l = D.y_n.values, D.conf_half_n.values, D.law_n.values
    rng = np.random.default_rng(0); cf = D.cfg.unique(); B = []
    rows_of = [np.where(D.cfg.values == c)[0] for c in cf]
    for _ in range(1000):
        pick = rng.integers(0, len(cf), len(cf)); ix = np.concatenate([rows_of[k] for k in pick])
        yy, mm, ll = y[ix], m[ix], l[ix]
        B.append((r2(yy, mm), np.mean(np.abs(yy - mm)), r2(yy, ll), np.mean(np.abs(yy - ll)), np.mean(np.abs(yy - ll)) - np.mean(np.abs(yy - mm))))
    B = np.array(B); q = lambda j: f"[{np.percentile(B[:, j], 2.5):+.2f}, {np.percentile(B[:, j], 97.5):+.2f}]"
    print(f"{name:58s} runs {len(D):4d} cfgs {len(cf):3d} | theory R2 {r2(y, m):+.2f} {q(0)} MAE {np.mean(np.abs(y - m)):.3f} | "
          f"curve R2 {r2(y, l):+.2f} {q(2)} MAE {np.mean(np.abs(y - l)):.3f} | MAE gain {np.mean(np.abs(y - l)) - np.mean(np.abs(y - m)):+.3f} {q(4)}")
small = T.st.isin(["phi4m", "smol3", "granite"]); mainst = T.st.isin(["qwen7", "llama8"])
inscope = (T.kl > 0) & (T.fmt == 1)
design = (T.c == 0.5) & (T.kl == 0.2) & (T.adv == "std") & (T.steps == 120)
oor = inscope & ~design
print("design recipe: credit 0.5, KL 0.2, std, 120 steps. 'out of recipe' = KL > 0, strict format, and at least one of these changed.\n")
summ(T[oor], "ALL out-of-recipe runs (KL > 0, format 1)")
summ(T[oor & mainst], "  main students (qwen7, llama8)")
summ(T[oor & small], "  small students (phi4m, smol3, granite)")
summ(T[oor & (T.c != 0.5)], "  credit != 0.5 (any student)")
summ(T[oor & (T.adv == "none")], "  Dr. GRPO (no std normalisation)")
summ(T[oor & (T.kl != 0.2)], "  KL != 0.2")
summ(T[oor & (T.steps != 120)], "  steps != 120")
print("\nfor reference, in-recipe confirmation (llama8 + small students, design recipe, all graders):")
summ(T[design & inscope & (T.main) & (T.st != "qwen7")], "  design recipe, not qwen7")
print("\nrecipe counts among out-of-recipe runs:")
print(T[oor].groupby(["adv", "kl", "c", "steps"]).size().to_string())
