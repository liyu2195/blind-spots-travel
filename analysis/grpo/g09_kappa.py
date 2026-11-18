"""(grpo) One global constant, if any: effective KL strength kappa * beta in the drift. Fit kappa on qwen7 main-grid runs (all graders with a
re-implemented grader; rule conf, threshold 1/2) by least squares on new-question abstention; report the fit on llama8 main grid and the KL
variants (KL 0.04-1.0) for the fitted kappa and for kappa = 1 (no constant).
Usage: nice -n 10 python3 -I g09_kappa.py <st> <ds> -> cache/kappa_<st>_<ds>.pkl ; then python3 -I g09_kappa.py report > out_g09_kappa.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, hashlib, glob, numpy as np
HERE = (_RA + "/grpo"); KS = (0.5, 0.75, 1.0, 1.5, 2.0)
if sys.argv[1] != "report":
    sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
    exec(open(f"{HERE}/g05_model.py").read().split("out = {}")[0])
    out = {}; seen = {}
    for run, r in sorted(D["rows"].items()):
        if r["alpha"] is None or r["fmt"] != "1" or r["kl"] <= 0 or r["force"] or r["switch"] or r["prompt"] != "idk" or r["temp"] != 0.8 or r["lr"] != 5e-5 or not r["std_home"]: continue
        a = np.where(np.isfinite(r["alpha"]), r["alpha"], 0.5).astype(float)
        h = hashlib.md5(a.tobytes() + f"{r['c']}{r['adv']}{r['kl']}".encode()).hexdigest()
        if h in seen: out[run] = out[seen[h]]; continue
        seen[h] = run; pref, agr = FE[r["fmt"]]; rec = {}
        for k in KS:
            p, _ = solve(a, pref, agr, r["c"], r["adv"], k * r["kl"], "conf"); rec[k] = float((p[new] > 0.5).mean())
        out[run] = rec
    pickle.dump(out, open(f"{HERE}/cache/kappa_{st}_{ds}.pkl", "wb")); print(st, ds, len(seen), "distinct", len(out), "runs")
else:
    import pandas as pd
    T = pd.read_csv(f"{HERE}/eval_table_k1.0.csv"); T["fmt"] = T.fmt.astype(str)
    P = {}
    for f in glob.glob(f"{HERE}/cache/kappa_*.pkl"): P.update(pickle.load(open(f, "rb")))
    T = T[T.run.isin(P)].copy()
    for k in KS: T[f"k{k}"] = T.run.map(lambda r: P[r][k])
    main = (T.kl == 0.2) & (T.c == 0.5) & (T.adv == "std") & (T.steps == 120)
    def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    Q = T[main & (T.st == "qwen7")]
    mse = {k: float(np.mean((Q.y_n - Q[f"k{k}"]) ** 2)) for k in KS}; kb = min(mse, key=mse.get)
    print("qwen7 main grid (discovery), MSE by kappa:", {k: round(v, 4) for k, v in mse.items()}, "-> kappa", kb)
    for nm, m in (("llama8 main grid", main & (T.st == "llama8")), ("small FK20 c 0.5", main & T.st.isin(["phi4m", "smol3", "granite"])),
                  ("KL != 0.2 (all students)", (T.kl != 0.2)), ("Dr. GRPO FK20D", T.adv == "none"), ("credit != 0.5", (T.c != 0.5) & (T.adv == "std") & (T.kl == 0.2))):
        D_ = T[m]
        if len(D_) < 3: continue
        print(f"  {nm:26s} n {len(D_):3d}: " + "  ".join(f"k{k}: R2 {r2(D_.y_n, D_[f'k{k}']):+.2f} MAE {np.mean(np.abs(D_.y_n - D_[f'k{k}'])):.3f}" for k in KS))
    D_ = T[T.kl != 0.2]
    print(f"  same KL != 0.2 group, leniency curve: R2 {r2(D_.y_n, D_.law_n):+.2f} MAE {np.mean(np.abs(D_.y_n - D_.law_n)):.3f}; runs by KL {D_.kl.value_counts().sort_index().to_dict()}")
