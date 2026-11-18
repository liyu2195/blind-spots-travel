"""(grpo) Readability test on PopQA: give the shared abstention rule one more readable feature set, the relation type of the question
(one-hot 'prop', from the question file), and solve the same stationarity (no fitted constants). If the misses of the conf rule on graders
whose rejections follow relation types (set-wise FN 'fnB80' etc., typeA/typeB FP, and LLM judges) shrink, the missing ingredient is what the
student can read about the grader, not the drift. Main grid (KL 0.2, c 0.5, std, 120 steps), PopQA, both students.
Usage: nice -n 10 python3 -I g11_type.py <st>  -> cache/type_<st>.pkl ; python3 -I g11_type.py report > out_g11_type.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, numpy as np
HERE = (_RA + "/grpo")
if sys.argv[1] != "report":
    st = sys.argv[1]; sys.argv = [sys.argv[0], st, "pq"]
    exec(open(f"{HERE}/g05_model.py").read().split("out = {}")[0])
    allq = {json.loads(l)["idx"]: json.loads(l) for l in open((_RD + "/pqa_1500.jsonl"))}
    props = [allq[int(i)]["prop"] for i in B["ids"]]; P = sorted(set(props))
    OH = np.array([[1.0 if p == q else 0.0 for q in P[1:]] for p in props])        # first type is the reference (absorbed by the bias)
    def solve_t(alpha, pref, agr, c, norm, beta):
        u0 = np.log(pref / (1 - pref)); z = np.log(agr / (1 - agr)); z = (z - z[home].mean()) / (z[home].std() + 1e-9)
        Phi = np.column_stack([np.ones(len(u0)), z, OH]); th = np.zeros(Phi.shape[1]); H = home
        for it in range(8000):
            p = sig(u0 + Phi @ th); g = Fi(p[H], alpha[H], c, norm) - beta * K * (p[H] - pref[H])
            step = Phi[H].T @ g / H.sum(); th = th + 0.5 * step
            if np.abs(step).max() < 1e-7: break
        return sig(u0 + Phi @ th)
    out = {}
    for run, r in sorted(D["rows"].items()):
        if r["alpha"] is None or r["kl"] != 0.2 or r["c"] != 0.5 or r["adv"] != "std" or r["steps"] != 120 or not r["std_home"] or r["fmt"] != "1" or r["force"] or r["prompt"] != "idk": continue
        a = np.where(np.isfinite(r["alpha"]), r["alpha"], 0.5).astype(float); pref, agr = FE["1"]
        p = solve_t(a, pref, agr, r["c"], r["adv"], r["kl"])
        out[run] = dict(type_n=float((p[new] > 0.5).mean()), type_h=float((p[home] > 0.5).mean()))
    pickle.dump(out, open(f"{HERE}/cache/type_{st}.pkl", "wb")); print(st, len(out), "runs;", len(P), "relation types")
else:
    import pandas as pd
    T = pd.read_csv(f"{HERE}/eval_table_k1.0.csv"); P = {}
    for st in ("qwen7", "llama8"): P.update(pickle.load(open(f"{HERE}/cache/type_{st}.pkl", "rb")))
    T = T[T.run.isin(P)].copy(); T["type_n"] = T.run.map(lambda r: P[r]["type_n"])
    T["g"] = np.where(T.fam.str.startswith("judge"), "LLM judges", np.where(T.fam.isin(["setFN", "setFP"]), "set-wise (by type / popularity / copied)", np.where(T.fam.isin(["gold", "uniFP", "uniFN", "uniMix"]), "random-error", "other")))
    def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    for (st, g), d in T.groupby(["st", "g"]):
        if len(d) < 3: continue
        print(f"{st:6s} {g:40s} n {len(d):3d}: conf rule R2 {r2(d.y_n, d.conf_half_n):+.2f} MAE {np.mean(np.abs(d.y_n - d.conf_half_n)):.3f} bias {np.mean(d.conf_half_n - d.y_n):+.3f} | conf + relation type R2 {r2(d.y_n, d.type_n):+.2f} MAE {np.mean(np.abs(d.y_n - d.type_n)):.3f} bias {np.mean(d.type_n - d.y_n):+.3f} | law R2 {r2(d.y_n, d.law_n):+.2f} MAE {np.mean(np.abs(d.y_n - d.law_n)):.3f}")
    d = T; print(f"all PopQA main-grid runs n {len(d)}: conf R2 {r2(d.y_n, d.conf_half_n):+.2f} | conf + type R2 {r2(d.y_n, d.type_n):+.2f} | law R2 {r2(d.y_n, d.law_n):+.2f}")
