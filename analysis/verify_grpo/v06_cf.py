"""(verify_grpo) Claim 3 (credit weight gamma). (a) My solver's counterfactuals: for every distinct grader configuration of the KL-0.2 strict-format
120-step runs, the conf-rule prediction of new-question abstention at c in {0.3, 0.5, 0.7, 0.85} x normaliser {std, none} (zero fitted constants).
(b) report: gamma in y = sigmoid(a + b (L - gamma c)) fitted to those counterfactuals and to ACTUAL runs (small students: std FK20 with credits
0.1-0.9; FK20D = Dr. GRPO with credits 0.3-0.7); per-sweep credit slopes, actual vs my model.
Usage: nice -n 10 python3 -I v06_cf.py <st> <ds> -> cache/vcf_<st>_<ds>.pkl ; python3 -I v06_cf.py report > out_v06_cf.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, pickle, hashlib, glob, numpy as np
HERE = (_RA + "/verify_grpo")
if sys.argv[1] != "report":
    src = open(f"{HERE}/v04_solve.py").read().split("out = {}; cache = {}")[0]
    sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]; exec(src)
    out = {}; seen = {}
    for run, r in sorted(D["rows"].items()):
        if r["alpha"] is None or r["prompt"] != "idk" or r["kl"] != 0.2 or r["fmt"] != "1" or r["steps"] != 120 or not r["std_home"] or r["force"] or r["switch"] or r["temp"] != 0.8 or r["lr"] != 5e-5:
            continue
        a = r["alpha"].astype(float); h = hashlib.md5(a.tobytes()).hexdigest()
        if h in seen: out[run] = dict(same_as=seen[h]); continue
        seen[h] = run; pref, agr = feats("1", "sm1"); u0 = np.log(pref / (1 - pref)); z = np.log(agr / (1 - agr)); z = (z - z[home].mean()) / z[home].std()
        Phi = np.column_stack([np.ones(len(u0)), z]); rec = dict(fam=r["fam"])
        for norm in ("std", "none"):
            for c in (0.3, 0.5, 0.7, 0.85):
                p, th, it, gres, stable, _ = solve(a, u0, pref, Phi, c, norm, 0.2)
                rec[(norm, c)] = float((p[new] > 0.5).mean())
        out[run] = rec
    pickle.dump(out, open(f"{HERE}/cache/vcf_{st}_{ds}.pkl", "wb")); print(st, ds, len(seen), "distinct grader configurations,", len(out), "runs")
else:
    import pandas as pd, warnings
    from scipy.optimize import curve_fit
    warnings.filterwarnings("ignore")
    T = pd.read_csv(f"{HERE}/eval_v.csv"); T["fmt"] = T.fmt.astype(str); Lm = dict(zip(T.run, T.L))
    CF = {}
    for f in glob.glob(f"{HERE}/cache/vcf_*.pkl"): CF.update(pickle.load(open(f, "rb")))
    rows = []
    for run, rec in CF.items():
        if "same_as" in rec or run not in Lm: continue
        for norm in ("std", "none"):
            for c in (0.3, 0.5, 0.7, 0.85):
                rows.append(dict(run=run, grp="main" if run.split("_")[0] in ("qwen7", "llama8") else "small", fam=rec["fam"], norm=norm, c=c, y=rec[(norm, c)], L=Lm[run]))
    U = pd.DataFrame(rows)
    lc3 = lambda X, a, b, g: 1 / (1 + np.exp(-(a + b * (X[0] - g * X[1]))))
    def fit(L, c, y):
        try:
            p, cov = curve_fit(lc3, np.vstack([L, c]), y, p0=[0, -8, 0.5], maxfev=50000); return p, np.sqrt(np.diag(cov))
        except Exception: return np.full(3, np.nan), np.full(3, np.nan)
    print("== 1. my model's counterfactual mean abstention on new questions (distinct grader configurations, KL 0.2)")
    print(U.pivot_table(index=["grp", "norm"], columns="c", values="y", aggfunc="mean").round(3).to_string())
    print("\n== 2. gamma fitted to my counterfactuals (claim: main std 0.22, none 0.93; small std 0.33, none 0.94)")
    for g in ("main", "small"):
        for n in ("std", "none"):
            d = U[(U.grp == g) & (U.norm == n)]; p, se = fit(d.L.values, d.c.values, d.y.values)
            print(f"   {g:5s} {n:4s}: gamma {p[2]:+.2f} (se {se[2]:.2f}) a {p[0]:+.2f} b {p[1]:+.2f} (n configs {d.run.nunique()})")
    sm = T.st.isin(["phi4m", "smol3", "granite"]) & (T.kl == 0.2) & (T.fmt == "1") & (T.steps == 120) & (T.temp == 0.8) & (T.lr == 5e-5) & T.fam.isin(["gold", "uniFP", "uniFN", "uniMix"])
    print("\n== 3. gamma fitted to ACTUAL small-student runs (claim: std 0.19 +- 0.05 on 144 runs; Dr. GRPO 0.64 +- 0.24 on 10 runs)")
    for n in ("std", "none"):
        d = T[sm & (T.adv == n)]; p, se = fit(d.L.values, d.c.values, d.y_n.values)
        print(f"   {n:4s}: gamma {p[2]:+.2f} (se {se[2]:.2f}) a {p[0]:+.2f} b {p[1]:+.2f}  n {len(d)} runs, of which c != 0.5: {int((d.c != 0.5).sum())}; credits {sorted(d.c.unique())}")
        if n == "std":
            bs = []
            rng = np.random.default_rng(3); g_ = d.groupby("cfg").indices; ks = list(g_)
            for _ in range(300):
                ii = np.concatenate([g_[k] for k in rng.choice(ks, len(ks))]); dd = d.iloc[ii]; bs.append(fit(dd.L.values, dd.c.values, dd.y_n.values)[0][2])
            print(f"        cluster bootstrap 95% interval for gamma (std): [{np.nanpercentile(bs, 2.5):+.2f}, {np.nanpercentile(bs, 97.5):+.2f}]")
    print("\n== 4. per-sweep credit slopes d(abstention)/dc on new questions: actual runs vs my model's prediction for the same runs")
    SW = T[sm].copy(); SW["grader"] = SW.run.str.extract(r"_FK20D?(sim\d\dfn\d\d)_")[0]
    for (st, adv, gr), d in SW.groupby(["st", "adv", "grader"]):
        if d.c.nunique() < 2: continue
        print(f"   {st:6s} {adv:4s} {gr:10s} runs {len(d):2d} credits {sorted(d.c.unique())}: actual {np.polyfit(d.c, d.y_n, 1)[0]:+.2f}  model {np.polyfit(d.c, d.conf_n_half, 1)[0]:+.2f}")
