"""(grpo) Is the credit-insensitivity a consequence of std normalisation?
(1) counterfactual model predictions (g07): abstention on new questions by normaliser and credit, the same grader configurations;
(2) the credit coefficient gamma in a leniency law y = 1 / (1 + exp(-(a + b (L - gamma c)))), fitted to the model's counterfactuals and to the
    actual runs (small students: std FK20 credit sweeps vs FK20D = Dr. GRPO; main students: qwen7 judge runs at c 0.5 / 0.7 / 0.85);
(3) per-sweep credit slopes, actual vs model. L = paid share of the untrained valid non-IDK samples on training questions (g06).
Usage: nice -n 10 python3 -I g08_credit.py > out_g08_credit.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import glob, pickle, numpy as np, pandas as pd, warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
HERE = (_RA + "/grpo")
T = pd.read_csv(f"{HERE}/eval_table_k1.0.csv"); T["fmt"] = T.fmt.astype(str)
Lmap = dict(zip(T.run, T.L)); ymap = dict(zip(T.run, T.y_n))
CF = {}
for f in glob.glob(f"{HERE}/cache/cf_*.pkl"): CF.update(pickle.load(open(f, "rb")))
rows = []
for run, rec in CF.items():
    src = rec.get("same_as", run); R = CF[src]
    st = run.split("_")[0]
    for norm in ("std", "none"):
        for c in (0.3, 0.5, 0.7, 0.85):
            rows.append(dict(run=run, src=src, st=st, fam=R["fam"], norm=norm, c=c, y=R[(norm, c, "n")], three=R[(norm, c, "three")], L=Lmap.get(run, np.nan)))
C = pd.DataFrame(rows); C = C[np.isfinite(C.L)]
U = C.drop_duplicates(["src", "norm", "c"])              # one row per distinct grader configuration
U = U.assign(grp=np.where(U.st.isin(["qwen7", "llama8"]), "main", "small"))
print("== 1. model counterfactuals (rule conf, KL 0.2, zero fitted constants): mean abstention on new questions over distinct grader configurations")
print(U.pivot_table(index=["grp", "norm"], columns="c", values="y", aggfunc="mean").round(3).to_string())
print("   share of IDK-containing groups with three reward levels (the only groups where the std advantage depends on c), at the model's stationary policy:")
print(U.pivot_table(index=["grp", "norm"], columns="c", values="three", aggfunc="mean").round(3).to_string())
sl = U.groupby(["grp", "norm", "src"]).apply(lambda g: np.polyfit(g.c, g.y, 1)[0]).reset_index(name="slope")
print("   per-configuration credit slope d abstention / dc (OLS over c 0.3-0.85): mean, [10%, 90%]")
for (g, n), d in sl.groupby(["grp", "norm"]): print(f"     {g:5s} {n:4s}: {d.slope.mean():+.3f} [{d.slope.quantile(.1):+.3f}, {d.slope.quantile(.9):+.3f}]  (n configs {len(d)})")
lc3 = lambda X, a, b, g: 1 / (1 + np.exp(-(a + b * (X[0] - g * X[1]))))
def fit_gamma(L, c, y):
    try:
        p, cov = curve_fit(lc3, np.vstack([L, c]), y, p0=[0, -8, 0.5], maxfev=50000); return p, np.sqrt(np.diag(cov))
    except Exception as e: return np.array([np.nan] * 3), np.array([np.nan] * 3)
print("\n== 2. credit coefficient gamma in y = logistic(a + b (L - gamma c))")
for g in ("main", "small"):
    for n in ("std", "none"):
        d = U[(U.grp == g) & (U.norm == n)]; p, se = fit_gamma(d.L.values, d.c.values, d.y.values)
        print(f"   model counterfactuals, {g:5s} students, {n:4s}: gamma {p[2]:+.2f} (se {se[2]:.2f})  a {p[0]:+.2f} b {p[1]:+.2f}  (n {len(d)})")
sm = T.st.isin(["phi4m", "smol3", "granite"]) & (T.kl == 0.2) & (T.fmt == "1") & (T.steps == 120) & (T.temp == 0.8) & (T.lr == 5e-5) & (T.prompt == "idk") & T.fam.isin(["gold", "uniFP", "uniFN", "uniMix"])
for n in ("std", "none"):
    d = T[sm & (T.adv == n)]
    p, se = fit_gamma(d.L.values, d.c.values, d.y_n.values)
    print(f"   ACTUAL small-student runs, {n:4s}: gamma {p[2]:+.2f} (se {se[2]:.2f})  a {p[0]:+.2f} b {p[1]:+.2f}  (n {len(d)} runs, credits {sorted(d.c.unique())})")
    dd = d[d.c != 0.5]
    print(f"      runs at c != 0.5: n {len(dd)}; mechanistic MAE {np.mean(np.abs(dd.y_n - dd.conf_half_n)):.3f} | law (gamma = 1, qwen7 curve) MAE {np.mean(np.abs(dd.y_n - dd.law_n)):.3f}")
# main students: only the qwen7 judge credit runs; fix (a, b) from the qwen7 uniform main-grid curve, fit gamma alone
lc = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
mainm = (T.kl == 0.2) & (T.c == 0.5) & (T.adv == "std") & (T.fmt == "1") & (T.steps == 120) & T.std_home & (T.force == 0) & (T.switch == 0) & (T.temp == 0.8) & (T.lr == 5e-5)
Dq = T[mainm & (T.st == "qwen7") & T.fam.isin(["gold", "uniFP", "uniFN", "uniMix"])]
pa, _ = curve_fit(lc, Dq.L - Dq.c, Dq.y_n, p0=[0, -8])
K = T[(T.st == "qwen7") & T.run.str.contains("FK20judgeQt50_idkc") & T.std_home & (T.fmt == "1")]
gs = np.linspace(-1, 1.5, 251); err = [np.sum((K.y_n - lc(K.L - g * K.c, *pa)) ** 2) for g in gs]
print(f"   ACTUAL qwen7 judge-Qt50 runs at c 0.5 / 0.7 / 0.85 (n {len(K)}), (a, b) fixed from the qwen7 uniform curve: gamma {gs[int(np.argmin(err))]:+.2f}")
print("\n== 3. the qwen7 judge credit runs: actual vs model (std = the trainer's setting) vs model under Dr. GRPO ('none', counterfactual)")
for r in K.sort_values(["ds", "c", "seed"]).itertuples():
    src = CF.get(r.run.replace(f"idkc{str(r.c).replace('0.', '0') if r.c != 0.85 else '085'}", "idkc05"), {})
    base = r.run.replace("_idkc07_", "_idkc05_").replace("_idkc085_", "_idkc05_"); R = CF.get(base, {}); R = CF.get(R.get("same_as", base), R)
    if not R: continue
    print(f"   {r.run:38s} c {r.c:.2f}  actual {r.y_n:.3f} | model std {R[('std', r.c, 'n')] if ('std', r.c, 'n') in R else float('nan'):.3f}  none {R[('none', r.c, 'n')] if ('none', r.c, 'n') in R else float('nan'):.3f}")
print("\n== 4. per-sweep credit slopes (OLS over runs; actual vs the model's prediction for the same runs)")
SW = T[sm | ((T.st == "qwen7") & T.run.str.contains("FK20judgeQt50_idkc") & T.std_home)].copy()
SW["grader"] = SW.run.str.extract(r"_(?:FK20D?)(sim\d\dfn\d\d|judgeQt50)_")[0]
for (st, adv, gr), d in SW.groupby(["st", "adv", "grader"]):
    if d.c.nunique() < 2: continue
    sa = np.polyfit(d.c, d.y_n, 1)[0]; sm_ = np.polyfit(d.c, d.conf_half_n, 1)[0]; slaw = np.polyfit(d.c, d.law_n, 1)[0]
    print(f"   {st:6s} {adv:4s} {gr:10s} runs {len(d):2d} credits {sorted(d.c.unique())}: actual {sa:+.2f}  model {sm_:+.2f}  law {slaw:+.2f}")
print("\n== 5. when c matters under std: share of three-level groups at the model's stationary policy (c 0.5, std), by grader family")
d = U[(U.norm == "std") & (U.c == 0.5)]
print(d.groupby(["grp", "fam"]).three.agg(["mean", "count"]).round(3).to_string())
sl2 = U[U.norm == "std"].groupby(["grp", "fam", "src"]).apply(lambda g: np.polyfit(g.c, g.y, 1)[0]).reset_index(name="slope")
print("   model credit slope (std) by family:", sl2.groupby(["grp", "fam"]).slope.mean().round(2).to_dict())
