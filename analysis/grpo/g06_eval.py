"""(grpo) Score the mechanistic predictions (g05_model) against each run's actual final abstention (share of final greedy answers that are IDK),
on new and on training questions, by run group; compare with the empirical leniency curve y = 1 / (1 + exp(-(a + b (L - c)))), fitted on
qwen7 main-grid random-error graders (as in the draft), L = share of the untrained student's valid non-IDK samples on training questions paid
by the run's grader. Discovery for choosing the rule / threshold variant: qwen7 main grid. Everything else is confirmation.
Usage: nice -n 10 python3 -I g06_eval.py [kappa] > out_g06_eval.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, glob, pickle, json, numpy as np, pandas as pd, warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
HERE = (_RA + "/grpo"); KAP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
rows = []
for f in sorted(glob.glob(f"{HERE}/cache/runs_*.pkl")):
    D = pickle.load(open(f, "rb")); st, ds = D["st"], D["ds"]
    P = {}
    for pf in glob.glob(f"{HERE}/cache/pred_{st}_{ds}*.pkl"): P.update(pickle.load(open(pf, "rb")))
    Bz = np.load(f"{HERE}/cache/base_{st}_{ds}.npz"); idk = Bz["idk"]; valid = Bz["valid"]
    nvalid_app = (valid & ~idk).sum(1); nnon = (~idk).sum(1)
    for run, r in D["rows"].items():
        if r["alpha"] is None or run not in P: continue
        a = r["alpha"].astype(float); H = D["home"] & np.isfinite(a)
        L = float(np.sum(a[H] * nnon[H]) / max(1, np.sum(nvalid_app[H])))
        d = dict(run=run, st=st, ds=ds, fam=r["fam"], c=r["c"], kl=r["kl"], adv=r["adv"], fmt=r["fmt"], steps=r["steps"], lr=r["lr"], temp=r["temp"],
                 std_home=r["std_home"], force=r["force"], switch=r["switch"], prompt=r["prompt"], seed=r["seed"], L=L,
                 y_n=float((r["out"][D["new"]] == 0).mean()), y_h=float((r["out"][D["home"]] == 0).mean()),
                 mal_n=float((r["out"][D["new"]] == 3).mean()), log_idk_end=float(np.nanmean(r["log_idk"][-10:])) if len(r["log_idk"]) else np.nan)
        for k, v in P[run].items():
            if isinstance(k, tuple) and k[0] == KAP and k[2] in ("half", "greedy", "mean_p"): d["_".join(map(str, k[1:]))] = v
        rows.append(d)
T = pd.DataFrame(rows)
T["tag"] = T.run.str.extract(r"^[a-z0-9]+_([A-Z]+\d*D?)")[0]
T["cfg"] = T.st + "|" + T.ds + "|" + T.run.str.replace(r"_s\d+_(h1|pq|nq)$", "", regex=True)      # grader configuration (all seeds together)
T = T[T.prompt == "idk"].reset_index(drop=True)                 # 'flag' prompt runs have no IDK reply by construction
main = (T.kl == 0.2) & (T.c == 0.5) & (T.adv == "std") & (T.fmt == "1") & (T.steps == 120) & T.std_home & (T.force == 0) & (T.switch == 0) & (T.temp == 0.8) & (T.lr == 5e-5)
T["main"] = main
uni = T.fam.isin(["gold", "uniFP", "uniFN", "uniMix"])
def r2(y, p): y = np.asarray(y, float); p = np.asarray(p, float); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan
def sc(y, p): y = np.asarray(y, float); p = np.asarray(p, float); return f"R2 {r2(y, p):+.2f}  MAE {np.mean(np.abs(y - p)):.3f}  bias {np.mean(p - y):+.3f}  corr {np.corrcoef(y, p)[0, 1] if len(y) > 2 and np.std(p) > 0 else np.nan:+.2f}"
lc = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
Dq = T[main & uni & (T.st == "qwen7")]
pa, _ = curve_fit(lc, Dq.L - Dq.c, Dq.y_n, p0=[0, -8], maxfev=20000)
T["law_n"] = lc(T.L - T.c, *pa)
pah, _ = curve_fit(lc, Dq.L - Dq.c, Dq.y_h, p0=[0, -8], maxfev=20000); T["law_h"] = lc(T.L - T.c, *pah)
print(f"leniency curve fitted on qwen7 main-grid random-error graders (n {len(Dq)}): a {pa[0]:+.2f} b {pa[1]:+.2f} (new); home a {pah[0]:+.2f} b {pah[1]:+.2f}")
VAR = [f"{rule}_{thr}" for rule in ("bias", "conf", "conf2", "free") for thr in ("half", "greedy")]
print(f"\n== 1. discovery (qwen7 main grid, all graders with a re-implemented grader, n {int((main & (T.st == 'qwen7')).sum())}): which rule / threshold, NEW questions")
for v in VAR:
    D_ = T[main & (T.st == "qwen7")]; print(f"   {v:14s} {sc(D_.y_n, D_[v + '_n'])}")
print("   leniency curve  ", sc(T[main & (T.st == 'qwen7')].y_n, T[main & (T.st == 'qwen7')].law_n))
CH = sys.argv[2] if len(sys.argv) > 2 else "conf_half"
print(f"\n== 2. confirmation, variant {CH} (zero fitted constants) vs leniency curve (2 constants, fitted on qwen7 uniform)")
G = {
 "llama8 main grid, all graders": main & (T.st == "llama8"),
 "llama8 main grid, random-error graders": main & (T.st == "llama8") & uni,
 "llama8 main grid, LLM judges": main & (T.st == "llama8") & T.fam.str.startswith("judge"),
 "qwen7 main grid, LLM judges": main & (T.st == "qwen7") & T.fam.str.startswith("judge"),
 "both main grid, hash": main & (T.fam == "hash"),
 "both main grid, set-wise FN/FP": main & T.fam.isin(["setFN", "setFP"]),
 "both main grid, blind/frozen keys": main & T.fam.isin(["blind", "frz"]),
 "both main grid, long-answer FP": main & (T.fam == "long"),
 "qwen7 judge credit 0.7/0.85": (T.st == "qwen7") & T.c.isin([0.7, 0.85]),
 "main students KL 0 (FK00)": T.st.isin(["qwen7", "llama8"]) & (T.kl == 0) & (T.fmt == "1"),
 "main students KL 0.5 / 1.0 (120 steps)": T.st.isin(["qwen7", "llama8"]) & T.kl.isin([0.5, 1.0]) & (T.steps == 120),
 "main students 360 steps (KL 0.2 / 0.5)": T.st.isin(["qwen7", "llama8"]) & (T.steps == 360),
 "small FK20 std c 0.5": T.st.isin(["phi4m", "smol3", "granite"]) & (T.kl == 0.2) & (T.adv == "std") & (T.c == 0.5) & (T.fmt == "1") & (T.steps == 120) & (T.temp == 0.8),
 "small FK20 std credit sweep c != 0.5": T.st.isin(["phi4m", "smol3", "granite"]) & (T.kl == 0.2) & (T.adv == "std") & (T.c != 0.5) & (T.fmt == "1") & (T.steps == 120),
 "small FK20D (Dr. GRPO, KL 0.2)": T.st.isin(["phi4m", "smol3"]) & (T.kl == 0.2) & (T.adv == "none") & (T.fmt == "1"),
 "small KL 0.04/0.1/0.4 (std)": T.st.isin(["phi4m", "smol3"]) & T.kl.isin([0.04, 0.1, 0.4]) & (T.fmt == "1") & (T.adv == "std"),
 "small no-KL strict (F, std)": T.st.isin(["phi4m", "smol3"]) & (T.kl == 0) & (T.fmt == "1") & (T.adv == "std") & (T.steps == 120) & (T.temp == 0.8) & (T.lr == 5e-5),
 "small old trainer (no KL, no format; std)": T.st.isin(["phi4m", "smol3"]) & (T.fmt == "0") & (T.adv == "std") & (T.switch == 0) & (T.force == 0),
 "small old trainer Dr. GRPO (no KL)": T.st.isin(["phi4m", "smol3"]) & (T.fmt == "0") & (T.adv == "none"),
}
rng = np.random.default_rng(7)
def ci(D_, col, n=1000):
    g = D_.groupby("cfg").indices; ks = list(g); o = []
    for _ in range(n):
        ii = np.concatenate([g[k] for k in rng.choice(ks, len(ks))]); y = D_.y_n.values[ii]; p = D_[col].values[ii]
        o.append((r2(y, p), np.mean(np.abs(y - p))))
    o = np.array(o); return np.nanpercentile(o, [2.5, 97.5], axis=0)
for nm, m in G.items():
    D_ = T[m]
    if len(D_) < 2: print(f"   {nm:44s} n {len(D_)}"); continue
    cm = ci(D_, CH + "_n"); cl = ci(D_, "law_n")
    print(f"   {nm:44s} n {len(D_):3d} (cfg {D_.cfg.nunique():3d}) mean abst {D_.y_n.mean():.3f} | mech {sc(D_.y_n, D_[CH + '_n'])} [R2 {cm[0,0]:+.2f}..{cm[1,0]:+.2f}, MAE {cm[0,1]:.3f}..{cm[1,1]:.3f}] | law {sc(D_.y_n, D_.law_n)} [R2 {cl[0,0]:+.2f}..{cl[1,0]:+.2f}, MAE {cl[0,1]:.3f}..{cl[1,1]:.3f}]")
print(f"\n== 3. the same on TRAINING questions (home), variant {CH}")
for nm, m in list(G.items())[:3] + [("qwen7 main grid, all graders", main & (T.st == "qwen7"))] + list(G.items())[12:16]:
    D_ = T[m]
    if len(D_) < 2: continue
    print(f"   {nm:44s} n {len(D_):3d} mean abst {D_.y_h.mean():.3f} | mech {sc(D_.y_h, D_[CH + '_h'])} | law {sc(D_.y_h, D_.law_h)}")
print("\n== 4. does the mechanism reproduce the leniency curve? fit the same logistic in (L - c) to the MECHANISTIC predictions (qwen7 uniform main grid)")
pm, _ = curve_fit(lc, Dq.L - Dq.c, Dq[CH + "_n"], p0=[0, -8], maxfev=20000)
print(f"   curve fitted to actual abstention: a {pa[0]:+.2f} b {pa[1]:+.2f};  fitted to mechanistic predictions: a {pm[0]:+.2f} b {pm[1]:+.2f}")
Dl = T[main & uni & (T.st == "llama8")]; pl_, _ = curve_fit(lc, Dl.L - Dl.c, Dl.y_n, p0=[0, -8], maxfev=20000); pml, _ = curve_fit(lc, Dl.L - Dl.c, Dl[CH + "_n"], p0=[0, -8], maxfev=20000)
print(f"   llama8: actual a {pl_[0]:+.2f} b {pl_[1]:+.2f};  mechanistic a {pml[0]:+.2f} b {pml[1]:+.2f}")
print("\n== 5. no KL (beta = 0, strict format): the drift has no interior attractor; runs end near an absorbing state (abstention < 0.02 or > 0.98)?")
NK = T[(T.kl == 0) & (T.fmt == "1")]
ext = (NK.y_n < 0.02) | (NK.y_n > 0.98); print(f"   {int(ext.sum())}/{len(NK)} no-KL runs end within 0.02 of 0 or 1 (KL 0.2 main grid + small FK20: {int(((T[main].y_n < 0.02) | (T[main].y_n > 0.98)).sum())}/{int(main.sum())})")
T.to_csv(f"{HERE}/eval_table_k{KAP}.csv", index=False)
