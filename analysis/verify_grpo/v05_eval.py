"""(verify_grpo) Score my solver's predictions (v04) against each run's actual final abstention (share of final greedy answers that are IDK) on new
and training questions, by run group (same group definitions as the claim, g06 / g09 of the grpo explorer, re-coded); the leniency curve
y = sigmoid(a + b (L - c)) refitted on qwen7 main-grid random-error graders (as claimed); zero-constant naive baselines; variant sensitivity;
direct comparison with the grpo explorer's own predictions (their eval table) on the same runs.
Usage: nice -n 10 python3 -I v05_eval.py > out_v05_eval.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import glob, pickle, numpy as np, pandas as pd, warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
HERE = (_RA + "/verify_grpo")
rows = []
for f in sorted(glob.glob(f"{HERE}/cache/va_*.pkl")):
    D = pickle.load(open(f, "rb")); st, ds = D["st"], D["ds"]; home, new = D["home"].astype(bool), D["new"].astype(bool)
    P = {}
    for pf in glob.glob(f"{HERE}/cache/vp_{st}_{ds}*.pkl"): P.update(pickle.load(open(pf, "rb")))
    B = np.load(f"{HERE}/cache/vb_{st}_{ds}.npz")
    for run, r in D["rows"].items():
        if r["alpha"] is None or run not in P or r["prompt"] != "idk": continue
        I = B["i1"] if r["fmt"] == "1" else B["i0"]; V = B["v1"] if r["fmt"] == "1" else np.ones_like(B["v1"])
        a = r["alpha"].astype(float); H = home & np.isfinite(a); nn = (~I).sum(1); nva = (V & ~I).sum(1)
        L = float(np.sum(a[H] * nn[H]) / max(1, np.sum(nva[H])))
        gi = B["gi1"] if r["fmt"] == "1" else B["gi0"]
        d = dict(run=run, st=st, ds=ds, fam=r["fam"], c=r["c"], kl=r["kl"], adv=r["adv"], fmt=r["fmt"], steps=r["steps"], lr=r["lr"], temp=r["temp"],
                 std_home=r["std_home"], force=r["force"], switch=r["switch"], seed=r["seed"], L=L,
                 y_n=float((r["out"][new] == 0).mean()), y_h=float((r["out"][home] == 0).mean()), base_n=float(gi[new].mean()),
                 myopic_n=float((a[H] < r["c"]).mean()))
        for key, v in P[run].items():
            if not isinstance(key, tuple): continue
            sm, rule, kap = key; nm = f"{rule}{'' if sm == 'sm1' else '_sm2'}{'' if kap == 1.0 else f'_k{kap}'}"
            for fld in ("n_half", "h_half", "n_greedy", "h_greedy"):
                if fld in v: d[f"{nm}_{fld}"] = v[fld]
            if rule == "conf" and sm == "sm1" and kap == 1.0: d["nroots"] = v["nroots"]; d["stable"] = v["stable"]; d["theta1"] = v["theta"][1]
        rows.append(d)
T = pd.DataFrame(rows)
T["cfg"] = T.st + "|" + T.ds + "|" + T.run.str.replace(r"_s\d+_(h1|pq|nq)$", "", regex=True)
main = (T.kl == 0.2) & (T.c == 0.5) & (T.adv == "std") & (T.fmt == "1") & (T.steps == 120) & T.std_home & (T.force == 0) & (T.switch == 0) & (T.temp == 0.8) & (T.lr == 5e-5)
uni = T.fam.isin(["gold", "uniFP", "uniFN", "uniMix"]); small = T.st.isin(["phi4m", "smol3", "granite"]); mains = T.st.isin(["qwen7", "llama8"])
def r2(y, p): y = np.asarray(y, float); p = np.asarray(p, float); return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
lc = lambda x, a, b: 1 / (1 + np.exp(-(a + b * x)))
Dq = T[main & uni & (T.st == "qwen7")]
pa, _ = curve_fit(lc, Dq.L - Dq.c, Dq.y_n, p0=[0, -8], maxfev=20000); T["law_n"] = lc(T.L - T.c, *pa)
pah, _ = curve_fit(lc, Dq.L - Dq.c, Dq.y_h, p0=[0, -8], maxfev=20000); T["law_h"] = lc(T.L - T.c, *pah)
print(f"leniency curve refitted on qwen7 main-grid random-error graders (n {len(Dq)}): a {pa[0]:+.2f} b {pa[1]:+.2f} (claim: a -0.84 b -8.27)")
rng = np.random.default_rng(11)
def ci(D_, col, ycol="y_n", n=1000):
    g = D_.groupby("cfg").indices; ks = list(g); o = []
    for _ in range(n):
        ii = np.concatenate([g[k] for k in rng.choice(ks, len(ks))]); o.append((r2(D_[ycol].values[ii], D_[col].values[ii]), np.mean(np.abs(D_[ycol].values[ii] - D_[col].values[ii]))))
    return np.nanpercentile(np.array(o), [2.5, 97.5], axis=0)
def sc(D_, col, ycol="y_n", boot=False):
    y, p = D_[ycol].values, D_[col].values; s = f"R2 {r2(y, p):+.2f} MAE {np.mean(np.abs(y - p)):.3f} bias {np.mean(p - y):+.3f}"
    if boot: c_ = ci(D_, col, ycol); s += f" [R2 {c_[0, 0]:+.2f}..{c_[1, 0]:+.2f}]"
    return s
G = {
 "qwen7 main grid, all graders (DISCOVERY)": main & (T.st == "qwen7"),
 "llama8 main grid, all graders": main & (T.st == "llama8"),
 "llama8 main grid, random-error graders": main & (T.st == "llama8") & uni,
 "llama8 main grid, LLM judges": main & (T.st == "llama8") & T.fam.str.startswith("judge"),
 "qwen7 main grid, LLM judges": main & (T.st == "qwen7") & T.fam.str.startswith("judge"),
 "both main grid, set-wise FN/FP": main & T.fam.isin(["setFN", "setFP"]),
 "both main grid, hash": main & (T.fam == "hash"),
 "both main grid, long-answer FP": main & (T.fam == "long"),
 "small FK20 std c 0.5": small & (T.kl == 0.2) & (T.adv == "std") & (T.c == 0.5) & (T.fmt == "1") & (T.steps == 120) & (T.temp == 0.8),
 "small FK20 std credit sweep c != 0.5": small & (T.kl == 0.2) & (T.adv == "std") & (T.c != 0.5) & (T.fmt == "1") & (T.steps == 120),
 "small FK20D (Dr. GRPO, KL 0.2)": small & (T.kl == 0.2) & (T.adv == "none") & (T.fmt == "1"),
 "KL != 0.2, > 0 (all students; claim n 42)": (T.kl > 0) & (T.kl != 0.2) & (T.fmt == "1") & (T.force == 0) & (T.switch == 0) & (T.temp == 0.8) & (T.lr == 5e-5) & T.std_home,
 "small KL 0.04/0.1/0.4 (std)": small & T.kl.isin([0.04, 0.1, 0.4]) & (T.fmt == "1") & (T.adv == "std"),
 "main students KL 0.5 / 1.0 (120 steps)": mains & T.kl.isin([0.5, 1.0]) & (T.steps == 120),
 "main students 360 steps (KL 0.2 / 0.5)": mains & (T.steps == 360),
 "qwen7 judge credit 0.7 / 0.85": (T.st == "qwen7") & T.c.isin([0.7, 0.85]),
 "main students KL 0 (FK00)": mains & (T.kl == 0) & (T.fmt == "1"),
 "small no-KL strict (std, T 0.8, 120 steps)": small & (T.kl == 0) & (T.fmt == "1") & (T.adv == "std") & (T.steps == 120) & (T.temp == 0.8) & (T.lr == 5e-5),
}
print("\n== 1. my solver (rule conf, threshold 1/2, kappa 1, zero constants) vs the leniency curve, NEW questions; cluster bootstrap by grader config")
for nm, m in G.items():
    D_ = T[m]
    if len(D_) < 2: print(f"   {nm:46s} n {len(D_)}"); continue
    print(f"   {nm:46s} n {len(D_):3d} cfg {D_.cfg.nunique():3d} y {D_.y_n.mean():.3f} | mech {sc(D_, 'conf_n_half', boot=True)} | law {sc(D_, 'law_n', boot=True)}")
print("\n== 2. the same on TRAINING questions")
for nm in ("llama8 main grid, all graders", "qwen7 main grid, all graders (DISCOVERY)", "small FK20 std c 0.5", "small FK20 std credit sweep c != 0.5", "small FK20D (Dr. GRPO, KL 0.2)"):
    D_ = T[G[nm]]; print(f"   {nm:46s} n {len(D_):3d} y_h {D_.y_h.mean():.3f} | mech {sc(D_, 'conf_h_half', 'y_h')} | law {sc(D_, 'law_h', 'y_h')}")
print("\n== 3. variant sensitivity (NEW questions): discovery vs confirmation groups")
VAR = ["conf_n_half", "conf_n_greedy", "conf2_n_half", "bias_n_half", "free_n_half", "conf_sm2_n_half", "conf_tcorr_n_half", "conf_k0.5_n_half", "conf_k0.75_n_half", "conf_k1.5_n_half", "conf_k2.0_n_half"]
for nm in ("qwen7 main grid, all graders (DISCOVERY)", "llama8 main grid, all graders", "small FK20 std c 0.5", "small FK20 std credit sweep c != 0.5", "small FK20D (Dr. GRPO, KL 0.2)", "KL != 0.2, > 0 (all students; claim n 42)"):
    D_ = T[G[nm]]
    print(f"   {nm} (n {len(D_)}):")
    print("      " + "  ".join(f"{v.replace('_n_half', '').replace('_n_', '_')}: {r2(D_.y_n, D_[v]):+.2f}" for v in VAR if v in D_ and D_[v].notna().all()))
print("\n== 4. zero-constant naive baselines (NEW questions): untrained greedy IDK share ('no change'); myopic share of training questions with alpha < c")
for nm in ("llama8 main grid, all graders", "small FK20 std c 0.5", "small FK20 std credit sweep c != 0.5", "small FK20D (Dr. GRPO, KL 0.2)", "KL != 0.2, > 0 (all students; claim n 42)"):
    D_ = T[G[nm]]
    rf = curve_fit(lc, D_.L - D_.c, D_.y_n, p0=[0, -8], maxfev=20000)[0] if len(D_) > 4 else None
    ins = f"  | curve refitted IN the group (2 constants, in-sample) R2 {r2(D_.y_n, lc(D_.L - D_.c, *rf)):+.2f}" if rf is not None else ""
    print(f"   {nm:46s} no-change {sc(D_, 'base_n')} | myopic {sc(D_, 'myopic_n')}{ins}")
print("\n== 5. reproduction: my predictions vs the grpo explorer's (their eval_table_k1.0.csv, conf_half_n) on the same runs (standard training sets)")
E = pd.read_csv((_RA + "/grpo/eval_table_k1.0.csv"))[["run", "conf_half_n", "y_n", "L"]].rename(columns={"conf_half_n": "theirs", "y_n": "their_y", "L": "their_L"})
M = T[T.std_home].merge(E, on="run")
print(f"   runs in both: {len(M)}; |mine - theirs| prediction: mean {np.mean(np.abs(M.conf_n_half - M.theirs)):.4f}, max {np.max(np.abs(M.conf_n_half - M.theirs)):.4f}, "
      f"share > 0.02: {np.mean(np.abs(M.conf_n_half - M.theirs) > 0.02):.3f}; outcome y_n max diff {np.max(np.abs(M.y_n - M.their_y)):.4f}; L max diff {np.max(np.abs(M.L - M.their_L)):.4f}")
big = M[np.abs(M.conf_n_half - M.theirs) > 0.02].sort_values("run")
for t in big.head(12).itertuples(): print(f"      {t.run:44s} mine {t.conf_n_half:.3f} theirs {t.theirs:.3f} actual {t.y_n:.3f} nroots {t.nroots} stable {t.stable}")
print("\n== 6. multiple stationary rules (multistart Newton, conf rule): runs with != 1 root, by group")
for nm in ("llama8 main grid, all graders", "qwen7 main grid, all graders (DISCOVERY)", "small FK20 std c 0.5"):
    D_ = T[G[nm]]; print(f"   {nm:46s} roots distribution {D_.nroots.value_counts().to_dict()}; unstable {int((~D_.stable.astype(bool)).sum())}")

print("\n== 7. seed-noise ceiling: R2 of predicting each run by the mean of its grader configuration's OTHER seeds (configs with >= 2 seeds)")
for nm in ("qwen7 main grid, all graders (DISCOVERY)", "llama8 main grid, all graders", "small FK20 std c 0.5", "small FK20 std credit sweep c != 0.5", "small FK20D (Dr. GRPO, KL 0.2)", "KL != 0.2, > 0 (all students; claim n 42)"):
    D_ = T[G[nm]].copy(); g = D_.groupby("cfg").y_n; cnt = g.transform("count"); loo = (g.transform("sum") - D_.y_n) / (cnt - 1)
    m2 = cnt >= 2
    if m2.sum() < 3: print(f"   {nm:46s} too few multi-seed configs"); continue
    print(f"   {nm:46s} runs in multi-seed configs {int(m2.sum())}/{len(D_)}: leave-one-seed-out R2 {r2(D_.y_n[m2], loo[m2]):+.2f} (mech on the same runs {r2(D_.y_n[m2], D_.conf_n_half[m2]):+.2f}); "
          f"within-config SD {np.sqrt(np.mean(D_[m2].groupby('cfg').y_n.var())):.3f}")
print("\n== 8. which questions abstain (per run, new questions, final IDK): AUROC of the model's p_q vs the untrained IDK share (ties by agreement) vs agreement")
from sklearn.metrics import roc_auc_score
res = []
for f in sorted(glob.glob(f"{HERE}/cache/va_*.pkl")):
    D = pickle.load(open(f, "rb")); st, ds = D["st"], D["ds"]; new_ = D["new"].astype(bool); B = np.load(f"{HERE}/cache/vb_{st}_{ds}.npz")
    P = {}
    for pf in glob.glob(f"{HERE}/cache/vp_{st}_{ds}*.pkl"): P.update(pickle.load(open(pf, "rb")))
    nI = B["i1"].sum(1); agr = (B["m1"] + 0.5) / (32 - nI + 1.0); pref = (nI + 0.5) / 33.0
    for run, r in D["rows"].items():
        if run not in P or "p_conf" not in P[run] or run not in set(T[G["llama8 main grid, all graders"] | G["qwen7 main grid, all graders (DISCOVERY)"] | G["small FK20 std c 0.5"]].run): continue
        y = r["out"][new_] == 0
        if y.mean() < 0.03 or y.mean() > 0.97: continue
        pq = P[run]["p_conf"][new_]
        res.append(dict(st=st, auc_model=roc_auc_score(y, pq), auc_pref=roc_auc_score(y, pref[new_] - 1e-6 * agr[new_]), auc_agr=roc_auc_score(y, -agr[new_])))
R_ = pd.DataFrame(res); print(R_.groupby("st")[["auc_model", "auc_pref", "auc_agr"]].mean().round(3).to_string()); print("   runs:", R_.groupby("st").size().to_dict())
T.to_csv(f"{HERE}/eval_v.csv", index=False)
