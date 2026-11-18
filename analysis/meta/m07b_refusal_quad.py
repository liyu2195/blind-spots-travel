"""(meta) [m07b = m07 with a QUADRATIC reference curve, robustness check] Grader-specific refusal map and its transfer, for every grader with structured false REJECTS.
Question split (per grader, home and new separately): among questions whose untrained greedy answer is right (loose key), "rejected" = the grader rejects
at least half of the untrained student's strictly-right samples (rej_q >= 0.5), "kept" = rej_q < 0.5.
contrast(run) = IDK rate on rejected - IDK rate on kept. Reference = the same contrast, same split, in uniform-grader runs (random rejects, no structure)
as a linear function of their overall IDK rate. effect = contrast - reference at the run's IDK rate. transfer = mean effect new / mean effect home.
For plreal<J> (rejections copied from judge J on trained questions) the new-question split is judge J's own rejections at its 35% threshold.
Writes cache/refusal.pkl. Inputs: cache/runs_<st>_<ds>.pkl, cache/hack.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, glob, json, numpy as np, pandas as pd
OUT = (_RA + "/meta/cache"); FIN = (_RX + "/final")
T = pd.read_pickle(f"{OUT}/hack.pkl").set_index("run")
rows = []
for f in sorted(glob.glob(f"{OUT}/runs_*_*.pkl")):
    if f.endswith("runs_table.pkl"): continue
    P = pickle.load(open(f, "rb"))
    if "SPECQ" not in P or P["st"] not in ("qwen7", "llama8"): continue
    st, ds = P["st"], P["ds"]; dsn = {"h1": "tqa", "pq": "pqa", "nq": "nqo"}[ds]
    Qf = P["Qf"]; R = P["R"]; UNI = [u for u in P["UNI"] if R[u]["kl"] == 0.2 and R[u]["c"] == 0.5]
    ucode = {part: np.array([Qf[i]["gcode"] for i in P[part.upper()]]) for part in ("home", "new")}
    # judge J at its a35 threshold: spec key used by the judge runs
    jkeys = {}
    for run, r in R.items():
        sp = r["spec"]
        if sp["kind"] == "judge" and "a35" in run and not sp.get("quorum"): jkeys[sp["J"]] = r["speckey"]
    for run, r in R.items():
        sp = r["spec"]; key = r["speckey"]
        d = dict(run=run)
        for part in ("home", "new"):
            rej = P["SPECQ"][key][part]["rej"]
            if part == "new" and sp["kind"] == "key" and sp.get("knows") and "real" in sp["knows"]:
                J = sp["knows"].split("real")[1][0]
                rej = P["SPECQ"][jkeys[J]]["new"]["rej"] if J in jkeys else rej
            ur = ucode[part] == 1; fin = np.isfinite(rej)
            A = ur & fin & (rej >= 0.5); B = ur & fin & (rej < 0.5)
            if A.sum() < 15 or B.sum() < 15: d[f"eff_{part}"] = np.nan; continue
            o = r[part]["o"]; con = (o[A] == 0).mean() - (o[B] == 0).mean()
            xs, ys = [], []
            for u in UNI:
                ou = R[u][part]["o"]; xs.append((ou == 0).mean()); ys.append((ou[A] == 0).mean() - (ou[B] == 0).mean())
            b = np.polyfit(xs, ys, 2); ref = np.polyval(b, (o == 0).mean())
            d[f"con_{part}"] = con; d[f"refcon_{part}"] = ref; d[f"eff_{part}"] = con - ref; d[f"nA_{part}"] = int(A.sum()); d[f"nB_{part}"] = int(B.sum())
            d[f"refsd_{part}"] = float(np.std(np.array(ys) - np.polyval(b, xs)))
        rows.append(d)
Rf = pd.DataFrame(rows).set_index("run")
T = T.join(Rf, how="left")
T.reset_index().to_pickle(f"{OUT}/refusal_quad.pkl")
S = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & np.isfinite(T.eff_home)]
rng = np.random.default_rng(1)
def boot_ratio(h, n, B=2000):
    h = np.asarray(h, float); n = np.asarray(n, float); m = np.isfinite(h) & np.isfinite(n); h, n = h[m], n[m]; k = len(h)
    if k < 2 or abs(h.mean()) < 1e-9: return np.nan, np.nan, np.nan
    bs = [n[i].mean() / h[i].mean() for i in (rng.integers(0, k, k) for _ in range(B)) if abs(h[i].mean()) > 1e-9]
    return n.mean() / h.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
pd.set_option("display.width", 250)
print("selective refusal of the grader's rejected-right questions beyond uniform-grader runs at the same IDK rate (qwen7 + llama8, standard runs)")
out = []
for (fam, sub), d in S.groupby(["fam", "sub"]):
    if len(d) < 2 or fam in ("keyFP",): continue
    dn = d[np.isfinite(d.eff_new)]
    rt, lo, hi = boot_ratio(dn.eff_home, dn.eff_new) if len(dn) >= 2 else (np.nan, np.nan, np.nan)
    out.append(dict(fam=fam, sub=sub, n=len(d), n_new=len(dn), FR=d.FR_home.mean(), nA_h=d.nA_home.mean(), con_h=d.con_home.mean(), ref_h=d.refcon_home.mean(), eff_h=d.eff_home.mean(),
                    nA_n=d.nA_new.mean(), eff_n=dn.eff_new.mean() if len(dn) else np.nan, transfer=rt, lo=lo, hi=hi))
O = pd.DataFrame(out)
print(O.round(3).to_string(index=False)); O.to_csv(f"{OUT}/refusal_by_grader_quad.csv", index=False)
for st in ("qwen7", "llama8"):
    print(f"\n{st} only")
    for (fam, sub), d in S[S.st == st].groupby(["fam", "sub"]):
        if fam in ("keyFP", "keyFN", "gold", "mixU", "hash", "long", "typeFP", "popFP"): continue
        dn = d[np.isfinite(d.eff_new)]
        rt, lo, hi = boot_ratio(dn.eff_home, dn.eff_new) if len(dn) >= 2 else (np.nan, np.nan, np.nan)
        print(f"  {fam:7s} {sub:13s} n {len(d):2d} effect home {d.eff_home.mean():+.3f} new {dn.eff_new.mean() if len(dn) else np.nan:+.3f} transfer {rt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
