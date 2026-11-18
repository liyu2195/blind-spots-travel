"""(meta) Grader-specific hack and its transfer. For every run r with grader g: paid share of r's wrong final answers under g, minus the paid share
under g of the wrong answers of UNIFORM-grader runs (random false accepts / rejects; they cannot learn g's preferences) at the same abstention
(linear reference curve paid ~ idk over those runs, per part). hack_home, hack_new; transfer = mean hack_new / mean hack_home per grader,
with a bootstrap over runs. Writes cache/hack.pkl. Inputs: cache/runs_<st>_<ds>.pkl, cache/law2.pkl."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, glob, json, numpy as np, pandas as pd
OUT = (_RA + "/meta/cache")
T = pd.read_pickle(f"{OUT}/law2.pkl").set_index("run")
rows = []
for f in sorted(glob.glob(f"{OUT}/runs_*_*.pkl")):
    if f.endswith("runs_table.pkl"): continue
    P = pickle.load(open(f, "rb"))
    if "SPECU" not in P: continue
    for run, r in P["R"].items():
        lst = [e for e in P["SPECU"][r["speckey"]] if e["kl"] == 0.2 and e["c"] == 0.5]
        d = dict(run=run)
        for part in ("home", "new"):
            x = np.array([e[f"idk_{part}"] for e in lst]); y = np.array([e[f"paid_{part}"] for e in lst]); m = np.isfinite(y)
            pr = r[part]["paid_wrong"]
            if m.sum() >= 6 and np.isfinite(pr):
                b = np.polyfit(x[m], y[m], 1); ref = np.polyval(b, r[part]["idk"])
                d[f"ref_paid_{part}"] = ref; d[f"hack_{part}"] = pr - ref; d[f"refslope_{part}"] = b[0]; d[f"refcnt_{part}"] = int(m.sum())
                d[f"ref_sd_{part}"] = float(np.std(y[m] - np.polyval(b, x[m])))
            else:
                d[f"hack_{part}"] = np.nan
        rows.append(d)
H = pd.DataFrame(rows).set_index("run")
T = T.join(H, how="left")
T.reset_index().to_pickle(f"{OUT}/hack.pkl")
S = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & np.isfinite(T.hack_home)]
rng = np.random.default_rng(0)
def boot_ratio(h, n, B=2000):
    h = np.asarray(h, float); n = np.asarray(n, float); m = np.isfinite(h) & np.isfinite(n); h, n = h[m], n[m]; k = len(h)
    if k < 2 or abs(h.mean()) < 1e-9: return np.nan, np.nan, np.nan
    bs = []
    for _ in range(B):
        i = rng.integers(0, k, k)
        if abs(h[i].mean()) > 1e-9: bs.append(n[i].mean() / h[i].mean())
    return n.mean() / h.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
pd.set_option("display.width", 250)
print("hack = paid share of wrong answers minus the uniform-run reference at the same abstention (qwen7 + llama8, standard runs)")
out = []
for (fam, sub), d in S.groupby(["fam", "sub"]):
    if len(d) < 2: continue
    rt, lo, hi = boot_ratio(d.hack_home, d.hack_new)
    out.append(dict(fam=fam, sub=sub, n=len(d), FA=d.FA_home.mean(), paid_h=d.paid_wrong_home.mean(), ref_h=d.ref_paid_home.mean(), hack_h=d.hack_home.mean(),
                    paid_n=d.paid_wrong_new.mean(), refp_n=d.ref_paid_new.mean(), hack_n=d.hack_new.mean(), transfer=rt, lo=lo, hi=hi))
O = pd.DataFrame(out).sort_values(["fam", "sub"])
print(O.round(3).to_string(index=False))
O.to_csv(f"{OUT}/hack_by_grader.csv", index=False)
for st in ("qwen7", "llama8"):
    print(f"\n{st} only:")
    for (fam, sub), d in S[S.st == st].groupby(["fam", "sub"]):
        if len(d) < 2 or fam in ("gold", "keyFP", "keyFN", "mixU"): continue
        rt, lo, hi = boot_ratio(d.hack_home, d.hack_new)
        print(f"  {fam:7s} {sub:14s} n {len(d):2d} hack home {d.hack_home.mean():+.3f} new {d.hack_new.mean():+.3f} transfer {rt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
