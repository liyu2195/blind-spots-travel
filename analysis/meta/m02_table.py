"""(meta) Flatten cache/runs_<st>_<ds>.pkl into one table, one row per run (cache/runs_table.pkl and .csv).
Adds the grader family, visits per question, the uniform-grader answer threshold t* = (c - FA) / (1 - FA - FR) and the key-trained reference
(mean of gold runs with the same student, dataset, KL and IDK credit) so that each grader-specific effect = run - reference."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pickle, glob, re, json, numpy as np, pandas as pd
OUT = (_RA + "/meta/cache")

def family(run, sp, cfg):
    k = sp["kind"]
    if k == "judge":
        J = sp["J"]; t = sp["t"]
        tag = "a35" if "a35" in run else ("a50" if "a50" in run else ("t50" if "t50" in run else "t?"))
        if sp.get("quorum"): return f"quorum", f"judge{J}{tag}q{sp['quorum']}"
        return "judge", f"judge{J}{tag}"
    if k == "api": return "api", "apiS55"
    if k == "alias":
        tf = sp["trainf"]
        if "blind" in tf: return "blind", "blind40"
        m = re.search(r"frz([A-Z]+)", tf); return "frz", "frz" + (m.group(1) if m else "?")
    if k == "key":
        fpk, fpu, fn, fnk, fnu, mode, kn = sp["fpk"], sp["fpu"], sp["fn"], sp["fnk"], sp["fnu"], sp["mode"], sp["knows"]
        if sp.get("quorum"): return "selfvote", "voteq"
        if mode == "hash": return "hash", f"hash{round(fpk*100):02d}fn{round(fn*100):02d}"
        if mode == "long": return "long", "long80"
        if mode == "det": return "det", "det"
        if kn and fnk is None and abs(fpk - fpu) < 1e-9: kn = None      # same false-accept rate on both sets: the set file is irrelevant
        if kn:
            if "type" in kn and fnk is None: return "typeFP", ("typeB" if fpk > 0 else "typeA") + f"{round(max(fpk, fpu)*100)}"
            if "pop" in kn: return "popFP", ("popH" if fpk > 0 else "popL") + f"{round(max(fpk, fpu)*100)}"
            if "type" in kn: return "typeFN", f"fnType{round(fnk*100)}/{round(fnu*100)}"
            if "hid" in kn: return "hidFN", f"fnHid{round(fnk*100)}/{round(fnu*100)}"
            m = re.search(r"knows_(real|scr|sharp)(\w)_", kn)
            if m: return "plFN", f"pl{m.group(1)}{m.group(2)}"
            return "other", kn
        if abs(fpk - fpu) > 1e-9: return "other", "fpk!=fpu"
        if fpk == 0 and fn == 0: return "gold", "gold"
        if "lenQ" in run: return "mixU", "lenQ"
        if fn == 0: return "keyFP", f"FP{round(fpk*100):02d}"
        if fpk == 0: return "keyFN", f"FN{round(fn*100):02d}"
        return "mixU", f"FP{round(fpk*100):02d}FN{round(fn*100):02d}"
    return "other", "other"

rows = []
for f in sorted(glob.glob(f"{OUT}/runs_*_*.pkl")):
    if f.endswith("runs_table.pkl"): continue
    P = pickle.load(open(f, "rb"))
    for run, r in P["R"].items():
        sp = r["spec"]; cfg = r["cfg"]; fam, sub = family(run, sp, cfg)
        d = dict(run=run, st=r["st"], ds=r["ds"], fam=fam, sub=sub, kind=sp["kind"], J=sp.get("J"), jt=sp.get("t"), c=r["c"], kl=r["kl"], seed=r["seed"], ck=r["ck"],
                 steps=r["steps"], trainf=r["trainf"], fmt=r["fmt"], adv=r["adv"], force=r["force"], temp=r["temp"], quorum=sp.get("quorum", 0),
                 from_name=int(bool(cfg.get("_from_name"))), lrate=cfg.get("LRATE"), topk=cfg.get("TOPK"), tag=re.match(r"^[a-z0-9]+_([A-Za-z]+\d*)", run).group(1))
        tf = r["trainf"] or ""
        m = re.search(r"_b(\d)_(\d+)\.jsonl", tf)
        if m: d["nq_train"] = int(m.group(2)); d["visits"] = 32 * (240 if m.group(1) == "2" and d["ck"] == 240 else 120) / int(m.group(2)); d["home_in_eval"] = 0
        else: d["nq_train"] = 750; d["visits"] = 32 * min(d["ck"], 10**6) / 750; d["home_in_eval"] = 1
        for part in ("home", "new"):
            for k, v in r[part].items():
                if k != "o": d[f"{k}_{part}"] = v
            dd = (r["desc"] or {}).get(part)
            for k in ("FA", "FR", "icc_FA", "icc_FR", "n_w", "n_r", "miss"):
                d[f"{k}_{part}"] = dd.get(k, np.nan) if dd else np.nan
        for k, v in r["log"].items(): d[k] = v
        for k, v in r.get("cross", {}).items(): d["x_" + k] = v
        for k, v in P.get("cross_u", {}).items(): d["x" + k] = v
        rows.append(d)
T = pd.DataFrame(rows)
# uniform-grader answer threshold on P(right): answer iff p (1 - FR) + (1 - p) FA > c
T["tstar"] = (T["c"] - T["FA_home"]) / (1 - T["FA_home"] - T["FR_home"])
T["tstar_c"] = T["tstar"].clip(0, 1)
# reference: gold runs, same st, ds, kl, c, standard 750-question training, final ck 120, strict format
gold = T[(T.fam == "gold") & (T.home_in_eval == 1) & (T.ck == 120) & (T.fmt == "1") & (T.temp == 0.8) & (T.adv == "std") & (T.steps.isin([120]) | T.tag.str.startswith("FK"))]
meas = [c for c in T.columns if c.endswith(("_home", "_new")) and T[c].dtype != object and not c.startswith(("FA_", "FR_", "icc_", "n_w", "n_r", "miss", "xu_"))]
ref = gold.groupby(["st", "ds", "kl", "c"])[meas].mean()
nref = gold.groupby(["st", "ds", "kl", "c"]).size()
keys = list(zip(T.st, T.ds, T.kl, T.c))
for mcol in meas:
    T[f"ref_{mcol}"] = [ref.loc[k, mcol] if k in ref.index else np.nan for k in keys]
T["n_ref"] = [int(nref.loc[k]) if k in nref.index else 0 for k in keys]
T.to_pickle(f"{OUT}/runs_table.pkl"); T.drop(columns=[]).to_csv(f"{OUT}/runs_table.csv", index=False)
print(T.shape); print(T.groupby(["st"]).size()); print(T.groupby("fam").size())
print("runs with a gold reference:", (T.n_ref > 0).sum())
print(nref)
