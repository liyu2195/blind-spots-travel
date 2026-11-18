"""(verify_rate) Flatten cache/runs_<st>_<ds>.pkl into cache/table.pkl (one row per run; per-question arrays kept in cache/qarr.pkl).
Adds: sub-grader label, 'std' flag (main grid: FMT 1, KL 0.2, 120 steps, standard 750 training questions, no forced IDK, std advantages, no extra keys).
Usage: python3 -I v2_table.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import glob, pickle, re, numpy as np, pandas as pd
HERE = (_RA + "/verify_rate")
rows, Q = [], {}
for f in sorted(glob.glob(f"{HERE}/cache/runs_*_*.pkl")):
    P = pickle.load(open(f, "rb"))
    for run, r in P["rows"].items():
        d = {k: v for k, v in r.items() if not k.startswith(("o_", "q_")) and k != "spec"}
        sp = r["spec"]; d.update({f"sp_{k}": v for k, v in sp.items() if k in ("kind", "J", "t", "fpk", "fpu", "fn", "fnk", "fnu", "mode", "knowsf", "alias", "ref")})
        d["base_qpaid_h"] = P["base_q"]["h"]; d["base_qpaid_n"] = P["base_q"]["n"]
        fam = r["fam"]
        if fam == "judge":
            tag = "a35" if "a35" in run else ("a50" if "a50" in run else ("t50" if "t50" in run else "t?"))
            sub = f"judge{sp['J']}{tag}"
        elif fam in ("gold", "uniFP", "uniFN", "uniMix"):
            sub = "lenQ" if "lenQ" in run else f"FP{round(sp['fpk']*100):02d}FN{round(sp['fn']*100):02d}"
        elif fam == "frz": sub = "frz" + re.search(r"frz([A-Z]+)", sp["alias"]).group(1)
        elif fam == "api": sub = "apiS55"
        else:
            m = re.match(r"^[a-z0-9]+_[A-Z]+\d*([A-Za-z0-9]+?)(f1)?_(idkc|flagc)", run); sub = m.group(1) if m else fam
        d["sub"] = sub
        rows.append(d); Q[run] = dict(q_mean=r.get("q_mean"), q_best=r.get("q_best"), o_h=r["o_h"], o_n=r["o_n"])
T = pd.DataFrame(rows)
T["std"] = (T.fmt == "1") & (T.kl == 0.2) & (T.steps == 120) & T.home_std & (T.force == 0) & (T.adv == "std") & (T.other_keys.apply(len) == 0) & T.tag.str.startswith("FK") \
           & ~T.fam.isin(["vote"]) & ~T.run.str.contains("f1_") & (T.sp_ref != True)
T["uni"] = T.fam.isin(["gold", "uniFP", "uniFN", "uniMix"])
T.to_pickle(f"{HERE}/cache/table.pkl"); pickle.dump(Q, open(f"{HERE}/cache/qarr.pkl", "wb"))
M = T[T["std"] & T.st.isin(["qwen7", "llama8"]) & (T.c == 0.5)]
print("runs", len(T), "| main grid (qwen7/llama8, std, c .5):", len(M))
print(M.groupby(["fam", "st"]).size().unstack().fillna(0).astype(int).to_string())
print("tags in main grid:", M.tag.value_counts().to_dict())
