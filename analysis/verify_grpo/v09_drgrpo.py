"""(verify_grpo) Risky test 3: the theory's zero-constant prediction of what removing the std normaliser does (Dr. GRPO, FK20D) for the same student,
grader, credit and seed, against the actual change; also the per-question escape threshold it implies. Small students (the only Dr. GRPO runs with KL).
Usage: python3 -I v09_drgrpo.py > out_v09_drgrpo.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import pandas as pd, numpy as np
T = pd.read_csv((_RA + "/verify_grpo/eval_v.csv")); T["fmt"] = T.fmt.astype(str)
base = (T.kl == 0.2) & (T.fmt == "1") & (T.steps == 120) & (T.temp == 0.8) & T.st.isin(["phi4m", "smol3"])
T["grader"] = T.run.str.extract(r"_FK20D?(sim\d\dfn\d\d)_")[0]
dr = T[base & (T.adv == "none")]; sd = T[base & (T.adv == "std")]
rows = []
for r in dr.itertuples():
    m = sd[(sd.st == r.st) & (sd.grader == r.grader) & (sd.c == r.c) & (sd.seed == r.seed)]
    if len(m) == 0: m = sd[(sd.st == r.st) & (sd.grader == r.grader) & (sd.c == r.c)]
    if len(m) == 0: continue
    rows.append(dict(run=r.run, c=r.c, act_dr=r.y_n, act_std=m.y_n.mean(), n_std=len(m), pred_dr=r.conf_n_half, pred_std=m.conf_n_half.mean()))
R = pd.DataFrame(rows); R["act_d"] = R.act_dr - R.act_std; R["pred_d"] = R.pred_dr - R.pred_std
print(R.round(3).to_string(index=False))
print(f"\nmatched pairs {len(R)}: actual change (Dr. GRPO - std) mean {R.act_d.mean():+.3f}; predicted {R.pred_d.mean():+.3f}; sign agreement {int((np.sign(R.act_d) == np.sign(R.pred_d)).sum())}/{len(R)}; "
      f"corr {np.corrcoef(R.act_d, R.pred_d)[0, 1]:+.2f}; MAE of the change {np.mean(np.abs(R.act_d - R.pred_d)):.3f}")
