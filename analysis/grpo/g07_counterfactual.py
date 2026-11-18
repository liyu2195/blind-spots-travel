"""(grpo) Counterfactual predictions of the same mechanistic model (g05_model, rule 'conf', threshold 1/2, zero fitted constants) for every
distinct grader configuration of the KL-0.2 strict-format runs, with the credit c and the advantage normaliser changed:
c in {0.3, 0.5, 0.7, 0.85}, normaliser in {std, none}, KL 0.2. Also the share of IDK-containing groups that have three reward levels
(IDK + accepted + rejected answers: the only groups in which the std-normalised advantage depends on c), at the model's stationary policy.
Usage: nice -n 10 python3 -I g07_counterfactual.py <st> <ds>  -> cache/cf_<st>_<ds>.pkl"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, hashlib, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
src = open((_RA + "/grpo/g05_model.py")).read().split("out = {}")[0]
exec(src)
CS = (0.3, 0.5, 0.7, 0.85)
def three_level_share(p, alpha):
    """P(group has >= 1 IDK, >= 1 accepted, >= 1 rejected) / P(group has >= 1 IDK and is not all IDK), k = 8, per question."""
    q1 = (1 - p) * alpha; q0 = (1 - p) * (1 - alpha)
    three = 1 - (1 - p) ** 8 - (1 - q1) ** 8 - (1 - q0) ** 8 + q0 ** 8 + q1 ** 8 + p ** 8     # inclusion-exclusion: >= 1 of each kind
    mixed_idk = 1 - (1 - p) ** 8 - p ** 8                                                        # >= 1 IDK and >= 1 answer
    return three, mixed_idk
seen = {}; out = {}
for run, r in sorted(D["rows"].items()):
    if r["alpha"] is None or r["kl"] != 0.2 or r["fmt"] != "1" or r["steps"] != 120 or not r["std_home"] or r["force"] or r["switch"] or r["prompt"] != "idk" or r["temp"] != 0.8 or r["lr"] != 5e-5:
        continue
    a = np.where(np.isfinite(r["alpha"]), r["alpha"], 0.5).astype(float)
    h = hashlib.md5(a.tobytes()).hexdigest()
    if h in seen: out[run] = dict(same_as=seen[h]); continue
    seen[h] = run; pref, agr = FE[r["fmt"]]; rec = dict(fam=r["fam"])
    for norm in ("std", "none"):
        for c in CS:
            p, it = solve(a, pref, agr, c, norm, 0.2, "conf")
            rec[(norm, c, "n")] = float((p[new] > 0.5).mean()); rec[(norm, c, "h")] = float((p[home] > 0.5).mean())
            th, mi = three_level_share(p[home], a[home]); rec[(norm, c, "three")] = float(th.sum() / max(mi.sum(), 1e-12))
    out[run] = rec
pickle.dump(out, open(f"{HERE}/cache/cf_{st}_{ds}.pkl", "wb"))
print(f"{st} {ds}: {len(seen)} distinct grader configurations, {len(out)} runs")
