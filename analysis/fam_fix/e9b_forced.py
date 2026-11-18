"""(fam_fix) e9b_forced: forced rejection at each operating point under three truth standards. Pool = the untrained student's valid, non-IDK sampled
answers (32 per question) on TRAINING questions (the pool the cuts are set on). right = strict key / loose key / loose key or Sonnet audit >= 4
(unaudited answers counted by the key alone; audited share printed). For a grader that accepts a share A of the pool, the least share of right
answers it must reject is floor = max(0, 1 - A / right). Per judge and cut: A, floor under each standard, and the share of audited-right answers the
judge actually rejects. Cuts: t50 (P(Yes) > 0.5), key-matched (final/thresholds_acc.json, PopQA only), a50, a35 (final/thresholds_<J>.json).
Output: e9b_forced_out.txt"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, json, pickle, numpy as np
sys.path.insert(0, (_RA + "/stats"))
from s00_common import *
TA = json.load(open((_RX + "/final/thresholds_acc.json"))); out = [__doc__]
for ds in ("pq", "h1", "nq"):
    for st in ("qwen7", "llama8"):
        B = pickle.load(open(f"{CACHE}/base_{st}_{ds}.pkl", "rb")); tr = np.asarray(B["train"], bool) if "train" in B else ~np.asarray(B["new"], bool)
        v = (np.asarray(B["valid"]) & ~np.asarray(B["idk"]))[tr][:, 1:]; ks = np.asarray(B["ks"])[tr][:, 1:][v]; kl = np.asarray(B["kl"])[tr][:, 1:][v]
        son = np.asarray(B["son"])[tr][:, 1:][v]; au = kl | (son >= 4)
        out.append(f"\n== {st} {ds}: pool {int(v.sum())} answers; right by strict key {ks.mean():.3f}, loose key {kl.mean():.3f}, key or audit {au.mean():.3f} (audited {np.mean(son > 0):.3f})")
        PJ = B["PJ"]
        for J in "QMGO":
            if J not in PJ: continue
            p = np.asarray(PJ[J])[tr][:, 1:][v]; TH = thresholds(J); key = f"{st}_{DSN[ds]}"
            cuts = [("t50", 0.5)] + ([("key", TA[J][key]["t_acc"])] if key in TA.get(J, {}) else []) + [(c, TH[key]["t_" + c]) for c in ("a50", "a35") if key in TH and "t_" + c in TH[key]]
            for cn, t in cuts:
                ok = np.isfinite(p); A = float((p[ok] > t).mean()); fl = lambda r: max(0.0, 1 - A / r)
                out.append(f"   judge {J} {cn:3s} (P > {t:.3f}): accepts {A:.3f} | least share of right answers rejected: strict key {fl(ks.mean()):.3f}, loose key {fl(kl.mean()):.3f}, key or audit {fl(au.mean()):.3f} | judge rejects {float((p[ok & au] <= t).mean()):.3f} of audited-right and accepts {float((p[ok & ~au] > t).mean()):.3f} of audited-wrong")
open("e9b_forced_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[1:]))
