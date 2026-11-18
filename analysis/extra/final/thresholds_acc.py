"""Accuracy-matched cut ("acc"): the same definition as thresholds2.py (quantile of the judge's P(Yes) over the untrained
student's valid, non-IDK sampled answers on TRAINED questions, rounded to three decimals) at the target acceptance a = the share of those answers the
key accepts (key_accepts in thresholds_<J>.json, strict key). At this cut a perfect grader would reject no right answer. Writes thresholds_acc.json
{"<J>": {"<st>_<ds>": {"key_accepts", "t_acc", "A_acc", "FP_acc", "FN_acc"}}}. Usage: python3 thresholds_acc.py"""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, numpy as np
sys.path.insert(0, (_RX + "/common"))
from rc import *
out = {}
for TAG in ("Q", "M", "G", "O"):
    out[TAG] = {}
    for ds in ("pqa",):
        probs, H1, HELD = setup(ds)
        for st in ("qwen7", "llama8"):
            f = f"{BASE}/score_{TAG}_{st}_{ds}_1500_n32_s1.jsonl"
            S = {(r["idx"], r["s"]): (r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5) for r in map(json.loads, open(f))}
            P, ok = [], []
            for r in map(json.loads, open(f"{BASE}/{st}_{ds}_1500_n32_s1.jsonl")):
                if r["s"] < 0 or r["idx"] not in H1: continue
                x = sa(r)
                if not x or IDK.match(x) or (r["idx"], r["s"]) not in S: continue
                P.append(S[(r["idx"], r["s"])]); ok.append(bool(C.tq_strict(x, set(probs[r["idx"]]["aliases"]))))
            P = np.array(P); ok = np.array(ok); a = float(ok.mean())
            t = float(np.clip(round(float(np.quantile(P, 1 - a)), 3), 0.001, 0.999))
            out[TAG][f"{st}_{ds}"] = {"key_accepts": a, "t_acc": t, "A_acc": float((P > t).mean()), "FP_acc": float((P[~ok] > t).mean()), "FN_acc": float((P[ok] <= t).mean())}
            print(f"{TAG} {st} {ds}: key accepts {a:.3f}; t_acc {t:.3f}; achieved {(P > t).mean():.3f}; accepts {(P[~ok] > t).mean():.3f} of key-wrong, rejects {(P[ok] <= t).mean():.3f} of key-right")
json.dump(out, open((_RX + "/final/thresholds_acc.json"), "w"), indent=1)
