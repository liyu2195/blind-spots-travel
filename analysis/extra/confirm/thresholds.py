"""Label-free thresholds of the open judge and its acceptance rates.
For each (student, dataset): P = P(Yes) / (P(Yes) + P(No)) of the judge for the untrained student's valid, non-IDK sampled answers
on TRAINED questions (32 samples per question). Threshold for target acceptance a = the (1 - a) quantile of P, rounded to three
decimals, kept inside [0.001, 0.999]. The trainer accepts at P > threshold. Labels (the key) are used only in the report columns."""
import os as _os  # paths below are relative to the repository root (the folder that holds .repo_root)
_ROOT = _os.environ.get("BLINDSPOTS_ROOT") or next(_d for _d in (_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd(), *[".."] * _k)) for _k in range(9)) if _os.path.exists(_os.path.join(_d, ".repo_root")))
_RA, _RP, _RX, _RR, _RD, _RC, _RT = (_ROOT + _s for _s in ("/analysis", "/paper", "/analysis/extra", "/runs", "/data", "/training/code", "/training"))
import sys, os, json, numpy as np
sys.path.insert(0, (_RX + "/common"))
from rc import *
out = {}
for ds in ("tqa", "pqa", "nqo"):
    probs, H1, HELD = setup(ds)
    for st in ("qwen7", "llama8"):
        f = f"{BASE}/score_Q_{st}_{ds}_1500_n32_s1.jsonl"
        if not os.path.exists(f): print(f"{st} {ds}: no score file yet"); continue
        S = {(r["idx"], r["s"]): (r["py"] / (r["py"] + r["pn"]) if r["py"] + r["pn"] > 1e-3 else 0.5) for r in map(json.loads, open(f))}
        P, ok = [], []; miss = 0
        for r in map(json.loads, open(f"{BASE}/{st}_{ds}_1500_n32_s1.jsonl")):
            if r["s"] < 0 or r["idx"] not in H1: continue
            x = sa(r)
            if not x or IDK.match(x): continue
            if (r["idx"], r["s"]) not in S: miss += 1; continue
            P.append(S[(r["idx"], r["s"])]); ok.append(bool(C.tq_strict(x, set(probs[r["idx"]]["aliases"]))))
        P = np.array(P); ok = np.array(ok)
        row = {"n": int(len(P)), "key_accepts": float(ok.mean()), "A_t50": float((P > 0.5).mean())}
        line = f"{st} {ds}: {len(P)} valid answers ({miss} without a score); key accepts {ok.mean():.3f}; judge at 0.5 accepts {(P > 0.5).mean():.3f} (accepts {(P[~ok] > 0.5).mean():.3f} of key-wrong, rejects {(P[ok] <= 0.5).mean():.3f} of key-right)"
        for a in (0.50, 0.35):
            t = float(np.clip(round(float(np.quantile(P, 1 - a)), 3), 0.001, 0.999)); A = float((P > t).mean())
            row[f"t_a{int(a * 100)}"] = t; row[f"A_a{int(a * 100)}"] = A
            line += f" | target {a:.2f}: threshold {t:.3f}, achieved {A:.3f} (accepts {(P[~ok] > t).mean():.3f} of key-wrong, rejects {(P[ok] <= t).mean():.3f} of key-right)"
        print(line); out[f"{st}_{ds}"] = row
json.dump(out, open((_RX + "/confirm/thresholds.json"), "w"), indent=1)
